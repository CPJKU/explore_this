"""
Model definitions for the Explore This! beat tracker.
"""

from collections import OrderedDict
from dataclasses import dataclass

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor, nn

from explore_this.batchable_dataclass import BatchableDataclass
from explore_this.model import roformer
from explore_this.model.grid import GridOutput, WindowedGrid
from explore_this.model.subgrid import BeatSubgrid, SubgridOutput
from explore_this.utils import replace_state_dict_key


@dataclass
class ModelOutput(BatchableDataclass):
    grid_features: Tensor
    grid_activations: Tensor

    grid_mask: Tensor

    subgrid_features: Tensor

    beat_activation: Tensor
    beat_mask: Tensor

    downbeat_activation: Tensor
    downbeat_mask: Tensor

    def get_masked_subgrid_features(self, index):
        m = self.grid_mask[index]
        return self.subgrid_features[index][m]


class ExploreThis(nn.Module):
    """
    A neural network model for beat tracking. It is composed of three main components:
    - a frontend that processes the input spectrogram,
    - a series of transformer blocks that process the output of the frontend,
    - a head that produces the final beat and downbeat predictions.

    Args:
        spect_dim (int): The dimension of the input spectrogram (default: 128).
        transformer_dim (int): The dimension of the main transformer blocks (default: 512).
        ff_mult (int): The multiplier for the feed-forward dimension in the transformer blocks (default: 4).
        n_layers (int): The number of transformer blocks (default: 6).
        head_dim (int): The dimension of each attention head for the partial transformers in the frontend and the transformer blocks (default: 32).
        stem_dim (int): The out dimension of the stem convolutional layer (default: 32).
        dropout (dict): A dictionary specifying the dropout rates for different parts of the model
            (default: {"frontend": 0.1, "transformer": 0.2}).
        sum_head (bool): Whether to use a SumHead for the final predictions (default: True) or plain independent projections.
        partial_transformers (bool): Whether to include partial frequency- and time-transformers in the frontend (default: True)
    """

    def __init__(
        self,
        spect_dim: int = 128,
        transformer_dim: int = 512,
        ff_mult: int = 4,
        n_layers: int = 6,
        head_dim: int = 32,
        stem_dim: int = 32,
        dropout: dict = {"frontend": 0.1, "transformer": 0.2},
        partial_transformers: bool = True,
        # Grid stuff
        grid_window_size: int = 50,
        # 400 bpm means ~7 bps which means 7/50 ~= 1/7 beats per 20ms. max_freq=0.2 should be safe, it corresponds to 600 bpm
        # that means min_freq is around 0.01 (1 peak per 2 seconds)
        grid_min_freq: float = 0.01,
        grid_bins_per_octave: int = 3,
        grid_pred_prob_spread: float = 0.05,
        grid_half_crossfade_frames: int = 4,
        # Subgrid stuff
        use_subgrid: bool = True,
        subgrid_transformer_layers=3,
        max_subgrid_meter: int = 4,
        max_subgrid_downbeat_meter: int = 15,
        # TESTING
        kaiming_init_grid=True,
        kaiming_grid_last_factor=0.01,
    ):
        super().__init__()
        self.use_subgrid = use_subgrid
        # shared rotary embedding for frontend blocks and transformer blocks
        rotary_embed = RotaryEmbedding(head_dim)

        # create the frontend
        # - stem
        stem = self.make_stem(spect_dim, stem_dim)
        spect_dim //= 4  # frequencies were convolved with stride 4
        # - three frontend blocks
        frontend_blocks = []
        dim = stem_dim
        for _ in range(3):
            frontend_blocks.append(
                self.make_frontend_block(
                    dim,
                    dim * 2,
                    partial_transformers,
                    head_dim,
                    rotary_embed,
                    dropout["frontend"],
                )
            )
            dim *= 2
            spect_dim //= 2  # frequencies were convolved with stride 2
        frontend_blocks = nn.Sequential(*frontend_blocks)
        # - linear projection to transformer dimensionality
        concat = Rearrange("b c f t -> b t (c f)")
        linear = nn.Linear(dim * spect_dim, transformer_dim)
        self.frontend = nn.Sequential(
            OrderedDict(stem=stem, blocks=frontend_blocks, concat=concat, linear=linear)
        )

        # create the transformer blocks
        assert transformer_dim % head_dim == 0, (
            "transformer_dim must be divisible by head_dim"
        )
        n_heads = transformer_dim // head_dim

        self.transformer_blocks = roformer.Transformer(
            dim=transformer_dim,
            depth=n_layers,
            heads=n_heads,
            attn_dropout=dropout["transformer"],
            ff_dropout=dropout["transformer"],
            rotary_embed=rotary_embed,
            ff_mult=ff_mult,
            dim_head=head_dim,
            norm_output=True,
        )

        # Create grid
        self.grid_processor = WindowedGrid(
            window_size=grid_window_size,
            min_freq=grid_min_freq,
            bins_per_octave=grid_bins_per_octave,
            gradient_spread=grid_pred_prob_spread,
            half_crossfade_frames=grid_half_crossfade_frames,
        )

        self.grid_block = GridBlock(
            grid_processor=self.grid_processor,
            input_channels=transformer_dim,
        )

        # Create subgrid
        if use_subgrid:
            # It is important that this is named something with subgrid because state-dict is filtered on the name later
            self.subgrid_block = SubgridBlock(
                max_beat_meter=max_subgrid_meter,
                max_downbeat_meter=max_subgrid_downbeat_meter,
                transformer_dim=transformer_dim,
                n_layers=subgrid_transformer_layers,
                dropout=dropout,
                ff_mult=ff_mult,
                head_dim=head_dim,
                n_heads=n_heads,
                rotary_embed=rotary_embed,
            )

            self.subgrid_processor = self.subgrid_block.subgrid_processor

        # init all weights
        with torch.no_grad():
            self.apply(self._init_weights)
            if kaiming_init_grid:
                self.grid_block.init_weights(kaiming_grid_last_factor)

    @staticmethod
    def make_stem(spect_dim: int, stem_dim: int) -> nn.Module:
        return nn.Sequential(
            OrderedDict(
                rearrange_tf=Rearrange("b t f -> b f t"),
                bn1d=nn.BatchNorm1d(spect_dim),
                add_channel=Rearrange("b f t -> b 1 f t"),
                conv2d=nn.Conv2d(
                    in_channels=1,
                    out_channels=stem_dim,
                    kernel_size=(4, 3),
                    stride=(4, 1),
                    padding=(0, 1),
                    bias=False,
                ),
                bn2d=nn.BatchNorm2d(stem_dim),
                activation=nn.GELU(),
            )
        )

    @staticmethod
    def make_frontend_block(
        in_dim: int,
        out_dim: int,
        partial_transformers: bool = True,
        head_dim: int | None = 32,
        rotary_embed: RotaryEmbedding | None = None,
        dropout: float = 0.1,
    ) -> nn.Module:
        if partial_transformers:
            if head_dim is None or rotary_embed is None:
                raise ValueError(
                    "Must specify head_dim and rotary_embed for using partial_transformers"
                )
            partial = PartialFTTransformer(
                dim=in_dim,
                dim_head=head_dim,
                n_head=in_dim // head_dim,
                rotary_embed=rotary_embed,
                dropout=dropout,
            )
        else:
            partial = nn.Identity()

        return nn.Sequential(
            OrderedDict(
                partial=partial,
                # conv block
                conv2d=nn.Conv2d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=(2, 3),
                    stride=(2, 1),
                    padding=(0, 1),
                    bias=False,
                ),
                # out_channels : 64, 128, 256
                # freqs : 16, 8, 4 (due to the stride=2)
                norm=nn.BatchNorm2d(out_dim),
                activation=nn.GELU(),
            )
        )

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def pre_grid_forward(self, x: Tensor) -> Tensor:
        x1 = self.frontend(x)
        x2 = self.transformer_blocks(x1)

        # if torch.any(torch.isnan(x1)):
        #     print(f"{x.mean(dim=(1,2))=}")
        #     print(f"{x.amin(dim=(1,2))=}")
        #     print(f"{x.amax(dim=(1,2))=}")
        #     print(f"{x1.mean(dim=(1,2))=}")
        #     raise ValueError("nan after frontend")
        # if torch.any(torch.isnan(x2)):
        #     print(f"{x.mean(dim=(1,2))=}")
        #     print(f"{x1.mean(dim=(1,2))=}")
        #     print(f"{x2.mean(dim=(1,2))=}")
        #     raise ValueError("nan after transformers")

        return x2

    def grid_forward(self, x: Tensor) -> GridOutput:
        x1 = self.grid_block.forward(x)
        # if x1.hasnan():
        #     print(f"{x.mean(dim=(1,2))=}")
        #     print(f"{x1.activation.mean(dim=(1))=}")
        #     print(f"{x1.features.mean(dim=(1,2,3))=}")
        #     print(f"{x1.mask.mean(dim=(1))=}")
        #     raise ValueError("nan after grid_forward")
        return x1

    def subgrid_forward(self, signal: Tensor, grid: GridOutput) -> SubgridOutput:
        x1 = self.subgrid_block.forward(
            signal,
            grid.mask,
            grid.waves,
        )

        # if x1.hasnan():
        #     print(f"{signal.mean(dim=(1,2))=}")
        #     print(f"{grid.mask.sum(dim=1)=}")
        #     print(f"{grid.waves.mean(dim=(1,2))=}")
        #     print(f"{grid.activation.mean(dim=1)=}")
        #     print(f"{x1.features.mean(dim=(1,2))=}")
        #     print(f"{x1.b_activation.mean(dim=(1))=}")
        #     print(f"{x1.beat_mask.sum(dim=(1))=}")
        #     print(f"{x1.db_activation.mean(dim=(1))=}")
        #     print(f"{x1.downbeat_mask.sum(dim=(1))=}")
        #     raise ValueError("nan after subgrid_forward")
        return x1

    def forward(self, x, detach_pregrid=False) -> ModelOutput:
        x = self.pre_grid_forward(x)
        if detach_pregrid:
            x = x.detach()
        y = self.grid_forward(x)

        if not self.use_subgrid:
            return ModelOutput(
                grid_features=y.features,
                grid_activations=y.activation,
                grid_mask=y.mask,
                beat_mask=None,  # type: ignore
                beat_activation=None,  # type: ignore
                downbeat_mask=None,  # type: ignore
                downbeat_activation=None,  # type: ignore
                subgrid_features=None,  # type: ignore
            )

        z = self.subgrid_forward(x, y)
        return ModelOutput(
            grid_features=y.features,
            grid_activations=y.activation,
            grid_mask=y.mask,
            beat_mask=z.beat_mask,
            beat_activation=z.b_activation,
            downbeat_activation=z.db_activation,
            downbeat_mask=z.downbeat_mask,
            subgrid_features=z.features,
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        return state_dict


class PartialRoformer(nn.Module):
    """
    Takes a (batch, channels, freqs, time) input, applies self-attention and
    a feed-forward block either only across frequencies or only across time.
    Returns a tensor of the same shape as the input.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        n_head: int,
        direction: str,
        rotary_embed: RotaryEmbedding,
        dropout: float,
    ):
        super().__init__()

        assert dim % dim_head == 0, "dim must be divisible by dim_head"
        assert dim // dim_head == n_head, "n_head must be equal to dim // dim_head"
        self.direction = direction[0].lower()
        if self.direction not in "ft":
            raise ValueError(f"direction must be F or T, got {direction}")
        self.attn = roformer.Attention(
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ff = roformer.FeedForward(dim, dropout=dropout)

    def forward(self, x):
        b = len(x)
        if self.direction == "f":
            pattern = "(b t) f c"
        elif self.direction == "t":
            pattern = "(b f) t c"
        x = rearrange(x, f"b c f t -> {pattern}")
        x = x + self.attn(x)
        x = x + self.ff(x)
        x = rearrange(x, f"{pattern} -> b c f t", b=b)
        return x


class PartialFTTransformer(nn.Module):
    """
    Takes a (batch, channels, freqs, time) input, applies self-attention and
    a feed-forward block once across frequencies and once across time. Same
    as applying two PartialRoformer() in sequence, but encapsulated in a single
    module. Returns a tensor of the same shape as the input.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        n_head: int,
        rotary_embed: RotaryEmbedding,
        dropout: float,
    ):
        super().__init__()

        assert dim % dim_head == 0, "dim must be divisible by dim_head"
        assert dim // dim_head == n_head, "n_head must be equal to dim // dim_head"
        # frequency directed partial transformer
        self.attnF = roformer.Attention(
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ffF = roformer.FeedForward(dim, dropout=dropout)
        # time directed partial transformer
        self.attnT = roformer.Attention(
            dim,
            heads=n_head,
            dim_head=dim_head,
            dropout=dropout,
            rotary_embed=rotary_embed,
        )
        self.ffT = roformer.FeedForward(dim, dropout=dropout)

    def forward(self, x):
        b = len(x)
        # frequency directed partial transformer
        x = rearrange(x, "b c f t -> (b t) f c")
        x = x + self.attnF(x)
        x = x + self.ffF(x)
        # time directed partial transformer
        x = rearrange(x, "(b t) f c ->(b f) t c", b=b)
        x = x + self.attnT(x)
        x = x + self.ffT(x)
        x = rearrange(x, "(b f) t c -> b c f t", b=b)
        return x


class GridBlock(nn.Module):
    def __init__(
        self,
        grid_processor: WindowedGrid,
        input_channels,
        hidden_size=50,
        n_layers=3,
    ) -> None:
        super().__init__()
        self.window_size = grid_processor.window_size
        self.grid_processor = grid_processor
        r1 = Rearrange("b (w t) ch -> b w (t ch)", t=self.window_size)
        r2 = Rearrange("b w (c three) -> b w c three", three=3)

        if n_layers == 1:
            self.layers = nn.Sequential(
                r1,
                nn.Linear(
                    self.window_size * input_channels,
                    self.grid_processor.n_bins * 3,
                ),
                r2,
                self.grid_processor,
            )
        elif n_layers > 1:
            self.layers = nn.Sequential(
                r1,
                nn.Linear(
                    self.window_size * input_channels,
                    hidden_size,
                ),
                nn.GELU(),
                *[
                    nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.GELU(),
                    )
                    for _ in range(n_layers - 2)
                ],
                nn.Linear(
                    hidden_size,
                    self.grid_processor.n_bins * 3,
                ),
                r2,
                self.grid_processor,
            )
        else:
            raise ValueError(f"{n_layers=} is not a valid number")

    def init_weights(self, factor):
        def _init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        self.apply(_init)
        self.layers[-3].weight.mul_(factor)  # type: ignore # Final linear layer

    def forward(self, x) -> GridOutput:
        y = self.layers(x)
        # y.activation has shape b t
        return y


class SubgridBlock(nn.Module):
    def __init__(
        self,
        max_beat_meter,
        max_downbeat_meter,
        transformer_dim,
        n_layers,
        n_heads,
        dropout,
        rotary_embed,
        ff_mult,
        head_dim,
    ) -> None:
        super().__init__()
        self.max_beat_meter = max_beat_meter
        self.max_downbeat_meter = max_downbeat_meter
        self.subgrid_processor = BeatSubgrid(
            max_beat_meter,
            max_downbeat_meter,
        )

        self.grid_embedding = nn.Linear(3, transformer_dim, bias=False)
        self.transformer = roformer.Transformer(
            dim=transformer_dim,
            depth=n_layers,
            heads=n_heads,
            attn_dropout=dropout["transformer"],
            ff_dropout=dropout["transformer"],
            rotary_embed=rotary_embed,
            ff_mult=ff_mult,
            dim_head=head_dim,
            norm_output=True,
        )

        self.final_projection = nn.Linear(
            transformer_dim, self.subgrid_processor.input_size
        )

    def forward(
        self,
        signal: torch.Tensor,  # B,T,C
        mask: torch.Tensor,  # B,T
        mask_waves: torch.Tensor,  # B,T,2
    ) -> SubgridOutput:
        # B,T,3
        full_grid_signal = torch.concat((mask[..., None].float(), mask_waves), dim=-1)
        # B,T,C
        extra = self.grid_embedding(full_grid_signal)
        # B,T,C
        x = signal + extra
        x = self.transformer(x)

        x = self.final_projection(x)

        y = self.subgrid_processor.forward(x, mask)
        return y
