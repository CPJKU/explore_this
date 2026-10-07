from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from beat_this.batchable_dataclass import BatchableDataclass
from beat_this.model.grid_space_embedding import space_size, advance_index, is_hit

from beat_this.viterbi import viterbi


@dataclass
class SubgridOutput(BatchableDataclass):
    # Sparse (B, T) tensors of probabilities of beat/downbeats, masked by the grid mask
    b_activation: Tensor
    db_activation: Tensor

    features: Tensor
    beat_mask: Tensor
    downbeat_mask: Tensor


class BeatSubgrid(nn.Module):
    def __init__(
        self,
        max_beat_meter: int = 4,
        max_downbeat_meter: int = 7,
        mask_method: Literal["threshold"] | Literal["viterbi"] = "threshold",
        mask_threshold=0.5,
        viterbi_transition_prob=0.1,
    ) -> None:
        super().__init__()

        # Calculate size
        beat_space_size = space_size(max_beat_meter)
        beat_indices = torch.arange(beat_space_size)
        advanced_beat_indices = advance_index(beat_indices)
        advances_to_beat = is_hit(advanced_beat_indices)

        downbeat_space_size = space_size(max_downbeat_meter)
        downbeat_indices = torch.arange(downbeat_space_size)

        combined_map = (downbeat_indices * beat_space_size)[
            :, None
        ] + advanced_beat_indices[None, :]

        combined_map[:, advances_to_beat] += (
            (advance_index(downbeat_indices) - downbeat_indices) * beat_space_size
        )[:, None]

        combined_map = combined_map.reshape(-1)
        is_beat = is_hit(beat_indices).repeat((downbeat_space_size, 1)).reshape(-1)

        is_downbeat = (
            is_hit(downbeat_indices)[:, None].repeat((1, beat_space_size)).reshape(-1)
        )
        is_downbeat = is_downbeat * is_beat

        self.max_beat_meter = max_beat_meter
        self.max_downbeat_meter = max_downbeat_meter

        self.beat_space_size = beat_space_size
        self.downbeat_space_size = downbeat_space_size
        self.input_size = beat_space_size * downbeat_space_size

        self.register_buffer("next_index", combined_map, persistent=False)  # int tensor

        self.register_buffer("is_beat", is_beat, persistent=False)  # bool tensor
        self.register_buffer("is_downbeat", is_downbeat, persistent=False)

        self.mask_method = mask_method
        self.mask_threshold = mask_threshold
        self.viterbi_transition_prob = max(viterbi_transition_prob, 1e-9)

    def split_combined_state(self, x) -> tuple[int, int]:
        """Returns the beat state index and downbeat state index, that can then be
        interpreted like in grid_space_embedding.py"""
        dbstate = x // self.beat_space_size
        bstate = x - dbstate * self.beat_space_size
        return bstate, dbstate

    def calculate_masks(self, x: Tensor, grid_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Calculate beat and downbeat masks from the grid mask. These are (B, T) boolean tensors."""

        # PLACEHOLDER
        if self.mask_method == "threshold":
            b_res = (x.softmax(dim=-1) * self.is_beat).sum(dim=-1)
            b_res = b_res * grid_mask

            db_res = (x.softmax(dim=-1) * self.is_downbeat).sum(dim=-1)
            db_res = db_res * grid_mask

            return b_res > self.mask_threshold, db_res > self.mask_threshold

        elif self.mask_method == "viterbi":
            # Transfer next_index to a transition matrix
            transitions = torch.zeros(
                (self.input_size, self.input_size), device=x.device
            )
            transitions[torch.arange(self.input_size), self.next_index] = 1
            transitions = (
                transitions * (1 - self.viterbi_transition_prob)
                + self.viterbi_transition_prob / self.input_size
            ).log()

            b_res = torch.zeros_like(grid_mask)
            sb_res = torch.zeros_like(grid_mask)

            for b in range(x.shape[0]):
                categoricals = x[b, grid_mask[b]]
                path = viterbi(categoricals, transitions)
                b_mask = self.is_beat[path]
                db_mask = self.is_downbeat[path]

                b_res[b, grid_mask[b]] = b_mask
                sb_res[b, grid_mask[b]] = db_mask

            return b_res.bool(), sb_res.bool()

        else:
            raise ValueError(f"Unknown mask method {self.mask_method}")

    def forward(self, x: Tensor, mask: Tensor) -> SubgridOutput:
        """
        x is (B, T, C) of logits
        mask is (B, T)
        """

        b_res = (x.softmax(dim=-1) * self.is_beat).sum(dim=-1)
        b_res = b_res * mask

        db_res = (x.softmax(dim=-1) * self.is_downbeat).sum(dim=-1)
        db_res = db_res * mask

        b_mask, db_mask = self.calculate_masks(x, mask)

        return SubgridOutput(
            b_activation=b_res,
            db_activation=db_res,
            features=x,
            beat_mask=b_mask,
            downbeat_mask=db_mask,
        )

    def backtrack_categorical(self, x):
        """
        x is (..., C)
        Find the previous catogorical that should have come before x.
        """
        return x[..., self.next_index]


class BeatSubgridPredictionLoss(nn.Module):
    """
    Takes in hit probabilities (B, T), the grid mask (B, T) and the ground truth (B, T).
    Finds which grid points best correspond to the ground truth, and uses those as positive targets.
    Then, compute difference between hit proabilities and these new targets.
    """

    def __init__(
        self,
        subgrid_processor: BeatSubgrid,
        downbeat_weight=1.0,
    ) -> None:
        super().__init__()
        self.subgrid_processor = subgrid_processor
        self.downbeat_weight = downbeat_weight
        # self.bce_loss = nn.BCELoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(
        self,
        b_activation,
        db_activation,
        grid_mask,
        b_truth,
        db_truth,
    ):
        """
        b_activation, db_activation is (B, T) of sparse probabilities,
        grid_mask is (B, T) boolean,
        *_truth is (B, T) boolean.
        For every true beat, we try to find a matching grid point within a certain leniency (e.g. 3 frames) and use that as a positive target.
        We then compute the loss as the difference between the hit probabilities and these new targets, masked by the grid mask.

        Currently not scaled by the number of grid-points
        """

        # (B, T)
        shifted_beats = assign_fast(b_truth, grid_mask)
        shifted_downbeats = assign_fast(db_truth, grid_mask)

        # Compute loss between shifted targets and hit probabilities, masked by grid mask
        sparse_b_loss = self.mse_loss(b_activation, shifted_beats.float()).mean(dim=-1)

        # Compute loss on db, but ignore excerpts without downbeats
        sparse_db_loss = self.mse_loss(db_activation, shifted_downbeats.float()).mean(
            dim=-1
        )

        # return sparse_b_loss.mean() + sparse_db_loss.mean()

        beats_per_piece = shifted_beats.sum(dim=-1)  # B
        downbeats_per_piece = shifted_downbeats.sum(dim=-1)  # B

        db_scaling = torch.where(
            (downbeats_per_piece > 0),
            (beats_per_piece / downbeats_per_piece),
            downbeats_per_piece,
        )  # b

        masked_loss = sparse_b_loss + sparse_db_loss * db_scaling * self.downbeat_weight
        return masked_loss.mean()


# Generated by ChatGPT
def assign_fast(x, y):
    B, T = x.shape

    z = torch.zeros_like(x)

    for b in range(B):
        y_pos = torch.where(y[b])[0]
        x_pos = torch.where(x[b])[0]

        if len(y_pos) == 0 or len(x_pos) == 0:
            continue

        # nearest via broadcasting
        dist = torch.abs(x_pos[:, None] - y_pos[None, :])
        nearest = y_pos[dist.argmin(dim=1)]

        z[b, nearest] = 1

    return z


class BeatSubgridRegularizationLoss(nn.Module):
    """
    Take the subgrid (B, T, C) and the grid mask (B, T)
    and evaluate the internal consistensy of the subgrid.
    """

    def __init__(
        self,
        subgrid_processor: BeatSubgrid,
        scale=1e-2,
    ) -> None:
        super().__init__()
        self.subgrid_processor = subgrid_processor
        self.scale = scale
        self.loss = nn.KLDivLoss(reduction="batchmean")
        # self.loss = nn.MSELoss()

    def forward(self, subgrid_features: Tensor, mask: Tensor) -> Tensor:
        """
        subgrid is (B, T, C) and mask is (B, T)
        1. Evaluate the subgrid at each mask point.
        2. Compare the difference between the predicted (advanced and shifted to the next grid point) categoricals and the actual ones.
        """
        B, T, C = subgrid_features.shape

        loss = torch.zeros((B,), device=mask.device)
        for b in range(B):
            m = mask[b]  # (T,)
            if m.sum() <= 1:
                continue

            s = subgrid_features[b]  # (T, C)

            categoricals = torch.softmax(s[m, :], dim=-1)  # [...,C]
            backtracked_categoricals = self.subgrid_processor.backtrack_categorical(
                categoricals
            )  # [..., C]

            # Downbeats are allowed to change to other downbeats, so we spread the probability of all downbeats evenly between each other
            mean_downbeat_prob_1 = categoricals[
                ..., self.subgrid_processor.is_downbeat
            ].mean(dim=-1)

            categoricals = categoricals.clone()
            categoricals[..., self.subgrid_processor.is_downbeat] = (
                mean_downbeat_prob_1[:, None]
            )

            mean_downbeat_prob_2 = backtracked_categoricals[
                ..., self.subgrid_processor.is_downbeat
            ].mean(dim=-1)
            backtracked_categoricals = backtracked_categoricals.clone()
            backtracked_categoricals[..., self.subgrid_processor.is_downbeat] = (
                mean_downbeat_prob_2[:, None]
            )

            # Only care about the matching ones
            categoricals = categoricals[:-1]
            backtracked_categoricals = backtracked_categoricals[1:]

            # Batchmean -> mean across first axis (grid point idx)
            # and sum across the second axis (categorical dimension)
            loss[b] = self.loss(
                torch.log(backtracked_categoricals + 1e-9), categoricals
            )

        return loss.mean() * self.scale
