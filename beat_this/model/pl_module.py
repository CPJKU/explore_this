"""
Pytorch Lightning module, wraps a BeatThis model along with losses, metrics and
optimizers for training.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningModule

from beat_this.model.grid import (
    WindowedGridRegularizationLoss,
    RecallLoss,
    LowProbLoss,
)

import beat_this.metrics as metrics

from beat_this.inference import split_predict_aggregate
from beat_this.model.beat_tracker import BeatThis, ModelOutput
from beat_this.model.subgrid import (
    BeatSubgridPredictionLoss,
    BeatSubgridRegularizationLoss,
)
from beat_this.utils import replace_state_dict_key


class PLBeatThis(LightningModule):
    def __init__(
        self,
        training_type: str,  # ["full", "grid", "subgrid"]
        spect_dim=128,
        fps=50,
        transformer_dim=512,
        ff_mult=4,
        n_layers=6,
        stem_dim=32,
        dropout={"frontend": 0.1, "transformer": 0.2},
        lr=0.0008,
        weight_decay=0.01,
        # pos_weights={"beat": 1, "downbeat": 1},
        head_dim=32,
        # loss_type="shift_tolerant_weighted_bce",
        warmup_steps=1000,
        max_epochs=100,
        eval_trim_beats=5,
        partial_transformers=True,
        grid_regularization_weight=0.1,
        grid_regularization_freq_scale=0.002,
        grid_regularization_phase_factor=2,
        grid_confidence_loss_weight=1e-5,
        grid_window_size: int = 50,
        grid_half_crossfade_frames: int = 5,
        grid_min_freq: float = 0.01,
        grid_bins_per_octave: int = 3,
        grid_consistensy_prob_spread: float = 0.05,
        grid_pred_prob_spread: float = 0.05,
        grid_downbeat_weight: float = 1,
        subgrid_transformer_layers: int = 3,
        max_subgrid_meter: int = 4,
        max_subgrid_downbeat_meter: int = 15,
        subgrid_regularization_scale=1e-2,
        subgrid_loss_scale=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.training_type = training_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.fps = fps
        # create model

        self.model = BeatThis(
            spect_dim=spect_dim,
            transformer_dim=transformer_dim,
            ff_mult=ff_mult,
            stem_dim=stem_dim,
            n_layers=n_layers,
            head_dim=head_dim,
            dropout=dropout,
            partial_transformers=partial_transformers,
            grid_window_size=grid_window_size,
            grid_min_freq=grid_min_freq,
            grid_bins_per_octave=grid_bins_per_octave,
            grid_pred_prob_spread=grid_pred_prob_spread,
            grid_half_crossfade_frames=grid_half_crossfade_frames,
            use_subgrid=training_type != "grid",
            subgrid_transformer_layers=subgrid_transformer_layers,
            max_subgrid_meter=max_subgrid_meter,
            max_subgrid_downbeat_meter=max_subgrid_downbeat_meter,
        )

        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs

        grid_model = self.model.grid_processor

        if self.training_type != "subgrid":
            self.grid_recall_loss = RecallLoss(
                downbeat_weight=grid_downbeat_weight,
            )

            self.grid_reg_loss = WindowedGridRegularizationLoss(
                grid_model=grid_model,
                scale=grid_regularization_weight,
                sinkhorn_max_iter=3,
                frequency_log_step=grid_regularization_freq_scale,
                phase_factor=grid_regularization_phase_factor,
                max_pointwise_distance=2,
                exponent=2,
                probability_spread=grid_consistensy_prob_spread,
            )

            self.low_prob_loss = LowProbLoss(
                grid_model=grid_model,
                scale=grid_confidence_loss_weight,
            )

        if self.training_type != "grid":
            self.subgrid_pred_loss = BeatSubgridPredictionLoss(
                self.model.subgrid_processor
            )
            self.subgrid_consistensy_loss = BeatSubgridRegularizationLoss(
                self.model.subgrid_processor,
                scale=subgrid_regularization_scale,
            )
            self.subgrid_loss_scale = subgrid_loss_scale

        # self.eval_trim_beats = eval_trim_beats

    def load_base_model(self, base_model_path, load_subgrid=True):
        try:
            sd = torch.load(base_model_path)["state_dict"]
        except FileNotFoundError:
            print(f"Base model file not found: {base_model_path}")
            return

        if not load_subgrid:
            sd = {k: v for k, v in sd.items() if "subgrid" not in k}

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            "Unexpected keys when loading state dict for subgrid training:",
            unexpected,
        )

    def _compute_loss(self, batch, model_prediction: ModelOutput) -> dict:
        losses = {}
        total_loss: torch.Tensor = 0  # type: ignore
        if self.training_type != "subgrid":
            reg_loss = self.grid_reg_loss.forward(model_prediction.grid_features)
            low_prob_loss = self.low_prob_loss.forward(model_prediction.grid_features)
            pred_loss = self.grid_recall_loss(
                model_prediction.grid_activations,
                batch["truth_beat"],
                batch["truth_downbeat"],
            )
            grid_losses = {
                "consistensy": reg_loss,
                "certainty": low_prob_loss,
                "total_regularization": reg_loss + low_prob_loss,
                "pred": pred_loss,
                "total": pred_loss + reg_loss + low_prob_loss,
            }

            total_loss = total_loss + grid_losses["total"]
            losses.update({f"grid_{k}": v for k, v in grid_losses.items()})

        if self.training_type != "grid":
            pred_loss = (
                self.subgrid_pred_loss.forward(
                    model_prediction.beat_activation,
                    model_prediction.downbeat_activation,
                    model_prediction.grid_mask,
                    batch["truth_beat"],
                    batch["truth_downbeat"],
                )
                * self.subgrid_loss_scale
            )

            consistensy_loss = (
                self.subgrid_consistensy_loss.forward(
                    model_prediction.subgrid_features, model_prediction.grid_mask
                )
                * self.subgrid_loss_scale
            )

            subgrid_losses = {
                "pred": pred_loss,
                "consistensy": consistensy_loss,
                "total": pred_loss + consistensy_loss,
                # "total": pred_loss,
            }

            losses.update({f"subgrid_{k}": v for k, v in subgrid_losses.items()})
            total_loss = total_loss + subgrid_losses["total"]

        assert torch.isfinite(total_loss).item()

        losses["total"] = total_loss
        return losses

    def _compute_metrics(
        self, batch, model_prediction: ModelOutput, step="val"
    ) -> dict[str, torch.Tensor]:
        all_metrics: dict[str, torch.Tensor] = {}
        if self.training_type != "subgrid":
            grid_beat_recall = metrics.mask_recall(
                model_prediction.grid_mask, batch["truth_beat"]
            )

            grid_downbeat_recall = metrics.mask_recall(
                model_prediction.grid_mask, batch["truth_downbeat"]
            )

            grid_beat_f_measure = metrics.f_measure(
                model_prediction.grid_mask, batch["truth_beat"]
            )

            grid_CMLc, grid_CMLt, grid_AMLc, grid_AMLt = metrics.continuity(
                model_prediction.grid_mask, batch["truth_beat"]
            )
            grid_metrics = {
                "recall": grid_beat_recall.mean(),
                "downbeat_recall": grid_downbeat_recall.mean(),
                "F-measure": grid_beat_f_measure.mean(),
                "AMLt": np.mean(grid_AMLt),
                "CMLt": np.mean(grid_CMLt),
            }
            all_metrics.update({f"grid_{k}": v for k, v in grid_metrics.items()})

        if self.training_type != "grid":
            bF_measure = metrics.f_measure(
                model_prediction.beat_mask, batch["truth_beat"]
            )
            bRecall = metrics.mask_recall(
                model_prediction.beat_mask, batch["truth_beat"]
            )
            bCMLc, bCMLt, bAMLc, bAMLt = metrics.continuity(
                model_prediction.beat_mask, batch["truth_beat"]
            )
            dbF_measure = metrics.f_measure(
                model_prediction.downbeat_mask, batch["truth_downbeat"]
            )
            dbRecall = metrics.mask_recall(
                model_prediction.downbeat_mask, batch["truth_downbeat"]
            )
            dbCMLc, dbCMLt, dbAMLc, dbAMLt = metrics.continuity(
                model_prediction.downbeat_mask, batch["truth_downbeat"]
            )

            subgrid_metrics = {
                "bF-measure": bF_measure.mean(),
                "bRecall": bRecall.mean(),
                "bAMLt": np.mean(bAMLt),
                "bCMLt": np.mean(bCMLt),
                "dbF-measure": dbF_measure.mean(),
                "dbRecall": dbRecall.mean(),
                "dbAMLt": np.mean(dbAMLt),
                "dbCMLt": np.mean(dbCMLt),
            }
            all_metrics.update({f"subgrid_{k}": v for k, v in subgrid_metrics.items()})
        # concatenate dictionaries

        return all_metrics

    def log_losses(self, losses: dict[str, torch.Tensor], batch_size, step="train"):
        # log for separate targets
        for target in losses.keys():
            if target == "total":
                prog_bar = True
                name = f"{step}_loss"
            else:
                prog_bar = False
                name = f"{step}_loss_{target}"

            self.log(
                name,
                losses[target].item(),
                prog_bar=prog_bar,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

    def log_metrics(self, metrics, batch_size, step="val"):
        for key, value in metrics.items():
            self.log(
                f"{step}_{key}",
                value,
                # prog_bar=key.startswith("F-measure"),
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):

        # DEBUGGING
        # self.log(
        #     "max subgrid parameter",
        #     max([p.abs().amax() for p in self.model.subgrid_block.parameters()]),
        #     prog_bar=True,
        #     on_step=True,
        # )
        # self.log(
        #     "sum of subgrid parameters",
        #     sum([p.abs().sum() for p in self.model.subgrid_block.parameters()]),
        #     prog_bar=True,
        #     on_step=True,
        # )
        # run the model
        # The model only knows whether it should calculate the subgrid or not, not whether it has
        # to detach the gradient. Therefore, we have to tell it here.
        model_prediction: ModelOutput = self.model.forward(
            batch["spect"], detach_pregrid=(self.training_type == "subgrid")
        )

        # compute loss
        losses = self._compute_loss(batch, model_prediction)
        self.log_losses(losses, len(batch["spect"]), "train")
        return losses["total"]

    def validation_step(self, batch, batch_idx):
        # run the model
        model_prediction: ModelOutput = self.model.forward(batch["spect"])
        # compute loss
        losses = self._compute_loss(batch, model_prediction)
        self.log_losses(losses, len(batch["spect"]), "val")

        # compute the metrics
        metrics = self._compute_metrics(batch, model_prediction, "val")

        self.log_metrics(metrics, batch["spect"].shape[0], "val")

    def test_step(self, batch, batch_idx):
        metrics, _, _, _ = self.predict_step(batch, batch_idx)
        self.log_metrics(metrics, batch["spect"].shape[0], "test")

    # TODO predict step
    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
        chunk_size: int = 1500,
        overlap_mode: str = "keep_first",
    ) -> Any:
        """
        Compute predictions and metrics for a batch (a dictionary with an "spect" key).
        It splits up the audio into multiple chunks of chunk size,
         which should correspond to the length of the sequence the model was trained with.
        Potential overlaps between chunks can be handled in two ways:
        by keeping the predictions of the excerpt coming first (overlap_mode='keep_first'), or
        by keeping the predictions of the excerpt coming last (overlap_mode='keep_last').
        Note that overlaps appear as the last excerpt is moved backwards
        when it would extend over the end of the piece.
        """
        if batch["spect"].shape[0] != 1:
            raise ValueError(
                "When predicting full pieces, only `batch_size=1` is supported"
            )
        if torch.any(~batch["padding_mask"]):
            raise ValueError(
                "When predicting full pieces, the Dataset must not pad inputs"
            )

        model_prediction: ModelOutput = split_predict_aggregate(
            batch["spect"][0],
            chunk_size,
            overlap_mode,
            self.model,
            end_goal="grid" if self.training_type == "grid" else "subgrid",
        )

        model_prediction = model_prediction.unsqueeze(0)
        # add the batch dimension back in the prediction for consistency

        # compute the metrics
        metrics = self._compute_metrics(batch, model_prediction, step="test")
        return metrics, model_prediction, batch["dataset"], batch["spect_path"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW
        # only decay 2+-dimensional tensors, to exclude biases and norms
        # (filtering on dimensionality idea taken from Kaparthy's nano-GPT)

        # No need to check for grid because then the subgrid doesn't exist anyways.
        if self.training_type == "subgrid":
            params = [p for n, p in self.named_parameters() if "subgrid" in n]
        else:
            params = self.parameters()

        params = [
            {
                "params": (p for p in params if p.requires_grad and p.ndim >= 2),
                "weight_decay": self.weight_decay,
            },
            {
                "params": (p for p in params if p.requires_grad and p.ndim <= 1),
                "weight_decay": 0,
            },
        ]

        optimizer = optimizer(params, lr=self.lr)

        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, self.warmup_steps, self.trainer.estimated_stepping_batches
        )

        result = dict(optimizer=optimizer)
        result["lr_scheduler"] = {"scheduler": self.lr_scheduler, "interval": "step"}  # type: ignore
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # remove _orig_mod prefixes for compiled models
        state_dict = replace_state_dict_key(state_dict, "_orig_mod.", "")
        return state_dict


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing over `max_iters` steps with `warmup` linear warmup steps.
    Optionally re-raises the learning rate for the final `raise_last` fraction
    of total training time to `raise_to` of the full learning rate, again with
    a linear warmup (useful for stochastic weight averaging).
    """

    def __init__(self, optimizer, warmup, max_iters, raise_last=0, raise_to=0.5):
        self.warmup = warmup
        self.max_num_iters = int((1 - raise_last) * max_iters)
        self.raise_to = raise_to
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        if step < self.max_num_iters:
            progress = step / self.max_num_iters
            lr_factor = 0.5 * (1 + np.cos(np.pi * progress))
            if step <= self.warmup:
                lr_factor *= step / self.warmup
        else:
            progress = (step - self.max_num_iters) / self.warmup
            lr_factor = self.raise_to * min(progress, 1)
        return lr_factor
