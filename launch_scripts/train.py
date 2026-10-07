import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from beat_this.dataset import BeatDataModule
from beat_this.model.pl_module import PLBeatThis


def main(args):
    # for repeatability
    seed_everything(args.seed, workers=True)

    print("Starting a new run with the following parameters:")
    print(args)

    params_str = (
        f"{'grid ' if args.training_type == 'grid' else ''}"
        f"{'noval ' if not args.val else ''}"
        + f"{'hung ' if args.hung_data else ''}"
        + f"{'fold' + str(args.fold) + ' ' if args.fold is not None else ''}"
        + f"-h{args.transformer_dim} "
        + f"-l{args.n_layers}{'+' + str(args.subgrid_transformer_layers) if args.training_type != 'grid' else ''} "
        + f"-gW{args.grid_window_size} "
    )
    if args.logger == "wandb":
        if args.resume_checkpoint and args.resume_id:
            wandb_args = dict(id=args.resume_id, resume="must")
        else:
            wandb_args = {}
        logger = WandbLogger(
            project="beat_this",
            name=f"{args.name} {params_str}".strip(),
            **wandb_args,  # type: ignore
        )
    else:
        logger = None

    if args.force_flash_attention:
        print("Forcing the use of the flash attention.")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    checkpoint_dir = (
        Path(__file__).parent.parent.relative_to(Path.cwd()) / "checkpoints"
    )
    augmentations = {}
    if args.tempo_augmentation:
        augmentations["tempo"] = {"min": -20, "max": 20, "stride": 4}
    if args.pitch_augmentation:
        augmentations["pitch"] = {"min": -5, "max": 6}
    if args.mask_augmentation:
        # kind, min_count, max_count, min_len, max_len, min_parts, max_parts
        augmentations["mask"] = {
            "kind": "permute",
            "min_count": 1,
            "max_count": 6,
            "min_len": 0.1,
            "max_len": 2,
            "min_parts": 5,
            "max_parts": 9,
        }

    datamodule = BeatDataModule(
        data_dir,
        batch_size=args.batch_size,
        train_length=args.train_length,
        spect_fps=args.fps,
        num_workers=args.num_workers,
        test_dataset="gtzan",
        length_based_oversampling_factor=args.length_based_oversampling_factor,
        augmentations=augmentations,
        hung_data=args.hung_data,
        no_val=not args.val,
        fold=args.fold,
    )
    datamodule.setup(stage="fit")

    # # compute positive weights
    # pos_weights = datamodule.get_train_positive_weights(widen_target_mask=3)
    # print("Using positive weights: ", pos_weights)
    dropout = {
        "frontend": args.frontend_dropout,
        "transformer": args.transformer_dropout,
    }
    pl_model = PLBeatThis(
        training_type=args.training_type,
        spect_dim=128,
        fps=50,
        transformer_dim=args.transformer_dim,
        ff_mult=4,
        n_layers=args.n_layers,
        stem_dim=32,
        dropout=dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        head_dim=32,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        eval_trim_beats=args.eval_trim_beats,
        partial_transformers=args.partial_transformers,
        grid_bins_per_octave=args.grid_bins_per_octave,
        grid_regularization_weight=args.grid_regularization_weight,
        grid_regularization_freq_scale=args.grid_reg_freq_scale,
        grid_regularization_phase_factor=args.grid_reg_phase_factor,
        grid_confidence_loss_weight=args.grid_confidence_loss_weight,
        grid_window_size=args.grid_window_size,
        grid_min_freq=args.grid_min_frequency,
        grid_consistensy_prob_spread=args.grid_consistensy_probability_spread,
        grid_pred_prob_spread=args.grid_prediction_probability_spread,
        grid_half_crossfade_frames=args.grid_crossfade,
        grid_downbeat_weight=args.grid_downbeat_weight,
        subgrid_transformer_layers=args.subgrid_transformer_layers,
        max_subgrid_meter=args.subgrid_max_meter,
        max_subgrid_downbeat_meter=args.subgrid_max_downbeat_meter,
        subgrid_regularization_scale=args.subgrid_regularization_scale,
        subgrid_loss_scale=args.subgrid_loss_scale,
    )

    if args.base_model is not None:
        pl_model.load_base_model(
            args.base_model, load_subgrid=(args.training_type != "subgrid")
        )

    for part in args.compile:
        if hasattr(pl_model.model, part):
            setattr(pl_model.model, part, torch.compile(getattr(pl_model.model, part)))
            print("Will compile model", part)
        else:
            # raise ValueError("The model is missing the part", part, "to compile")
            print(
                f"The model is missing the part {part} to compile, skipping compilation of this part."
            )

    callbacks: list = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            every_n_epochs=1,
            dirpath=str(checkpoint_dir),
            filename=f"{args.name} S{args.seed} {params_str}".strip(),
        ),
    ]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=[args.gpu],
        num_sanity_val_steps=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        precision="16-mixed",
        accumulate_grad_batches=args.accumulate_grad_batches,
        # gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=args.val_frequency,
        limit_train_batches=args.limit_train_batches or 1.0,
        limit_val_batches=args.limit_val_batches or 1.0,
        limit_test_batches=args.limit_test_batches or 1.0,
    )

    trainer.fit(pl_model, datamodule, ckpt_path=args.resume_checkpoint)
    trainer.test(pl_model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--force-flash-attention", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--compile",
        action="store",
        nargs="*",
        type=str,
        default=[
            "frontend",
            "transformer_blocks",
            "grid_processor",
            "grid_block",
            "subgrid_block",
        ],  # TODO
        help="Which model parts to compile, among frontend, transformer_encoder, grid_processor, grid_block, subgrid_block",
    )
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--transformer-dim", type=int, default=512)
    parser.add_argument(
        "--frontend-dropout",
        type=float,
        default=0.1,
        help="dropout rate to apply in the frontend",
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.2,
        help="dropout rate to apply in the main transformer blocks",
    )
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logger", type=str, choices=["wandb", "none"], default="none")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--fps", type=int, default=50, help="The spectrograms fps.")

    parser.add_argument(
        "--warmup-steps", type=int, default=1000, help="warmup steps for optimizer"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=100, help="max epochs for training"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="batch size for training"
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)

    parser.add_argument(
        "--train-length",
        type=int,
        default=1500,
        help="maximum seq length for training in frames",
    )
    parser.add_argument(
        "--eval-trim-beats",
        metavar="SECONDS",
        type=float,
        default=5,
        help="Skip the first given seconds per piece in evaluating (default: %(default)s)",
    )
    parser.add_argument(
        "--val-frequency",
        metavar="N",
        type=int,
        default=5,
        help="validate every N epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--tempo-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use precomputed tempo aumentation",
    )
    parser.add_argument(
        "--pitch-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use precomputed pitch aumentation",
    )
    parser.add_argument(
        "--mask-augmentation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use online mask aumentation",
    )
    parser.add_argument(
        "--partial-transformers",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use Partial transformers in the frontend",
    )
    parser.add_argument(
        "--length-based-oversampling-factor",
        type=float,
        default=0.65,
        help="The factor to oversample the long pieces in the dataset. Set to 0 to only take one excerpt for each piece.",
    )
    parser.add_argument(
        "--val",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Train on all data, including validation data, escluding test data. The validation metrics will still be computed, but they won't carry any meaning.",
    )
    parser.add_argument(
        "--hung-data",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Limit the training to Hung et al. data. The validation will still be computed on all datasets.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="If given, the CV fold number to *not* train on (0-based).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random number generators.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Resume training from a local checkpoint.",
    )
    parser.add_argument(
        "--resume-id",
        type=str,
        default=None,
        help="When resuming with --resume-checkpoint, optionally provide the wandb id to continue logging to.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="Limit the number of batches to be trained on, should be a positive integer",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Limit the number of batches to be validated on, should be a positive integer",
    )
    parser.add_argument(
        "--limit-test-batches",
        type=int,
        default=None,
        help="Limit the number of batches to be tested on, should be a positive integer",
    )

    parser.add_argument(
        "--training-type",
        type=str,
        choices=["full", "grid", "subgrid"],
        default="full",
        help="What part of the model should be trained. If subgrid is chosen, a base-model is required.",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="The path to the base model checkpoint. "
        "This will be loaded with strict=False, but hyperparameters for architecture "
        "should still give same shape of tensors for this to work.",
    )

    parser.add_argument(
        "--grid-regularization-weight",
        type=float,
        default=0.1,
        help="Weight for the grid regularization loss",
    )

    parser.add_argument(
        "--grid-window-size",
        type=int,
        default=100,
        help="Width of a single grid-window in frames",
    )

    parser.add_argument(
        "--grid-min-frequency",
        type=float,
        default=0.01,
        help="The minimum frequency (in peaks per frame) possible for the grid",
    )

    parser.add_argument(
        "--grid-consistensy-probability-spread",
        type=float,
        default=0.4,
        help="How much the regularization should care about lesser-used bins, 0 means no extra care and 1 means equal caring of all bins. ",
    )

    parser.add_argument(
        "--grid-bins-per-octave",
        type=int,
        default=3,
        help="How many frequency bins there are per octave for the grid",
    )

    parser.add_argument(
        "--grid-prediction-probability-spread",
        type=float,
        default=0.4,
        help="How much the prediction loss should care about lesser-used bins, 0 means no extra care and 1 means equal caring of all bins.",
    )

    parser.add_argument(
        "--grid-crossfade",
        type=int,
        default=4,
        help="How many frames of overlap the grid activation windows will be calculated with, and then smoothed over.",
    )

    parser.add_argument(
        "--sharp-activations",
        type=bool,
        default=False,
        help="Whether the normalized cosine wave should be squared before recall loss.",
    )

    parser.add_argument(
        "--grid-confidence-loss-weight",
        type=float,
        default=1e-5,
        help="The scale of the sum(1/bin_probability) loss",
    )

    parser.add_argument(
        "--grid-downbeat-weight",
        type=float,
        default=2.0,
        help="What the relative loss should be on downbeat compared to beats (1 is no extra weight)",
    )

    parser.add_argument(
        "--grid-reg-freq-scale",
        type=float,
        default=0.002,
        help="The inverse weighing of frequency jumps in the regularization.",
    )

    parser.add_argument(
        "--grid-reg-phase-factor",
        type=float,
        default=2,
        help="The weighing of phase jumps in the regularization.",
    )

    parser.add_argument(
        "--subgrid-max-meter",
        type=int,
        default=4,
        help="The maximum allowed grid points per beat.",
    )

    parser.add_argument(
        "--subgrid-max-downbeat-meter",
        type=int,
        default=15,
        help="The maximum allowed beats per downbeat.",
    )

    parser.add_argument(
        "--subgrid-transformer-layers",
        type=int,
        default=3,
        help="How many transformer layers are used between the grid and the subgrid.",
    )
    parser.add_argument(
        "--subgrid-regularization-scale",
        type=float,
        default=1e-2,
        help="The weight of regularization loss for the subgrid.",
    )
    parser.add_argument(
        "--subgrid-loss-scale",
        type=float,
        default=1.0,
        help="Scaling factor for subgrid loss, to control relative importance with grid when training-type=full.",
    )

    args = parser.parse_args()
    main(args)
