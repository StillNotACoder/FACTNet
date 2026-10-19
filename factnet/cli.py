"""Parse experiment configuration without importing the training runtime."""

import argparse
import os
from pathlib import Path


def create_parser():
    parser = argparse.ArgumentParser(description="Train or evaluate FACTNet.")
    parser.add_argument("--config", type=Path, help="YAML configuration; CLI options override it")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument(
        "--eval", action="store_const", const="eval", dest="mode", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--dataset", choices=("sevir", "meteo", "meteonet", "shanghai", "cikm"), default="shanghai"
    )
    parser.add_argument("--data-path", "--data_path", type=Path)
    parser.add_argument("--output-dir", "--output_dir", type=Path, help="Experiment directory")
    parser.add_argument(
        "--checkpoint",
        "--ckpt_milestone",
        type=Path,
        help="Model .pt file for evaluation or initialization",
    )
    parser.add_argument(
        "--resume", type=Path, help="Full checkpoint directory saved by this runner"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-use", "--gpu_use", nargs="+", help="Visible CUDA device IDs, e.g. 0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img-size", "--img_size", type=int, default=128)
    parser.add_argument("--frames-in", "--frames_in", type=int, default=5)
    parser.add_argument("--frames-out", "--frames_out", type=int, default=None)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    parser.add_argument("--stride", type=int, default=13, help="SEVIR sequence stride")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3, help="MD branch depth")
    parser.add_argument("--ie-layers", type=int, default=3)
    parser.add_argument("--spec-num", "--spec_num", type=int, default=20)
    parser.add_argument("--pha-weight", "--pha_weight", type=float, default=0.01)
    parser.add_argument("--amp-weight", "--amp_weight", type=float, default=0.01)
    parser.add_argument("--anet-weight", "--anet_weight", type=float, default=0.1)
    parser.add_argument("--aw-stop-step", "--aw_stop_step", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size", "--batch_size", type=int, default=4, help="Microbatch size per process"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.90)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("constant", "linear", "cosine"), default="cosine")
    parser.add_argument("--warmup-steps", "--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad-acc-step", "--grad_acc_step", type=int, default=1)
    parser.add_argument(
        "--mixed-precision", "--mixed_precision", choices=("no", "fp16", "bf16"), default="no"
    )
    parser.add_argument("--ema-rate", "--ema_rate", type=float, default=0.95)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-ema", action="store_true", help="Evaluate the EMA model; raw weights are the default"
    )
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-visualizations", type=int, default=8)
    parser.add_argument(
        "--validation-batches", type=int, default=None, help="Debug-only validation limit"
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--wandb", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Synthetic forward/backward check without radar data",
    )
    return parser


def parse_args(argv=None):
    parser = create_parser()
    known, _ = parser.parse_known_args(argv)
    if known.config:
        import yaml

        config = yaml.safe_load(known.config.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            parser.error("The YAML configuration must be a mapping.")
        config = {key.replace("-", "_"): value for key, value in config.items()}
        actions = {action.dest: action for action in parser._actions}
        unknown = config.keys() - actions.keys()
        if unknown:
            parser.error(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        for key, value in config.items():
            action = actions[key]
            if action.type is not None and value is not None:
                config[key] = action.type(value)
            if action.choices and value not in action.choices:
                parser.error(f"Invalid value for {key}: {value}")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if args.dataset == "meteonet":
        args.dataset = "meteo"
    if args.frames_out is None:
        args.frames_out = 10 if args.dataset == "cikm" else 20
    if args.data_path is None:
        value = os.getenv(f"FACTNET_DATA_{args.dataset.upper()}")
        args.data_path = Path(value) if value else None
    positive = (
        "img_size",
        "frames_in",
        "frames_out",
        "dim",
        "layers",
        "ie_layers",
        "spec_num",
        "epochs",
        "batch_size",
        "grad_acc_step",
        "aw_stop_step",
        "log_every",
        "stride",
    )
    if any(getattr(args, key) <= 0 for key in positive):
        parser.error(
            "Image sizes, sequence lengths, depths, batch sizes, strides, and steps must be positive."
        )
    if args.num_workers < 0 or args.warmup_steps < 0 or args.max_visualizations < 0:
        parser.error("Worker, warmup, and visualization counts must be nonnegative.")
    if args.validation_batches is not None and args.validation_batches <= 0:
        parser.error("validation-batches must be positive.")
    if not 0 <= args.ema_rate < 1 or not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        parser.error("EMA rate and optimizer betas must lie in [0,1).")
    if (
        args.lr <= 0
        or min(args.weight_decay, args.pha_weight, args.amp_weight, args.anet_weight) < 0
    ):
        parser.error("Learning rate must be positive and loss weights/weight decay nonnegative.")
    if args.resume and args.checkpoint:
        parser.error("Use --resume for full continuation or --checkpoint for weights, not both.")
    if args.mode == "eval" and not args.checkpoint and not args.smoke_test:
        parser.error("Evaluation requires --checkpoint.")
    if args.mode == "eval" and args.resume:
        parser.error("--resume is a training option.")
    if not args.data_path and not args.smoke_test:
        parser.error("Specify --data-path or FACTNET_DATA_<DATASET>.")
    if args.use_ema and not args.ema:
        parser.error("--use-ema requires EMA to be enabled.")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.gpu_use:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu_use)
    from .runner import Experiment

    experiment = Experiment(args)
    try:
        if args.smoke_test:
            experiment.smoke_test()
        elif args.mode == "eval":
            experiment.evaluate_test()
        else:
            experiment.train()
    finally:
        experiment.close()
