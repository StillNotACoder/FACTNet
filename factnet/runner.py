"""Training, evaluation, and checkpoint lifecycle for FACTNet."""

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    broadcast_object_list,
    set_seed,
)
from tqdm.auto import tqdm

from datasets.get_datasets import build_dataloaders
from models.factnet import get_model
from utils.metrics import Evaluator


LOGGER = logging.getLogger(__name__)


def build_scheduler(optimizer, kind, warmup_steps, total_steps):
    def multiplier(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        if kind == "constant":
            return 1.0
        if kind == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class Experiment:
    def __init__(self, args):
        self.args = args
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        self.accelerator = Accelerator(
            cpu=args.device == "cpu",
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.grad_acc_step,
            log_with="wandb" if args.wandb != "disabled" else None,
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=False),
                InitProcessGroupKwargs(timeout=timedelta(seconds=5400)),
            ],
        )
        set_seed(args.seed)
        self.model = get_model(
            input_shape=(args.img_size, args.img_size),
            T_in=args.frames_in,
            T_out=args.frames_out,
            dim=args.dim,
            n_layers=args.layers,
            ie_layers=args.ie_layers,
            spec_num=args.spec_num,
            pha_weight=args.pha_weight,
            anet_weight=args.anet_weight,
            amp_weight=args.amp_weight,
            aweight_stop_steps=args.aw_stop_step,
        )
        self.completed_epochs, self.global_step, self.best_csi = 0, 0, -1.0
        self.ema = None
        self.train_loader = self.val_loader = self.test_loader = None
        self.optimizer = self.scheduler = None
        if args.resume:
            default_output = args.resume.resolve().parent.parent
        else:
            timestamp = [datetime.now().strftime("%Y%m%d-%H%M%S")]
            broadcast_object_list(timestamp)
            default_output = Path("outputs") / args.dataset / timestamp[0]
        self.output_dir = Path(args.output_dir or default_output).resolve()
        conflict = [
            args.mode == "train"
            and not args.resume
            and not args.smoke_test
            and self.accelerator.is_main_process
            and (self.output_dir / "config.json").exists()
        ]
        broadcast_object_list(conflict)
        if conflict[0]:
            raise FileExistsError(
                f"Experiment already exists: {self.output_dir}. Use --resume or another output directory."
            )
        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(message)s",
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(self.output_dir / "run.log", encoding="utf-8"),
                ],
            )
            config_name = (
                "eval_config.json"
                if args.mode == "eval"
                else "resume_config.json"
                if args.resume
                else "config.json"
            )
            (self.output_dir / config_name).write_text(
                json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
            )
        self.accelerator.wait_for_everyone()
        if args.wandb != "disabled":
            self.accelerator.init_trackers(
                "FACTNet",
                config=json.loads(json.dumps(vars(args), default=str)),
                init_kwargs={"wandb": {"mode": args.wandb}},
            )
        LOGGER.info(
            "Device: %s; model parameters: %.3fM",
            self.accelerator.device,
            sum(p.numel() for p in self.model.parameters()) / 1e6,
        )
        if args.smoke_test:
            self.model = self.accelerator.prepare(self.model)
            return
        if args.mode == "train":
            self._prepare_training()
        else:
            self.model = self.accelerator.prepare(self.model)
            self._build_ema()
        if args.checkpoint:
            self._load_weights(args.checkpoint)
        if args.resume:
            self._resume(args.resume)

    def _loaders(self, splits):
        return build_dataloaders(
            self.args.dataset,
            data_path=self.args.data_path,
            img_size=self.args.img_size,
            seq_len=self.args.frames_in + self.args.frames_out,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            stride=self.args.stride,
            splits=splits,
            seed=self.args.seed,
            pin_memory=self.accelerator.device.type == "cuda",
        )

    def _prepare_training(self):
        requested = ("train", "val") if self.args.validate else ("train",)
        train, val, _, self.visualizer, self.pixel_scale, self.thresholds = self._loaders(requested)
        self.train_loader = self.accelerator.prepare(train)
        self.val_loader = self.accelerator.prepare(val) if val is not None else None
        if len(self.train_loader) == 0:
            raise ValueError("Training loader is empty; check the split and batch size.")
        updates_per_epoch = math.ceil(len(self.train_loader) / self.args.grad_acc_step)
        self.total_updates = self.args.epochs * updates_per_epoch
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay,
        )
        # Advance once per optimizer update on each process. Do not apply the
        # process-count multiplier used by an AcceleratedScheduler.
        self.scheduler = build_scheduler(
            self.optimizer, self.args.scheduler, self.args.warmup_steps, self.total_updates
        )
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.accelerator.register_for_checkpointing(self.scheduler)
        self._build_ema()

    def _build_ema(self):
        if self.args.ema:
            from ema_pytorch import EMA

            self.ema = EMA(
                self.accelerator.unwrap_model(self.model), beta=self.args.ema_rate, update_every=20
            )
            self.ema.to(self.accelerator.device)
            self.accelerator.register_for_checkpointing(self.ema)

    def _load_weights(self, path):
        path = Path(path)
        if path.is_dir():
            path = path / "model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint)
        if state and all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        self.accelerator.unwrap_model(self.model).load_state_dict(state, strict=True)
        if self.args.mode == "train":
            self.accelerator.unwrap_model(self.model).training_steps.zero_()
        if self.ema is not None:
            if self.args.mode == "train":
                self.ema.ema_model.load_state_dict(
                    self.accelerator.unwrap_model(self.model).state_dict()
                )
            elif "ema" in checkpoint:
                # Earlier EMA snapshots also lack the new loss-step buffer.
                ema_state = checkpoint["ema"].copy()
                for key, value in self.ema.state_dict().items():
                    if key.endswith("training_steps"):
                        ema_state.setdefault(key, value)
                self.ema.load_state_dict(ema_state, strict=True)
            else:
                if self.args.use_ema:
                    raise ValueError("This checkpoint has no EMA weights; omit --use-ema.")
                self.ema.ema_model.load_state_dict(
                    self.accelerator.unwrap_model(self.model).state_dict()
                )
        LOGGER.info("Loaded model weights from %s", path)

    def _resume(self, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        expected = metadata["config"]
        if metadata["num_processes"] != self.accelerator.num_processes:
            raise ValueError("Resume must use the same number of processes as the saved run.")
        # A changed scheduler or model size is not a continuation of the same run.
        for name in (
            "dataset",
            "img_size",
            "frames_in",
            "frames_out",
            "dim",
            "layers",
            "ie_layers",
            "spec_num",
            "batch_size",
            "grad_acc_step",
            "epochs",
            "lr",
            "beta1",
            "beta2",
            "weight_decay",
            "scheduler",
            "warmup_steps",
            "ema",
            "ema_rate",
            "mixed_precision",
            "pha_weight",
            "amp_weight",
            "anet_weight",
            "aw_stop_step",
            "seed",
            "stride",
            "validate",
            "validation_batches",
        ):
            if expected[name] != getattr(self.args, name):
                raise ValueError(
                    f"Resume configuration mismatch for {name}: {expected[name]!r} vs {getattr(self.args, name)!r}."
                )
        self.accelerator.load_state(str(directory))
        self.completed_epochs = metadata["completed_epochs"]
        self.global_step = metadata["global_step"]
        self.best_csi = metadata["best_csi"]

    def _save(self, name):
        directory = self.output_dir / "checkpoints" / name
        self.accelerator.save_state(str(directory), safe_serialization=False)
        if self.accelerator.is_main_process:
            state = self.accelerator.get_state_dict(self.model)
            weights = {"model": state}
            if self.ema is not None:
                weights["ema"] = self.ema.state_dict()
            torch.save(weights, directory / "model.pt")
            metadata = {
                "completed_epochs": self.completed_epochs,
                "global_step": self.global_step,
                "best_csi": self.best_csi,
                "config": vars(self.args),
                "num_processes": self.accelerator.num_processes,
            }
            (directory / "state.json").write_text(
                json.dumps(metadata, default=str, indent=2) + "\n", encoding="utf-8"
            )
        self.accelerator.wait_for_everyone()

    def _split_batch(self, batch):
        if batch.ndim != 5 or batch.shape[1] < self.args.frames_in + self.args.frames_out:
            raise ValueError(f"Expected radar sequence [B,T,1,H,W], got {tuple(batch.shape)}.")
        return batch[:, : self.args.frames_in], batch[
            :, self.args.frames_in : self.args.frames_in + self.args.frames_out
        ]

    def train(self):
        for epoch in range(self.completed_epochs, self.args.epochs):
            self.model.train()
            self.train_loader.set_epoch(epoch)
            if self.train_loader.generator is not None:
                self.train_loader.generator.manual_seed(self.args.seed + epoch)
            progress = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.args.epochs}",
                disable=not self.accelerator.is_main_process,
            )
            for index, batch in enumerate(progress):
                inputs, target = self._split_batch(batch)
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        _, losses = self.model(
                            inputs, target, compute_loss=True, training_step=self.global_step
                        )
                        # Accelerator divides by grad_acc_step; correct its final
                        # incomplete accumulation group to average the actual microbatches.
                        remainder = len(self.train_loader) % self.args.grad_acc_step
                        final_partial = remainder and index >= len(self.train_loader) - remainder
                        loss = losses["total_loss"] * (
                            self.args.grad_acc_step / remainder if final_partial else 1.0
                        )
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    if (
                        self.accelerator.sync_gradients
                        and not self.accelerator.optimizer_step_was_skipped
                    ):
                        self.scheduler.step()
                        if self.ema is not None:
                            self.ema.update()
                        self.global_step += 1
                    if self.accelerator.sync_gradients:
                        self.accelerator.unwrap_model(self.model).training_steps.fill_(
                            self.global_step
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                if self.accelerator.sync_gradients and self.global_step % self.args.log_every == 0:
                    logs = {key: float(value.detach()) for key, value in losses.items()}
                    logs["lr"] = self.optimizer.param_groups[0]["lr"]
                    self.accelerator.log(logs, step=self.global_step)
                    progress.set_postfix(loss=logs["total_loss"], lr=logs["lr"])
                    if self.accelerator.is_main_process:
                        LOGGER.info("Epoch %s update %s: %s", epoch + 1, self.global_step, logs)
            self.completed_epochs = epoch + 1
            if self.val_loader is not None:
                scores = self._evaluate(
                    self.val_loader, "validation", epoch + 1, image_metrics=False
                )
                if scores["csi"] > self.best_csi:
                    self.best_csi = scores["csi"]
                    self._save("best")
            self._save("last")

    @torch.no_grad()
    def _evaluate(self, loader, split, tag, image_metrics=True):
        model = (
            self.ema.ema_model if self.args.use_ema else self.accelerator.unwrap_model(self.model)
        )
        model.eval()
        evaluator = (
            Evaluator(
                self.args.frames_out,
                self.pixel_scale,
                self.thresholds,
                compute_image_metrics=image_metrics,
            )
            if self.accelerator.is_main_process
            else None
        )
        destination = self.output_dir / split / str(tag)
        visualized = 0
        for index, batch in enumerate(
            tqdm(loader, desc=split, disable=not self.accelerator.is_main_process)
        ):
            inputs, targets = self._split_batch(batch)
            with self.accelerator.autocast():
                predictions, _ = model(inputs, compute_loss=False)
            targets, predictions = self.accelerator.gather_for_metrics((targets, predictions))
            if evaluator is not None:
                evaluator.evaluate(targets.float(), predictions.float())
                if self.args.visualize:
                    for target, prediction in zip(targets, predictions):
                        if visualized >= self.args.max_visualizations:
                            break
                        self.visualizer(
                            prediction,
                            target,
                            destination / "samples" / f"{visualized:06d}",
                            save_colored=True,
                        )
                        visualized += 1
            if (
                split == "validation"
                and self.args.validation_batches is not None
                and index + 1 >= self.args.validation_batches
            ):
                break
        payload = [evaluator.done(json_save_dir=destination) if evaluator is not None else None]
        broadcast_object_list(payload)
        if self.accelerator.is_main_process:
            LOGGER.info(
                "%s results: %s",
                split,
                {k: v for k, v in payload[0].items() if isinstance(v, (int, float))},
            )
        return payload[0]

    def evaluate_test(self):
        _, _, loader, self.visualizer, self.pixel_scale, self.thresholds = self._loaders(("test",))
        self.test_loader = self.accelerator.prepare(loader)
        return self._evaluate(self.test_loader, "test", Path(self.args.checkpoint).stem)

    def smoke_test(self):
        shape = (1, self.args.frames_in, 1, self.args.img_size, self.args.img_size)
        inputs = torch.rand(shape, device=self.accelerator.device)
        target = torch.rand((1, self.args.frames_out, *shape[2:]), device=self.accelerator.device)
        self.model.train()
        with self.accelerator.autocast():
            prediction, losses = self.model(inputs, target, compute_loss=True, training_step=0)
        self.accelerator.backward(losses["total_loss"])
        if not torch.isfinite(losses["total_loss"]):
            raise RuntimeError("Smoke test produced a non-finite loss.")
        self.accelerator.print(
            f"FACTNet smoke test passed: output={tuple(prediction.shape)}, loss={float(losses['total_loss'].detach()):.6f}"
        )

    def close(self):
        self.accelerator.end_training()
