"""Train the custom pretrained dual-stream RGB-thermal segmentation model on MFNet."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import my_model as model_module
from my_model import ARCHITECTURE_ID, build_model
import dataset as dataset_module
from dataset import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DATA_ROOT = "../MFNet_dataset"
DEFAULT_CHECKPOINT_DIR = "checkpoints/resnet18_crossmodal_attention_v2_1"
DEFAULT_PLOT_DIR = "plots/resnet18_crossmodal_attention_v2_1"
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
RGB_T_MEAN = (0.485, 0.456, 0.406)
RGB_T_STD = (0.229, 0.224, 0.225)

MFNET_CLASS_COUNTS = np.asarray(
    [
        442504816,  # 0: Unlabeled
        21804852,   # 1: Car
        6648728,    # 2: Person
        2055576,    # 3: Bike
        3097628,    # 4: Curve
        2627332,    # 5: Car Stop
        727236,     # 6: Guardrail
        844474,     # 7: Color Cone
        1378958,    # 8: Bump
    ],
    dtype=np.float64,
)

MFNET_COLORMAP = np.asarray(
    [
        [0, 0, 0],
        [220, 20, 60],
        [0, 200, 0],
        [30, 144, 255],
        [255, 215, 0],
        [255, 20, 147],
        [0, 255, 255],
        [255, 140, 0],
        [138, 43, 226],
    ],
    dtype=np.uint8,
)
IGNORE_COLOR = np.asarray([128, 128, 128], dtype=np.uint8)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Fixed input shapes benefit from cuDNN autotuning. Runs are seeded but not bitwise deterministic.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


class SegmentationMetrics:
    """Confusion-matrix metrics. `miou` includes class 0; `miou_foreground` covers classes 1..8."""

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=DEVICE)

    def reset(self) -> None:
        self.conf_matrix.zero_()

    @torch.no_grad()
    def update(self, logits_or_preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = logits_or_preds.argmax(dim=1) if logits_or_preds.ndim == 4 else logits_or_preds
        p = preds.reshape(-1).long()
        t = targets.reshape(-1).long()
        valid = (t >= 0) & (t < self.num_classes) & (t != self.ignore_index)
        valid &= (p >= 0) & (p < self.num_classes)
        if valid.any():
            indices = t[valid] * self.num_classes + p[valid]
            self.conf_matrix += torch.bincount(
                indices, minlength=self.num_classes * self.num_classes
            ).reshape(self.num_classes, self.num_classes)

    @staticmethod
    def _class_scores(conf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conf = conf.float()
        tp = torch.diag(conf)
        gt = conf.sum(dim=1)
        pred = conf.sum(dim=0)
        union = gt + pred - tp
        iou = torch.where(union > 0, tp / union.clamp_min(1.0), torch.zeros_like(tp))
        denom = gt + pred
        dice = torch.where(denom > 0, 2.0 * tp / denom.clamp_min(1.0), torch.zeros_like(tp))
        return iou, dice

    @torch.no_grad()
    def compute(self) -> Dict[str, Any]:
        conf = self.conf_matrix
        iou, dice = self._class_scores(conf)
        valid_iou = (conf.sum(0) + conf.sum(1) - conf.diag()) > 0
        valid_dice = (conf.sum(0) + conf.sum(1)) > 0
        miou = iou[valid_iou].mean().item() if valid_iou.any() else 0.0
        dice_mean = dice[valid_dice].mean().item() if valid_dice.any() else 0.0

        fg_iou = iou[1:]
        fg_union = (conf.sum(0) + conf.sum(1) - conf.diag())[1:]
        fg_valid = fg_union > 0
        fg_miou = fg_iou[fg_valid].mean().item() if fg_valid.any() else 0.0

        total = conf.sum().float()
        pixel_acc = (conf.diag().sum().float() / total).item() if total.item() > 0 else 0.0
        return {
            "miou": float(miou),
            "miou_foreground": float(fg_miou),
            "pixel_acc": float(pixel_acc),
            "dice": float(dice_mean),
            "iou_per_class": [float(v) for v in iou.cpu().tolist()],
            "dice_per_class": [float(v) for v in dice.cpu().tolist()],
        }


class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for stability under CUDA autocast.
        probs = F.softmax(logits.float(), dim=1)
        valid = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        safe_targets = targets.masked_fill(~valid, 0).long()
        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        one_hot = one_hot * valid_f

        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        cardinality = (probs + one_hot).sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        # Skip classes absent from both prediction and target for this batch.
        present = cardinality > self.eps
        if not present.any():
            return logits.sum() * 0.0
        return 1.0 - dice[present].mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = MulticlassDiceLoss(num_classes, ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits.float(), targets) + self.dice_weight * self.dice(logits, targets)


def get_class_weights(method: str = "auto", device: torch.device = DEVICE) -> Optional[torch.Tensor]:
    if method == "none":
        return None
    freqs = MFNET_CLASS_COUNTS / MFNET_CLASS_COUNTS.sum()
    if method == "auto":
        weights = (1.0 / np.maximum(freqs, 1e-8)) ** (1.0 / 3.0)
        weights = weights / weights.mean()
    elif method == "mfb":
        weights = np.median(freqs) / np.maximum(freqs, 1e-8)
        weights = np.clip(weights, 0.05, 10.0)
        weights = weights / weights.mean()
    elif method == "enet":
        weights = 1.0 / np.log(1.10 + freqs)
        weights = weights / weights.mean()
    else:
        raise ValueError(f"Unknown class-weight method: {method}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(enabled and device.type == "cuda"))


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _config_snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "architecture_id": ARCHITECTURE_ID,
        "train": vars(args),
        "dataset": {
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "ignore_index": IGNORE_INDEX,
            "default_image_size": [args.img_h, args.img_w],
        },
    }


def save_checkpoint(
    path: str,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    history: Dict[str, list],
    best_metric_val: float,
    best_epoch: int,
    early_best: float,
    stale_epochs: int,
    config: Dict[str, Any],
    best_val_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "architecture_id": ARCHITECTURE_ID,
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "history": copy.deepcopy(history),
        "best_metric_val": float(best_metric_val),
        "best_epoch": int(best_epoch),
        "early_best": float(early_best),
        "stale_epochs": int(stale_epochs),
        "best_val_metrics": best_val_metrics or {},
        "config_snapshot": config,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    arch_id = checkpoint.get("architecture_id")
    if arch_id != ARCHITECTURE_ID:
        raise RuntimeError(
            f"Checkpoint architecture mismatch: file has {arch_id!r}, but current model is "
            f"{ARCHITECTURE_ID!r}. Do not resume from the old MFNet checkpoint. Start a new "
            "run in the new resnet18_crossmodal_attention_v2 checkpoint directory."
        )

    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    rng = checkpoint.get("rng_state", {})
    try:
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
    except Exception as exc:
        print(f"[Warning] Could not restore all random states: {exc}")

    print(
        f"[checkpoint] Resumed compatible architecture at epoch {checkpoint['epoch']} "
        f"(best epoch={checkpoint.get('best_epoch', -1)}, best mIoU={checkpoint.get('best_metric_val', 0):.4f})."
    )
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint.get("global_step", 0)),
        checkpoint.get("history") or {"train_step": [], "val_step": [], "train_epoch": [], "val_epoch": []},
        float(checkpoint.get("best_metric_val", -float("inf"))),
        int(checkpoint.get("best_epoch", -1)),
        float(checkpoint.get("early_best", -float("inf"))),
        int(checkpoint.get("stale_epochs", 0)),
        checkpoint.get("best_val_metrics", {}),
    )


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        out[mask == class_id] = MFNET_COLORMAP[class_id]
    out[mask == IGNORE_INDEX] = IGNORE_COLOR
    out[(mask < 0) | ((mask >= NUM_CLASSES) & (mask != IGNORE_INDEX))] = IGNORE_COLOR
    return out


@torch.no_grad()
def save_epoch_predictions(
    model: nn.Module,
    loader,
    epoch: int,
    device: torch.device,
    output_dir: str,
    num_samples: int = 4,
    use_amp: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    examples = []
    for images, targets in loader:
        images_dev = images.to(device, non_blocking=True)
        with _autocast(device, use_amp):
            logits = model(images_dev)
        pred = logits.argmax(dim=1).cpu()
        for i in range(images.shape[0]):
            examples.append((images[i].cpu(), targets[i].cpu(), pred[i]))
            if len(examples) >= max(1, num_samples):
                break
        if len(examples) >= max(1, num_samples):
            break
    if not examples:
        return

    fig, axes = plt.subplots(len(examples), 4, figsize=(16, 3.2 * len(examples)), squeeze=False)
    for idx, (image, target, pred) in enumerate(examples):
        image_np = image.numpy().transpose(1, 2, 0)
        rgb = np.clip(image_np[:, :, :3], 0, 1)
        thermal = np.clip(image_np[:, :, 3], 0, 1)
        axes[idx, 0].imshow(rgb)
        axes[idx, 1].imshow(thermal, cmap="gray", vmin=0, vmax=1)
        axes[idx, 2].imshow(_mask_to_rgb(pred.numpy()))
        axes[idx, 3].imshow(_mask_to_rgb(target.numpy()))
        axes[idx, 0].set_ylabel(f"Sample {idx + 1}", fontsize=10, fontweight="bold")
        for ax in axes[idx]:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, title in zip(axes[0], ("RGB", "Thermal", "Predicted mask", "Ground truth")):
        ax.set_title(title, fontweight="bold")
    patches = [
        mpatches.Patch(color=MFNET_COLORMAP[i] / 255.0, label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=NUM_CLASSES, fontsize=8,
               bbox_to_anchor=(0.5, -0.005), frameon=True)
    fig.suptitle(f"Epoch {epoch:03d} · sample predictions", fontweight="bold")
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    out_path = os.path.join(output_dir, f"epoch_{epoch:03d}_predictions.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "latest_predictions.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved predictions: {out_path}")


def plot_history(history: Dict[str, list], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    train_epochs = history.get("train_epoch", [])
    val_epochs = history.get("val_epoch", [])
    train_steps = history.get("train_step", [])

    if train_steps:
        xs = [item["global_step"] for item in train_steps]
        ys = [item["loss"] for item in train_steps]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(xs, ys, alpha=0.45, label="Train loss (logged steps)")
        if len(ys) >= 5:
            window = min(20, max(5, len(ys) // 10))
            smooth = np.convolve(ys, np.ones(window) / window, mode="valid")
            ax.plot(xs[window - 1:], smooth, linewidth=2, label=f"Moving average ({window})")
        ax.set(title="Step-level Training Loss", xlabel="Global step", ylabel="Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "step_train_loss.png"), dpi=150)
        plt.close(fig)

    if train_epochs and val_epochs:
        tx = [entry["epoch"] for entry in train_epochs]
        vx = [entry["epoch"] for entry in val_epochs]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tx, [entry["loss"] for entry in train_epochs], "o-", label="Train")
        ax.plot(vx, [entry["loss"] for entry in val_epochs], "s--", label="Validation")
        ax.set(title="Epoch Loss", xlabel="Epoch", ylabel="Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "epoch_loss.png"), dpi=150)
        plt.close(fig)

        for key, title, ylabel, filename in (
            ("miou", "Mean IoU (all 9 classes)", "mIoU", "miou_all_classes.png"),
            ("miou_foreground", "Foreground mIoU (classes 1–8)", "mIoU", "miou_foreground.png"),
            ("pixel_acc", "Pixel Accuracy", "Accuracy", "pixel_accuracy.png"),
            ("dice", "Mean Dice", "Dice", "dice_score.png"),
        ):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(tx, [entry.get(key, 0.0) for entry in train_epochs], "o-", label="Train")
            ax.plot(vx, [entry.get(key, 0.0) for entry in val_epochs], "s--", label="Validation")
            ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, filename), dpi=150)
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for ax, key, title in zip(
            axes, ("miou", "pixel_acc", "dice"),
            ("Mean IoU (9 classes)", "Pixel accuracy", "Mean Dice")
        ):
            ax.plot(tx, [entry.get(key, 0.0) for entry in train_epochs], "o-", label="Train")
            ax.plot(vx, [entry.get(key, 0.0) for entry in val_epochs], "s--", label="Validation")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
        fig.suptitle("Epoch Metrics")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "epoch_metrics.png"), dpi=150)
        plt.close(fig)

        lrs = [entry.get("lr_decoder", None) for entry in train_epochs]
        if any(value is not None for value in lrs):
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(tx, lrs, linewidth=2, label="Decoder/fusion LR")
            ax.plot(tx, [entry.get("lr_encoder", 0.0) for entry in train_epochs],
                    linewidth=2, label="Encoder LR")
            ax.set(title="Learning Rate by Epoch", xlabel="Epoch", ylabel="Learning rate")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "learning_rate.png"), dpi=150)
            plt.close(fig)

        # Classwise scores from the best validation epoch, not the final epoch.
        best_entry = max(val_epochs, key=lambda entry: entry.get("miou", -1))
        iou_values = best_entry.get("iou_per_class", [])
        if iou_values:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(CLASS_NAMES[:len(iou_values)], iou_values)
            ax.set(title=f"Validation IoU by Class (epoch {best_entry['epoch']})",
                   ylabel="IoU", ylim=(0, 1))
            ax.tick_params(axis="x", rotation=35)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "per_class_iou.png"), dpi=150)
            plt.close(fig)

    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    print(f"[plots] Training curves and history saved under '{output_dir}'.")


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epochs: int,
    start_epoch: int,
    global_step: int,
    history: Dict[str, list],
    best_metric_val: float,
    best_epoch: int,
    early_best: float,
    stale_epochs: int,
    device: torch.device,
    checkpoint_dir: str,
    plot_dir: str,
    config: Dict[str, Any],
    log_every: int = 20,
    vis_every: int = 5,
    vis_samples: int = 4,
    patience: int = 12,
    min_delta: float = 0.0015,
    use_amp: bool = True,
    save_every: int = 10,
) -> Dict[str, list]:
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    if not history:
        history = {"train_step": [], "val_step": [], "train_epoch": [], "val_epoch": []}

    train_metrics = SegmentationMetrics()
    val_metrics = SegmentationMetrics()
    model.to(device)
    print(f"\nStarting training: epochs {start_epoch}->{epochs} | device={device} | AMP={use_amp and device.type == 'cuda'}")
    print(f"Train batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Plots: {plot_dir}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_metrics.reset()
        train_loss_total = torch.zeros((), device=device, dtype=torch.float32)
        train_sample_count = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch:03d}/{epochs:03d} [train]", dynamic_ncols=True)

        for step, (images, targets) in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch = images.shape[0]

            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if scaler.is_enabled():
                # Backpropagate exactly once; a second backward frees/reuses the same graph incorrectly.
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            train_loss_total += loss.detach().float() * batch
            train_sample_count += batch
            train_metrics.update(logits.detach(), targets)
            global_step += 1

            if (step + 1) % log_every == 0 or (step + 1) == len(train_loader):
                running_loss = (train_loss_total / max(1, train_sample_count)).item()
                running_metrics = train_metrics.compute()
                lr_enc = optimizer.param_groups[0]["lr"]
                lr_dec = optimizer.param_groups[1]["lr"]
                pbar.set_postfix(
                    loss=f"{running_loss:.3f}",
                    mIoU=f"{running_metrics['miou']:.3f}",
                    fg_mIoU=f"{running_metrics['miou_foreground']:.3f}",
                    lr=f"{lr_dec:.1e}",
                )
                history["train_step"].append({
                    "epoch": epoch,
                    "step": step + 1,
                    "global_step": global_step,
                    "loss": float(running_loss),
                    "miou": float(running_metrics["miou"]),
                    "miou_foreground": float(running_metrics["miou_foreground"]),
                    "lr_encoder": float(lr_enc),
                    "lr_decoder": float(lr_dec),
                })

        train_metrics_result = train_metrics.compute()
        train_loss = (train_loss_total / max(1, train_sample_count)).item()
        train_epoch_record = {
            "epoch": epoch,
            "loss": float(train_loss),
            **train_metrics_result,
            "lr_encoder": float(optimizer.param_groups[0]["lr"]),
            "lr_decoder": float(optimizer.param_groups[1]["lr"]),
        }
        history["train_epoch"].append(train_epoch_record)

        model.eval()
        val_metrics.reset()
        val_loss_total = torch.zeros((), device=device, dtype=torch.float32)
        val_sample_count = 0
        pbar_val = tqdm(enumerate(val_loader), total=len(val_loader),
                        desc=f"Epoch {epoch:03d}/{epochs:03d} [val]", dynamic_ncols=True)
        with torch.no_grad():
            for step, (images, targets) in pbar_val:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                batch = images.shape[0]
                with _autocast(device, use_amp):
                    logits = model(images)
                    loss = criterion(logits, targets)
                val_loss_total += loss.detach().float() * batch
                val_sample_count += batch
                val_metrics.update(logits, targets)
                if (step + 1) % log_every == 0 or (step + 1) == len(val_loader):
                    running_metrics = val_metrics.compute()
                    running_loss = (val_loss_total / max(1, val_sample_count)).item()
                    pbar_val.set_postfix(
                        loss=f"{running_loss:.3f}",
                        mIoU=f"{running_metrics['miou']:.3f}",
                        fg_mIoU=f"{running_metrics['miou_foreground']:.3f}",
                    )
                    history["val_step"].append({
                        "epoch": epoch,
                        "step": step + 1,
                        "global_step": global_step,
                        "loss": float(running_loss),
                        "miou": float(running_metrics["miou"]),
                        "miou_foreground": float(running_metrics["miou_foreground"]),
                    })

        val_metrics_result = val_metrics.compute()
        val_loss = (val_loss_total / max(1, val_sample_count)).item()
        val_epoch_record = {
            "epoch": epoch,
            "loss": float(val_loss),
            **val_metrics_result,
        }
        history["val_epoch"].append(val_epoch_record)

        # Reduce LR when validation mIoU stalls.
        scheduler.step(val_metrics_result["miou"])
        current_metric = val_metrics_result["miou"]
        raw_improved = current_metric > best_metric_val
        significant_improvement = current_metric > early_best + min_delta

        if raw_improved:
            best_metric_val = current_metric
            best_epoch = epoch

        if significant_improvement:
            early_best = current_metric
            stale_epochs = 0
        else:
            stale_epochs += 1

        lr_enc = optimizer.param_groups[0]["lr"]
        lr_dec = optimizer.param_groups[1]["lr"]
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train loss={train_loss:.4f} mIoU={train_metrics_result['miou']:.4f} "
            f"fg_mIoU={train_metrics_result['miou_foreground']:.4f} | "
            f"Val loss={val_loss:.4f} mIoU={val_metrics_result['miou']:.4f} "
            f"fg_mIoU={val_metrics_result['miou_foreground']:.4f} "
            f"pixel_acc={val_metrics_result['pixel_acc']:.4f} dice={val_metrics_result['dice']:.4f} | "
            f"LR(enc/dec)={lr_enc:.2e}/{lr_dec:.2e}"
        )
        print("  Val IoU/class: " + " | ".join(
            f"{CLASS_NAMES[i]}:{score:.3f}"
            for i, score in enumerate(val_metrics_result["iou_per_class"])
        ))

        # Save a full resumable checkpoint every epoch, isolated from old MFNet checkpoints.
        ckpt_args = dict(
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            history=history,
            best_metric_val=best_metric_val,
            best_epoch=best_epoch,
            early_best=early_best,
            stale_epochs=stale_epochs,
            config=config,
            best_val_metrics=val_metrics_result if raw_improved else {},
        )
        save_checkpoint(os.path.join(checkpoint_dir, "last_checkpoint.pt"), **ckpt_args)
        if raw_improved:
            save_checkpoint(os.path.join(checkpoint_dir, "best_checkpoint.pt"), **ckpt_args)
            print(f"  [checkpoint] New best validation mIoU={best_metric_val:.4f} at epoch {best_epoch}.")

        # Periodic model-only checkpoints are smaller and useful for inference snapshots.
        if save_every > 0 and epoch % save_every == 0:
            torch.save(
                {
                    "architecture_id": ARCHITECTURE_ID,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics_result,
                    "config_snapshot": config,
                },
                os.path.join(checkpoint_dir, f"model_epoch_{epoch:03d}.pt"),
            )

        if raw_improved or epoch % max(1, vis_every) == 0 or epoch == epochs:
            save_epoch_predictions(
                model, val_loader, epoch, device, plot_dir, vis_samples, use_amp=use_amp
            )

        # Atomic history snapshot means progress is recoverable if the run is interrupted.
        history_path = os.path.join(plot_dir, "training_history.json")
        temp_history = history_path + ".tmp"
        with open(temp_history, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        os.replace(temp_history, history_path)

        if patience > 0 and stale_epochs >= patience:
            print(
                f"[early stopping] Validation mIoU did not improve by {min_delta:.4f} "
                f"for {patience} epochs. Best epoch={best_epoch}, mIoU={best_metric_val:.4f}."
            )
            break

    plot_history(history, plot_dir)
    print(f"[finished] Best validation mIoU={best_metric_val:.4f} at epoch {best_epoch}.")
    print(f"[finished] Resumable checkpoint: {os.path.join(checkpoint_dir, 'last_checkpoint.pt')}")
    print(f"[finished] Best checkpoint: {os.path.join(checkpoint_dir, 'best_checkpoint.pt')}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the pretrained custom RGB-thermal cross-modal segmentation model on MFNet."
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Learning rate for fusion/decoder; pretrained encoders use 0.1x.")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume only from a last_checkpoint.pt created by this architecture.")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--plot-dir", type=str, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-h", type=int, default=480)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--vis-every", type=int, default=5,
                        help="Save sample predictions every N epochs; best epochs are always saved.")
    parser.add_argument("--vis-samples", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--patience", type=int, default=12,
                        help="Early-stopping patience based on validation mIoU; 0 disables it.")
    parser.add_argument("--min-delta", type=float, default=0.0015,
                        help="Minimum mIoU improvement that resets early stopping.")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Write a compact model-only checkpoint every N epochs; 0 disables.")
    parser.add_argument("--class-weights", choices=("auto", "mfb", "enet", "none"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not download/use ImageNet ResNet-18 weights (not recommended for this small dataset).")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed-precision training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = DEVICE
    use_amp = not args.no_amp and device.type == "cuda"

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.log_every < 1:
        raise ValueError("--log-every must be >= 1")

    print("=" * 78)
    print("Custom Dual-Stream ResNet-18 + Multi-Scale Cross-Modal Attention (v2)")
    print(f"Architecture={ARCHITECTURE_ID} | device={device} | AMP={use_amp}")
    print(f"Batch={args.batch_size} | epochs={args.epochs} | output resolution=({args.img_h},{args.img_w})")
    print(f"ImageNet pretrained encoders={not args.no_pretrained}")
    print("=" * 78)

    model = build_model(
        num_classes=NUM_CLASSES,
        in_channels=4,
        pretrained=not args.no_pretrained,
    ).to(device)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Model trainable parameters: {trainable_params:,}")

    encoder_params = list(model.rgb_encoder.parameters()) + list(model.thermal_encoder.parameters())
    encoder_ids = {id(param) for param in encoder_params}
    fusion_decoder_params = [
        param for param in model.parameters() if id(param) not in encoder_ids and param.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr * 0.1, "name": "pretrained_encoders"},
            {"params": fusion_decoder_params, "lr": args.lr, "name": "fusion_decoder"},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        threshold=0.001,
        threshold_mode="abs",
        min_lr=[1e-6, 1e-5],
    )
    scaler = _make_grad_scaler(use_amp)
    class_weights = get_class_weights(args.class_weights, device=device)
    criterion = CombinedLoss(NUM_CLASSES, IGNORE_INDEX, class_weights=class_weights)

    history: Dict[str, list] = {"train_step": [], "val_step": [], "train_epoch": [], "val_epoch": []}
    start_epoch = 1
    global_step = 0
    best_metric = -float("inf")
    best_epoch = -1
    early_best = -float("inf")
    stale_epochs = 0
    best_val_metrics: Dict[str, Any] = {}

    if args.resume:
        (
            start_epoch,
            global_step,
            history,
            best_metric,
            best_epoch,
            early_best,
            stale_epochs,
            best_val_metrics,
        ) = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )

    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        img_size=(args.img_h, args.img_w),
        num_workers=args.num_workers,
        auto_download=True,
    )

    config = _config_snapshot(args)
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epochs=args.epochs,
        start_epoch=start_epoch,
        global_step=global_step,
        history=history,
        best_metric_val=best_metric,
        best_epoch=best_epoch,
        early_best=early_best,
        stale_epochs=stale_epochs,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        plot_dir=args.plot_dir,
        config=config,
        log_every=args.log_every,
        vis_every=args.vis_every,
        vis_samples=args.vis_samples,
        patience=args.patience,
        min_delta=args.min_delta,
        use_amp=use_amp,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
