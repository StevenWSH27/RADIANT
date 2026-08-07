"""Train RADIANT on collaborative multi-modal fusion data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from radiant_config import RADIANTConfig
from radiant_dataset import RADIANTTrainingDataset
from radiant_model import RADIANT, RADIANTLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RADIANT using IVIF and MFIF collaborative pairs."
    )
    parser.add_argument(
        "--train-data-root",
        "--trainDataRoot",
        dest="train_data_root",
        default="./train_data",
        help="root containing the four expected training subdirectories",
    )
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--run-dir", default="./runs")
    parser.add_argument("--resume", default=None, help="checkpoint to resume")
    parser.add_argument("--epochs", type=int, default=RADIANTConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=RADIANTConfig.batch_size)
    parser.add_argument("--image-size", type=int, default=RADIANTConfig.Height)
    parser.add_argument("--lr", type=float, default=RADIANTConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=RADIANTConfig.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=RADIANTConfig.grad_clip)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=RADIANTConfig.num_tasks)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument("--amp", action="store_true", help="enable CUDA mixed precision")
    parser.add_argument(
        "--dry-run", action="store_true", help="stop after one optimization step"
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(
            "resume expects a recovered training checkpoint containing model_state"
        )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_loss", float("inf")))


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss, options):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "architecture": "radiant",
        "epoch": epoch,
        "best_loss": best_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": vars(options),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def scalar_losses(losses):
    return {name: float(value.detach().cpu()) for name, value in losses.items()}


def main():
    options = parse_args()
    if options.epochs < 1 or options.batch_size < 1 or options.image_size < 8:
        raise ValueError("epochs/batch-size must be positive and image-size must be >= 8")

    seed_everything(options.seed)
    device = resolve_device(options.device)
    amp_enabled = bool(options.amp and device.type == "cuda")

    transform = transforms.Compose(
        [transforms.Resize((options.image_size, options.image_size)), transforms.ToTensor()]
    )
    dataset = RADIANTTrainingDataset(options.train_data_root, transform=transform)
    if options.max_samples is not None:
        if options.max_samples < 1:
            raise ValueError("max-samples must be positive")
        dataset = Subset(dataset, range(min(options.max_samples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=options.num_workers > 0,
    )

    model = RADIANT(
        RADIANTConfig.s,
        RADIANTConfig.n,
        RADIANTConfig.channel,
        RADIANTConfig.stride,
        num_tasks=options.num_tasks,
    ).to(device)
    criterion = RADIANTLoss().to(device)
    optimizer = AdamW(
        model.parameters(), lr=options.lr, weight_decay=options.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, options.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_loss = float("inf")
    if options.resume:
        start_epoch, best_loss = load_checkpoint(
            options.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    checkpoint_dir = Path(options.checkpoint_dir)
    run_dir = Path(options.run_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.jsonl"

    print(
        f"Training {len(dataset)} samples on {device}; "
        f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}; "
        f"amp={amp_enabled}"
    )

    for epoch in range(start_epoch, options.epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        epoch_start = time.time()

        for step, batch in enumerate(loader, start=1):
            infrared, visible, focus_near, focus_far = [
                tensor.to(device, non_blocking=True) for tensor in batch
            ]
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                ivif_outputs = model.forward_with_radiant(
                    infrared,
                    visible,
                    x_mfif=focus_near,
                    y_mfif=focus_far,
                    training_tag=1,
                    task_id=0,
                )
                ivif_loss, ivif_items = criterion(
                    ivif_outputs,
                    {
                        "source1": infrared,
                        "source2": visible,
                        "target_rec": visible,
                    },
                )

                mfif_outputs = model.forward_with_radiant(
                    focus_near,
                    focus_far,
                    x_mfif=infrared,
                    y_mfif=visible,
                    training_tag=2,
                    task_id=1,
                )
                mfif_loss, mfif_items = criterion(
                    mfif_outputs,
                    {
                        "source1": focus_near,
                        "source2": focus_far,
                        "target_rec": focus_far,
                    },
                )
                total_loss = 0.5 * (ivif_loss + mfif_loss)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            value = float(total_loss.detach().cpu())
            running_loss += value
            steps += 1
            if step == 1 or step % options.log_every == 0:
                print(
                    f"epoch={epoch + 1}/{options.epochs} step={step}/{len(loader)} "
                    f"loss={value:.6f} lr={optimizer.param_groups[0]['lr']:.3e}"
                )

            record = {
                "epoch": epoch + 1,
                "step": step,
                "total": value,
                "ivif": scalar_losses(ivif_items),
                "mfif": scalar_losses(mfif_items),
            }
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

            if options.dry_run:
                break

        scheduler.step()
        epoch_loss = running_loss / max(steps, 1)
        improved = epoch_loss < best_loss
        if improved:
            best_loss = epoch_loss

        save_checkpoint(
            checkpoint_dir / "radiant_latest.pt",
            model,
            optimizer,
            scheduler,
            epoch + 1,
            best_loss,
            options,
        )
        if improved:
            save_checkpoint(
                checkpoint_dir / "radiant_best.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_loss,
                options,
            )
        if (epoch + 1) % options.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"radiant_epoch-{epoch + 1:03d}.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_loss,
                options,
            )

        print(
            f"epoch={epoch + 1} mean_loss={epoch_loss:.6f} "
            f"elapsed={time.time() - epoch_start:.1f}s"
        )
        if options.dry_run:
            break


if __name__ == "__main__":
    main()
