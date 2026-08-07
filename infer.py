"""Fuse paired images with RADIANT or a retained legacy baseline checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from radiant_config import RADIANTConfig
from radiant_model import RADIANT, TwoBranchesFusionNet


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
TASK_IDS = {
    "ivif": 0,
    "mfif": 1,
    "meif": 2,
    "nir-vis": 3,
    "remote": 4,
    "medical": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(description="RADIANT image fusion inference")
    parser.add_argument(
        "--source1-root", "--test_ir_root", dest="source1_root", required=True
    )
    parser.add_argument(
        "--source2-root", "--test_vis_root", dest="source2_root", required=True
    )
    parser.add_argument("--save-path", "--save_path", dest="save_path", default="./outputs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--architecture", choices=("auto", "radiant", "legacy"), default="auto"
    )
    parser.add_argument("--task", choices=tuple(TASK_IDS), default="ivif")
    parser.add_argument("--source1-rgb", "--IR_IS_RGB", dest="source1_rgb", action="store_true")
    parser.add_argument("--source2-rgb", "--VIS_IS_RGB", dest="source2_rgb", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def default_checkpoint():
    candidates = (
        Path("checkpoints/radiant_best.pt"),
        Path("checkpoints/radiant_latest.pt"),
        Path("checkpoints/legacy_gifnet.model"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "no checkpoint found; pass --checkpoint or train a model into ./checkpoints"
    )


def _load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def load_model(checkpoint_path, architecture, device):
    payload = _load_payload(checkpoint_path)
    if isinstance(payload, dict) and "model_state" in payload:
        state = payload["model_state"]
        saved_architecture = payload.get("architecture")
        config = payload.get("config") or {}
    elif isinstance(payload, dict) and payload and all(
        torch.is_tensor(value) for value in payload.values()
    ):
        state = payload
        saved_architecture = None
        config = {}
    else:
        raise ValueError("unsupported checkpoint format")

    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    if architecture == "auto":
        radiant_keys = (
            "decomposition.",
            "task_memory.",
            "multi_grain_interaction.",
        )
        is_radiant = saved_architecture == "radiant" or any(
            key.startswith(radiant_keys) for key in state
        )
        architecture = "radiant" if is_radiant else "legacy"

    if architecture == "radiant":
        model = RADIANT(
            RADIANTConfig.s,
            RADIANTConfig.n,
            RADIANTConfig.channel,
            RADIANTConfig.stride,
            num_tasks=int(config.get("num_tasks", RADIANTConfig.num_tasks)),
        )
    else:
        model = TwoBranchesFusionNet(
            RADIANTConfig.s,
            RADIANTConfig.n,
            RADIANTConfig.channel,
            RADIANTConfig.stride,
        )

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"checkpoint is incompatible with architecture={architecture}: {error}"
        ) from error
    model.to(device).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Loaded {architecture} checkpoint from {checkpoint_path} ({parameters:,} parameters)")
    return model, architecture


def _natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _image_index(root, recursive):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    paths = root.rglob("*") if recursive else root.iterdir()
    index = {}
    for path in paths:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            relative = path.relative_to(root)
            key = relative.with_suffix("").as_posix()
            if key in index:
                raise RuntimeError(f"duplicate image stem under {root}: {key}")
            index[key] = path
    if not index:
        raise FileNotFoundError(f"no supported images found in: {root}")
    return index


def paired_images(source1_root, source2_root, recursive=False):
    source1 = _image_index(source1_root, recursive)
    source2 = _image_index(source2_root, recursive)
    common = sorted(set(source1) & set(source2), key=lambda value: _natural_key(Path(value)))
    if not common:
        raise RuntimeError("the two input directories contain no matching relative stems")
    missing_1 = len(set(source2) - set(source1))
    missing_2 = len(set(source1) - set(source2))
    if missing_1 or missing_2:
        print(
            "Warning: skipped unmatched inputs "
            f"(source1 missing={missing_1}, source2 missing={missing_2})"
        )
    return [(key, source1[key], source2[key]) for key in common]


def load_source(path, rgb, target_size=None):
    image = Image.open(path)
    if target_size is not None and image.size != target_size:
        image = image.resize(target_size, Image.Resampling.BICUBIC)
    if rgb:
        ycbcr = np.asarray(image.convert("RGB").convert("YCbCr"), dtype=np.float32)
        luminance = ycbcr[:, :, 0]
        chroma = (ycbcr[:, :, 1], ycbcr[:, :, 2])
    else:
        luminance = np.asarray(image.convert("L"), dtype=np.float32)
        chroma = None
    tensor = torch.from_numpy(luminance / 255.0).unsqueeze(0).unsqueeze(0)
    return tensor, chroma, image.size


def fuse_chroma(first, second):
    if first is None:
        return second
    if second is None:
        return first
    fused = []
    for channel_1, channel_2 in zip(first, second):
        weight_1 = np.abs(channel_1 - 128.0)
        weight_2 = np.abs(channel_2 - 128.0)
        denominator = weight_1 + weight_2
        channel = np.where(
            denominator > 1e-6,
            (channel_1 * weight_1 + channel_2 * weight_2) / np.maximum(denominator, 1e-6),
            128.0,
        )
        fused.append(channel)
    return tuple(fused)


def run_model(model, architecture, source1, source2, task_id):
    if architecture == "radiant":
        return model(
            source1,
            source2,
            task_id=task_id,
            training_tag=1 if task_id == 0 else 2,
            update_memory=False,
        )
    features = model.forward_encoder(source1, source2)
    task_features = model.forward_MultiTask_branch(
        fea_com_ivif=features, fea_com_mfif=features, trainingTag=2
    )
    return model.forward_mixed_decoder(features, task_features)


def save_result(path, luminance, chroma):
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(luminance * 255.0, 0, 255).astype(np.uint8)
    if chroma is None:
        Image.fromarray(y, mode="L").save(path)
        return
    cb = np.clip(chroma[0], 0, 255).astype(np.uint8)
    cr = np.clip(chroma[1], 0, 255).astype(np.uint8)
    ycbcr = Image.fromarray(np.stack((y, cb, cr), axis=-1), mode="YCbCr")
    ycbcr.convert("RGB").save(path)


def main():
    options = parse_args()
    device = resolve_device(options.device)
    checkpoint = Path(options.checkpoint) if options.checkpoint else default_checkpoint()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    model, architecture = load_model(checkpoint, options.architecture, device)
    output_root = Path(options.save_path)

    pairs = paired_images(
        options.source1_root, options.source2_root, recursive=options.recursive
    )
    with torch.inference_mode():
        for index, (relative_stem, source1_path, source2_path) in enumerate(pairs, start=1):
            source1, chroma1, size = load_source(source1_path, options.source1_rgb)
            source2, chroma2, _ = load_source(
                source2_path, options.source2_rgb, target_size=size
            )
            source1 = source1.to(device=device, dtype=torch.float32)
            source2 = source2.to(device=device, dtype=torch.float32)
            fused = run_model(
                model, architecture, source1, source2, TASK_IDS[options.task]
            )
            fused = fused[0, 0].clamp(0, 1).cpu().numpy()
            chroma = fuse_chroma(chroma1, chroma2)
            output_path = output_root / f"{relative_stem}{source1_path.suffix.lower()}"
            save_result(output_path, fused, chroma)
            print(f"[{index}/{len(pairs)}] {output_path}")


if __name__ == "__main__":
    main()
