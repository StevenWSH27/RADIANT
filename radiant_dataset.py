"""Paired training datasets for RADIANT."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Optional

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _index_images(directory: Path):
    if not directory.is_dir():
        raise FileNotFoundError(f"training directory does not exist: {directory}")
    indexed = {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not indexed:
        raise FileNotFoundError(f"no supported images found in: {directory}")
    return indexed


class RADIANTTrainingDataset(Dataset):
    """Four-stream collaborative RADIANT training dataset.

    Expected layout::

        root/infrared/train/*
        root/visible/train/*
        root/visible_focus_near/train/*
        root/visible_focus_far/train/*

    Files are paired by stem rather than by a hard-coded ``1.jpg`` sequence,
    so gaps and non-JPEG inputs are supported.
    """

    stream_directories = (
        "infrared/train",
        "visible/train",
        "visible_focus_near/train",
        "visible_focus_far/train",
    )

    def __init__(
        self,
        root,
        image_numbers: Optional[Iterable[int | str]] = None,
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform
        indices = [
            _index_images(self.root / relative)
            for relative in self.stream_directories
        ]

        common_stems = set(indices[0])
        for index in indices[1:]:
            common_stems.intersection_update(index)

        if image_numbers is not None:
            requested = [str(number) for number in image_numbers]
            missing = [stem for stem in requested if stem not in common_stems]
            if missing:
                preview = ", ".join(missing[:5])
                raise FileNotFoundError(
                    f"{len(missing)} requested samples are incomplete; "
                    f"first missing stems: {preview}"
                )
            stems = requested
        else:
            stems = sorted(common_stems, key=_natural_key)

        if not stems:
            raise RuntimeError(
                "the four training directories do not contain any matching file stems"
            )
        self.samples = [tuple(index[stem] for index in indices) for stem in stems]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images = [Image.open(path).convert("L") for path in self.samples[idx]]
        if self.transform is not None:
            images = [self.transform(image) for image in images]
        return tuple(images)
