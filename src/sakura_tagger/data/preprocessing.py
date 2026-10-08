"""Shared training/inference full-frame RGB preprocessing, without crop/pad."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps
import torch

from ..adapter import IMAGENET_MEAN, IMAGENET_STD
from .buckets import BucketChoice, BucketSelector


@dataclass(frozen=True)
class PreprocessedImage:
    tensor: torch.Tensor  # [3,H,W], ImageNet-normalized
    bucket: BucketChoice


def prepare_rgb(image: Image.Image, alpha_background=(255, 255, 255)) -> Image.Image:
    """Composite transparency on the existing canvas; this adds no border."""
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (*alpha_background, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


class NativePreprocessor:
    def __init__(self, selector: BucketSelector | None = None, *, alpha_background=(255, 255, 255)):
        self.selector = selector or BucketSelector()
        if self.selector.config.alignment != 16:
            raise ValueError("DINOv3 native image preprocessing requires alignment=16")
        self.alpha_background = alpha_background
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> PreprocessedImage:
        oriented = ImageOps.exif_transpose(image)
        bucket = self.selector.select(*oriented.size)
        rgb = prepare_rgb(oriented, self.alpha_background)
        resized = rgb.resize(bucket.target_size, resample=Image.Resampling.BICUBIC)
        pixels = np.array(resized, dtype=np.float32, copy=True) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        return PreprocessedImage((tensor - self.mean) / self.std, bucket)
