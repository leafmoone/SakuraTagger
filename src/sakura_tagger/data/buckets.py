"""Deterministic full-frame resolution choices with explicit aspect error."""
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class BucketConfig:
    target_pixels: int = 512 * 512
    max_pixels: int = 272 * 1024
    max_side: int = 2048
    alignment: int = 16
    preferred_error: float = 0.01
    max_error: float = 0.02

    def __post_init__(self):
        if min(self.target_pixels, self.max_pixels, self.max_side, self.alignment) <= 0:
            raise ValueError("Pixel, side and alignment settings must be positive")
        if not 0 <= self.preferred_error <= self.max_error:
            raise ValueError("Require 0 <= preferred_error <= max_error")


@dataclass(frozen=True)
class BucketChoice:
    original_size: tuple[int, int]  # width, height after EXIF correction
    target_size: tuple[int, int]
    aspect_error: float  # symmetric relative ratio error; zero is exact
    patch_count: int
    pixel_budget_deviation: float  # signed fraction from target_pixels


class BucketAspectError(ValueError):
    pass


@lru_cache(maxsize=32)
def _landscape_candidates(config):
    step = config.alignment
    candidates = []
    for height in range(step, config.max_side + 1, step):
        max_width = min(config.max_side, config.max_pixels // height)
        for width in range(height, max_width + 1, step):
            candidates.append((width, height))
    if not candidates:
        raise ValueError("No aligned bucket fits max_pixels and max_side")
    return tuple(candidates)


class BucketSelector:
    def __init__(self, config: BucketConfig | None = None):
        self.config = config or BucketConfig()
        self._candidates = _landscape_candidates(self.config)

    def select(self, width: int, height: int) -> BucketChoice:
        if width <= 0 or height <= 0:
            raise ValueError("Original width and height must be positive")
        portrait = height > width
        ratio = max(width, height) / min(width, height)
        ranked = []
        for w, h in self._candidates:
            candidate_ratio = w / h
            error = max(candidate_ratio / ratio, ratio / candidate_ratio) - 1
            ranked.append((error, w, h))
        preferred = [item for item in ranked if item[0] <= self.config.preferred_error]
        valid = preferred or [item for item in ranked if item[0] <= self.config.max_error]
        if not valid:
            best = min(item[0] for item in ranked)
            raise BucketAspectError(
                f"No bucket fits aspect {width}:{height} within {self.config.max_error:.2%}; "
                f"best aspect error={best:.2%}, max_side={self.config.max_side}, max_pixels={self.config.max_pixels}"
            )
        # Once inside an acceptable error band, favor the intended token budget.
        error, w, h = min(valid, key=lambda item: (
            abs(item[1] * item[2] - self.config.target_pixels), item[0], item[1], item[2]
        ))
        target = (h, w) if portrait else (w, h)
        pixels = w * h
        return BucketChoice((width, height), target, error,
                            pixels // self.config.alignment ** 2,
                            pixels / self.config.target_pixels - 1)
