"""Deterministic full-frame resolution choices with explicit aspect error."""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
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


@lru_cache(maxsize=32)
def _ratio_index(config):
    candidates = tuple(sorted(_landscape_candidates(config), key=lambda item: item[0] / item[1]))
    return tuple(w / h for w, h in candidates), candidates


@lru_cache(maxsize=65536)
def _choose_ratio(config, numerator, denominator):
    ratio = numerator / denominator
    ratios, candidates = _ratio_index(config)
    for limit in (config.preferred_error, config.max_error):
        # Widen by an ULP before applying the original exact error expression.
        # This preserves candidates at floating-point band boundaries.
        lower = math.nextafter(ratio / (1 + limit), -math.inf)
        upper = math.nextafter(ratio * (1 + limit), math.inf)
        left, right = bisect_left(ratios, lower), bisect_right(ratios, upper)
        valid = []
        for w, h in candidates[left:right]:
            error = max((w / h) / ratio, ratio / (w / h)) - 1
            if error <= limit:
                valid.append((error, w, h))
        if valid:
            return min(valid, key=lambda item: (
                abs(item[1] * item[2] - config.target_pixels), item[0], item[1], item[2]))
    position = bisect_left(ratios, ratio)
    neighbors = ratios[max(0, position - 1):position + 1]
    best = min(max(candidate / ratio, ratio / candidate) - 1 for candidate in neighbors)
    raise BucketAspectError(
        f"No bucket fits aspect {numerator}:{denominator} within {config.max_error:.2%}; "
        f"best aspect error={best:.2%}, max_side={config.max_side}, max_pixels={config.max_pixels}"
    )


class BucketSelector:
    def __init__(self, config: BucketConfig | None = None):
        self.config = config or BucketConfig()
        self._candidates = _landscape_candidates(self.config)
        _ratio_index(self.config)

    def select(self, width: int, height: int) -> BucketChoice:
        if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
            raise ValueError("Original width and height must be positive integers")
        divisor = math.gcd(width, height)
        error, w, h = _choose_ratio(self.config, max(width, height) // divisor, min(width, height) // divisor)
        target = (h, w) if height > width else (w, h)
        pixels = w * h
        return BucketChoice((width, height), target, error,
                            pixels // self.config.alignment ** 2,
                            pixels / self.config.target_pixels - 1)
