"""All legacy and new-task inputs from a single DINOv3 invocation."""
from dataclasses import dataclass
from torch import Tensor


@dataclass(frozen=True)
class FeatureBundle:
    global_features: Tensor
    cls_tokens: Tensor
    patch_tokens: Tensor
    artist_logits: Tensor | None = None
    style_embedding: Tensor | None = None
