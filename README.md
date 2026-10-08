# SakuraTagger

A standalone PyTorch integration of Kaloscope's frozen DINOv3 artist classifier
and style projector. ComfyUI, torchvision, a GPU, and a training dataset are not
required. The upstream source remains an unmodified, pinned Git submodule.

## Install from this checkout

Python 3.11 or newer is required. Install a CPU build of PyTorch first when no
GPU is available:

```sh
git submodule update --init --recursive
python -m venv .venv
. .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

The editable checkout is required because the adapter reads the pinned source
submodule beside `src/`; a standalone wheel does not contain third-party code.

## Official model package

Place `model.safetensors`, `config.json`, and `class_mapping.csv` from the
[v1-artist-classifier package](https://huggingface.co/heathcliff01/Kaloscope3.0-preview/tree/main/v1-artist-classifier)
in `checkpoints/v1-artist-classifier/`, or provide your own directory argument.
Weights and generated data are ignored by Git. Missing files, unexpected or
missing tensor keys, key collisions, and mismatched class mappings are errors;
there is no random-weight fallback.

```python
from sakura_tagger import load_kaloscope
model = load_kaloscope("checkpoints/v1-artist-classifier")  # CPU by default
# images: float [B, 3, H, W], H/W divisible by 16, normalized using
# mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225).
style, artist_logits = model(images, return_both=True)
```

Tensor inputs must already be RGB and ImageNet-normalized. No crop, padding,
or forced square resizing is performed. Full-image preprocessing is described
separately as the native bucket API becomes available. Official outputs are
256-dimensional style vectors and 44,129 artist logits, based on 1,536-dimensional
CLS-plus-mean features. Artist input uses the original `l2_sqrt_dim` normalization;
the style projector receives the original pooled features.

The development-only **Synthetic** fixture uses a small randomly initialized
upstream DINOv3. It is kept in local, Git-ignored tests and is not an official
checkpoint, accuracy benchmark, or substitute for real weight verification.

See [upstream sources and license boundaries](docs/UPSTREAM.md) before use or
redistribution. This project does not grant a blanket MIT or Apache license.

## Shared frozen features

```python
from sakura_tagger.model import KaloscopeBackbone
backbone = KaloscopeBackbone(model)
features = backbone.forward_features(images)
# features.global_features: [B, 1536]
# features.cls_tokens: [B, 768]; patch_tokens: [B, (H/16)*(W/16), 768]
# features.artist_logits: [B, 44129]; style_embedding: [B, 256]
```

Every feature bundle shares exactly one DINOv3 forward. The original backbone,
artist head, and style projector remain frozen and in evaluation mode even when
a containing training model calls `train()`. Features are ordinary detached
tensors that newly added heads can consume with normal autograd. The complete
original state, including auxiliary temperature and bias tensors, remains in the
wrapper's `state_dict()` and supports strict reload. Tiny Synthetic models use
smaller dimensions derived from their own configuration.

## Native resolution buckets

```python
from PIL import Image
from sakura_tagger.data import BucketConfig, BucketSelector, NativePreprocessor
selector = BucketSelector(BucketConfig(target_pixels=512*512, max_pixels=278528,
                                      max_side=2048, max_error=0.02))
preprocess = NativePreprocessor(selector)
prepared = preprocess(Image.open("image.png"))
features = backbone(prepared.tensor.unsqueeze(0))
print(prepared.bucket)  # Original/target W,H, aspect error, patches, budget deviation
```

Use this same preprocessor for training and inference. It applies EXIF orientation,
composites existing transparent pixels onto white (configurable), converts to RGB,
resizes the **complete** image canvas, and applies ImageNet normalization. It does
not crop or add padding. Both dimensions are multiples of 16. Finite aligned
buckets introduce a reported small aspect-ratio deviation; the error is symmetric
under width/height exchange, `max(target_ratio/original_ratio,
original_ratio/target_ratio)-1`.

The selector searches all legal aligned buckets within the side/pixel limits.
It first considers candidates within 1% aspect error, then up to the configured
2% limit if necessary, and within that band chooses area closest to the target.
The default maximum allows a small pixel-budget fluctuation; set `max_pixels` to
`target_pixels` for a strict upper bound. Unrepresentable ratios raise an error
with the best available deviation instead of silently cropping or padding.
These dynamically generated buckets are provisional, not a final dataset-derived
bucket list.

`BucketBatchSampler(sizes, batch_size, selector=selector)` groups indices into
same-size batches. Supply `(width,height)` sizes **after EXIF correction** and use
the same selector in preprocessing. Optional `shuffle=True`, `seed`, and
`set_epoch()` give reproducible ordering; `drop_last=True` drops the final short
batch of each bucket, otherwise every index appears once.

Preserving legacy weight computations does not establish unchanged accuracy:
Kaloscope's original 512-square center-crop preprocessing differs from these
full-frame dynamic inputs. Real-data evaluation and threshold calibration are
still needed; no accuracy equivalence is claimed.
