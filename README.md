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
