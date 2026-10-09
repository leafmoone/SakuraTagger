# SakuraTagger V1

Three new classification modules on one strictly loaded Kaloscope/DINOv3
backbone. The original 44,129-class Artist head and 256D Style projector remain
registered and frozen. The pinned upstream submodule is unmodified; no ComfyUI,
torchvision, GPU or formal training dataset is needed for CPU engineering checks.

## Install

Python 3.11+ and an editable checkout are required because the adapter loads the
pinned upstream beside `src/`. A standalone wheel omits that submodule.

```sh
git submodule update --init --recursive
python -m venv .venv
. .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m sakura_tagger.selfcheck
```

Selfcheck uses a tiny, random **12-block upstream DINOv3** with small class counts.
It performs synthetic CPU forwards/backwards and optimizer/checkpoint checks,
never formal model training. It reports real weights as `NOT_VERIFIED` unless an
existing official package is explicitly supplied:

```sh
python -m sakura_tagger.selfcheck --real-model-dir checkpoints/v1-artist-classifier
```

Missing real weights are never replaced with random weights. The optional real
check verifies strict CPU loading and output parity, not real-data accuracy or
GPU performance. Generated checkpoints, data, temporary tests and logs are Git
ignored. The durable acceptance entry above is part of the supported program.

## Original Kaloscope compatibility

Place `model.safetensors`, `config.json`, and `class_mapping.csv` from the
[official v1 package](https://huggingface.co/heathcliff01/Kaloscope3.0-preview/tree/main/v1-artist-classifier)
in a directory you choose. Files and all tensor keys are strictly validated,
including auxiliary temperature/bias tensors and the original normalization
protocol. No weights are downloaded automatically.

```python
from sakura_tagger import load_kaloscope
from sakura_tagger.model import KaloscopeBackbone
legacy = load_kaloscope("checkpoints/v1-artist-classifier")
backbone = KaloscopeBackbone(legacy)
features = backbone(images)  # RGB/ImageNet-normalized float [B,3,H,W]
```

Image H/W must be divisible by 16. Features are CLS+patch mean `[B,1536]`, CLS
`[B,768]`, native-grid patch tokens `[B,N,768]`, optional raw original Artist
logits `[B,44129]` and Style `[B,256]`. A feature bundle invokes DINOv3 once.
Direct backbone calls include the legacy outputs by default; callers can pass
`include_artist=False, include_style=False`. The original Artist path is FP32 L2
normalization followed by multiplication by `sqrt(1536)` and its frozen linear
head. Style receives the original unnormalized pooled features. Synthetic
fixtures derive their smaller dimensions from their own architecture.

The complete original state remains registered and independently strict-loadable.
T2-modified backbone states belong to new training checkpoints; do not overwrite
the original package. Original Artist/Style parameters never directly update,
although their predictions can change when T2 changes their input features.

## Three physical modules

- General: 16 learned queries, 256D attention with four heads; query concatenation
  is learned down to 512D, added to a global residual, then classified. No second
  General contrastive projection or 4096D-to-vocabulary classifier is created.
- Identity: shared 1536→512 adapter and 512→(Character+Copyright) classifier.
  Domain-local outputs retain separate IDs, masks, losses and metrics. Both are
  multi-label; equal names across domains are distinct labels.
- Other Artist: direct 1536D normalized input to its own linear classifier. It
  does not consume original Artist logits or Style vectors.

Candidate counts are 37,679 General, 50,369 Character + 17,664 Copyright, and
21,506 Other Artist. Final sizes come from the versioned vocabulary. The
candidate new heads have 91,355,378 trainable parameters; selfcheck reports each
module and actual synthetic frozen/trainable counts separately.

```python
from sakura_tagger.data import Vocabulary
from sakura_tagger.model import MultiTaskModel
vocabulary = Vocabulary.read("data/vocabulary.json")
model = MultiTaskModel(backbone, vocabulary.layout())
output = model(images, head_selection=["general", "character", "copyright"])
# output.logits contains raw requested classification outputs.
model.eval()
prediction = model.predict(images, ["general", "other_artist"], top_k=20)
# Each classification result: {"class_ids": [B,K], "scores": [B,K]}.
```

Selectable names are `general`, `character`, `copyright`, `other_artist`,
`artist`, `style_embedding`. Omitted selection requests all. General attention,
Identity and unrelated legacy branches are physically skipped when unselected;
requesting either identity domain executes the shared Identity module once.
Prediction defaults to Top-K, not full dense probabilities; `dense=True` is an
explicit opt-in. Original Artist retains its original softmax semantics, and all
four new domains use independent sigmoids. Uncalibrated original/other Artist
scores are never combined into one ranking. Call `eval()` before inference.
`configs/v1_dev.yaml` is a formal candidate architecture/training configuration;
old per-semantic-group neural heads have been removed.

## Native full-frame buckets

```python
from PIL import Image
from sakura_tagger.data import BucketSelector, NativePreprocessor
selector = BucketSelector()
prepared = NativePreprocessor(selector)(Image.open("image.png"))
prediction = model.predict(prepared.tensor.unsqueeze(0), ["general"])
print(prepared.bucket)
```

The shared preprocessor applies EXIF orientation, composites existing alpha onto
white (configurable), converts to RGB and resizes the complete canvas. It never
crops or pads. Native buckets use aligned 16px dimensions near 512² pixels, with
reported symmetric ratio error. The preferred error band is 1%, maximum 2% by
default; closest area wins within the first usable band. Impossible ratios raise
an error. Indexed ratio bands plus a bounded aspect cache replace full scans;
the baseline tie-break and portrait/landscape symmetry are preserved.

The official 512-square center crop differs from these full-frame inputs, so
preserving weight computations does **not** establish accuracy equivalence.
Dataset evaluation and threshold calibration remain necessary.

## Dataset, losses and training API

Read the [dataset/mask and sampler contract](docs/DATASET.md) and
[training/checkpoint interface](docs/TRAINING.md). SakuraPool supplies images and
annotations. Tagger never crawls, downloads, silently reorders IDs, or interprets
missing annotations as negatives.

T1 (`frozen_heads`) trains only the three new modules with detached backbone
features. T2 (`finetune_last4`) enables DINOv3 blocks 8–11 and final norm; blocks
0–7, embeddings and original Artist/Style stay frozen/eval. Last-four/norm
modules follow train/eval mode, and their input path stays differentiable.
Optional teacher preservation uses a separately loaded **original**, frozen
Kaloscope only when explicitly configured; it is never a second student.

Masked ASL defaults to gamma-positive 0, gamma-negative 4 and clipping 0.05.
Character and Copyright are normalized/weighted independently. Masks are
mandatory for provided targets. Chunked recomputation limits redundant FP32 loss
intermediates; all-empty masks return differentiable zero. Trusted Other Artist
single-label CE and multi-label ASL are explicit modes. Optional identity metric
loss requires caller-supplied relations and is disabled by default.

Trainer supports gradient accumulation, CPU BF16/CUDA AMP configuration,
mask-aware evaluation, stage-aware optimizer updates, and complete boundary
checkpoints. No training starts merely by importing or constructing it. This
repository's V1 acceptance scope is CPU engineering only; real weights, GPU,
large-scale distributed throughput, formal data and calibrated accuracy need
separate validation.

See [upstream/license boundaries](docs/UPSTREAM.md) before use or redistribution.
This project does not grant a blanket MIT or Apache license.
