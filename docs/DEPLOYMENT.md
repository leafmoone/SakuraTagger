# Strict deployment packages and output policies

`load_predictor` reconstructs and checks a self-contained inference asset package:
student weights including original modules, exact Vocabulary, architecture/base
configuration, stage, Other Artist training mode, native preprocessing, explicit
thresholds or frozen validation calibration, optional SourceMembership and
optional Release-bound NO_TRAIN_POSITIVE metadata. The installed Tagger code and
pinned upstream architecture are still required. No model weights or data are
downloaded on load, and tensor shapes never guess the loss/probability mode.

```python
from sakura_tagger import load_predictor
predictor = load_predictor("outputs/tagger-v1")
# Existing tensor API remains valid; inputs are native preprocessed RGB tensors.
candidates = predictor.predict(images, ["general", "artist"], top_k=20)
# A fixed Top-K candidate list does not promise complete General annotation.
tags = predictor.predict(images, ["general", "character", "copyright"],
                         output_mode="threshold", max_tags=100,
                         source_filter=["danbooru"])
# PIL input follows exactly the package's full-frame preprocessing contract.
result = predictor.predict_image(opened_pil_image, ["general"], top_k=20)
```

## Output contract

- `top_k` is the existing default: each classifier returns `class_ids` and `scores`
  shaped `[B,K]`, with original domain-local IDs. `top_k=20` is a compatibility
  default for candidate retrieval, not a completeness or quality claim.
- `dense=True` or `output_mode="dense"` preserves the old contract: no source
  filter returns `[B,C]` tensors; source filtering returns aligned original IDs
  plus scores. The matrix dimensions and Vocabulary order are never truncated.
- `output_mode="threshold"` returns `class_ids` and `scores` as lists of 1D tensors,
  one per image. Empty lists of predictions are supported. Results are sorted by
  score, with original-ID tie order, then optionally capped by positive `max_tags`.
  It requires explicit thresholds for every requested classification domain.
  Style vectors remain unchanged and do not take class thresholds.

Source filtering precedes Top-K, thresholds and caps. General/Character/Copyright
use sigmoid. Original Artist and CE-trained (`single_label`) Other Artist always
compute full-vocabulary softmax before any filtering; retained scores are never
renormalized. Legacy `multilabel` Other Artist uses its declared sigmoid mode.
The two artist candidate lists stay separate; there is no gate or unified ranking.

The direct model API accepts the same new options while retaining old defaults.
Explicit domain and class thresholds use original IDs:

```python
thresholds = {
    "general": {"default": 0.5, "per_class": {"12": 0.65}},
    "character": 0.6,
}
output = model.predict(images, ["general", "character"],
                       output_mode="threshold", thresholds=thresholds)
```

These numbers are illustrative, uncalibrated settings. A package can instead
embed an E7 validation-fitted calibration artifact. It must match the actual
model-state digest, checkpoint, Vocabulary and preprocessing. Its per-domain
fallback and class thresholds are loaded without fitting or adjustment. Package-
default threshold output is restricted to the artifact's declared original class
IDs and source-derived class scope; a domain fallback never silently extends to
uncalibrated labels. An explicit call-time threshold override is a deliberate
uncalibrated policy and may use a broader class scope. Artist
thresholds are not invented from multilabel calibration: request those domains
in Top-K mode or supply their thresholds explicitly.

## NO_TRAIN_POSITIVE and exact Vocabulary

`untrained_label_metadata(index)` derives unsupported original IDs from the fixed
Release's independent train-positive support for General, Character, Copyright
and Other Artist. It is bound to that Release and exact Vocabulary. By default,
these IDs cannot appear in Top-K or threshold output even if the requested K
exceeds the number of supported classes. Source-filtered dense output excludes
them too. Unfiltered dense output keeps `[B,C]` and zeroes unsupported scores;
metadata records why they are unavailable. This is output policy, not a change to
weights, IDs or classifier dimensions, and does not renormalize softmax.

Original Artist is never suppressed because the new Release lacks examples of
some old classes. Old models without an explicit metadata binding keep their
prior behavior. `suppress_untrained=False` is an explicit diagnostic opt-out; it
does not mean unsupported labels have been learned. SourceMembership is read once
at package load and retained as an immutable in-memory index; there is no per-batch
Parquet access or unbounded device/source cache.

`model.bind_vocabulary(vocabulary)` binds exact canonical class names/order as
well as dimensions and the original Artist mapping. The formal sparse trainer
sets this binding from its Release. A legacy unbound model needs an explicit,
trusted Vocabulary binding before export. Rebinding to a same-size reordered or
renamed Vocabulary fails. Loading normally uses the package's exact Vocabulary;
an optional `expected_vocabulary` must match it byte-canonically.

## Export and package validation

```python
from sakura_tagger.deployment import export_predictor, untrained_label_metadata
export_predictor(model, vocabulary, "outputs/tagger-v1",
                 checkpoint_id=checkpoint_identity,
                 preprocessor=preprocessor,
                 calibration=frozen_validation_calibration,
                 untrained=untrained_label_metadata(index),
                 source_membership=membership)
```

Choose calibration or explicit `thresholds`, not both. Without either, Top-K/dense
remain available and threshold output requires call-time settings. Export preserves
already bound untrained metadata and SourceMembership by default. It writes a new
version directory through a temporary directory and refuses to overwrite an old
package. No optimizer state, TAR cache or training records are required to load.

`manifest.json` binds actual model/Vocabulary identities, configuration and each
package-local file's size/SHA-256. `model.safetensors` is strict-loaded, including
every original Artist/Style tensor. Loader checks versions, files, weight digest,
class/layout agreement, preprocessing, explicit Other Artist mode, calibration
and supported-label metadata. Membership is checked against the package's exact
Vocabulary, including same-size label name/order changes. Missing/extra tensor
keys or malformed/incompatible metadata fail rather than silently falling back.
These integrity hashes do not authenticate an untrusted publisher's identity.

## CPU acceptance

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m sakura_tagger.selfcheck --suite deployment
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m sakura_tagger.selfcheck --suite source
```

The original 15 source checks remain unchanged. New checks cover filtered
thresholds/caps/empty outputs, full artist softmax, unsupported new-label policy,
strict dense/Top-K/threshold package round trips, RGBA/PIL preprocessing, actual
validation-calibration packaging, exact Vocabulary and membership rejection,
malformed modes/structure/files/preprocessing, Identity-local and legacy mode.
Five seeded baseline tensor-mode comparisons are bitwise. The separate PIL and
seeded Identity-local reconstruction checks require exact class IDs and use
explicit score `atol=1e-7, rtol=1e-6`, reporting measured errors rather than assuming
bitwise floating-point parity across separately allocated upstream models. Real Kaloscope
weights, real source membership, production data quality and GPU execution remain
`NOT_VERIFIED`.
