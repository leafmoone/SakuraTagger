# Training and recovery API

The API prepares engineering paths. Importing it does not start a training run.
No formal training or CUDA run is part of the V1 CPU acceptance.

```python
from torch.utils.data import DataLoader
from sakura_tagger.data import ManifestDataset, BucketBatchSampler, repeat_factors
from sakura_tagger.training import Trainer, TrainerConfig, LossConfig, checkpoint_metadata

dataset = ManifestDataset("data/manifest.jsonl", "data/metadata.json", "data/vocabulary.json")
factors, frequencies = repeat_factors(dataset.records, dataset.vocabulary)
sampler = BucketBatchSampler(
    [(row["width"], row["height"]) for row in dataset.records], 8,
    selector=dataset.preprocessor.selector, shuffle=True, seed=42, repeat_weights=factors)
loader = DataLoader(dataset, batch_sampler=sampler)
# model is the strictly loaded Kaloscope-backed MultiTaskModel from README.
metadata = checkpoint_metadata(model, dataset.vocabulary, dataset.metadata,
                               dataset.preprocessor.selector.config, "your-original-Kaloscope-version")
trainer = Trainer(model, metadata, config=TrainerConfig(accumulation_steps=4),
                  loss_config=LossConfig(chunk_size=4096), device="cpu")
# Explicitly call these only when authorized and actual data is ready:
# summary = trainer.train_epoch(loader, sampler=sampler)
# trainer.save_checkpoint("outputs/t1.ckpt", sampler=sampler)
# metrics = trainer.evaluate(validation_loader)
```

`checkpoint_metadata` binds original Kaloscope version, dataset schema, dataset
ID, canonical manifest contents **and order**, vocabulary SHA, physical layout,
bucket/stage settings and all ordered class mappings. Original Artist mapping
must exactly match the loaded legacy model, including names and IDs.
`ManifestDataset` computes its manifest fingerprint once over record metadata;
no image byte hashing is performed in the hot path. Treat manifest order as
immutable while resuming: a changed order, label set or source record changes the
fingerprint and must not silently resume an earlier sampler position.

`TrainerConfig` selects stage, learning rate, decay, accumulation, gradient clip
and AMP (`off`, `bf16`, or CUDA-only `fp16`). CPU is default. `set_stage()` is
allowed only at optimizer boundaries and rebuilds optimizer membership while
preserving shared parameters' Adam state. New T2 backbone parameters are included.
Do not build an optimizer once from only T1 parameters and silently reuse it for
T2. The trainer is a single-process training API; the distributed sampler is an
integration interface, not a claimed fully tested multi-GPU launch system.
External DDP integrations must handle unequal rank lengths with join and align
accumulation/optimizer boundaries. Distributed/GPU behavior is NOT_VERIFIED.

## Batches and supervision

A batch has `images: [B,3,H,W]`, and dictionaries `targets`, `masks` over the four
logical domains. All multi-label targets/masks are `[B,C]`, mask dtype bool.
Every supplied target needs an explicit mask; absent domains are skipped.
Unknown NaN targets are excluded before evaluating a loss. Observed targets must
be finite binary values. Each logical loss divides by its observed element count,
then applies its independent configured weight. Gradient accumulation averages
microbatch losses, with a correctly rescaled final partial accumulation. It does
not claim an element-weighted mean across unequal-supervision microbatches.

General `LossConfig.general_groups` can map semantic names to General IDs for
separate detached loss statistics. These groups do not introduce physical heads.
`LossConfig.weights` independently weights General, Character, Copyright and
Other Artist. Default ASL settings are gamma 0/4 and negative clip 0.05.
Class chunks are converted to FP32 only as needed, with non-reentrant checkpoint
recomputation to avoid retaining every FP32 loss intermediate until backward.
Dense low-precision logits themselves are still produced by the classifiers.

`other_artist_mode="single_label"` requires trusted [B] integer class targets and
[B] boolean sample masks; known single-artist rows imply an exclusive class over
the full Other Artist vocabulary. `trusted_single_artist(targets, masks)` converts
canonical multi-label tensors only where **all** labels are observed and exactly
one positive exists. Partially labeled or multi-artist rows remain masked out.
The default `multilabel` mode uses masked ASL and supports trusted multi-artist
annotation. Do not interpret missing artist labels as evidence of exclusivity.

Evaluation returns separate domain losses, valid counts and masked correct/TP/FP/FN
counts, including Character and Copyright independently. The default threshold
is sigmoid 0.5 (zero raw logits), a placeholder requiring calibration. Empty
supervision reports zero loss/count. No validation accuracy is implied by a
successful synthetic run.

## Optional auxiliary objectives

All preservation weights default to zero. With any nonzero preservation weight,
provide an independently loaded `KaloscopeBackbone(original_legacy)` as `teacher`;
zero weights require `teacher=None`. Shared student/teacher storage and mismatched
legacy mappings are rejected. The teacher is forced frozen/eval, runs under
no_grad, and is never optimized. This exception is the only additional DINOv3
model: the ordinary student still has one backbone and one forward.

`TrainerConfig.preservation` independently weights `global_features` MSE,
`style_embedding` MSE and `artist_logits` KL divergence. Only required teacher
legacy branches run. Student preservation gradients can flow through frozen
Artist/Style operations into allowed T2 blocks. Teacher state is included in a
teacher-enabled training checkpoint so it remains reproducible.

`identity_metric_weight=0` is default. To enable it, supply an `identity_metric`
callable and each batch's `identity_relations`. The callable receives shared
Identity-adapter features and the caller-owned relation object and returns a
scalar loss. Relationships are never invented or hardcoded; callers own their
device placement and semantics. No contrastive pretraining stage is started.

## Checkpoints

`save_checkpoint(path, sampler=...)` requires a completed optimizer boundary.
If accumulation is pending, complete it or explicitly `flush()` first; the API
refuses to discard partial gradients. Save as `.pt` or `.ckpt`, separately from
the original `.safetensors` package. Writes are atomic within the chosen folder.

The checkpoint includes model, optimizer, AMP scaler, teacher when enabled,
Python/NumPy/PyTorch RNG, epoch/microstep/optimizer step, exact sampler configuration
and its acknowledged cursor. `train_batch(..., sampler=...)` advances only after
successful backward. DataLoader prefetch never moves the saved cursor. An epoch
completion advances the epoch and resets the sampler for the next one.

Create a trainer with the matching stage/configuration, then call
`load_checkpoint(path, sampler=...)`. Schema/vocabulary/base model/dataset/order/
layout/buckets/stage/settings are checked before model mutation. Tensor layouts
are checked before strict load. Loading uses PyTorch `weights_only=True`.
Stage changes, new datasets and optimizer-policy changes are not transparent
resume operations; initialize a deliberate new run rather than weakening checks.

The CPU selfcheck verifies exact next-update equality after optimizer/RNG/sampler
restore, rejects metadata mismatch and partial-accumulation saves, and exercises
both T1/T2 gradient boundaries, independent teacher, BF16 and actual image reads.
