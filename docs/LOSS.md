# Sparse Positive–Unknown engineering objective

`training.positive_unknown` implements one WAN-inspired engineering variant.
It is **not an unbiased estimator of true PU risk**, does not solve missing labels
by proof, and has no real-data accuracy guarantee. Weak negatives are a training
assumption only; no sampled term is written back to the dataset.

## Per-image objective

For each of General, Character and Copyright independently:

```
P = observed positives, all retained
N = explicitly confirmed negatives, possibly empty
U = eligible trained classes minus P, N, and explicit forbidden classes

L_i = mean_P softplus(-z) + lambda_u * mean_sampled_U softplus(z)
      + lambda_n * mean_N softplus(z)
```

Empty terms contribute a differentiable zero. The weak-U term requires at least
one P for that image/domain. N-only rows can still contribute their explicit N
term when lambda_n > 0. Completely unknown rows have zero numerator and zero
valid count; they never get a full-vocabulary negative gradient. P-only rows
remain valid even when U is empty. A row with N only and lambda_n = 0 is inactive.

`DomainLoss` returns the **sum of these image objectives** and the number of
valid images. Its mean is numerator/count. `SparseLossResult` retains separate
Character/Copyright terms and applies explicit task weights only when combining
the domain means. The supervision-only `sparse_valid_counts` helper can count a
complete accumulation window before constructing any forward graph. DDP uses
these numerators/counts, not an average of rank means or label counts.

Other Artist uses only the explicit trusted CE targets/validity from the Release
contract. It never enters the weak-negative loss. Original Artist/Style have no
new classification target in this objective.

## Scope and sampling

`WeakNegativeScope.from_index(index)` uses only positive-supported classes from
the frozen Release's unique train samples. `NO_TRAIN_POSITIVE` classes cannot
enter U, but every explicit P is kept. To constrain U by vocabulary source,
pass a validated `SourceMembership` and explicit run `source_filter` to that
factory and select `weak_negative_scope: source_membership` in the loss config.
Missing indexes, unknown sources, or mismatched Vocabulary/Release identities
fail. The actual allowed IDs and source selection contribute to the saved scope
identity. Image-level source strings are never consulted to infer label scope.

Sampling is uniform without replacement. K >= |U| uses all U exactly. The sampled
mean is never multiplied by K or |U|. There is no hard-negative mining,
frequency-weighted sampling, or pseudo-labeling in this implementation.
`SamplingKey(sample_id, epoch, occurrence_id, seed)` derives a stable domain seed
with SHA256 rather than Python's process-randomized hash. Worker order and
prefetch cannot advance this sampling state. Occurrences of one sample have
separate keys, and resume can reconstruct a key without replaying earlier RNG.

`WeakNegativeConfig` requires explicit lambda_u, lambda_n and sample_size. Modes:

- `full_reference`: exact full-U objective
- `chunked_reference`: the same objective with bounded class chunks
- `sampled_weak_negative`: uniform sampled-U objective
- `masked_asl`: known P/N only, reusing the legacy ASL function per image

The new sparse ASL wrapper averages known entries per image and then effective
images. The old `masked_asymmetric_loss` and `multitask_loss` APIs remain unchanged
and use their existing known-label normalization. Old multilabel checkpoints
continue through the legacy `LossConfig`; this new Release boundary deliberately
requires `other_artist_mode: single_label` rather than guessing the mode.

Softplus and CE/logsumexp use FP32. Excluded unknown/forbidden entries, including
masked NaNs, are gathered away before arithmetic. Selected nonfinite logits are
a real training failure, not a reason to reinterpret facts as negatives. Class
chunks can be recomputed during backward. The model still generates full logits:
**loss sampling does not imply sampled-row classifier multiplication**.

## Actual API

```python
from sakura_tagger.training import (
    WeakNegativeScope, SparseLossConfig, sparse_multitask_loss,
)
scope = WeakNegativeScope.from_index(index)
config = SparseLossConfig.from_dict(configuration["sparse_loss"])
result = sparse_multitask_loss(
    outputs.logits, records, release=index.release, scope=scope, config=config,
    epoch=epoch, occurrence_ids=occurrence_ids,
)
result.total.backward()  # single effective batch; DDP composes numerators/counts
```

The baseline and Identity-local files contain runnable engineering examples with
lambda_u=1, lambda_n=1, K=2048. These values are **not tuned or recommended as
optimal on the real dataset**. Constructors/imports perform no training.

`python -m sakura_tagger.selfcheck --suite positive_unknown` checks enumerated
subset expectations and gradients against the reference, sign/exclusion tests,
N-only/empty/one-class/BF16/NaN cases, deterministic keys, train-support/source
restrictions, sparse multitask CE, and a tiny CPU learnable fixture. Synthetic
optimization success does not establish real missing-label quality.
