# Fixed-cohort evaluation and calibration

The `sakura_tagger.evaluation` API evaluates a declared set of unique published
validation, calibration or test sample IDs. It does not choose a changing random
cohort or evaluate the entire dataset by default. `EvaluationCohort` binds the
Release, Vocabulary, ordered sample IDs and sparse labels, split, class/support
scope, protocol and optional real SourceMembership source union. The same identity
must be used for checkpoint comparisons. `configs/evaluation.yaml` is an explicit
engineering example; it does not select a real cohort or validated quality gates.

## Metrics and missing annotations

- `observed_label_proxy` treats unobserved labels as evaluation zero only within
  the declared cohort/class scope. Results are named observed-label AP and are
  incomplete-annotation proxies. No dataset facts or training targets are changed.
- `verified_scope` uses only actual sparse positive and explicit negative facts.
  Unknown pairs are omitted. A class without enough positive evidence or a
  comparable negative is `UNAVAILABLE`, with the exact reason and support counts.
  No manual full-vocabulary negative-labeling task is required.

AP uses the complete raw logits for each class over the whole cohort. It groups
tied scores and computes non-interpolated precision weighted by recall increments,
matching [scikit-learn average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
No batch AP, rank AP, class-block AP or truncated Top-K values are averaged to
approximate global AP. `merge_score_partitions` verifies exact fixed-cohort ID
coverage and merges raw scores before metric calculation.

Macro AP averages available classes. Reports include available/declared/total
class counts and every original class ID's status, support, AP and recall. Micro
AP performs one global ordering over every protocol-known pair in the explicit
`available_classes_all_protocol_known_pairs` class scope. Availability depends on
fixed labels/support rules, not on checkpoint scores. If nothing is computable,
the aggregate is null rather than an invented zero. Change protocol/cohort/scope
only as a separately reported evaluation, never silently between checkpoints.

General, Character and Copyright have independent Macro/Micro AP and frequency
band recall/AP. Bands are 0, 1, 2–9, 10–99, 100–999 and 1000+ positives in the
Release's independent unique train samples, before RFS. Recall uses an explicit
probability threshold or a frozen calibration artifact. Reports record every
class's effective threshold, complete evaluation settings and the fixed-threshold
policy or frozen calibration identity/method. A separate evaluation identity
changes with these settings; threshold-only changes preserve the score-independent
cohort/feature-cache identity. It is computed for known
positive labels even where AP has no negative comparison.

Other Artist and Original Artist are separate trusted-single-artist Top-1, Top-5
and Macro Top-1 reports, never a merged ranking. Missing, conflicting, unconfirmed
or out-of-scope targets are excluded. Original Artist requires explicit sample or
Release single-artist confirmation too. Top-K scans classifier blocks and keeps
original IDs, with ID-order tie breaking. An optional fixed original reference
model/cache reports old Artist metric differences on the exact same cohort and
class scope; otherwise the comparison is `NOT_VERIFIED`.

Style uses cosine-similarity Recall@K only with a supplied reference gallery and
explicit relevant gallery IDs for each query. Same sample IDs are excluded.
Queries without available gallery positives are counted separately; absence of
a gallery is `NOT_VERIFIED`. Synthetic retrieval checks are not real Style quality.

## Feature caching and explicit resource limits

```python
from sakura_tagger.evaluation import (
    EvaluationConfig, EvaluationCohort, build_feature_cache, evaluate,
)
cohort = EvaluationCohort(index, fixed_sample_ids, split="val",
                         config=EvaluationConfig(ram_bytes=512 * 1024**2,
                                                 disk_bytes=2 * 1024**3))
# Batches contain images plus sample_ids, exactly in cohort order, once each.
cache = build_feature_cache(model, batches, cohort, "cache/eval/features",
                            checkpoint_id=checkpoint_identity,
                            preprocessing=preprocessing_identity)
report = evaluate(model, cache, cohort, spool_dir="cache/eval/spool")
```

Each image batch runs the student backbone once. General/Identity task features,
global features and Style vectors are stored as FP32 memory-mapped arrays; patch
tokens and full classification outputs are not retained. Evaluation applies the
terminal classifier in bounded class and image-row blocks. Teacher/reference
feature construction, when requested, is a separate explicit image pass.

A cache binds both the supplied checkpoint identity and a digest of the actual
loaded model state, plus cohort, preprocessing, structure and Other Artist mode.
A different checkpoint or model state cannot silently reuse the cache. Reopen
with `FeatureCache(path, binding=feature_binding(...))`; evaluator checks the
current model/cohort binding before use. No cross-checkpoint feature reuse is
claimed, even when an architecture happens to match.

The implementation deliberately supports an exact small-cohort route. It spools
only one multilabel domain's complete scores to temporary disk, calculates AP,
then releases that spool. Exact Micro AP needs global sorting; it refuses work
before score generation when the configured RAM budget is insufficient. Its
conservative planning allowance is 96 bytes per cohort/class pair, 2048 bytes per
reported class, plus class-block work buffers. Disk budgeting includes compact
feature bytes and one domain score spool. Empty class scopes remain well-defined.
Actual free disk is checked before allocation too. There is no histogram AP
fallback masquerading as exact AP and no unbounded dense matrix allocation.

These are incremental evaluator/feature/score working-array budgets, not total
process or GPU-memory guarantees: the already loaded model, dataset reader and
backbone activations need their own machine capacity. Reduce the explicitly fixed
cohort/class scope or raise budgets intentionally after inspecting the estimate.
`row_batch`, `class_block`, `max_samples`, RAM and disk bytes are configurable.
Cached features/spools are runtime artifacts and are not committed to Git.

## Validation-only thresholds, leakage and model selection

`fit_calibration(model, cache, cohort, spool_dir=...)` accepts only validation or
calibration splits. It maximizes protocol-specific F1, choosing the highest
threshold for tied optima. A class below configurable positive/negative support
falls back to its domain's validation-fitted threshold. If the whole domain lacks
comparable evidence, it is marked unavailable and uses an explicitly uncalibrated
0.5 default. No low-support per-class optimum is claimed.

The JSON-compatible artifact freezes checkpoint/model/Vocabulary/Release,
preprocessing, protocol, source/class scope, cohort/support settings, method,
thresholds and calibration image SHA/known-family evidence. `evaluate` validates
that binding and can apply it to test, but never refits there. AP ranking remains
independent of the threshold; recall changes according to the frozen threshold.
Calibration and test must be SHA- and known-family-disjoint.

Cohort creation streams train metadata once to check exact image-SHA and known
family overlaps, rejecting observed leakage. Without complete family evidence,
it reports SHA-disjoint and family `NOT_VERIFIED`. Near-duplicate status strings
are reported, but without verified duplicate-group identities they cannot prove
near-duplicate disjointness. The evaluator does not invent missing family data.

Model selection defaults to `REPORT_ONLY` with `general.macro_ap` as the primary
metric. A caller may configure named minimums for other domains and original
Artist reference deltas. Only those explicit gates are assessed; the result never
means general production readiness. There is no mechanical 127K-class all-task
average serving as the sole selection rule, and real quality floors remain unset.

## CPU acceptance

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m sakura_tagger.selfcheck --suite evaluation
```

This checks tied AP against the independent scikit-learn oracle, a global Micro
example differing from averaged class AP, raw partition merge equivalence,
protocol/unavailable differences, train-only frequencies, actual cached feature
and classifier outputs, one backbone invocation per image batch, binding/budget
rejections, separate artists and reference deltas, validation-only threshold
fitting with low-support fallback and frozen test use, leakage evidence and
optional Style retrieval. Real-data accuracy, missing-label recovery, reference
gallery quality and quality gates remain `NOT_VERIFIED` or unconfigured.
