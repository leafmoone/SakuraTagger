# Source membership V1

This optional inference-only sidecar associates existing vocabulary labels with
original tag sources. It is **not image annotation provenance**: an image's
source does not prove that all its labels belong to that source vocabulary.
Cross-source training, four supervision domains, five vocabulary ID spaces,
three new classification modules and the original Artist/Style computations are
unchanged. No relabeling, vocabulary regeneration or string-based guessing is
performed. Same names in different domains remain distinct.

## Parquet schema and binding

Suggested release-relative path: `vocab/source_membership.parquet`.
Required columns:

- `domain`: Arrow string or large_string; one of `general`, `character`,
  `copyright`, `other_artist`, `artist`
- `class_id`: Arrow integer; non-null original zero-based domain-local ID
- `source`: Arrow string or large_string; non-null canonical lowercase source
  identifier matching `[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*`, e.g. `danbooru`,
  `zerochan`, `gamecg`

Optional columns: `namespace`, `evidence_type`, each Arrow string/large_string
with optional null values. They provide evidence context, never separate class
identity. Unknown or duplicate columns and incorrect column types are rejected.
Required fields may not contain null values. An empty table is valid with the
explicit schema, but it declares no known sources.

Required UTF-8 Arrow schema metadata (byte keys and values):

- `schema_version`: `sakura-tagger-source-membership-v1`
- `vocab_version`: exact `vocabulary.document['version']`, currently
  `sakura-tagger-vocabulary-v1`
- `vocab_sha256`: existing `vocabulary.sha256`, unchanged

The reader checks version and SHA against the caller's authoritative Vocabulary,
then validates every relationship. Negative/out-of-range/noninteger IDs, unknown
domains, noncanonical source strings and duplicate `(domain,class_id,source)`
relationships are errors. Different evidence fields do not permit duplicate
relationships. A label may belong to multiple sources. No entry implies no
membership, not an automatically inferred source.

`artist` refers only to the original 44,129-class Kaloscope Artist vocabulary;
`other_artist` refers only to the new head. Synthetic vocabularies may use smaller
counts. `set_source_membership` verifies new-head dimensions and exact original
Artist mapping/count against the loaded model.

Use the Vocabulary belonging to the loaded training checkpoint (its existing
`vocab_sha256` and `class_id_mapping` metadata), not an arbitrary same-size
vocabulary. A bare model has only new-head dimensions, so it cannot infer those
new domains' semantic identities from tensor shapes. The caller's authoritative
Vocabulary is mandatory for both constructors below; Tagger neither regenerates
it nor overrides its digest.

## Python interface

```python
from sakura_tagger.data import SourceMembership, SOURCE_MEMBERSHIP_VERSION

# Real I/O; requires the optional `parquet` extra.
membership = SourceMembership.read("vocab/source_membership.parquet", vocabulary)

# Alternative in-memory synthetic construction; no PyArrow dependency.
membership = SourceMembership.from_records(
    [{"domain": "general", "class_id": 2, "source": "danbooru"},
     {"domain": "general", "class_id": 2, "source": "zerochan"}],
    vocabulary,
    schema_version=SOURCE_MEMBERSHIP_VERSION,
    vocab_version=vocabulary.document["version"],
    vocab_sha256=vocabulary.sha256,
)
ids = membership.allowed_ids("general", ["danbooru", "zerochan"])  # (2,)
label = membership.class_name("general", 2)
model.set_source_membership(membership)
```

Construction reads and validates once. Immutable snapshots of names and sorted
per-source/per-domain candidate IDs remain in memory, independent of later edits
to the input records/Vocabulary. Inference does not open the sidecar or recompute
the Vocabulary hash. Multi-source unions use cached IDs, not a Parquet scan.
`set_source_membership(None)` detaches the index. Membership is not a parameter,
buffer, training state or state_dict/checkpoint entry; bind it again after loading
a model in a new process. Checkpoint format and strict tensor loading stay intact.

## Prediction contract

`model.predict(images, head_selection=None, *, top_k=20, dense=False,
source_filter=None)` supports:

- `None`: existing whole-vocabulary behavior and result types
- a known source string: that source's candidates in each requested domain
- a nonempty list (or tuple) of known source strings: sorted deduplicated union;
  repeated sources and overlapping IDs appear once

Unknown sources, malformed filters and absent membership fail before visual
inference, including style-only requests with an explicit source filter. There
is no fallback to an unfiltered result. A known source may have no candidates in
a requested domain, which is a valid empty result.

Each output domain remains its own dictionary key; classification values are:

- Top-K: `{'class_ids': LongTensor[B,K], 'scores': Tensor[B,K]}` in descending
  probability order, `K=min(top_k,candidate_count)`. IDs are original vocabulary
  IDs, never candidate-subset column numbers. Empty results have shape `[B,0]`.
- Unfiltered `dense=True`: the original `Tensor[B,C]` probability result.
- Filtered `dense=True`: `{'class_ids': LongTensor[B,M], 'scores': Tensor[B,M]}`,
  aligned in ascending original ID order. M is the union candidate count.
  All rows carry the same candidate IDs. Empty results have shape `[B,0]`.

The result key supplies Domain; `scores` supplies Probability; original IDs
resolve through `membership.class_name(domain, int(class_id))`. A caller may
label a presentation group `danbooru::general`, `danbooru::character`,
`danbooru::copyright`, or `danbooru::artist`; this is a source-filter label only,
not a replacement namespace or new ID space.

`style_embedding` is the unchanged style tensor, never an ID/score dictionary.
The official projector remains 256D. Selection retains one DINOv3 forward and
skips unrequested branches; source filtering occurs before Top-K over candidates.

## Probability and checkpoint configuration

Original Artist always computes softmax over the **complete** original Artist
vocabulary before candidate selection. Other Artist uses full-vocabulary softmax
when `other_artist_mode='single_label'`, matching CE; legacy `multilabel` uses
sigmoid. General, Character and Copyright always use independent sigmoid.
Filtered softmax scores are not renormalized and need not sum to one. Original
and Other Artist have no cross-head calibration and are never ranked together.

A fresh model preserves the previous `multilabel` default. For a single-label
checkpoint use `MultiTaskModel(backbone, layout,
other_artist_mode='single_label')` or `model.set_other_artist_mode('single_label')`.
Trainer synchronizes this nonparametric inference setting from its effective
`LossConfig.other_artist_mode`. Existing checkpoint loss settings and validation
are unchanged; standalone state_dict loading cannot infer a training objective,
so callers must restore the mode from that checkpoint's loss configuration.
This fixes CE inference probability semantics without changing Forward or Loss.

## Producer handoff and verification limits

SakuraPool may append this sidecar independently of existing TAR shards in a new
Dataset Release. Supply actual original-vocabulary membership and provenance,
plus the exact existing Vocabulary version and digest. Do not derive membership
from image source or regenerate/reorder label IDs. This feature does not modify
SakuraPool, SakuraMoon, or active data production.

`python -m sakura_tagger.selfcheck` performs the 15 synthetic source-filter cases
alongside existing CPU checks. Install `.[parquet]` to include the actual Parquet
write/read/schema-validation case; otherwise that case reports `NOT_VERIFIED`.
Real source membership and real model/data accuracy remain `NOT_VERIFIED` until
those assets are supplied. No weights/datasets are downloaded and no GPU/formal
training is launched by this feature.
