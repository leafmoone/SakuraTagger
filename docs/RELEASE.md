# Reference Release and sparse sample index

Status: **REAL_PRODUCER_HANDOFF_NOT_VERIFIED**. This is the explicitly named
`sakura-tagger-reference-release-v1` fixture/interchange contract, not an assertion
that the producer's existing P2 schema implements it. A real producer handoff
needs an explicit adapter after its actual release documents and samples are
available. Existing TAR files must not be repacked merely to match this reference.
The older `ManifestDataset` API remains supported for small datasets.

## Immutable inventory

`DatasetRelease.read(release_path)` loads one JSON snapshot and its bound
Vocabulary. Required fields:

- `schema_version`, `status: published`, `dataset_id`, `release_id`
- `vocabulary_path`: a portable relative path to the existing canonical JSON
- `vocab_sha256`: the canonical `Vocabulary.sha256`, including ordered IDs/names
- `shards`: `shard_id`, `path`, `size_bytes`, `sha256`, `status: published`
- `indices`: `path`, `size_bytes`, `sha256`, `status: published`
- `supervision_assumptions.single_artist_confirmed`: explicit boolean; default
  false. This is a run/data contract assumption, not a rewritten image fact

Optional `vocabulary_csv_sha256` holds individual input CSV **file-byte** hashes,
keyed by domain (`general`, `character`, `copyright`, `other_artist`, `artist`).
These are a different hash convention from the canonical Vocabulary SHA and
never substitute for it. The five ID spaces remain separate and unchanged.
Other Artist names must exclude exact Original Artist names.

`source` names `provider: local`, or an explicit `modelscope`, `huggingface` or
`loopback` repository and fixed revision. Mutable main/master/latest aliases are
rejected. No credentials, URL guesses, network request or directory scanning is
performed by this module. A remote revision must identify the immutable release
the caller intends; remote downloading is a separate provider responsibility.

Every shard must be an uncompressed `.tar`, at most **5,368,709,120 bytes (5 GiB)**
including archive overhead. Unpublished and `.partial` paths are rejected.
Artifact size/SHA is checked at index-build or provider publication boundaries,
not every optimizer step. `source_membership_path`, when present, is loaded using
the existing `sakura-tagger-source-membership-v1` contract and the bound Vocabulary.
Image annotation provenance never implies a label's vocabulary membership.

## Sparse Parquet rows

`reference_index_schema(vocabulary, release_id)` returns the actual PyArrow
schema, versioned `sakura-tagger-reference-index-v1`. Its file metadata binds the
schema version, release ID and canonical vocabulary SHA.

- Identity: `sample_id`, `image_sha256`, `split` (train/val/test/calibration)
- Eligibility: `status` (published/planned), `shard_id`, nullable `file_path`
- In this reference, non-null `file_path` is the declared shard's relative path
- Pair: `image_member`, `json_member`
- Unsigned 64-bit extents: `image_offset_data`, `image_size`,
  `json_offset_data`, `json_size`. Offsets address member data, never headers
- Nullable uint32 `raw_width/raw_height` and `effective_width/effective_height`
- `dimension_status`: unverified/confirmed/conflict_rebucketed. Raw P2 dimensions
  are not presumed post-EXIF. Each pair is independently known or unknown; only
  actual decode evidence can confirm/correct effective dimensions
- `general_ids`, `character_ids`, `copyright_ids`, `other_artist_ids`,
  `original_artist_ids`: sparse, unique, domain-local int32 ID lists
- Each multilabel domain also has sparse `negative_ids` and `forbidden_ids`
  fields, prefixed with that domain. Empty negatives do not imply all other
  labels are known negative. Weak negatives never get written to these facts
- `source_provenance`: actual image/annotation-source strings
- `family_id`, `near_duplicate_status`: explicit strings, `UNKNOWN` allowed
- Nullable `single_artist_confirmed`: sample-level evidence. Null inherits the
  Release assumption, true confirms, and explicit false overrides a true Release
  default and excludes the sample from CE and Other Artist train support
- `artist_evidence`: source/domain/class_id/label structs. Domain is `artist`,
  `other_artist`, or `out_of_vocabulary`; OOV evidence keeps its label and null ID

Planned rows and rows with null `file_path` are skipped even if other metadata
exists. Published rows cannot refer to an undeclared TAR. Sample IDs, image SHAs,
member identities, and non-overlapping extents are checked once while building
the index. This detects known duplicate metadata; it does not invent family
annotations or claim family-disjointness from SHA uniqueness.

## Trusted single-artist conversion

`single_artist_target` and `collate_artist_targets` perform the actual conversion:

- A single Other Artist ID plus the explicit Release or sample confirmation:
  `[B] int64` target and `[B] bool` validity; no vocabulary-wide dense mask
- A unique Original Artist: retains original IDs; Other Artist CE is invalid
- Missing or OOV artist: CE invalid
- Distinct claims across sources/domains: conflict retained, CE invalid
- One unconfirmed observed positive: CE invalid in the generic API

The collation result has `targets`, `valid`, `original_ids`, and reason `status`.
Invalid targets use -100, and callers must use the validity mask. The persisted
raw sparse facts and source evidence are not changed by conversion.

## Bounded local lookup

```python
from sakura_tagger.data import DatasetRelease, build_local_index
release = DatasetRelease.read("data/release.json")
index = build_local_index(release, "cache/release.sqlite", batch_size=4096)
record = index[0]
for ordinal, record in index.iter_records(split="train"):
    pass
train_counts = index.positive_support("general")
index.close()
```

Construction streams bounded Parquet batches to a sparse on-disk SQLite index.
Only per-class support counters live in memory, not millions of record dicts or
a full-dataset dense target matrix. Lookup uses independent lazy process/thread
connections and is pickle-safe for workers. Existing indices require the same
Release/Vocabulary identity and a completion marker. A failed build never
publishes a completed index.

Support counts use unique published **train** samples before RFS. Other Artist
counts use only trusted CE-valid rows, excluding conflict/unconfirmed/Original-only
claims. Validation/test/planned rows cannot mark a class as trained. Original
Artist's existing trained vocabulary is not masked based on new-data support.

Actual decode-dimension corrections should be kept as a separate local derived
asset bound to this index identity, leaving the immutable released facts intact.
Sampling must use those actual dimensions before stacking a batch; unresolved
raw/effective mismatch must not be retried unchanged each epoch.

Run focused synthetic contract checks with
`python -m sakura_tagger.selfcheck --suite release`. This metadata suite does not
claim actual TAR byte reads, real producer compatibility or remote 5 GiB upload.
