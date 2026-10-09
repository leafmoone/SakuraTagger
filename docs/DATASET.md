# SakuraPool dataset interface v1

SakuraTagger reads existing data; it never crawls or downloads images. Metadata,
vocabulary and JSONL manifest are separate UTF-8 files. `ManifestDataset` takes
their paths and an optional image root. Width/height are original dimensions
after EXIF orientation. Train, validation and test use the **same** vocabulary.

Metadata requires `schema_version: "sakura-tagger-dataset-v1"`, `dataset_id`,
`provider: "SakuraPool"` and `vocab_sha256`. The SHA is computed once over canonical
JSON (`sort_keys=True`, compact separators, UTF-8, `ensure_ascii=False`) of the
entire vocabulary. This does not hash/re-read every image during training.

Vocabulary has `version: "sakura-tagger-vocabulary-v1"`, `domains` containing
`general`, `character`, `copyright`, `other_artist` arrays, and an
`original_artist` array. Entries are `{"class_id": 0, "name": "label"}`. Each array
must already be ordered with contiguous zero-based IDs. The reader never sorts
or renumbers entries. Identical names in Character and Copyright remain separate
domain-qualified labels. Identity physical classifier order is Character then
Copyright, each retaining its own zero-based logical IDs. Optional per-entry
`groups` metadata can describe General semantic groups without more heads.
`Vocabulary.layout()` derives final class counts; candidate config counts are not
a substitute for a final vocabulary.

Every JSONL record requires:

- `image_id`, `image_path`, positive integer `width`, `height`
- `source`, `RID` (string or integer), lowercase `image_sha256`
- `split`: `train`, `val` or `test`; `family_id` (including `"UNKNOWN"`)
- `positive_labels`, `supervision_scope`, `unknown_mask` dictionaries

Label dictionaries use the four logical domain names and domain-local class IDs.
`positive_labels` values are lists. `supervision_scope` values can be `"all"`
(explicit exhaustive annotation), `"positive_only"`, `"unknown"`, or a list of
known class IDs. `unknown_mask` lists additional unknown IDs to exclude from the
scope. Positives cannot be marked unknown. Absent domains are wholly unknown;
known-negative supervision requires an explicitly present positive-label list,
which may be empty. Thus missing annotations never silently become negatives.
Target tensors are float [C]; masks are bool [C]. PyTorch's normal collation
produces the [B,C] dictionaries accepted by training losses.

The manifest reader rejects duplicate image IDs/content hashes across splits,
invalid IDs and schema/SHA mismatches. It does not reorder records or infer new
labels. Image byte hashes are supplied by SakuraPool, not recomputed per sample.
RFS statistics use **only the final deduplicated training records**, count each
positive class once per image, and retain domain-qualified keys. The factor is
`min(max_repeat_factor, max(1, sqrt(threshold/frequency)))`, maximized over an
image's positives (1 for images without positives). Defaults are 0.001 and 3.

## Batching and recovery

`BucketBatchSampler` creates one globally seeded occurrence stream, groups it
into homogeneous-resolution batches, then assigns whole batches by rank.
Default stochastic rounding uses each image at least once and turns repeat
factors into integral multiplicities. Explicit `repeat_mode="replacement"` plus
`epoch_size` instead uses weighted draws; realized per-image draw counts can
exceed the weight cap 3, and omission in that draw mode is intentional. The cap
bounds weights, not all possible sampling procedures' realized repetition.

Ranks never receive the same occurrence. There is **no implicit padding or
rank-balancing drop**, so rank lengths may differ. Distributed callers must use
DDP join (and coordinate gradient accumulation) or reject uneven workloads.
`drop_last=True` explicitly drops each bucket's short final batch. It is not the
default. Epoch seed is `seed+epoch`.

Call `advance()` only after a consumed batch completes. The sampler does not
advance on iterator yield, since DataLoader prefetch could otherwise skip unseen
work after a restart. Save `state_dict()` together with a completed optimizer
step. Resume validates configuration, epoch and consumed rank-local cursor.
`set_epoch()` starts a new epoch and resets the cursor. `occurrence_batches()` is
available to audit intentional repeat occurrences separately from image IDs.
