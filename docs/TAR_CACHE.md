# TAR providers, bounded cache, and actual image dimensions

The reader consumes the immutable [reference Release](RELEASE.md). It never
extracts a complete archive, contacts Pool per image, scans a live upload folder,
or requires the production machine to remain online.

## Actual byte reads

`TarReader` has sequential `iter_members()` and random `read_member()` /
`read_sample()` APIs. It opens its own descriptor and uses `os.pread`, retrying
short reads until the exact extent is read or raising on EOF. Both paths check
the data offset, preceding regular-file header/checksum, name, size and bounds.
The size cap is exactly 5,368,709,120 bytes including archive overhead. `.partial`
inputs, links, directories, compressed archives, mismatched headers and unsafe
member paths are rejected. The reference fixture uses USTAR regular members;
extended-name metadata that cannot be verified from the indexed header fails
explicitly rather than silently reading a different member.

The configured `max_member_bytes` (default 128 MiB) bounds each in-memory read
and can be raised for a producer's verified image requirements. It is independent
of the 5 GiB **shard** cap. Whole-shard SHA validation occurs once at cache
publication. Random reads do not rehash the archive or the entire image corpus.

`TarDataset` reads indexed images and spot-checks paired JSON against frozen
Vocabulary/sparse labels. The reference JSON schema is explicitly named
`sakura-tagger-reference-sample-v1`; it is not claimed to be the real producer's
format. `json_checks_per_shard` is configurable and at least one. Every indexed
JSON header still gets range/type/name validation. Real producer JSON needs its
own adapter after handoff.

## Providers and immutable revisions

- `LocalShardProvider` streams the declared published local file
- `LoopbackShardProvider` is restricted to unauthenticated localhost HTTP
  fixtures; it is not a claimed production URL protocol
- `HuggingFaceShardProvider` supports public dataset repositories and requires a
  **full 40-hex commit**. Branches, tags and abbreviated revisions are rejected

The Hugging Face adapter uses the official
[`huggingface_hub.hf_hub_url`](https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.hf_hub_url)
API with dataset type, explicit commit and official endpoint, then streams into
the cache's one owned partial file. It does not create a second SDK download
cache, read token files, or infer credentials. The CPU lock pins the actual
`huggingface-hub==2.2.0`. Missing/private access fails explicitly; authentication
is a separate authorized integration, not an implicit environment lookup.

`ModelScopeShardProvider` uses the official dataset `repo` download route with
an explicit full commit (the `repo/files` route may return Git LFS pointers).
Only descriptors in the selected release inventory are admitted. It streams into
the same bounded cache, without SDK caches or credential lookup. The SakuraTool
producer format has an explicit metadata/TAR-JSON adapter; this does not establish
full-release reader acceptance, GPU performance or model quality.
See [selected dataset setup](../README.md#selected-sakuratool-dataset-modelscope).

## Cache ownership, capacity and termination

`ShardCache(root, release, provider, capacity_bytes=..., job_id=...)` owns only
its selected job subdirectory. Release identity and budget/watermarks must agree
across workers sharing that directory. An unrelated directory is never adopted
or cleaned. Portable paths are configured by the caller; no machine paths or
GPU counts are embedded in the implementation.
The shared-worker cache uses POSIX `fcntl` locks and fails explicitly on platforms
without them. Importing legacy Vocabulary/SourceMembership APIs does not require
those locks. No Windows machine was used for this CPU acceptance.

- One interprocess lock per shard permits only one writer
- A short global state lock protects reservations and reader lease tokens
- Full expected bytes are reserved before downloading, so partial + ready +
  leased TAR bytes never exceed the hard configured capacity
- Size/SHA is computed while streaming, then the partial is atomically renamed
- Only ready files without live reader leases can be evicted
- High/low watermarks trigger LRU eviction; one large shard may exceed the high
  watermark but may never exceed hard capacity
- Full caches apply finite backpressure (`CacheBackpressure`), not an unlimited
  queue or an unbounded wait. Consumption/releasing leases allows a retry
- An ended writer's partial is cleaned within this job; dead-process leases are
  reclaimed. This is not a general machine/process monitoring service
- A missing or externally changed unleased ready file is downloaded again from
  its fixed Release. A leased externally changed file fails explicitly
- Eviction is unrelated to committed training progress and never rewrites it

`prefetch_count`, `workers`, `wait_timeout`, overall transfer timeout, socket
`read_timeout`, and retries are configurable. Timeouts must be finite and
positive. A long whole-shard transfer budget does not imply one equally long
uninterruptible socket read. `prefetch(ids)` schedules at most the pending bound;
`wait_prefetch()` drains results and surfaces failures. `close()` requests
cancellation and joins bounded provider operations, including epoch-tail work.
The budget covers TAR data/reservations; small lock/state bookkeeping is separate.
Callers must provision filesystem headroom for that bookkeeping and derived
indices. No detached download or training task is created.

## EXIF, RGBA and safe re-bucketing

The existing deterministic `NativePreprocessor` handles full-frame EXIF
orientation and alpha compositing on the original canvas. It does not add a
border, crop, flip, or silently turn an arbitrary image into a square. The
actual decode size determines its aligned rectangular bucket.

`DimensionCorrections(path, release)` stores derived raw and effective dimensions
once per sample, including conflicts with released metadata and alpha/format
observations. It is bound to the Release identity; released facts remain intact.
SQLite handles are lazily reused per process/thread and recreated after worker
serialization or fork, avoiding a new connection per image.
Repeated epochs reuse the correction and do not fail forever on the same stale
P2 width/height assumption. A genuinely changing decode within one immutable
Release is rejected.

`rebucket_prepared(samples, batch_size, max_open_buckets=...)` groups actual tensor
shapes in bounded buffers **before** `torch.stack`. It preserves every sample and
emits smaller tail groups. `collate_prepared` refuses a mixed-shape batch.
Occurrence IDs are retained separately for the scheduler and loss.

`decoder_capabilities()` probes JPEG/PNG/WebP/AVIF registration and tiny local
encode/decode behavior. Unsupported formats are reported; no source image is
transcoded or deleted.

## Focused acceptance

`python -m sakura_tagger.selfcheck --suite tar` executes real tiny
TAR+JSON+Parquet reads, sequential/offset parity, short-read handling, a sparse
file with >2^32 data offsets and exact 5 GiB/rejection boundaries, partial/corrupt
cache cases, lease protection, finite backpressure, eviction/redownload,
prefetch tail completion, and two spawned processes sharing one loopback
download. It also checks EXIF/RGBA/dimension corrections and actual re-bucketing.
The sparse boundary fixture allocates under 1 MiB of disk blocks and is never
hashed as 5 GiB; it is not a production large-file upload benchmark.
