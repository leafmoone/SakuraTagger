# Repeat factors and shard-local windows

`build_sampling_assets(index, path)` writes a versioned, Release-bound manifest
and read-only NumPy mmap: 16 bytes per unique train sample (ordinal + FP64 factor).
It reads sparse index rows, not images. The boundary validates the asset digest
once. Checkpoints contain its identity, not its sizes, weights, or occurrences.

For class frequency `f_c = unique_train_positive_count / unique_train_images`,
`r_c = min(3, max(1, sqrt(0.001 / f_c)))`; an image takes the maximum of its valid
General, Character, Copyright, and trusted CE-valid Other Artist classes. No
positive means factor 1. Original Artist, unconfirmed/conflicting artist claims,
planned rows, and val/test samples do not enter these frequencies. Raw facts and
source provenance remain unchanged. The default threshold is 0.001 and cap is 3.

The default count is `floor(r_i) + Bernoulli(frac(r_i))`, exactly 1–3 exposures per
sample per epoch. The sampler does not copy files or use replacement draws. Its
PCG64 seeds are derived with SHA256 from the configured seed, epoch, window, and
purpose. `occurrence_id = ordinal * 3 + repetition` is distinct from sample ID;
epoch also participates in downstream weak-negative sampling seeds.

## Window and cursor protocol

`ShardWindowSampler` shuffles shard order and contiguous bounded windows inside
each shard each epoch. It mixes images within each actual aspect bucket, arranging
repetitions while that shard is resident. This is **window-local shuffle, not a
uniform global permutation**. Same-image batch repeats are never introduced;
consecutive repeats are avoided where another image in the bucket is available.
Small shards/buckets can make separation impossible and yield smaller batches.

1. Call `prepare_window()`. Only this explicit operation builds the bounded plan.
2. Inspect `window_info`, or `batch_at(control_step)` and `records_for(batch)` to
   compute whole optimizer-window supervision counts without retaining graphs.
3. Consume `window_loader(dataset, sampler, ...)`. It returns planned metadata and
   a real batch, or `None` for a control-only tail. It never acknowledges work.
4. After a complete optimizer window has been handled, call `advance(n)` with its
   logical control-step count. `len` and `advance` are O(1) on the prepared plan.
5. Call `next_window()` only after the current window is fully acknowledged. At
   epoch end, explicitly call `set_epoch(next_epoch)`.

All ranks reconstruct the same bounded global plan, then take disjoint batches.
Each gets `ceil(real_batches / world_size)` control steps. Missing tail steps are
explicitly control-only with no sample or exposure; no real sample is padded,
replicated, or dropped for rank alignment. E5 owns coordinated DDP execution.

## Geometry and replay

The plan uses `TarDataset.inspect_dimensions`, which performs one actual decode
on first encounter and records raw/post-EXIF geometry in the Release-bound local
correction store. It does not resize or retain a window of image tensors. That
first encounter costs **one extra decode**; later planning reads the correction.
Normal loading still decodes and verifies the same dimensions. Re-bucketing thus
occurs before batching, without an image-shape mismatch exception every epoch.
Published P2 dimensions are never treated as authoritative decoded geometry.

A checkpoint saves asset/config identity, epoch, window, seed algorithm, and only
the acknowledged cursor. Restore replays at most the current bounded window.
Newly persisted corrections and worker lookahead cannot alter its order because
cold and warm planning both use actual dimensions. Iterator read-ahead, cache
publication, and worker decode are not optimizer progress. Changed world size,
asset, preprocessing, or scheduling settings require an explicit new run.

`window_loader` defaults to normal PyTorch shared-memory tensor transport. The
explicit `worker_transport: numpy` alternative sends the collated CPU array and
plain sparse metadata through ordinary multiprocessing pipes, then wraps the
array as a tensor in the parent. This involves serialization/copy overhead and
is useful where local tensor-FD IPC is unavailable; it changes neither values nor
logical cursor semantics. A loader-local seeded torch.Generator isolates iterator
creation and worker base seeds from the model's global RNG. Worker waiting has a
finite configurable timeout.
Some restricted runtimes lack the local socket capability required by PyTorch's
default tensor-FD transport. The explicit NumPy mode uses ordinary permitted
pipes instead; selecting it does not establish that shared-tensor transport works.
Validate each worker/backend capability in the chosen runtime, report unsupported
capabilities separately, and do not change security settings as a transport fix.

## Reports and limits

The asset manifest reports factor=1, intermediate, and factor=3 image shares,
plus independent-train count bands (1, 2–9, 10–99, 100–999, 1000+) before sampling
and expected positive-label exposures afterward. `window_info` reports realized
exposures and band counts; sum these once over committed windows for realized
epoch totals, never on replay/prefetch. A warning is explicit when at least half
of images saturate at 3: proportional rebalancing is limited. No threshold is
automatically changed, and these statistics are not a quality measurement.

The legacy `BucketBatchSampler` API remains available for local manifests. Its
expanded plan is now compact arrays shared by lazy rank views, not repeated
per-rank Python tuple lists. Its cursor operations no longer slice the epoch.
Version-1 sampler state is accepted and retains the old random sequence; new
state binds a one-time compact input digest instead of embedding all sizes and
weights. Input arrays are copied once into read-only snapshots so caller mutations
cannot invalidate the cached identity. Legacy replacement mode remains explicitly distinct from the strict
1–3 stochastic-rounding guarantee.

Run the durable synthetic CPU check:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m sakura_tagger.selfcheck --suite sampling
```
