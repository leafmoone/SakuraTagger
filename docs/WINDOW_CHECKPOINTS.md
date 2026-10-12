# Exact optimizer-boundary checkpoints

`WindowTrainer.save_checkpoint(path, sampler=sampler)` and
`load_checkpoint(path, sampler=sampler)` are collective operations: every rank
calls with the same shared-filesystem directory. A fresh checkpoint directory is
required; existing complete or failed attempts are never overwritten. A failed
attempt without its completion manifest is not resumable.

```python
# Explicitly configured trainer and sampler; saving never starts training.
trainer.save_checkpoint("checkpoints/run-1/update-100", sampler=sampler)
# Reconstruct the same model, Release, loss scope, trainer and sampler first.
trainer.load_checkpoint("checkpoints/run-1/update-100", sampler=sampler)
# Discard old DataLoader iterators. run() recreates them from consumed progress.
trainer.run(sampler)
```

Each rank atomically writes its own `rank-00000.pt`-style file with Python, NumPy,
Torch and (when used) local CUDA RNG plus the deterministic sampler's epoch,
window and consumed cursor. Rank 0 writes `common.pt` with the complete student
state, AdamW moments/groups, successful-update scheduler, GradScaler and lifetime
attempted/successful/skipped/exposure counters. Each file is flushed before its
atomic rename. Only after every rank reports success does rank 0 atomically
publish `manifest.json` with complete status and each file's size/SHA-256.

DDP replicas share the common model and optimizer state. Rank-specific random
state and input progress are never inferred from rank 0. Save rejects incomplete
optimizer windows, pending gradients or disagreement between ranks about the
contract, successful-step schedule, scaler or consumed boundary. A skipped but
fully processed update is a valid boundary with its input cursor acknowledged.

## Strict identity and recovery

The manifest binds Vocabulary/Release/loss-scope identities, complete effective
loss and training settings (including Other Artist mode), stage, structure,
base architecture, world size, deterministic sampling assets/settings, full-frame
preprocessing version/normalization/bucket/alpha background, device type and
optional teacher identity. Model tensor names, shapes and dtypes are checked.
Changing world size, extending the Release, reordering Vocabulary, switching
Identity-local, changing preprocessing or stage is an explicit new run/migration,
not an exact resume. The model and all optimizer state must fit the selected
machine; this is not a sharded-checkpoint or elastic-training implementation.

Load checks the completion manifest, required rank files and checksums, then
validates RNG and prospective model/optimizer/scheduler/scaler and sampler state
before changing the live trainer. A rejected checkpoint leaves the live weights,
optimizer, consumed cursor and RNG unchanged. Replaying a prospective bounded
sampler window may refill its cache or dimension-correction database; this is not
optimizer progress. No untrusted pickle objects are accepted: loading uses
`weights_only=True`.

Prefetched images, worker lookahead and completed TAR downloads never advance
the saved cursor. TAR cache files are not checkpoint contents. They can be
refetched from the same fixed published Release on resume. No dataset-wide
rehashing is introduced. SourceMembership remains an external vocabulary-bound
sidecar, never a model training parameter.

## Preservation teacher

When preservation is enabled, pass an independent frozen original teacher and
`teacher_source`, a stable recoverable package/revision or synthetic fixture
identifier. Its loaded tensor state is hashed once at construction, and the source,
architecture and hash become part of the training/checkpoint contract. Restore
requires reloading that teacher with the same identity; teacher weights are not
copied into every checkpoint. A source string alone is not proof of matching
weights. The fixture identifier used in selfcheck is explicitly synthetic and is
not an official Kaloscope weight claim. Without `teacher_source`, training can
run but checkpoint saving refuses until the caller supplies a recoverable source.

## Acceptance

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m sakura_tagger.selfcheck --suite checkpoint
# Includes actual same-world two-rank resume when standard Gloo is available:
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m sakura_tagger.selfcheck --suite ddp
```

The focused CPU test checks T1/T2 next IDs/occurrences/buckets, RNG draws, loss,
counters, model and Adam updates against uninterrupted execution. In this same
single-process fixture the comparison is bitwise, including a short tail, and
restoration refetches into a separate cold cache. It also checks finite malformed
checkpoint cases, missing files/manifest, checksum and contract changes, bad
cursor/RNG/optimizer/model state, partial-window rejection, failed writes and
fixed-teacher rebinding without duplication. These checks do not establish
cross-device bitwise equality.

The actual two-rank worker contains the equivalent resume exercise. On a runtime
where Gloo is blocked (including the recorded `getifaddrs: Operation not permitted`
restriction), report `CPU_DDP_RESUME: NOT_VERIFIED`; single-process success does not
establish distributed acceptance. CUDA resume and real-weight/real-data validation
remain separate `NOT_VERIFIED` checks. No formal training is performed here.
