# Explicit optimizer windows and distributed training

`WindowTrainer` is the common single-process / torchrun training implementation
for immutable sparse Releases. Construction never starts training. `run` requires
a finite epoch or attempted-window limit even if a successful-update limit exists;
an all-skipped run must still terminate. Empty train sets fail explicitly. The
baseline configurations deliberately leave stop limits unset and therefore cannot
start a run accidentally. Example LR/schedule values are not quality-tuned.

## Objective and execution

For domain d, compute supervision-only valid-image counts over every planned
microbatch on every rank in the **actual optimizer window**, before forward.
With global count N_d, each rank contributes `world_size * local_numerator / N_d`
to backward; DDP then averages gradients. There is no further division by the
configured accumulation length. A short final window uses its real counts.
Zero-count domains contribute finite, output-connected differentiable zeros.
See [LOSS.md](LOSS.md) for the per-image weak-negative engineering objective and
its incomplete-label assumptions; it is not an unbiased population PU estimator.

All ranks select the same four logical branches. The student backbone runs once
per real microbatch. A rank without a real tail batch performs an explicit
zero-loss control forward; it has no sample ID and contributes zero counts and
exposures. No real image is silently duplicated or dropped to align ranks.
`no_sync` encloses forward **and** backward until the final control step. There is
no reliance on DDP.join for custom denominator collectives or accumulation.

Ordinary data/forward failures are coordinated before backward and fail the
current attempt on every rank. An unfinished attempt leaves the logical cursor
uncommitted and the trainer outside a saveable boundary. Selected nonfinite values
are aggregated as device-side finite flags; excluded NaNs never enter the loss.
Nonfinite losses/gradients skip the whole attempted update consistently, clear
gradients, acknowledge that completed attempt's input, and leave the successful
update scheduler unchanged. Process death or backend failure is bounded by the
explicit distributed timeout; this is not fault-tolerant elastic training.

## Optimizer, AMP and stages

AdamW groups are `backbone_last4`, `general`, `identity`, `other_artist`, with an
independent LR for each. Bias and tensors with at most one dimension (norm/scalar
parameters) receive zero weight decay. Frozen old Artist/Style parameters never
enter the optimizer. T1 freezes the entire original model. T2 trains only its
last four backbone blocks and final normalization, alongside the new heads.

If a physical new module is globally inactive for the whole window, its gradients
are set to None before the optimizer step: General and Other Artist separately;
Identity only when both Character and Copyright are inactive. This prevents
AdamW decay on empty modules. A fully empty active-gradient set skips the update.
Backbone activity follows the active new domains and any enabled preservation.

AMP supports off, CPU/CUDA BF16, and CUDA-only FP16. The order is scaled backward,
unscale, aggregate finite/norm check and clipping, global skip decision, optimizer
step, then scaler update. FP16 global overflow uses a consistent backoff across
ranks. **CUDA/NCCL/FP16 execution remains NOT_VERIFIED.** Warmup/cosine LR advances
only after a successful update. Attempted windows, successful/skipped updates,
real exposures, and control steps are separate counters.

`set_stage` requires a complete boundary, retires the DDP reducer, changes
trainable membership, and rebuilds both optimizer and reducer. Shared Adam moments
are retained; the stage's LR schedule restarts. Lifetime counters remain, so an
explicit new phase must choose appropriate limits. Do not retain/use an old DDP
wrapper after a stage transition. [Window checkpoints](WINDOW_CHECKPOINTS.md) save complete multi-rank boundary
state and publish a completion manifest only after every rank's files succeed.

## Optional preservation and loaders

Preservation requires an independent frozen original Kaloscope teacher, matching
Original Artist mapping and feature dimensions without shared parameter storage.
It sees the exact same real input tensor as the student and adds one explicit
teacher backbone forward. Per-image MSE / old-Artist KL contributions use the
same whole-window real-image denominator. The teacher is never optimized. This
does not guarantee old Artist/Style quality is unchanged; the old preprocessing
regression protocol and real-weight validation remain necessary.

Zero DataLoader workers remain the default. CUDA multiworker runs request the
supported spawn context, not Linux fork, because NCCL is not fork-safe. CPU workers
can use normal shared tensor memory or the explicit NumPy pipe transport described
in [SAMPLING.md](SAMPLING.md). Each loader has a private seeded generator, so its
construction cannot consume the model's global RNG. Worker tensor-FD IPC and
distributed backend capabilities must be validated in the chosen runtime; a
NumPy pipe test does not establish shared-tensor transport availability.

## Acceptance commands

```sh
# Exact windows, legacy compatibility, empty gradients, BF16 and frozen teacher:
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m sakura_tagger.selfcheck --suite window_trainer
# Actual two-rank torchrun CPU/Gloo, synthetic local TARs only:
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m sakura_tagger.selfcheck --suite ddp
```

The distributed suite launches the same module through `torch.distributed.run`,
uses TCP loopback and one compute thread per rank, and compares actual T1/T2
updates against identical single-process effective windows with explicit
atol=2e-6, rtol=2e-5. It also exercises unequal supervision/buckets, locally empty
domains, a control-only rank tail, a short accumulation tail, stage rebuild,
ordinary data failure, and one-rank nonfinite skip. A single-process mathematical
comparison alone is never reported as a CPU_DDP_PASS.

Primary reference: [PyTorch DDP documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).
