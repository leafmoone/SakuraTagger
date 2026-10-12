# Explicit command-line workflows

The actual unified entry is `python -m sakura_tagger`, or `sakura-tagger` after
installing this editable checkout. Importing the package or loading a config never
starts training. All commands accept `--config`, `--run-dir` and portable path
flags. Run `python -m sakura_tagger --help` for the complete parser.

## Configuration and run records

`configs/baseline.yaml` is the global-Identity baseline;
`configs/identity_local.yaml` is the only optional local-attention candidate.
`configs/T1.yaml` and `configs/T2.yaml` explicitly choose their training stage.
`configs/evaluation.yaml` extends the baseline with fixed-cohort evaluation options.
YAML `extends` paths resolve relative to the config file. Other relative paths
resolve from the working directory. `${ENVIRONMENT_VARIABLE}` strings are expanded
explicitly; unresolved variables fail. Paths are never tied to a user, OS, GPU
count or production machine. CLI path flags override `runtime` fields.

Release Vocabulary counts are authoritative: candidate counts in the architecture
example are replaced with the bound Vocabulary's actual dimensions, with all IDs
and names preserved. Feature widths, structure version and Identity mode remain
explicit. Exact resume checks the complete saved training contract, including stop
limits, losses and scheduler; changing that contract is not an implicit resume.

Every command writes `resolved-rank-N.json` and `result-rank-N.json` under its run
directory, with command arguments, resolved config, Python/library/device versions
and result. Training also records the effective model/loss/sampler contract and
streams window counters to JSONL rather than keeping an unbounded history in RAM.
Runtime artifacts live in ignored `outputs/`, `cache/`, `checkpoints/` or caller-
selected directories; none are project source files.

## Offline metadata preparation

Set paths to already published assets. These examples use shell environment
variables and must be supplied with the user's actual immutable Release/index:

```sh
PYTHONPATH=src .venv/bin/python -m sakura_tagger preflight --config configs/T1.yaml --release "$SAKURA_RELEASE" --run-dir outputs/preflight
PYTHONPATH=src .venv/bin/python -m sakura_tagger build-local-index --config configs/T1.yaml --release "$SAKURA_RELEASE" --index "$SAKURA_INDEX" --run-dir outputs/index
```

`inspect` is an alias of `preflight`. It reads Release/Vocabulary and published
Parquet schema footers, checks their metadata binding and configurable cache/
cohort budgets, and reports whether a finite training endpoint is configured.
It reads no images and downloads nothing. `build-local-index` streams published
Parquet into the compact local query index and RFS assets, without image reads or
TAR repackaging. Final real producer schema compatibility remains unverified until
an explicit handoff adapter/sample is actually tested.

## Explicit training and exact resume

Defaults deliberately leave stop limits null. `train` refuses unless an explicit
finite epoch or attempted-window limit is set; `max_optimizer_steps` alone cannot
prevent an all-skipped loop. No formal T1/T2 training is part of repository
acceptance. For a user-authorized real run, existing real weights are mandatory:

```sh
PYTHONPATH=src .venv/bin/python -m sakura_tagger train --config configs/T1.yaml --release "$SAKURA_RELEASE" --index "$SAKURA_INDEX" --model-dir "$SAKURA_MODEL_DIR" --max-epochs 1 --run-dir outputs/t1-run
```

The same engine supports torchrun when the chosen runtime supports its backend:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 -m sakura_tagger train --config configs/T1.yaml --release "$SAKURA_RELEASE" --index "$SAKURA_INDEX" --model-dir "$SAKURA_MODEL_DIR" --max-epochs 1 --run-dir outputs/t1-ddp
```

These are real-data usage examples, not commands executed during CPU engineering
acceptance. Every rank requires the same explicitly shared run/checkpoint path.
CPU uses Gloo and CUDA uses NCCL. GPU paths exist but were not executed here.
`runtime.distributed_timeout_seconds` (or `--distributed-timeout-seconds`) is
explicitly configurable; the examples use 600 seconds. Choose it to cover the
longest bounded checkpoint/periodic-evaluation wait on the selected machine. It
does not enable elastic recovery or make a stalled backend healthy.
Sampling asset publication is single-writer rank 0, with coordinated failure
before rank-local readers proceed. No credentials or network security settings
are changed to make a backend work.

Use `--checkpoint PATH` with the same model, settings and endpoint to restore an
intermediate complete optimizer boundary. Prefetch is replayed from the consumed
cursor. A partial accumulation is never resumable. Changing world size, Release,
Vocabulary, architecture, losses or endpoint requires an explicit new run/migration.

For a new T2 stage initialized from T1 weights, use `--initialize-from PATH` with
`configs/T2.yaml` and a new run directory. It carries model weights from the same
Release/structure while intentionally starting new optimizer/scheduler/counters.
It cannot be combined with exact `--checkpoint` resume. T2 is not started
implicitly, and original Artist/Style parameters remain frozen.

`run.checkpoint_every_updates` and `run.evaluate_every_updates` are optional
positive successful-update intervals. Evaluation requires the fixed `cohort`
settings described below. Rank 0 evaluates the underlying model at a complete
boundary; all ranks coordinate callback success/failure. Final training always
publishes a complete all-rank checkpoint. Optional preservation requires a fixed
teacher source plus independently loaded original teacher weights.

## Evaluation, calibration, export and prediction

Set `cohort.sample_ids_file` to a JSON list of fixed published IDs, choose its
`split` (`val`, `calibration`, or `test`), protocol/class/source scope and budgets.
Exact small-cohort score sorting refuses an insufficient budget. The following
commands use a completed training checkpoint and the existing frozen Release:

```sh
PYTHONPATH=src .venv/bin/python -m sakura_tagger evaluate --config configs/evaluation.yaml --release "$SAKURA_RELEASE" --index "$SAKURA_INDEX" --checkpoint "$SAKURA_CHECKPOINT" --run-dir outputs/evaluation
PYTHONPATH=src .venv/bin/python -m sakura_tagger calibrate --config configs/evaluation.yaml --release "$SAKURA_RELEASE" --index "$SAKURA_INDEX" --checkpoint "$SAKURA_CHECKPOINT" --output outputs/calibration.json --run-dir outputs/calibration
PYTHONPATH=src .venv/bin/python -m sakura_tagger export --config configs/T1.yaml --release "$SAKURA_RELEASE" --index "$SAKURA_INDEX" --checkpoint "$SAKURA_CHECKPOINT" --calibration outputs/calibration.json --output outputs/predictor --run-dir outputs/export
PYTHONPATH=src .venv/bin/python -m sakura_tagger predict --package outputs/predictor --image "$SAKURA_IMAGE" --heads general character copyright --output-mode threshold --max-tags 100 --run-dir outputs/predict
```

A completed checkpoint already includes full student weights, so inference
extraction does not require the original model directory. It is not training-
state resume and does not transfer optimizer/world-size state. Evaluation and
export reject mismatched Release/Vocabulary or the entire saved preprocessing
identity, including version, normalization mean/std, buckets and alpha background. Calibration only
fits validation/calibration, and the deployment package enforces its class scope.
See [evaluation](EVALUATION.md), [deployment](DEPLOYMENT.md) and
[checkpoint](WINDOW_CHECKPOINTS.md) contracts for formats and limits.

The explicit `--synthetic` flag is required to create or consume synthetic model
checkpoints in train/evaluate/calibrate/export. It is never inferred from missing
real weights. The integrated selfcheck below exercises these commands with a
locally generated synthetic Release and real TAR/Parquet reads.

## CPU acceptance and actual timings

These self-contained commands need no real data, model directory or credentials:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m sakura_tagger selfcheck --suite integration --run-dir outputs/cli-integration
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m sakura_tagger benchmark --iterations 2 --run-dir outputs/cli-benchmark
```

Integration invokes the actual command parser for preflight/index/train/evaluate/
calibrate/export/predict. It reads a synthetic published Release through TAR/cache/
bucket/loss, saves intermediate complete-window checkpoints, resumes to the same
final model/counters, evaluates, freezes calibration, exports and predicts an RGBA
image. It exercises periodic evaluation/checkpoints, configuration records,
finite-endpoint refusal and strict real/synthetic distinction. The real torchrun
selfcheck worker includes the equivalent distributed command pipeline.

Benchmark uses actual wall-clock stages for synthetic Release creation, first
shape planning, TAR/decode/preprocess/collation, backbone, task features, terminal
classifiers, a complete optimizer window and checkpoint writing. CUDA timing
synchronizes if explicitly requested on a capable machine. Only synthetic CPU
timings were run; they do not measure production throughput or establish GPU
performance. Parameter totals are not speed estimates.

The original `python -m sakura_tagger.selfcheck` entry and suites remain supported.
Full acceptance normally includes actual two-rank Gloo. In an environment with the
already verified backend restriction, run the explicitly limited aggregate:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m sakura_tagger selfcheck --suite all --skip-ddp-reason "Existing CPU/Gloo backend fails getifaddrs: Operation not permitted; not retried" --run-dir outputs/pretraining-cpu-acceptance
```

`--skip-ddp-reason` requires a nonempty reason, makes no backend attempt, and records
`NOT_VERIFIED`; it never catches a failed test and converts it to PASS. Other
failures still fail the command. The limited run distinguishes available CPU
suite/single-process E2E success from unverified actual DDP/full distributed E2E.
Strict all-required-verification `CODE_COMPLETE` remains verification-incomplete
while Gloo is unexecuted. Real handoff, remote production transfer, real weights,
CUDA/GPU quality/performance and family/near-duplicate data quality remain
separate `NOT_VERIFIED` items. Formal training remains `NOT_STARTED`.
