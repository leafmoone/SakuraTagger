"""Explicit portable commands; no training, downloads or CUDA work on import."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import torch
import torch.distributed as dist
import yaml

from .adapter import load_kaloscope, load_state_strict, model_options
from .data import (DatasetRelease, SparseReleaseIndex, build_local_index, build_sampling_assets,
                   SamplingAssets, ShardCache, provider_for_release, DimensionCorrections,
                   TarDataset, ShardWindowSampler, BucketConfig, BucketSelector, NativePreprocessor)
from .deployment import (export_predictor, load_predictor, untrained_label_metadata, preprocessing_identity)
from .evaluation import (EvaluationConfig, EvaluationCohort, FeatureCache, build_feature_cache,
                         feature_binding, evaluate, fit_calibration)
from .model import MultiTaskModel, KaloscopeBackbone, HeadLayout
from .model.registry import LOGICAL_DOMAINS
from .training import WindowTrainer, WindowTrainerConfig, SparseLossConfig, WeakNegativeScope, initialize_distributed
from .training.window_checkpoint import VERSION as CHECKPOINT_VERSION, _read_file, state_digest


def _merge(left, right):
    result = deepcopy(left)
    for key, value in right.items():
        result[key] = _merge(result[key], value) if isinstance(result.get(key), dict) and isinstance(value, dict) else deepcopy(value)
    return result


def read_config(path, _seen=None):
    if path is None:
        return {}
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError('Configuration extends cycle')
    document = yaml.safe_load(path.read_text()) or {}
    if not isinstance(document, dict):
        raise ValueError('Configuration must be a mapping')
    parent = document.pop('extends', None)
    result = _merge(read_config(path.parent / parent, seen | {path}) if parent else {}, document)
    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            value = os.path.expandvars(value)
            if '${' in value:
                raise ValueError(f'Unresolved environment variable in configuration: {value}')
        return value
    return expand(result)


def versions(device):
    result = {'python': platform.python_version(), 'platform': platform.platform(), 'device': str(device),
              'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda}
    for name in ('numpy', 'Pillow', 'PyYAML', 'pyarrow', 'scikit-learn', 'safetensors'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = 'NOT_INSTALLED'
    return result


def _json(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=_json, ensure_ascii=False, indent=2, allow_nan=False))


def _runtime(config, name, *, required=True):
    value = config.get('runtime', {}).get(name)
    if required and not value:
        raise ValueError(f'Explicit runtime.{name} is required in config or CLI')
    return value


def _release_index(config):
    release = DatasetRelease.read(_runtime(config, 'release'))
    return release, SparseReleaseIndex(_runtime(config, 'index'), release=release)


def _preprocessor(config):
    return NativePreprocessor(BucketSelector(BucketConfig(**config.get('preprocessing', {}).get('bucket', {}))),
                              alpha_background=tuple(config.get('preprocessing', {}).get('alpha_background', [255, 255, 255])))


def _model(config, release, *, synthetic, device):
    if synthetic:
        from .selfcheck import synthetic_fixture
        legacy = synthetic_fixture(config.get('seed', 0))
    else:
        legacy = load_kaloscope(_runtime(config, 'model_dir'))
    options = {key: value for key, value in config.get('head_layout', {}).items() if key not in LOGICAL_DOMAINS}
    layout = release.vocabulary.layout(**options)
    model = MultiTaskModel(KaloscopeBackbone(legacy), layout,
                           other_artist_mode=config.get('sparse_loss', {}).get('other_artist_mode', 'single_label'))
    return model.bind_vocabulary(release.vocabulary).to(device)


def _checkpoint_model(path, release, *, synthetic, device):
    """Explicit inference extraction, not an exact training-state/world migration."""
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('version') != CHECKPOINT_VERSION or manifest.get('complete') is not True:
        raise ValueError('Inference requires a complete window checkpoint')
    common = _read_file(path, manifest['common'], 'common.pt')
    binding = common['binding']
    if common['version'] != CHECKPOINT_VERSION or binding != manifest['binding'] or binding['training']['release'] != release.identity:
        raise ValueError('Checkpoint Release/Vocabulary binding mismatch')
    if bool(binding['synthetic_base']) != synthetic:
        raise ValueError('Synthetic checkpoint requires --synthetic; real weights never use a synthetic fallback')
    state = common['model']
    legacy = load_state_strict({key.removeprefix('backbone.legacy.'): value for key, value in state.items()
                               if key.startswith('backbone.legacy.')}, model_options(binding['base_model_config']))
    legacy.package_config, legacy.synthetic = binding['base_model_config'], binding['synthetic_base']
    legacy.class_mapping = {row['class_id']: row['name'] for row in release.vocabulary.document['original_artist']}
    model = MultiTaskModel(KaloscopeBackbone(legacy), HeadLayout(**binding['training']['layout']),
                           other_artist_mode=binding['training']['loss']['other_artist_mode'])
    model.load_state_dict(state, strict=True, assign=True)
    model.bind_vocabulary(release.vocabulary).set_stage(binding['training']['trainer']['stage'])
    return model.to(device).eval(), binding


def _dataset(config, index):
    cache_settings = dict(config.get('cache', {}))
    root = cache_settings.pop('root', 'cache/shards')
    timeout = cache_settings.pop('transfer_timeout', 3600.)
    read_timeout = cache_settings.pop('read_timeout', 30.)
    max_member_bytes = cache_settings.pop('max_member_bytes', 128 * 1024**2)
    cache_settings.setdefault('capacity_bytes', 10 * 1024**3)
    cache = ShardCache(root, index.release, provider_for_release(index.release, timeout=timeout, read_timeout=read_timeout), **cache_settings)
    corrections = DimensionCorrections(_runtime(config, 'dimensions', required=False) or str(Path(root) / 'dimensions.sqlite'), index.release)
    dataset = TarDataset(index, cache, corrections=corrections, preprocessor=_preprocessor(config), max_member_bytes=max_member_bytes)
    return dataset, cache


def _cohort(config, index):
    options = config.get('cohort', {})
    ids = options.get('sample_ids')
    if ids is None:
        path = options.get('sample_ids_file')
        if not path:
            raise ValueError('cohort.sample_ids or sample_ids_file must declare a fixed evaluation cohort')
        ids = json.loads(Path(path).read_text())
    return EvaluationCohort(index, ids, split=options.get('split', 'val'), config=EvaluationConfig(**config.get('evaluation', {})),
        class_ids=options.get('class_ids'), source_membership=index.release.source_membership(), source_filter=options.get('source_filter'))


def _evaluation_batches(dataset, cohort):
    wanted = set(cohort.sample_ids)
    ordinals = {row['sample_id']: ordinal for ordinal, row in dataset.index.iter_records(split=cohort.split) if row['sample_id'] in wanted}
    pending, ids, shape = [], [], None
    for sample_id in cohort.sample_ids:
        value = dataset[ordinals[sample_id]]['images']
        if pending and (value.shape != shape or len(pending) == cohort.config.row_batch):
            yield {'images': torch.stack(pending), 'sample_ids': ids}
            pending, ids = [], []
        pending.append(value); ids.append(sample_id); shape = value.shape
    if pending:
        yield {'images': torch.stack(pending), 'sample_ids': ids}


def _evaluate_current(model, config, index, dataset, run_dir, *, calibrate=False, checkpoint_id=None):
    cohort = _cohort(config, index)
    checkpoint_id = checkpoint_id or state_digest(model)
    options = config.get('feature_cache', {})
    cache_path = Path(options.get('path') or run_dir / 'features')
    preprocessing = preprocessing_identity(dataset.preprocessor)
    if cache_path.exists():
        cache = FeatureCache(cache_path, binding=feature_binding(model, cohort, checkpoint_id, preprocessing))
    else:
        cache = build_feature_cache(model, _evaluation_batches(dataset, cohort), cohort, cache_path,
                                    checkpoint_id=checkpoint_id, preprocessing=preprocessing)
    spool = options.get('spool_dir') or run_dir / 'spool'
    if calibrate:
        return fit_calibration(model, cache, cohort, spool_dir=spool, **config.get('calibration', {}))
    calibration_path = _runtime(config, 'calibration', required=False)
    calibration = json.loads(Path(calibration_path).read_text()) if calibration_path else None
    return evaluate(model, cache, cohort, spool_dir=spool, calibration=calibration, selection=config.get('selection'))


def _train(config, args, run_dir, device):
    release, index = _release_index(config)
    dataset, cache = _dataset(config, index)
    try:
        loss = SparseLossConfig.from_dict(config['sparse_loss'])
        scope = WeakNegativeScope.from_index(index, source_membership=release.source_membership(),
                                             source_filter=loss.source_filter if loss.weak_negative_scope == 'source_membership' else None)
        settings = WindowTrainerConfig(**config.get('window_trainer', {}))
        # Refuse before any model forward/image read when no finite endpoint exists.
        if settings.max_epochs is None and settings.max_attempted_windows is None:
            raise ValueError('train requires an explicit finite max_epochs or max_attempted_windows')
        initialization = _runtime(config, 'initialize_from', required=False)
        if initialization and _runtime(config, 'checkpoint', required=False):
            raise ValueError('Choose exact resume or explicit new-stage weight initialization, not both')
        if initialization:
            model, _ = _checkpoint_model(initialization, release, synthetic=args.synthetic, device=device)
            expected_layout = release.vocabulary.layout(**{key: value for key, value in config.get('head_layout', {}).items() if key not in LOGICAL_DOMAINS})
            if model.layout.to_dict() != expected_layout.to_dict():
                raise ValueError('New-stage initialization cannot silently change model structure')
        else:
            model = _model(config, release, synthetic=args.synthetic, device=device)
        config['head_layout'] = model.layout.to_dict()
        config['window_trainer'] = asdict(settings)
        config['sparse_loss'] = loss.to_dict()
        teacher, teacher_source = None, None
        if any(settings.preservation.values()):
            teacher_source = _runtime(config, 'teacher_source')
            if args.synthetic:
                from .selfcheck import synthetic_fixture
                teacher = KaloscopeBackbone(synthetic_fixture(config.get('teacher_seed', 0)))
            else:
                teacher = KaloscopeBackbone(load_kaloscope(_runtime(config, 'teacher_model_dir')))
        trainer = WindowTrainer(model, release, scope, loss, config=settings, device=device,
                                teacher=teacher, teacher_source=teacher_source)
        sampling = config.get('sampling', {})
        assets_path = sampling.get('assets', 'cache/sampling')
        error = None
        if trainer.rank == 0:
            try:
                build_sampling_assets(index, assets_path, **config.get('rfs', {}))
            except Exception as failure:
                error = failure
        trainer._raise_together(error, 'sampling asset publication')
        error, assets = None, None
        try:
            assets = SamplingAssets(assets_path, index)
        except Exception as failure:
            error = failure
        trainer._raise_together(error, 'sampling asset read')
        sampler = ShardWindowSampler(assets, dataset, sampling.get('batch_size', 8),
            window_samples=sampling.get('window_samples', 4096), seed=sampling.get('seed', 0), rank=trainer.rank, world_size=trainer.world_size)
        checkpoint = _runtime(config, 'checkpoint', required=False)
        if checkpoint:
            trainer.load_checkpoint(checkpoint, sampler=sampler)
        frequencies = config.get('run', {})
        for name in ('checkpoint_every_updates', 'evaluate_every_updates'):
            value = frequencies.get(name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f'{name} must be positive or null')
        if frequencies.get('evaluate_every_updates'):
            _cohort(config, index)  # Validate configured evaluation before training starts.
        def boundary(current, current_sampler, result):
            update = result['counters']['successful_updates']
            if result['success'] and frequencies.get('checkpoint_every_updates') and update % frequencies['checkpoint_every_updates'] == 0:
                current.save_checkpoint(run_dir / 'checkpoints' / f'update-{update}', sampler=current_sampler)
            if current.rank == 0:
                with (run_dir / 'windows.jsonl').open('a') as stream:
                    stream.write(json.dumps(result, allow_nan=False) + '\n')
                if result['success'] and frequencies.get('evaluate_every_updates') and update % frequencies['evaluate_every_updates'] == 0:
                    evaluation_config = deepcopy(config)
                    evaluation_dir = run_dir / f'evaluation-{update}'
                    evaluation_config['feature_cache'] = {'path': str(evaluation_dir / 'features'), 'spool_dir': str(evaluation_dir / 'spool')}
                    report = _evaluate_current(current.model, evaluation_config, index, dataset, evaluation_dir)
                    _write(evaluation_dir / 'report.json', report)
        _write(run_dir / f'effective-training-rank-{trainer.rank}.json', {'contract': trainer.contract(), 'sampler': sampler.state_dict(),
                                                                       'versions': versions(device)})
        counters = trainer.run(sampler, num_workers=sampling.get('num_workers', 0), worker_transport=sampling.get('worker_transport', 'shared_memory'),
                               worker_timeout=sampling.get('worker_timeout', 60.), on_boundary=boundary)
        final_path = run_dir / 'checkpoints' / 'final'
        trainer.save_checkpoint(final_path, sampler=sampler)
        return {'counters': counters, 'checkpoint': str(final_path), 'checkpoint_id': state_digest(trainer.model),
                'artist_conversion': index.metadata['stats']['artist_conversion'], 'synthetic': args.synthetic,
                'formal_training': 'NOT_STARTED' if args.synthetic else 'EXPLICIT_REAL_RUN',
                'distributed_world_size': trainer.world_size}
    finally:
        cache.close(); dataset.corrections.close(); index.close()


def run_command(args):
    config = read_config(args.config)
    config.setdefault('runtime', {})
    for name in ('release', 'index', 'model_dir', 'checkpoint', 'initialize_from', 'package', 'calibration'):
        value = getattr(args, name, None)
        if value is not None:
            config['runtime'][name] = value
    if args.command == 'train':
        for name in ('max_epochs', 'max_optimizer_steps', 'max_attempted_windows'):
            if getattr(args, name, None) is not None:
                config.setdefault('window_trainer', {})[name] = getattr(args, name)
    device = torch.device(args.device or config['runtime'].get('device', 'cpu'))
    config['runtime']['device'] = str(device)
    if args.command == 'train':
        if args.distributed_timeout_seconds is not None:
            config['runtime']['distributed_timeout_seconds'] = args.distributed_timeout_seconds
        device = initialize_distributed(device, timeout_seconds=config['runtime'].get('distributed_timeout_seconds', 120))
    rank = dist.get_rank() if dist.is_initialized() else 0
    run_dir = args.run_dir or config['runtime'].get('run_dir')
    if run_dir is None and dist.is_initialized():
        raise ValueError('torchrun requires one explicit shared --run-dir')
    run_dir = Path(run_dir or Path('outputs') / args.command / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]))
    run_dir.mkdir(parents=True, exist_ok=True)
    context = {'command': args.command, 'arguments': vars(args), 'resolved_config': config, 'versions': versions(device)}
    _write(run_dir / f'resolved-rank-{rank}.json', context)
    torch.manual_seed(config.get('seed', 0))
    started = time.perf_counter()
    if args.command == 'prepare-dataset':
        from .data.sakuratool import prepare_release
        source = config.get('dataset', {})
        release = prepare_release(source['repository'], source['revision'], source['manifest'],
                                  source['output'])
        result = {'release': str(release.root / 'release.json'), 'identity': release.identity,
                  'samples': release.document['producer_sample_counts'], 'images_read': 0,
                  'formal_training': 'NOT_STARTED'}
    elif args.command in ('inspect', 'preflight'):
        release = DatasetRelease.read(_runtime(config, 'release'))
        from .data.release import reference_index_schema
        import pyarrow.parquet as pq
        schemas = []
        expected_schema = reference_index_schema(release.vocabulary, release.document['release_id'],
                                               producer_format=release.document.get('producer_format'))
        for item in release.document['indices']:
            schema = pq.read_schema(release.local_path(item['path']))
            if (not schema.equals(expected_schema, check_metadata=False)
                    or any((schema.metadata or {}).get(key) != value for key, value in expected_schema.metadata.items())):
                raise ValueError('Published Parquet schema/Vocabulary/Release footer mismatch')
            schemas.append(item['path'])
        capacity = config.get('cache', {}).get('capacity_bytes', 10 * 1024**3)
        largest = max(item['size_bytes'] for item in release.shards.values())
        if type(capacity) is not int or capacity < largest:
            raise ValueError('Cache byte budget cannot hold the largest published shard')
        evaluation_config = EvaluationConfig(**config.get('evaluation', {}))
        config['evaluation'] = asdict(evaluation_config)
        trainer_config = WindowTrainerConfig(**config.get('window_trainer', {}))
        if 'sparse_loss' in config:
            SparseLossConfig.from_dict(config['sparse_loss'])
        scope = config.get('cohort', {})
        sample_ids = scope.get('sample_ids')
        if sample_ids is None and scope.get('sample_ids_file'):
            sample_ids = json.loads(Path(scope['sample_ids_file']).read_text())
        estimate = None
        if sample_ids:
            class_scope = scope.get('class_ids')
            classes = max(len(class_scope[name]) if class_scope else len(release.vocabulary.document['domains'][name])
                          for name in ('general', 'character', 'copyright'))
            estimate = 96 * len(sample_ids) * classes + 2048 * classes + 64 * len(sample_ids) * min(classes, evaluation_config.class_block)
            if len(sample_ids) > evaluation_config.max_samples or estimate > evaluation_config.ram_bytes:
                raise ValueError('Declared fixed cohort exceeds exact evaluation RAM/sample budget')
        result = {'parquet_schema_footers': schemas, 'exact_micro_ram_estimate': estimate,
                  'finite_train_endpoint_configured': trainer_config.max_epochs is not None or trainer_config.max_attempted_windows is not None,
                  'release': release.identity, 'vocabulary_classes': {k: len(v) for k, v in release.vocabulary.class_id_mapping.items()},
                  'published_shards': len(release.shards), 'published_tar_bytes': sum(item['size_bytes'] for item in release.shards.values()),
                  'cache_budget_bytes': config.get('cache', {}).get('capacity_bytes'),
                  'evaluation_budget': config.get('evaluation', {}), 'images_read': 0,
                  'producer_handoff': 'REAL_PRODUCER_HANDOFF_NOT_VERIFIED'}
    elif args.command == 'build-local-index':
        release = DatasetRelease.read(_runtime(config, 'release'))
        index = build_local_index(release, _runtime(config, 'index'))
        assets = build_sampling_assets(index, config.get('sampling', {}).get('assets', str(run_dir / 'sampling')), **config.get('rfs', {}))
        result = {'index': _runtime(config, 'index'), 'stats': index.metadata['stats'], 'sampling_assets': str(assets.path), 'rfs': assets.metadata['stats'], 'images_read': 0}
        index.close()
    elif args.command == 'train':
        result = _train(config, args, run_dir, device)
    elif args.command in ('evaluate', 'calibrate', 'export'):
        release, index = _release_index(config)
        model, binding = _checkpoint_model(_runtime(config, 'checkpoint'), release, synthetic=args.synthetic, device=device)
        config['head_layout'] = model.layout.to_dict()
        dataset, cache = _dataset(config, index)
        try:
            requested = preprocessing_identity(dataset.preprocessor)
            normalized = {'version': requested['version'], 'mean': dataset.preprocessor.mean.flatten().tolist(),
                          'std': dataset.preprocessor.std.flatten().tolist()}
            if (normalized != binding['preprocessing']
                    or requested['bucket'] != binding['sampler']['preprocessing']['bucket']
                    or requested['alpha_background'] != binding['sampler']['preprocessing']['alpha_background']):
                raise ValueError('Checkpoint and requested preprocessing differ (version/mean/std/bucket/alpha)')
            checkpoint_id = state_digest(model)
            if args.command == 'export':
                output = args.output or str(run_dir / 'predictor')
                calibration_path = _runtime(config, 'calibration', required=False)
                calibration = json.loads(Path(calibration_path).read_text()) if calibration_path else None
                manifest = export_predictor(model, release.vocabulary, output, checkpoint_id=checkpoint_id, preprocessor=dataset.preprocessor,
                    calibration=calibration, thresholds=config.get('prediction', {}).get('thresholds') if calibration is None else None,
                    untrained=untrained_label_metadata(index), source_membership=release.source_membership())
                result = {'manifest': str(manifest), 'package': str(Path(manifest).parent), 'checkpoint_id': checkpoint_id}
            else:
                result = _evaluate_current(model, config, index, dataset, run_dir, calibrate=args.command == 'calibrate', checkpoint_id=checkpoint_id)
                if args.output:
                    _write(args.output, result)
        finally:
            cache.close(); dataset.corrections.close(); index.close()
    elif args.command == 'predict':
        from PIL import Image
        predictor = load_predictor(_runtime(config, 'package'), device=device)
        with Image.open(args.image) as image:
            result = predictor.predict_image(image, args.heads, output_mode=args.output_mode, top_k=args.top_k,
                max_tags=args.max_tags, source_filter=args.sources, thresholds=config.get('prediction', {}).get('thresholds'))
        if args.output:
            _write(args.output, result)
    elif args.command == 'selfcheck':
        from .selfcheck import run_selfcheck
        result = run_selfcheck(args.suite, real_model_dir=args.real_model_dir, skip_ddp_reason=args.skip_ddp_reason)
    elif args.command == 'benchmark':
        from .benchmark import benchmark_synthetic
        result = benchmark_synthetic(device=device, iterations=args.iterations)
    else:
        raise ValueError('Unknown command')
    result = {'command': args.command, 'elapsed_seconds': time.perf_counter() - started, 'result': result}
    _write(run_dir / f'resolved-rank-{rank}.json', context)
    _write(run_dir / f'result-rank-{rank}.json', result)
    if rank == 0:
        print(json.dumps(result, default=_json, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    for name in ('prepare-dataset', 'inspect', 'preflight', 'build-local-index', 'train', 'evaluate', 'calibrate', 'predict', 'export', 'selfcheck', 'benchmark'):
        command = commands.add_parser(name)
        command.add_argument('--config', type=Path)
        command.add_argument('--run-dir', type=Path)
        command.add_argument('--device')
        command.add_argument('--release')
        command.add_argument('--index')
        command.add_argument('--model-dir')
        command.add_argument('--checkpoint')
        command.add_argument('--initialize-from')
        command.add_argument('--package')
        command.add_argument('--calibration')
        command.add_argument('--synthetic', action='store_true', help='Explicit tiny synthetic model/checkpoint; never an official weight fallback')
        command.add_argument('--output')
        if name == 'train':
            command.add_argument('--distributed-timeout-seconds', type=float, help='Collective timeout must cover bounded periodic evaluation/checkpoint work')
            command.add_argument('--max-epochs', type=int)
            command.add_argument('--max-optimizer-steps', type=int)
            command.add_argument('--max-attempted-windows', type=int)
        if name == 'predict':
            command.add_argument('--image', required=True)
            command.add_argument('--heads', nargs='+')
            command.add_argument('--sources', nargs='+')
            command.add_argument('--output-mode', choices=('top_k', 'threshold', 'dense'), default='top_k')
            command.add_argument('--top-k', type=int, default=20)
            command.add_argument('--max-tags', type=int)
        if name == 'selfcheck':
            command.add_argument('--suite', default='all')
            command.add_argument('--real-model-dir')
            command.add_argument('--skip-ddp-reason', help='Explicit environmental NOT_VERIFIED skip; no attempted backend probe')
        if name == 'benchmark':
            command.add_argument('--iterations', type=int, default=3)
    return result


def main():
    args = parser().parse_args()
    try:
        run_command(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
