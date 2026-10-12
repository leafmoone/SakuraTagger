"""Boundary-only, all-rank complete checkpoints for the sparse window trainer.

Files contain only safe-loader tensors and primitive state. Cache contents and
teacher weights are deliberately external to each checkpoint.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist

VERSION = 'sakura-tagger-window-checkpoint-v1'


def state_digest(module):
    """Bind fixed teacher weights once at construction, not on every update/save."""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def teacher_identity(teacher, source):
    if teacher is None:
        if source is not None:
            raise ValueError('A teacher source requires an enabled teacher')
        return None
    if source is None:
        return None  # Training is allowed; saving requires a recoverable source.
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Teacher source must identify a fixed recoverable package or fixture')
    return {'source': source, 'state_sha256': state_digest(teacher),
            'package_config': deepcopy(teacher.legacy.package_config)}


def _rng_state(device):
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), state[2], state[3], state[4]),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def _validate_rng(state, device):
    random.Random().setstate(state['python'])
    np.random.RandomState().set_state((state['numpy'][0], np.asarray(state['numpy'][1], dtype=np.uint32),
                                     *state['numpy'][2:]))
    torch.Generator().set_state(state['torch'])
    if (state['cuda'] is not None) != (device.type == 'cuda'):
        raise ValueError('Checkpoint RNG device differs; use an explicit new run')
    if state['cuda'] is not None:
        torch.Generator(device=device).set_state(state['cuda'])


def _restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state((state['numpy'][0], np.asarray(state['numpy'][1], dtype=np.uint32), *state['numpy'][2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def _descriptor(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return {'path': path.name, 'size': path.stat().st_size, 'sha256': digest.hexdigest()}


def _atomic_write(path, writer):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.partial.', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open('wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _binding(trainer, sampler):
    signature = sampler._signature().copy()
    signature.pop('rank')
    preprocessing = sampler.dataset.preprocessor
    return {'training': trainer.contract(), 'sampler': signature,
            'base_model_config': deepcopy(trainer.model.backbone.legacy.package_config),
            'synthetic_base': bool(trainer.model.backbone.legacy.synthetic),
            'preprocessing': {'version': 'native-full-frame-exif-rgb-bicubic-v1',
                              'mean': preprocessing.mean.flatten().tolist(),
                              'std': preprocessing.std.flatten().tolist()},
            'device_type': trainer.device.type, 'teacher': trainer.teacher_identity}


def _boundary(trainer, sampler):
    if not trainer.at_boundary:
        raise ValueError('Checkpoint requires a complete optimizer-window boundary')
    if (sampler.rank, sampler.world_size) != (trainer.rank, trainer.world_size):
        raise ValueError('Checkpoint sampler rank/world identity mismatch')
    if trainer.teacher is not None and trainer.teacher_identity is None:
        raise ValueError('Checkpoint preservation requires an explicit fixed teacher_source')
    if any(parameter.grad is not None for parameter in trainer.model.parameters()):
        raise ValueError('Pending gradients cannot be checkpointed or overwritten')


def _raise(trainer, error, phase):
    # Checkpoint rejection must not clear pending gradients or otherwise mutate
    # the live training state, unlike a failed training attempt.
    if not trainer._all_true(error is None):
        message = None if error is None else f'{type(error).__name__}: {error}'
        messages = [message]
        if trainer.distributed:
            messages = [None] * trainer.world_size
            dist.all_gather_object(messages, message)
        from .window_trainer import CoordinatedTrainingError
        raise CoordinatedTrainingError(f'{phase} failed across ranks: {messages}') from error


def save_checkpoint(trainer, path, sampler):
    """All ranks call with one new shared directory; rank 0 publishes last.

    Failed/incomplete directories are retained for inspection and cannot resume.
    Never overwrite an existing checkpoint, including an incomplete attempt.
    """
    path = Path(path)
    error, request = None, None
    try:
        _boundary(trainer, sampler)
        request = {'path': str(path.resolve()), 'binding': _binding(trainer, sampler),
                   'counters': trainer.counters, 'scheduler': trainer.scheduler.state_dict(),
                   'scaler': trainer.scaler.state_dict(),
                   'position': [sampler.epoch, sampler.window, sampler.cursor]}
    except Exception as failure:
        error = failure
    _raise(trainer, error, 'checkpoint boundary')
    if trainer.distributed:
        requests = [None] * trainer.world_size
        dist.all_gather_object(requests, request)
        if any(value != requests[0] for value in requests):
            raise ValueError('Ranks disagree on checkpoint destination, contract, or consumed boundary')
    error = None
    if trainer.rank == 0:
        try:
            path.mkdir(parents=True, exist_ok=False)
        except Exception as failure:
            error = failure
    _raise(trainer, error, 'checkpoint directory')
    error, local = None, None
    try:
        rank_path = path / f'rank-{trainer.rank:05d}.pt'
        rank_state = {'version': VERSION, 'rank': trainer.rank, 'world_size': trainer.world_size,
                      'sampler': sampler.state_dict(), 'rng': _rng_state(trainer.device)}
        _atomic_write(rank_path, lambda stream: torch.save(rank_state, stream))
        local = _descriptor(rank_path)
        if trainer.rank == 0:
            common = {'version': VERSION, 'binding': request['binding'], 'counters': dict(trainer.counters),
                      'model': trainer.model.state_dict(), 'optimizer': trainer.optimizer.state_dict(),
                      'scheduler': trainer.scheduler.state_dict(), 'scaler': trainer.scaler.state_dict()}
            _atomic_write(path / 'common.pt', lambda stream: torch.save(common, stream))
    except Exception as failure:
        error = failure
    _raise(trainer, error, 'checkpoint files')
    ranks = [local]
    if trainer.distributed:
        ranks = [None] * trainer.world_size
        dist.all_gather_object(ranks, local)
    error = None
    if trainer.rank == 0:
        try:
            manifest = {'version': VERSION, 'complete': True, 'world_size': trainer.world_size,
                        'binding': request['binding'], 'common': _descriptor(path / 'common.pt'), 'ranks': ranks}
            body = json.dumps(manifest, sort_keys=True, indent=2).encode()
            _atomic_write(path / 'manifest.json', lambda stream: stream.write(body))
        except Exception as failure:
            error = failure
    _raise(trainer, error, 'checkpoint completion manifest')
    return path / 'manifest.json'


def _read_file(path, descriptor, name):
    if (not isinstance(descriptor, dict) or descriptor.get('path') != name
            or type(descriptor.get('size')) is not int or descriptor['size'] < 1):
        raise ValueError('Invalid checkpoint file descriptor')
    target = path / name
    if _descriptor(target) != descriptor:
        raise ValueError(f'Checkpoint file size/checksum mismatch: {name}')
    return torch.load(target, map_location='cpu', weights_only=True)


def _prepare_optimizer(trainer, state):
    optimizer = trainer._new_optimizer()
    current = optimizer.state_dict()['param_groups']
    groups = state['param_groups']
    if len(groups) != len(current):
        raise ValueError('Checkpoint optimizer groups changed')
    for actual, expected in zip(groups, current):
        if (set(actual) != set(expected) or len(actual['params']) != len(expected['params'])
                or any(actual[name] != expected[name] for name in expected if name not in ('params', 'lr'))):
            raise ValueError('Checkpoint optimizer membership/configuration changed')
    optimizer.load_state_dict(state)
    for parameter, values in optimizer.state.items():
        if set(values) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Invalid AdamW state')
        for name in ('exp_avg', 'exp_avg_sq'):
            if values[name].shape != parameter.shape or values[name].dtype != parameter.dtype:
                raise ValueError('Invalid AdamW moment dimensions/type')
        if values['step'].numel() != 1 or not torch.isfinite(values['step']) or values['step'] < 0:
            raise ValueError('Invalid AdamW step')
    return optimizer


def load_checkpoint(trainer, path, sampler):
    """Validate all prospective state before changing live weights, cursor or RNG."""
    from .window_trainer import UpdateSchedule
    path = Path(path)
    error, prepared = None, None
    try:
        _boundary(trainer, sampler)
        manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        if (manifest.get('version') != VERSION or manifest.get('complete') is not True
                or manifest.get('world_size') != trainer.world_size
                or manifest.get('binding') != _binding(trainer, sampler)):
            raise ValueError('Incomplete or incompatible checkpoint; use an explicit new run/migration')
        if len(manifest['ranks']) != trainer.world_size:
            raise ValueError('Missing rank checkpoint state')
        # Every rank confirms all required files are present. Each rank hashes its
        # own RNG file and the common model once, not the TARs or whole dataset.
        for rank, descriptor in enumerate(manifest['ranks']):
            if descriptor.get('path') != f'rank-{rank:05d}.pt' or not (path / descriptor['path']).is_file():
                raise ValueError('Incomplete rank checkpoint files')
        common = _read_file(path, manifest['common'], 'common.pt')
        local = _read_file(path, manifest['ranks'][trainer.rank], f'rank-{trainer.rank:05d}.pt')
        if (common.get('version') != VERSION or common.get('binding') != manifest['binding']
                or local.get('version') != VERSION or local.get('rank') != trainer.rank
                or local.get('world_size') != trainer.world_size):
            raise ValueError('Checkpoint payload identity mismatch')
        expected = trainer.model.state_dict()
        if set(common['model']) != set(expected) or any(
                not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype
                for name, value in common['model'].items()):
            raise ValueError('Checkpoint model keys, dimensions or dtype mismatch')
        counters = common['counters']
        if (set(counters) != set(trainer.counters) or any(type(value) is not int or value < 0 for value in counters.values())
                or counters['attempted_windows'] != counters['successful_updates'] + counters['skipped_updates']):
            raise ValueError('Invalid checkpoint update counters')
        prospective = sampler.prepare_restore(local['sampler'])
        _validate_rng(local['rng'], trainer.device)
        optimizer = _prepare_optimizer(trainer, common['optimizer'])
        scheduler = UpdateSchedule(optimizer, trainer.config)
        scheduler.load_state_dict(common['scheduler'])
        if scheduler.completed > counters['successful_updates']:
            raise ValueError('Scheduler exceeds successful update count')
        scaler = torch.amp.GradScaler(trainer.device.type, enabled=trainer.config.amp == 'fp16')
        if bool(common['scaler']) != scaler.is_enabled():
            raise ValueError('Checkpoint GradScaler mode mismatch')
        scaler.load_state_dict(common['scaler'])
        prepared = common, local, prospective, optimizer, scheduler, scaler
    except Exception as failure:
        error = failure
    _raise(trainer, error, 'checkpoint validation')
    if trainer.distributed:
        positions = [None] * trainer.world_size
        dist.all_gather_object(positions, [str(path.resolve()), common['counters'],
                                         prospective.epoch, prospective.window, prospective.cursor])
        if any(value != positions[0] for value in positions):
            raise ValueError('Ranks selected different checkpoint boundaries')
    common, local, prospective, optimizer, scheduler, scaler = prepared
    trainer.model.load_state_dict(common['model'], strict=True)
    trainer.optimizer, trainer.scheduler, trainer.scaler = optimizer, scheduler, scaler
    trainer.counters = dict(common['counters'])
    sampler.__dict__.update(prospective.__dict__)
    _restore_rng(local['rng'], trainer.device)
    return dict(trainer.counters)
