"""Actual bounded synthetic stage timing; not a real model throughput claim."""
from pathlib import Path
import tempfile
import time

import torch

from .data import window_loader
from .selfcheck_ddp import training_fixture, make_components


def benchmark_synthetic(*, device='cpu', iterations=3):
    device = torch.device(device)
    if type(iterations) is not int or not 1 <= iterations <= 100:
        raise ValueError('Benchmark iterations must be 1..100')
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Benchmark supports CPU/CUDA')
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    timings = {}
    def measure(name, call):
        sync(); begin = time.perf_counter()
        value = call()
        sync(); timings.setdefault(name, []).append(time.perf_counter() - begin)
        return value
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        measure('synthetic_release_creation', lambda: training_fixture(root / 'published'))
        trainer, sampler, cache = make_components(root, stage='T2')
        model = trainer.model.to(device)
        try:
            measure('first_decode_bucket_plan', sampler.prepare_window)
            _, batch = measure('TAR_read_decode_preprocess_collate', lambda: next(iter(window_loader(sampler.dataset, sampler))))
            images = batch['images'].to(device)
            model.eval()
            for _ in range(iterations):
                with torch.no_grad():
                    features = measure('backbone', lambda: model.backbone(images, include_artist=False, include_style=False))
                    general = measure('general_features', lambda: model.general.features(features.global_features, features.patch_tokens))
                    identity = measure('identity_features', lambda: model.identity.features(features.global_features, features.patch_tokens))
                    measure('terminal_classifiers', lambda: (model.general.classify_features(general), model.identity.classify_features(identity), model.other_artist(features.global_features)))
            # Actual common optimizer engine includes sparse loss/backward/step.
            # Reconstruct on the selected device; no synthetic CPU run is called a GPU benchmark.
            from .training import WindowTrainer
            trainer = WindowTrainer(model, trainer.release, trainer.scope, trainer.loss_config, config=trainer.config, device=device)
            measure('complete_optimizer_window', lambda: trainer.train_window(sampler, iter(window_loader(sampler.dataset, sampler))))
            measure('boundary_checkpoint', lambda: trainer.save_checkpoint(root / 'checkpoint', sampler=sampler))
            return {'synthetic': True, 'device': str(device), 'iterations': iterations,
                    'seconds': {name: {'samples': values, 'mean': sum(values) / len(values)} for name, values in timings.items()},
                    'note': 'Tiny synthetic CPU/GPU engineering timings only; no production throughput or parameter-count speed claim',
                    'real_model_performance': 'NOT_VERIFIED'}
        finally:
            cache.close()


if __name__ == '__main__':
    import json
    print(json.dumps(benchmark_synthetic(), indent=2))
