"""Real CLI-parser synthetic Release-to-predict flow; optional actual Gloo route."""
from contextlib import redirect_stdout
from io import StringIO
import json
import shutil
from pathlib import Path
import tempfile

from PIL import Image
import torch
import torch.distributed as dist
import yaml

from .selfcheck_ddp import training_fixture


def _prepare(root):
    from .cli import read_config
    root.mkdir(parents=True, exist_ok=True)
    release = training_fixture(root / 'published', splits=['train'] * 5 + ['val'] * 2 + ['test'] * 2, build_assets=False)
    config = read_config(Path(__file__).resolve().parents[2] / 'configs/baseline.yaml')
    config.update(seed=71, runtime={'release': str(root / 'published/release.json'), 'index': str(root / 'index.sqlite'),
                                  'dimensions': str(root / 'dimensions.sqlite'), 'device': 'cpu'},
                  preprocessing={'bucket': {'target_pixels': 1024, 'max_pixels': 4096, 'max_side': 64}},
                  cohort={'split': 'val', 'sample_ids': ['tar_5', 'tar_6']},
                  evaluation={'row_batch': 2, 'class_block': 3, 'ram_bytes': 2 * 1024**2, 'disk_bytes': 4 * 1024**2},
                  calibration={'min_class_positives': 2, 'min_class_negatives': 2},
                  run={'checkpoint_every_updates': 1, 'evaluate_every_updates': 1})
    config['head_layout'].update(feature_dim=32, attention_dim=16)
    config['cache'].update(root=str(root / 'cache'), capacity_bytes=65536)
    config['sampling'].update(assets=str(root / 'assets'), batch_size=2, window_samples=32, num_workers=0)
    config['window_trainer'].update(stage='T2', accumulation_steps=1, warmup_updates=0, schedule_updates=8, max_epochs=1)
    # Actual fixture IDs are read from its Parquet contract, never guessed.
    import pyarrow.parquet as pq
    rows = pq.read_table(root / 'published/index.parquet').to_pylist()
    config['cohort']['sample_ids'] = [row['sample_id'] for row in rows if row['split'] == 'val']
    path = root / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    Image.new('RGBA', (48, 32), (100, 150, 200, 120)).save(root / 'input.png')
    return path


def _call(root, config_path, command, *extra):
    from .cli import parser, run_command
    args = parser().parse_args([command, '--config', str(config_path), '--run-dir', str(root / command), *extra])
    with redirect_stdout(StringIO()):
        return run_command(args)['result']


def _pipeline(root, config_path, *, distributed=False):
    rank = dist.get_rank() if distributed else 0
    train = _call(root, config_path, 'train', '--synthetic')
    config = yaml.safe_load(config_path.read_text())
    # Same settings; resume an intermediate complete optimizer-window boundary.
    resume = _call(root / 'resume', config_path, 'train', '--synthetic', '--checkpoint', str(root / 'train/checkpoints/update-1'))
    assert train['counters'] == resume['counters'] and train['checkpoint_id'] == resume['checkpoint_id']
    if rank == 0:
        evaluation = _call(root, config_path, 'evaluate', '--synthetic', '--checkpoint', train['checkpoint'])
        calibration_path = root / 'calibration.json'
        calibration = _call(root, config_path, 'calibrate', '--synthetic', '--checkpoint', train['checkpoint'], '--output', str(calibration_path))
        exported = _call(root, config_path, 'export', '--synthetic', '--checkpoint', train['checkpoint'], '--calibration', str(calibration_path), '--output', str(root / 'package'))
        prediction = _call(root, config_path, 'predict', '--package', exported['package'], '--image', str(root / 'input.png'),
                           '--heads', 'general', '--output-mode', 'threshold', '--max-tags', '3')
        assert evaluation['selection']['status'] == 'REPORT_ONLY'
        assert calibration['split'] == 'val'
        assert len(prediction['outputs']['general']['class_ids']) == 1
        assert len(prediction['outputs']['general']['class_ids'][0]) <= 3
        assert (root / 'train/evaluation-1/report.json').is_file()
        for command in ('train', 'evaluate', 'calibrate', 'export', 'predict'):
            assert (root / command / 'resolved-rank-0.json').is_file()
        result = {'Release_TAR_cache_bucket_loss_train_checkpoint_resume_evaluate_calibrate_export_predict': 'PASS',
                  'exact_resume_final_model_hash_and_counters': 'PASS', 'periodic_checkpoint_and_evaluation': 'PASS',
                  'resolved_config_library_device_records': 'PASS', 'artist_conversion': train['artist_conversion'],
                  'training_counters': train['counters'], 'world_size': train['distributed_world_size']}
    else:
        result = None
    if distributed:
        values = [result]
        dist.broadcast_object_list(values, src=0)
        result = values[0]
    return result


def check_integration():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = _prepare(root)
        preflight = _call(root, config, 'preflight')
        indexed = _call(root, config, 'build-local-index')
        assert preflight['images_read'] == indexed['images_read'] == 0
        result = _pipeline(root, config)
        # Default null stop limits must fail without starting a model/data forward.
        document = yaml.safe_load(config.read_text())
        document['window_trainer'].update(max_epochs=None, max_attempted_windows=None)
        refused = root / 'no-endpoint.yaml'; refused.write_text(yaml.safe_dump(document))
        try:
            _call(root / 'refused', refused, 'train', '--synthetic')
        except ValueError as error:
            assert 'finite' in str(error)
        else:
            raise AssertionError('Unbounded training configuration accepted')
        try:
            _call(root / 'wrong-mode', config, 'evaluate', '--checkpoint', str(root / 'train/checkpoints/final'))
        except ValueError as error:
            assert '--synthetic' in str(error)
        else:
            raise AssertionError('Synthetic checkpoint silently used as real weights')
        # Complete checkpoint with changed normalization must never be silently
        # interpreted using the CLI's canonical ImageNet preprocessing.
        changed = root / 'changed-normalization'
        shutil.copytree(root / 'train/checkpoints/final', changed)
        common = torch.load(changed / 'common.pt', weights_only=True)
        common['binding']['preprocessing']['mean'][0] += .1
        torch.save(common, changed / 'common.pt')
        manifest = json.loads((changed / 'manifest.json').read_text())
        manifest['binding'] = common['binding']
        from .training.window_checkpoint import _descriptor
        manifest['common'] = _descriptor(changed / 'common.pt')
        (changed / 'manifest.json').write_text(json.dumps(manifest))
        for command in ('evaluate', 'export'):
            try:
                _call(root / 'wrong-preprocessing', config, command, '--synthetic', '--checkpoint', str(changed))
            except ValueError as error:
                assert 'preprocessing differ' in str(error)
            else:
                raise AssertionError('Changed checkpoint normalization silently accepted')
        result['whole_checkpoint_preprocessing_binding'] = 'PASS'
        # Explicit new T2 run imports T1 weights without carrying counters/state.
        phase_config = yaml.safe_load(config.read_text())
        phase_config['window_trainer']['stage'] = 'T1'
        phase_config['run'] = {}
        t1_config = root / 'T1.yaml'; t1_config.write_text(yaml.safe_dump(phase_config))
        t1 = _call(root / 'stage1', t1_config, 'train', '--synthetic')
        phase_config['window_trainer']['stage'] = 'T2'
        t2_config = root / 'T2.yaml'; t2_config.write_text(yaml.safe_dump(phase_config))
        t2 = _call(root / 'stage2', t2_config, 'train', '--synthetic', '--initialize-from', t1['checkpoint'])
        assert t1['counters']['successful_updates'] == t2['counters']['successful_updates']
        effective = json.loads((root / 'stage2/train/effective-training-rank-0.json').read_text())
        assert effective['contract']['trainer']['stage'] == 'finetune_last4'
        result['explicit_T1_to_T2_new_run_initialization'] = 'PASS'
        from .benchmark import benchmark_synthetic
        result['synthetic_cpu_benchmark'] = benchmark_synthetic(iterations=2)
        result['no_endpoint_and_no_synthetic_fallback'] = 'PASS'
        result['CPU_SINGLE_PROCESS_E2E'] = 'PASS'
        result['CPU_DDP_E2E'] = 'NOT_VERIFIED; actual Gloo suite is a separate required check'
        return result


def check_distributed_pipeline(root):
    if dist.get_rank() == 0:
        config = _prepare(root)
        _call(root, config, 'preflight')
        _call(root, config, 'build-local-index')
    dist.barrier()
    return _pipeline(root, root / 'config.yaml', distributed=True)


if __name__ == '__main__':
    print(json.dumps(check_integration(), indent=2))
