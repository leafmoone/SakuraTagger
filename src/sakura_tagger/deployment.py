"""Strict self-contained inference packages; loss modes are never guessed."""
from copy import deepcopy, copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file, save_file

from .adapter import load_state_strict, model_options, IMAGENET_MEAN, IMAGENET_STD
from .data import Vocabulary, SourceMembership, BucketConfig, BucketSelector, NativePreprocessor
from .data.release import artifact_descriptor, verify_artifact
from .evaluation.cohort import identity
from .model import HeadLayout, KaloscopeBackbone, MultiTaskModel
from .model.prediction import normalize_thresholds
from .model.registry import LOGICAL_DOMAINS
from .training.window_checkpoint import state_digest

VERSION = 'sakura-tagger-deployment-v1'


def preprocessing_identity(preprocessor):
    if (not torch.equal(preprocessor.mean, torch.tensor(IMAGENET_MEAN).view(3, 1, 1))
            or not torch.equal(preprocessor.std, torch.tensor(IMAGENET_STD).view(3, 1, 1))):
        raise ValueError('Deployment requires the declared native ImageNet normalization')
    return {'version': 'native-full-frame-exif-rgb-bicubic-v1', 'bucket': asdict(preprocessor.selector.config),
            'alpha_background': list(preprocessor.alpha_background), 'mean': list(IMAGENET_MEAN), 'std': list(IMAGENET_STD),
            'crop': False, 'padding': False}


def _preprocessor(document):
    if (set(document) != {'version', 'bucket', 'alpha_background', 'mean', 'std', 'crop', 'padding'}
            or document['version'] != 'native-full-frame-exif-rgb-bicubic-v1'
            or document['mean'] != list(IMAGENET_MEAN) or document['std'] != list(IMAGENET_STD)
            or document['crop'] is not False or document['padding'] is not False
            or len(document['alpha_background']) != 3
            or any(type(value) is not int or not 0 <= value <= 255 for value in document['alpha_background'])):
        raise ValueError('Unsupported deployment preprocessing identity')
    return NativePreprocessor(BucketSelector(BucketConfig(**document['bucket'])), alpha_background=tuple(document['alpha_background']))


def untrained_label_metadata(index):
    return {'version': 'sakura-tagger-untrained-labels-v1', 'vocab_sha256': index.release.vocabulary.sha256,
            'release': index.release.identity, 'reason': 'NO_TRAIN_POSITIVE',
            'ids': {domain: np.flatnonzero(index.positive_support(domain) == 0).tolist() for domain in LOGICAL_DOMAINS}}


def _membership_write(membership, vocabulary, path):
    if membership.vocab_sha256 != vocabulary.sha256:
        raise ValueError('Deployment membership exact Vocabulary mismatch')
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = [{'domain': domain, 'class_id': class_id, 'source': source}
            for domain in vocabulary.class_id_mapping for source in membership.sources
            for class_id in membership.allowed_ids(domain, source)]
    schema = pa.schema([('domain', pa.string()), ('class_id', pa.int32()), ('source', pa.string())], metadata={
        b'schema_version': membership.schema_version.encode(), b'vocab_version': membership.vocab_version.encode(),
        b'vocab_sha256': membership.vocab_sha256.encode()})
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _calibrated_thresholds(calibration, *, vocabulary, checkpoint_id, model_sha256, preprocessing, counts):
    value = deepcopy(calibration)
    digest = value.pop('identity', None)
    if (digest != identity(value) or value.get('version') != 'sakura-tagger-thresholds-v1'
            or value.get('vocab_sha256') != vocabulary.sha256 or value.get('checkpoint') != checkpoint_id
            or value.get('model_sha256') != model_sha256 or value.get('preprocessing') != preprocessing
            or value.get('split') not in ('val', 'calibration') or set(value.get('domains', {})) != {'general', 'character', 'copyright'}):
        raise ValueError('Calibration does not bind this checkpoint/Vocabulary/preprocessing')
    if set(value.get('class_ids', {})) != set(counts) or not isinstance(value.get('sources'), list):
        raise ValueError('Calibration must preserve explicit class/source scope')
    for domain, ids in value['class_ids'].items():
        if ids != sorted(set(ids)) or any(type(label) is not int or not 0 <= label < counts[domain] for label in ids):
            raise ValueError('Invalid calibration original-ID scope')
        if domain in value['domains'] and set(value['domains'][domain]['per_class']) != {str(label) for label in ids}:
            raise ValueError('Calibration threshold coverage differs from its fixed class scope')
    settings = {domain: {'default': item['domain_threshold'], 'per_class': item['per_class']}
                for domain, item in value['domains'].items()}
    return normalize_thresholds(settings, counts)


def export_predictor(model, vocabulary, output, *, checkpoint_id, preprocessor,
                     thresholds=None, calibration=None, untrained=None, source_membership=None):
    """Export only an already vocabulary-bound model to a new local directory."""
    if model._vocabulary_sha256 != vocabulary.sha256:
        raise ValueError('Export requires weights already bound to this exact Vocabulary')
    model.bind_vocabulary(vocabulary)
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError('Explicit checkpoint identity required')
    preprocessing = preprocessing_identity(preprocessor)
    _preprocessor(preprocessing)
    counts = {domain: len(entries) for domain, entries in vocabulary.class_id_mapping.items()}
    model_sha = state_digest(model)
    if calibration is not None and thresholds is not None:
        raise ValueError('Use a frozen calibration or explicit thresholds, not ambiguous overrides')
    settings = (_calibrated_thresholds(calibration, vocabulary=vocabulary, checkpoint_id=checkpoint_id,
                model_sha256=model_sha, preprocessing=preprocessing, counts=counts) if calibration is not None
                else normalize_thresholds({} if thresholds is None else thresholds, counts))
    # Validation on a lightweight policy copy never changes the live model.
    untrained = model._untrained_labels if untrained is None else untrained
    untrained = json.loads(json.dumps(untrained)) if untrained is not None else None
    source_membership = model._source_membership if source_membership is None else source_membership
    policy = copy(model)
    policy.set_untrained_labels(untrained)
    normalized_untrained = deepcopy(policy._untrained_labels)
    output = Path(output)
    if output.exists():
        raise ValueError('Deployment output exists; use a new explicit version')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + '.partial.', dir=output.parent))
    try:
        (temporary / 'vocabulary.json').write_text(json.dumps(vocabulary.document, ensure_ascii=False, indent=2))
        save_file({name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}, str(temporary / 'model.safetensors'))
        files = {name: artifact_descriptor(temporary / name, temporary) for name in ('model.safetensors', 'vocabulary.json')}
        if source_membership is not None:
            _membership_write(source_membership, vocabulary, temporary / 'source_membership.parquet')
            files['source_membership.parquet'] = artifact_descriptor(temporary / 'source_membership.parquet', temporary)
        if calibration is not None:
            (temporary / 'calibration.json').write_text(json.dumps(calibration, ensure_ascii=False, indent=2))
            files['calibration.json'] = artifact_descriptor(temporary / 'calibration.json', temporary)
        manifest = {'version': VERSION, 'checkpoint_id': checkpoint_id, 'model_sha256': model_sha,
                    'vocab_sha256': vocabulary.sha256, 'layout': model.layout.to_dict(),
                    'base_model_config': model.backbone.legacy.package_config,
                    'synthetic_base': bool(model.backbone.legacy.synthetic), 'stage': model.backbone.stage,
                    'other_artist_mode': model.other_artist_mode, 'preprocessing': preprocessing,
                    'calibration_scope': {'class_ids': calibration['class_ids'], 'sources': calibration['sources']} if calibration is not None else None,
                    'thresholds': settings, 'threshold_status': 'FROZEN_VALIDATION_CALIBRATION' if calibration is not None else 'EXPLICIT_UNCALIBRATED' if settings else 'NOT_CONFIGURED',
                    'untrained_labels': normalized_untrained, 'files': files}
        manifest['identity'] = identity(manifest)
        (temporary / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return output / 'manifest.json'


class Predictor:
    def __init__(self, model, vocabulary, preprocessor, manifest):
        self.model, self.vocabulary, self.preprocessor, self.manifest = model, vocabulary, preprocessor, manifest

    def predict(self, images, head_selection=None, *, output_mode=None, thresholds=None, **options):
        if output_mode == 'threshold' and thresholds is None:
            thresholds = self.manifest['thresholds']
            scope = self.manifest['calibration_scope']
            if scope is not None:
                requested = options.pop('class_scope', None)
                options['class_scope'] = {name: list(ids) if requested is None or name not in requested
                                         else [class_id for class_id in ids if class_id in requested[name]]
                                         for name, ids in scope['class_ids'].items()}
        return self.model.predict(images, head_selection, output_mode=output_mode, thresholds=thresholds, **options)

    def predict_image(self, image, head_selection=None, **options):
        if not isinstance(image, Image.Image):
            raise TypeError('predict_image requires an opened PIL image')
        prepared = self.preprocessor(image)
        result = self.predict(prepared.tensor.unsqueeze(0).to(next(self.model.parameters()).device), head_selection, **options)
        return {'outputs': result, 'bucket': asdict(prepared.bucket)}


def load_predictor(path, *, device='cpu', expected_vocabulary=None):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    document = dict(manifest)
    digest = document.pop('identity', None)
    if document.get('version') != VERSION or digest != identity(document):
        raise ValueError('Invalid deployment manifest identity/version')
    required = {'model.safetensors', 'vocabulary.json'}
    if not required <= set(document['files']) or set(document['files']) - (required | {'source_membership.parquet', 'calibration.json'}):
        raise ValueError('Deployment package has missing or unsupported files')
    for name, descriptor in document['files'].items():
        if descriptor.get('path') != name:
            raise ValueError('Deployment file descriptor must use its exact package-local name')
        verify_artifact(path / name, descriptor)
    vocabulary = Vocabulary.read(path / 'vocabulary.json')
    if vocabulary.sha256 != document['vocab_sha256'] or expected_vocabulary is not None and expected_vocabulary.sha256 != vocabulary.sha256:
        raise ValueError('Deployment requires the exact bound Vocabulary, not same-size replacements')
    layout = HeadLayout(**document['layout'])
    vocabulary.validate_layout(layout)
    preprocessor = _preprocessor(document['preprocessing'])
    if document['other_artist_mode'] not in ('single_label', 'multilabel'):
        raise ValueError('Package must declare its Other Artist training mode')
    state = load_file(str(path / 'model.safetensors'), device='cpu')
    legacy_state = {name.removeprefix('backbone.legacy.'): value for name, value in state.items() if name.startswith('backbone.legacy.')}
    legacy = load_state_strict(legacy_state, model_options(document['base_model_config']))
    legacy.class_mapping = {entry['class_id']: entry['name'] for entry in vocabulary.document['original_artist']}
    legacy.synthetic, legacy.package_config = document['synthetic_base'], document['base_model_config']
    model = MultiTaskModel(KaloscopeBackbone(legacy), layout, other_artist_mode=document['other_artist_mode'])
    model.load_state_dict(state, strict=True, assign=True)
    if state_digest(model) != document['model_sha256']:
        raise ValueError('Deployment model state digest mismatch')
    model.bind_vocabulary(vocabulary).set_stage(document['stage']).set_untrained_labels(document['untrained_labels'])
    counts = {domain: len(entries) for domain, entries in vocabulary.class_id_mapping.items()}
    settings = normalize_thresholds(document['thresholds'], counts)
    if 'calibration.json' in document['files']:
        calibration = json.loads((path / 'calibration.json').read_text())
        calibrated = _calibrated_thresholds(calibration, vocabulary=vocabulary, checkpoint_id=document['checkpoint_id'],
                         model_sha256=document['model_sha256'], preprocessing=document['preprocessing'], counts=counts)
        if (calibrated != settings or document['threshold_status'] != 'FROZEN_VALIDATION_CALIBRATION'
                or document.get('calibration_scope') != {'class_ids': calibration['class_ids'], 'sources': calibration['sources']}):
            raise ValueError('Package thresholds disagree with its frozen calibration')
    elif (document['threshold_status'] != ('EXPLICIT_UNCALIBRATED' if settings else 'NOT_CONFIGURED')
          or document.get('calibration_scope') is not None):
        raise ValueError('Package threshold calibration status is inconsistent')
    if 'source_membership.parquet' in document['files']:
        model.set_source_membership(SourceMembership.read(path / 'source_membership.parquet', vocabulary))
    return Predictor(model.to(device).eval(), vocabulary, preprocessor, manifest)
