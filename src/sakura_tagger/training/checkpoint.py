"""Strict V1 training checkpoint contract, separate from original Kaloscope files."""
from dataclasses import asdict
import os
import re
from pathlib import Path
import tempfile

import torch

from ..data.contract import SCHEMA_VERSION, validate_metadata

CHECKPOINT_VERSION = 'sakura-tagger-checkpoint-v1'


def checkpoint_metadata(model, vocabulary, dataset_metadata, bucket_config, base_kaloscope_version):
    validate_metadata(dataset_metadata, vocabulary)
    vocabulary.validate_layout(model.layout)
    legacy = model.backbone.legacy
    expected = {entry['class_id']: entry['name'] for entry in vocabulary.document['original_artist']}
    if expected != getattr(legacy, 'class_mapping', None) or len(expected) != model.backbone.artist_classes:
        raise ValueError('Original Artist vocabulary does not match the loaded Kaloscope class mapping')
    if not isinstance(base_kaloscope_version, str) or not base_kaloscope_version:
        raise ValueError('Explicit base Kaloscope version is required')
    manifest_sha = dataset_metadata.get('manifest_sha256')
    if not isinstance(manifest_sha, str) or not re.fullmatch('[0-9a-f]{64}', manifest_sha):
        raise ValueError('Checkpoint requires an immutable ordered manifest_sha256')
    return {'base_kaloscope_version': base_kaloscope_version, 'dataset_schema_version': SCHEMA_VERSION,
            'dataset_id': dataset_metadata['dataset_id'], 'manifest_sha256': manifest_sha,
            'vocab_sha256': vocabulary.sha256, 'head_layout': model.layout.to_dict(),
            'bucket_config': asdict(bucket_config), 'stage_config': {'stage': model.backbone.stage},
            'class_id_mapping': vocabulary.class_id_mapping}


def save_payload(path, payload):
    path = Path(path)
    if path.suffix not in ('.pt', '.ckpt'):
        raise ValueError('Training checkpoints must use .pt or .ckpt, never an original model.safetensors')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_payload(path, expected_metadata):
    # weights_only rejects arbitrary Python object execution in untrusted files.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('version') != CHECKPOINT_VERSION:
        raise ValueError('Unsupported training checkpoint version')
    if payload.get('metadata') != expected_metadata:
        raise ValueError('Checkpoint metadata mismatch (dataset/vocabulary/layout/bucket/stage/base)')
    return payload
