"""SakuraPool's read-only, versioned JSON vocabulary + JSONL manifest contract."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import torch
from torch.utils.data import Dataset
from PIL import Image

from ..model.registry import HeadLayout, LOGICAL_DOMAINS
from .preprocessing import NativePreprocessor

SCHEMA_VERSION = 'sakura-tagger-dataset-v1'
VOCAB_VERSION = 'sakura-tagger-vocabulary-v1'


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def _ids(values, count, field):
    if not isinstance(values, list) or any(type(i) is not int or not 0 <= i < count for i in values) or len(set(values)) != len(values):
        raise ValueError(f'{field}: expected unique in-range integer IDs')
    return values


@dataclass(frozen=True)
class Vocabulary:
    document: dict

    def __post_init__(self):
        if self.document.get('version') != VOCAB_VERSION:
            raise ValueError('Unsupported vocabulary version')
        domains = self.document.get('domains', {})
        if set(domains) != set(LOGICAL_DOMAINS):
            raise ValueError('Vocabulary requires exactly four logical domains')
        for name, entries in (*domains.items(), ('artist', self.document.get('original_artist', []))):
            if not isinstance(entries, list) or not entries:
                raise ValueError(f'{name}: vocabulary cannot be empty')
            if any(type(entry.get('class_id')) is not int for entry in entries) or [entry.get('class_id') for entry in entries] != list(range(len(entries))):
                raise ValueError(f'{name}: class_id must be contiguous and already ordered; Tagger never reorders')
            if any(not isinstance(entry.get('name'), str) or not entry['name'] for entry in entries):
                raise ValueError(f'{name}: class names must be nonempty strings')
            if len({entry['name'] for entry in entries}) != len(entries):
                raise ValueError(f'{name}: duplicate names within a domain')
        # Duplicate names ACROSS domains are deliberately independent IDs.

    @classmethod
    def read(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    @property
    def sha256(self):
        return _sha(self.document)

    @property
    def class_id_mapping(self):
        return {**{name: entries for name, entries in self.document['domains'].items()},
                'artist': self.document['original_artist']}

    def layout(self, **options):
        return HeadLayout(**{name: len(entries) for name, entries in self.document['domains'].items()}, **options)

    def validate_layout(self, layout):
        if any(getattr(layout, name) != len(self.document['domains'][name]) for name in LOGICAL_DOMAINS):
            raise ValueError('Head layout class counts disagree with vocabulary')


def validate_metadata(metadata, vocabulary):
    if metadata.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Dataset schema_version must be sakura-tagger-dataset-v1')
    if metadata.get('vocab_sha256') != vocabulary.sha256:
        raise ValueError('Dataset vocabulary SHA mismatch')
    if not isinstance(metadata.get('dataset_id'), str) or not metadata['dataset_id']:
        raise ValueError('dataset_id is required')
    if metadata.get('provider') != 'SakuraPool':
        raise ValueError('Dataset provider must be SakuraPool')


def validate_record(record, vocabulary):
    required = ('image_id', 'image_path', 'width', 'height', 'source', 'RID', 'image_sha256',
                'split', 'family_id', 'positive_labels', 'supervision_scope', 'unknown_mask')
    missing = set(required) - record.keys()
    if missing:
        raise ValueError(f'Manifest missing fields: {sorted(missing)}')
    for name in ('image_id', 'image_path', 'source', 'family_id'):
        if not isinstance(record[name], str) or not record[name]:
            raise ValueError(f'{name} must be a nonempty string (family_id may be UNKNOWN)')
    if not isinstance(record['RID'], (str, int)) or isinstance(record['RID'], bool):
        raise ValueError('RID must be a string or integer')
    if any(type(record[name]) is not int or record[name] < 1 for name in ('width', 'height')):
        raise ValueError('Original width/height must be positive integers after EXIF correction')
    if record['split'] not in ('train', 'val', 'test'):
        raise ValueError('Invalid train/val/test split')
    if not isinstance(record['image_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', record['image_sha256']):
        raise ValueError('image_sha256 must be lowercase SHA256')
    for name in ('positive_labels', 'supervision_scope', 'unknown_mask'):
        if not isinstance(record[name], dict) or set(record[name]) - set(LOGICAL_DOMAINS):
            raise ValueError(f'{name} must be a dictionary over the four logical domains')
    for domain in LOGICAL_DOMAINS:
        count = len(vocabulary.document['domains'][domain])
        positive = _ids(record['positive_labels'].get(domain, []), count, domain)
        unknown = _ids(record['unknown_mask'].get(domain, []), count, domain)
        scope = record['supervision_scope'].get(domain, 'unknown')
        if isinstance(scope, list):
            known = _ids(scope, count, domain)
            if not set(positive) <= set(known):
                raise ValueError(f'{domain}: positives must be inside explicit supervision_scope')
        elif scope not in ('all', 'positive_only', 'unknown'):
            raise ValueError(f'{domain}: invalid supervision_scope')
        if (scope == 'all' or isinstance(scope, list) and scope) and domain not in record['positive_labels']:
            raise ValueError(f'{domain}: fully supervised negatives require an explicit positive_labels list')
        if scope == 'unknown' and positive:
            raise ValueError(f'{domain}: positive labels require explicit supervision scope')
        if set(positive) & set(unknown):
            raise ValueError(f'{domain}: positive/unknown labels conflict')
    return record


def encode_supervision(record, vocabulary):
    """Missing domains become all-unknown, never implicit negatives."""
    validate_record(record, vocabulary)
    targets, masks = {}, {}
    for domain in LOGICAL_DOMAINS:
        count = len(vocabulary.document['domains'][domain])
        target = torch.zeros(count)
        mask = torch.zeros(count, dtype=torch.bool)
        positive = record['positive_labels'].get(domain, [])
        scope = record['supervision_scope'].get(domain, 'unknown')
        target[positive] = 1
        if scope == 'all':
            mask[:] = True
        elif scope == 'positive_only':
            mask[positive] = True
        elif isinstance(scope, list):
            mask[scope] = True
        mask[record['unknown_mask'].get(domain, [])] = False
        targets[domain], masks[domain] = target, mask
    return targets, masks


def read_manifest(path, metadata, vocabulary):
    validate_metadata(metadata, vocabulary)
    records, seen_ids, seen_hashes = [], set(), set()
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = validate_record(json.loads(line), vocabulary)
                if record['image_id'] in seen_ids or record['image_sha256'] in seen_hashes:
                    raise ValueError('Manifest must be globally deduplicated by image_id and image_sha256')
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f'Manifest line {line_number}: {error}') from error
            seen_ids.add(record['image_id'])
            seen_hashes.add(record['image_sha256'])
            records.append(record)
    return records


def manifest_fingerprint(records):
    """One metadata-only pass binds record contents and order, never image bytes."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode())
        digest.update(b'\n')
    return digest.hexdigest()


class ManifestDataset(Dataset):
    """Read existing images only; never crawl, download, or reorder labels."""
    def __init__(self, manifest_path, metadata_path, vocabulary_path, *, root=None,
                 split='train', preprocessor=None):
        self.vocabulary = Vocabulary.read(vocabulary_path)
        self.metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
        records = read_manifest(manifest_path, self.metadata, self.vocabulary)
        fingerprint = manifest_fingerprint(records)
        if 'manifest_sha256' in self.metadata and self.metadata['manifest_sha256'] != fingerprint:
            raise ValueError('Dataset manifest SHA mismatch')
        self.metadata = {**self.metadata, 'manifest_sha256': fingerprint}
        if split not in ('train', 'val', 'test'):
            raise ValueError('Invalid dataset split')
        self.records = [record for record in records if record['split'] == split]
        self.root = Path(root) if root is not None else Path(manifest_path).parent
        self.preprocessor = preprocessor or NativePreprocessor()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record['image_path']) as image:
            prepared = self.preprocessor(image)
        if prepared.bucket.original_size != (record['width'], record['height']):
            raise ValueError(f"{record['image_id']}: manifest and post-EXIF image dimensions disagree")
        targets, masks = encode_supervision(record, self.vocabulary)
        return {'images': prepared.tensor, 'targets': targets, 'masks': masks,
                'image_id': record['image_id']}
