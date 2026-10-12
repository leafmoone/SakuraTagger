"""Explicit SakuraTool immutable-release-v1 → Tagger sparse Release adapter.

Only manifest-listed indices and shards are selected. No directory discovery,
image download, implicit negatives, global-ID rebasing, or training occurs here.
"""
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path

from .contract import VOCAB_VERSION, Vocabulary
from .modelscope import read_metadata, dataset_file_url
from .release import (DatasetRelease, RELEASE_VERSION, MULTILABEL_DOMAINS,
                      artifact_descriptor, reference_index_schema, relative_path,
                      validate_sparse_row)

PRODUCER_FORMAT = 'sakuratool-immutable-release-v1'
DOMAINS = ('general', 'character', 'copyright', 'other_artist', 'artist_existing')


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def convert_row(raw, vocabulary):
    """Retain the original row losslessly alongside normalized training facts."""
    positive = _json(raw['frozen_raw_positive_ids'])
    scope = _json(raw['supervision_scope'])
    provenance = _json(raw['source_provenance'])
    if set(positive) != set(DOMAINS):
        raise ValueError('Producer positives must use the five independent ID spaces')
    artist = scope['artist']['other_artist']
    if artist.get('task') != 'SINGLE_LABEL' or artist.get('unknown_is_not_negative') is not True:
        raise ValueError('Unsupported producer artist supervision')
    confirmed = artist.get('status') == 'KNOWN' and artist.get('reason') == 'KNOWN_SINGLE_ARTIST'
    if confirmed and (positive['artist_existing'] or positive['other_artist'] != [raw['other_artist_id']]):
        raise ValueError('Producer single-artist scalar disagrees with frozen evidence')
    if raw['original_artist_ids'] != positive['artist_existing']:
        raise ValueError('Original Artist IDs disagree with frozen evidence')
    row = {key: raw[key] for key in ('sample_id', 'image_sha256', 'split', 'shard_id', 'family_id')}
    row.update(status='published', file_path=raw['shard_path'],
               image_member=raw['image_member_name'], json_member=raw['json_member_name'],
               image_offset_data=raw['image_offset'], image_size=raw['image_size'],
               json_offset_data=raw['json_offset'], json_size=raw['json_size'],
               raw_width=raw['width'], raw_height=raw['height'],
               effective_width=None, effective_height=None, dimension_status='unverified',
               near_duplicate_status='UNKNOWN', original_artist_ids=positive['artist_existing'],
               other_artist_ids=positive['other_artist'], single_artist_confirmed=confirmed,
               artist_conflict=artist.get('reason') == 'MULTIPLE_RELIABLE_ARTIST_IDENTITIES',
               artist_evidence=[], source_provenance=list(dict.fromkeys(p['source'] for p in provenance)),
               producer_record=json.dumps(raw, ensure_ascii=False, separators=(',', ':')))
    for domain in MULTILABEL_DOMAINS:
        if scope[domain].get('unlabeled') != 'UNKNOWN_NOT_NEGATIVE':
            raise ValueError('Unsupported producer unknown supervision')
        if raw[domain + '_ids'] != positive[domain]:
            raise ValueError('Producer sparse positives disagree with frozen positives')
        row[domain + '_ids'] = positive[domain]
        row[domain + '_negative_ids'] = scope[domain]['confirmed_negative']
        row[domain + '_forbidden_ids'] = []
    return row


def validate_sample(metadata, row, release):
    """Compare producer TAR JSON against the pinned, losslessly retained index."""
    raw = json.loads(row['producer_record'])
    expected = {key: raw[key] for key in ('sample_id', 'image_sha256', 'split', 'width', 'height',
                                          'family_id', 'other_artist_id', 'original_artist_id', 'original_artist_ids')}
    positive = _json(raw['frozen_raw_positive_ids'])
    # Producer positive_ids masks conflicted Other Artist labels for CE; its
    # frozen_raw_positive_ids retains every observed identity for auditing.
    positive = {**positive, 'other_artist': [] if raw['other_artist_id'] is None else [raw['other_artist_id']]}
    expected.update(format='sakuratool-tar-sample-v1',
                    frozen_raw_positive_ids=_json(raw['frozen_raw_positive_ids']),
                    positive_ids=positive,
                    supervision_scope=_json(raw['supervision_scope']),
                    source_provenance=_json(raw['source_provenance']))
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('SakuraTool TAR JSON and immutable producer index disagree')


def prepare_release(repository, revision, manifest_path, output, *, read_file=None):
    """Materialize only small pinned metadata; publish local release.json last.

    read_file(path) is an optional offline snapshot reader. All listed source
    hashes are checked once on ingest. No extra rehash is done at train steps.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    dataset_file_url(repository, revision, manifest_path)  # Validate identity even offline.
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory; existing release data is not replaced')
    output.mkdir(parents=True, exist_ok=True)
    read_file = read_file or (lambda path: read_metadata(repository, revision, path))
    manifest_bytes = read_file(relative_path(manifest_path))
    manifest = json.loads(manifest_bytes)
    if manifest.get('format') != PRODUCER_FORMAT or manifest.get('repository') != repository:
        raise ValueError('Unsupported producer release or repository mismatch')
    if manifest.get('planned_rows_trainable') is not False:
        raise ValueError('Planned rows must not be trainable')

    def read_verified(path, sha, size=None):
        data = read_file(relative_path(path))
        if (size is not None and len(data) != size) or hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f'Producer metadata size/SHA mismatch: {path}')
        return data

    domains = {}
    csv_hashes = {}
    for domain in DOMAINS:
        descriptor = manifest['vocabulary'][domain]
        data = read_verified(descriptor['path'], descriptor['sha256'])
        entries = [{'class_id': int(row['class_id']), 'name': row['class_name']}
                   for row in csv.DictReader(StringIO(data.decode('utf-8-sig')))]
        if len(entries) != descriptor['classes']:
            raise ValueError('Producer vocabulary class count mismatch')
        domains[domain] = entries
        csv_hashes['artist' if domain == 'artist_existing' else domain] = descriptor['sha256']
    original = domains.pop('artist_existing')
    vocabulary = Vocabulary({'version': VOCAB_VERSION, 'domains': domains, 'original_artist': original})
    document = {'schema_version': RELEASE_VERSION, 'status': 'published', 'dataset_id': repository,
                'release_id': manifest['dataset_version'], 'vocab_sha256': vocabulary.sha256,
                'vocabulary_path': 'vocabulary.json', 'vocabulary_csv_sha256': csv_hashes,
                'source': {'provider': 'modelscope', 'repository': repository, 'revision': revision},
                'producer_format': PRODUCER_FORMAT, 'producer_manifest_path': manifest_path,
                'producer_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                'producer_vocabulary_hash': manifest['vocab_hash'],
                'supervision_assumptions': {'single_artist_confirmed': False},
                'shards': [], 'indices': []}
    for shard in manifest['shard_list']:
        if shard.get('state') != 'REMOTE_VERIFIED':
            raise ValueError('Producer shard has not been published and verified')
        document['shards'].append({'shard_id': shard['shard_id'], 'path': shard['relative_path'],
                                  'size_bytes': shard['actual_bytes'], 'sha256': shard['tar_sha256'],
                                  'status': 'published'})
    # A temporary descriptor inventory lets the existing validator check rows
    # before writing converted Parquet. Replace these with local hashes below.
    for index in manifest['index_list']:
        document['indices'].append({'path': index['path'], 'size_bytes': index['bytes'],
                                    'sha256': index['sha256'], 'status': 'published'})
    release = DatasetRelease(document, vocabulary, root=output)
    schema = reference_index_schema(vocabulary, document['release_id'], producer_format=PRODUCER_FORMAT)
    shard_splits = {item['shard_id']: item['split'] for item in manifest['shard_list']}
    shard_counts = {item['shard_id']: 0 for item in manifest['shard_list']}
    converted, total, split_counts = [], 0, {}
    for descriptor in manifest['index_list']:
        data = read_verified(descriptor['path'], descriptor['sha256'], descriptor['bytes'])
        parquet = pq.ParquetFile(BytesIO(data))
        metadata = parquet.schema_arrow.metadata or {}
        if metadata.get(b'format') != b'sakuratool-sample-index-v1' or metadata.get(b'offset_semantics') != b'absolute TAR data offset; not header':
            raise ValueError('Unsupported producer index format/offset semantics')
        path = release.local_path(descriptor['path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with pq.ParquetWriter(path, schema) as writer:
            for batch in parquet.iter_batches(batch_size=256):
                rows = []
                for raw in batch.to_pylist():
                    shard = release.shards.get(raw['shard_id'])
                    if shard is None or raw['tar_sha256'] != shard['sha256'] or raw['tar_size'] != shard['size_bytes']:
                        raise ValueError('Producer index is outside the selected shard inventory')
                    if raw['split'] != shard_splits[raw['shard_id']]:
                        raise ValueError('Producer sample split disagrees with its shard')
                    shard_counts[raw['shard_id']] += 1
                    row = validate_sparse_row(convert_row(raw, vocabulary), release)
                    rows.append(row)
                    split_counts[row['split']] = split_counts.get(row['split'], 0) + 1
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                count += len(rows)
        if count != descriptor['rows']:
            raise ValueError('Producer index row count mismatch')
        total += count
        converted.append({**artifact_descriptor(path, output), 'status': 'published'})
    if total != manifest['sample_counts'] or split_counts != manifest['split_counts']:
        raise ValueError('Producer release sample/split counts mismatch')
    if any(shard_counts[item['shard_id']] != item['sample_count'] for item in manifest['shard_list']):
        raise ValueError('Producer shard sample count mismatch')
    document['indices'] = converted
    document['producer_sample_counts'] = split_counts
    (output / 'vocabulary.json').write_text(json.dumps(vocabulary.document, ensure_ascii=False), encoding='utf-8')
    (output / 'producer_manifest.json').write_bytes(manifest_bytes)
    (output / 'release.json').write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
    return DatasetRelease.read(output / 'release.json')
