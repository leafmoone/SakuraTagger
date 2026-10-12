"""Explicit reference Release contract; not a claim of producer handoff compatibility.

Frozen inventories are authoritative. No discovery of live production directories,
image requests, or training occurs here. The old manifest API remains separate.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import tempfile
import threading
from types import MappingProxyType

import numpy as np
import torch

from .contract import Vocabulary, _ids, _sha
from ..model.registry import LOGICAL_DOMAINS

RELEASE_VERSION = 'sakura-tagger-reference-release-v1'
INDEX_VERSION = 'sakura-tagger-reference-index-v1'
LOCAL_INDEX_VERSION = 'sakura-tagger-local-index-v1'
MAX_SHARD_BYTES = 5_368_709_120  # Exactly 5 GiB, including TAR headers/trailer.
HANDOFF_STATUS = 'REAL_PRODUCER_HANDOFF_NOT_VERIFIED'
MULTILABEL_DOMAINS = ('general', 'character', 'copyright')
SPLITS = ('train', 'val', 'test', 'calibration')


def _digest(value, field):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError(f'{field} must be a lowercase SHA256')
    return value


def relative_path(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Expected a nonempty portable relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or ':' in value or '.partial' in value or value != str(path):
        raise ValueError(f'Unsafe or unpublished artifact path: {value!r}')
    return value


def artifact_descriptor(path, root):
    """Create a size/SHA descriptor at a fixture or publication boundary only."""
    path, root = Path(path), Path(root)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': path.relative_to(root).as_posix(), 'size_bytes': path.stat().st_size,
            'sha256': digest.hexdigest()}


def verify_artifact(path, descriptor):
    """Validate once at index-build/download boundaries, never each train step."""
    path = Path(path)
    if path.stat().st_size != descriptor['size_bytes']:
        raise ValueError(f'Artifact size mismatch: {descriptor["path"]}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != descriptor['sha256']:
        raise ValueError(f'Artifact SHA mismatch: {descriptor["path"]}')


class DatasetRelease:
    """Snapshot of an explicitly published reference Release and its Vocabulary."""
    def __init__(self, document, vocabulary, *, root='.'):
        document = json.loads(json.dumps(document))
        if document.get('schema_version') != RELEASE_VERSION:
            raise ValueError('Unsupported Release schema; a real producer needs an explicit adapter')
        if document.get('status') != 'published':
            raise ValueError('Only a completely published immutable Release can be read')
        for key in ('dataset_id', 'release_id'):
            if not isinstance(document.get(key), str) or not document[key]:
                raise ValueError(f'Release requires {key}')
        if document.get('vocab_sha256') != vocabulary.sha256:
            raise ValueError('Release canonical Vocabulary SHA mismatch')
        relative_path(document['vocabulary_path'])
        csv_shas = document.get('vocabulary_csv_sha256', {})
        if not isinstance(csv_shas, dict) or set(csv_shas) - set(vocabulary.class_id_mapping):
            raise ValueError('CSV file SHAs must use the five independent vocabulary domains')
        for domain, value in csv_shas.items():
            _digest(value, f'{domain} CSV file SHA')
        original_names = {entry['name'] for entry in vocabulary.document['original_artist']}
        if original_names & {entry['name'] for entry in vocabulary.document['domains']['other_artist']}:
            raise ValueError('Other Artist names must exclude exact Original Artist names')
        assumptions = document.get('supervision_assumptions', {})
        if type(assumptions.get('single_artist_confirmed', False)) is not bool:
            raise ValueError('single_artist_confirmed must be an explicit boolean assumption')
        self.single_artist_confirmed = assumptions.get('single_artist_confirmed', False)
        source = document.get('source', {'provider': 'local'})
        if source.get('provider') not in ('local', 'modelscope', 'huggingface', 'loopback'):
            raise ValueError('Unsupported shard provider')
        if source['provider'] != 'local':
            for key in ('repository', 'revision'):
                if not isinstance(source.get(key), str) or not source[key] or source[key] in ('main', 'master', 'latest'):
                    raise ValueError('Remote sources require explicit fixed repository/revision identity')
        inventories = {}
        for group, suffix in (('shards', '.tar'), ('indices', '.parquet')):
            values = document.get(group)
            if not isinstance(values, list) or not values:
                raise ValueError(f'Release requires a nonempty published {group} inventory')
            paths, ids = set(), set()
            for item in values:
                path = relative_path(item['path'])
                if not path.endswith(suffix) or path in paths:
                    raise ValueError(f'Invalid or duplicate {group} path')
                paths.add(path)
                size = item.get('size_bytes')
                if type(size) is not int or size < 1 or group == 'shards' and size > MAX_SHARD_BYTES:
                    raise ValueError(f'Invalid {group} size; TAR limit is 5 GiB including archive overhead')
                _digest(item.get('sha256'), f'{group} SHA')
                if item.get('status') != 'published':
                    raise ValueError(f'Unpublished {group} inventory entry')
                if group == 'shards':
                    shard_id = item.get('shard_id')
                    if not isinstance(shard_id, str) or not shard_id or shard_id in ids:
                        raise ValueError('Shard IDs must be nonempty and unique')
                    ids.add(shard_id)
            inventories[group] = tuple(MappingProxyType(item) for item in values)
        self._document = document
        self.root = Path(root).resolve()
        self.vocabulary = Vocabulary(json.loads(json.dumps(vocabulary.document)))
        self.sha256 = _sha(document)
        self.shards = MappingProxyType({item['shard_id']: item for item in inventories['shards']})
        self.indices = inventories['indices']
        self.source = MappingProxyType(source)

    @classmethod
    def read(cls, path):
        path = Path(path)
        document = json.loads(path.read_text(encoding='utf-8'))
        vocabulary = Vocabulary.read(path.parent / relative_path(document['vocabulary_path']))
        return cls(document, vocabulary, root=path.parent)

    @property
    def document(self):
        return json.loads(json.dumps(self._document))

    def __getstate__(self):
        return {'document': self.document, 'vocabulary': self.vocabulary.document, 'root': str(self.root)}

    def __setstate__(self, state):
        self.__init__(state['document'], Vocabulary(state['vocabulary']), root=state['root'])

    @property
    def identity(self):
        return {'release_schema': RELEASE_VERSION, 'dataset_id': self._document['dataset_id'],
                'release_id': self._document['release_id'], 'release_sha256': self.sha256,
                'vocab_sha256': self.vocabulary.sha256}

    def local_path(self, path):
        result = (self.root / relative_path(path)).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError('Release artifact resolves outside its declared root')
        return result

    def source_membership(self):
        from .source_membership import SourceMembership
        path = self._document.get('source_membership_path')
        return None if path is None else SourceMembership.read(self.local_path(path), self.vocabulary)


def reference_index_schema(vocabulary, release_id):
    """Physical sparse Parquet schema. uint64 values are TAR data offsets."""
    import pyarrow as pa
    fields = [(name, pa.string()) for name in (
        'sample_id', 'image_sha256', 'split', 'status', 'shard_id', 'file_path',
        'image_member', 'json_member', 'dimension_status', 'family_id', 'near_duplicate_status')]
    fields += [(name, pa.uint64()) for name in
               ('image_offset_data', 'image_size', 'json_offset_data', 'json_size')]
    fields += [(name, pa.uint32()) for name in
               ('raw_width', 'raw_height', 'effective_width', 'effective_height')]
    fields += [(domain + '_ids', pa.list_(pa.int32())) for domain in (*LOGICAL_DOMAINS, 'original_artist')]
    fields += [(domain + '_' + suffix, pa.list_(pa.int32()))
               for domain in MULTILABEL_DOMAINS for suffix in ('negative_ids', 'forbidden_ids')]
    fields += [('source_provenance', pa.list_(pa.string())), ('single_artist_confirmed', pa.bool_()),
               ('artist_evidence', pa.list_(pa.struct([
                   ('source', pa.string()), ('domain', pa.string()), ('class_id', pa.int32()), ('label', pa.string())])))]
    metadata = {b'schema_version': INDEX_VERSION.encode(), b'vocab_sha256': vocabulary.sha256.encode(),
                b'release_id': release_id.encode()}
    return pa.schema(fields, metadata=metadata)


@dataclass(frozen=True)
class ArtistTarget:
    target: int
    valid: bool
    original_ids: tuple
    conflict: bool
    reason: str


def single_artist_target(record, vocabulary, *, release_confirmed=False):
    """Convert facts to trusted CE; a lone observed positive alone is insufficient."""
    if type(release_confirmed) is not bool:
        raise ValueError('Release confirmation must be an explicit boolean')
    other = set(_ids(record.get('other_artist_ids') or [], len(vocabulary.document['domains']['other_artist']), 'other_artist'))
    original = set(_ids(record.get('original_artist_ids') or [], len(vocabulary.document['original_artist']), 'original_artist'))
    claims = {('other_artist', item) for item in other} | {('artist', item) for item in original}
    for evidence in record.get('artist_evidence') or []:
        domain, class_id = evidence.get('domain'), evidence.get('class_id')
        if not isinstance(evidence.get('source'), str) or not evidence['source']:
            raise ValueError('Artist evidence requires its actual annotation source')
        if domain in ('artist', 'other_artist'):
            count = len(vocabulary.class_id_mapping[domain])
            _ids([class_id], count, 'artist evidence')
            claims.add((domain, class_id))
            (original if domain == 'artist' else other).add(class_id)
        elif domain == 'out_of_vocabulary' and isinstance(evidence.get('label'), str) and evidence['label']:
            if class_id is not None:
                raise ValueError('Out-of-vocabulary evidence cannot have an in-vocabulary class ID')
            claims.add((domain, evidence['label']))
        else:
            raise ValueError('Invalid artist evidence domain or unknown label')
    sample_confirmed = record.get('single_artist_confirmed')
    if sample_confirmed is not None and type(sample_confirmed) is not bool:
        raise ValueError('Sample confirmation must be a boolean or None')
    # None inherits the Release assumption; an explicit sample-level False
    # overrides it, so contradictory evidence cannot silently enter CE/support.
    confirmed = release_confirmed if sample_confirmed is None else sample_confirmed
    if len(claims) > 1:
        return ArtistTarget(-100, False, tuple(sorted(original)), True, 'conflict')
    if not claims:
        return ArtistTarget(-100, False, (), False, 'missing')
    domain, target = next(iter(claims))
    if domain == 'artist':
        return ArtistTarget(-100, False, tuple(sorted(original)), False, 'original_only')
    if domain == 'out_of_vocabulary':
        return ArtistTarget(-100, False, (), False, 'out_of_vocabulary')
    if not confirmed:
        return ArtistTarget(-100, False, (), False, 'unconfirmed')
    return ArtistTarget(target, True, (), False, 'other_artist_valid')


def collate_artist_targets(records, release):
    converted = [single_artist_target(row, release.vocabulary, release_confirmed=release.single_artist_confirmed)
                 for row in records]
    return {'targets': torch.tensor([item.target for item in converted], dtype=torch.int64),
            'valid': torch.tensor([item.valid for item in converted], dtype=torch.bool),
            'original_ids': [item.original_ids for item in converted],
            'status': [item.reason for item in converted]}


def validate_sparse_row(row, release):
    """Normalize one published row; return None for an explicit planned-only row."""
    if row.get('status') == 'planned' or row.get('file_path') is None:
        return None
    if row.get('status') != 'published':
        raise ValueError('Index row must explicitly be published or planned')
    row = dict(row)
    if not isinstance(row.get('sample_id'), str) or not row['sample_id']:
        raise ValueError('Published sample requires sample_id')
    _digest(row.get('image_sha256'), 'image_sha256')
    if row.get('split') not in SPLITS:
        raise ValueError('Unknown sample split')
    if row.get('shard_id') not in release.shards:
        raise ValueError('Sample refers to an undeclared/unpublished shard')
    shard = release.shards[row['shard_id']]
    if relative_path(row['file_path']) != shard['path']:
        raise ValueError('Reference file_path must equal its published shard path')
    for prefix in ('image', 'json'):
        relative_path(row[prefix + '_member'])
        offset, size = row[prefix + '_offset_data'], row[prefix + '_size']
        if type(offset) is not int or type(size) is not int or offset < 512 or offset % 512 or size < 1 or offset + size > shard['size_bytes']:
            raise ValueError('Invalid uint64 member data extent (not a TAR header offset)')
    if row['image_member'] == row['json_member'] or not row['json_member'].endswith('.json'):
        raise ValueError('Each sample requires distinct image and JSON members')
    if row.get('dimension_status') not in ('unverified', 'confirmed', 'conflict_rebucketed'):
        raise ValueError('Dimension status must distinguish raw from actual post-EXIF dimensions')
    for name in ('raw_width', 'raw_height', 'effective_width', 'effective_height'):
        if row.get(name) is not None and (type(row[name]) is not int or row[name] < 1):
            raise ValueError('Raw/effective dimensions must be positive or unknown')
    for prefix in ('raw', 'effective'):
        if (row.get(prefix + '_width') is None) != (row.get(prefix + '_height') is None):
            raise ValueError('Each dimension pair must be known or unknown together')
    if row['dimension_status'] != 'unverified' and row.get('effective_width') is None:
        raise ValueError('Confirmed/rebucketed dimensions need actual post-EXIF width and height')
    for domain in (*LOGICAL_DOMAINS, 'original_artist'):
        entries = (release.vocabulary.document['original_artist'] if domain == 'original_artist'
                   else release.vocabulary.document['domains'][domain])
        row[domain + '_ids'] = _ids(row.get(domain + '_ids') or [], len(entries), domain)
        if domain in MULTILABEL_DOMAINS:
            for suffix in ('negative_ids', 'forbidden_ids'):
                row[domain + '_' + suffix] = _ids(row.get(domain + '_' + suffix) or [], len(entries), domain)
            if set(row[domain + '_ids']) & set(row[domain + '_negative_ids']):
                raise ValueError('A label cannot be both positive and explicitly negative')
    for name in ('source_provenance', 'artist_evidence'):
        row[name] = row.get(name) or []
        if not isinstance(row[name], list):
            raise ValueError(f'{name} must be a list')
    if any(not isinstance(value, str) or not value for value in row['source_provenance']):
        raise ValueError('Image provenance must contain actual nonempty source identifiers')
    for name in ('family_id', 'near_duplicate_status'):
        if not isinstance(row.get(name), str) or not row[name]:
            raise ValueError(f'{name} must be explicit (UNKNOWN is allowed)')
    if row.get('single_artist_confirmed') is not None and type(row['single_artist_confirmed']) is not bool:
        raise ValueError('Sample single-artist evidence must be boolean')
    single_artist_target(row, release.vocabulary, release_confirmed=release.single_artist_confirmed)
    return row


def build_local_index(release, output, *, batch_size=4096):
    """One bounded Parquet pass into an atomic disk-backed sparse query index.

    Raw records never become a dataset-sized Python list or dense label matrix.
    Train support counts are based on unique published train samples before RFS.
    """
    import pyarrow.parquet as pq
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    output = Path(output)
    if output.exists():
        return SparseReleaseIndex(output, release=release)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=output.name + '.partial.', dir=output.parent)
    os.close(handle)
    counts = {domain: Counter() for domain in LOGICAL_DOMAINS}
    splits, artists, dimensions = Counter(), Counter(), Counter()
    skipped, count = 0, 0
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript('''
            CREATE TABLE samples (ordinal INTEGER PRIMARY KEY, sample_id TEXT UNIQUE NOT NULL,
                image_sha256 TEXT UNIQUE NOT NULL, split TEXT NOT NULL, shard_id TEXT NOT NULL,
                width INTEGER, height INTEGER, record TEXT NOT NULL);
            CREATE INDEX samples_split ON samples(split, ordinal);
            CREATE INDEX samples_shard ON samples(shard_id, ordinal);
            CREATE TABLE members (shard_id TEXT NOT NULL, name TEXT NOT NULL, offset INTEGER NOT NULL,
                size INTEGER NOT NULL, UNIQUE(shard_id, name), UNIQUE(shard_id, offset));
            CREATE TABLE support (domain TEXT NOT NULL, class_id INTEGER NOT NULL, count INTEGER NOT NULL,
                PRIMARY KEY(domain, class_id));
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')
        expected = reference_index_schema(release.vocabulary, release.identity['release_id'])
        for descriptor in release.indices:
            path = release.local_path(descriptor['path'])
            verify_artifact(path, descriptor)
            parquet = pq.ParquetFile(path)
            actual = parquet.schema_arrow
            # Parquet names list children "element" while Arrow may use "item";
            # compare physical types separately from the required file binding.
            if (not actual.equals(expected, check_metadata=False)
                    or any((actual.metadata or {}).get(key) != value for key, value in expected.metadata.items())):
                raise ValueError('Parquet schema or frozen Vocabulary/Release binding mismatch')
            for batch in parquet.iter_batches(batch_size=batch_size):
                for raw in batch.to_pylist():
                    row = validate_sparse_row(raw, release)
                    if row is None:
                        skipped += 1
                        continue
                    artist = single_artist_target(row, release.vocabulary,
                                                  release_confirmed=release.single_artist_confirmed)
                    connection.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?)', (
                        count, row['sample_id'], row['image_sha256'], row['split'], row['shard_id'],
                        row['effective_width'], row['effective_height'], json.dumps(row, separators=(',', ':'))))
                    for prefix in ('image', 'json'):
                        connection.execute('INSERT INTO members VALUES (?,?,?,?)', (
                            row['shard_id'], row[prefix + '_member'], row[prefix + '_offset_data'], row[prefix + '_size']))
                    count += 1
                    splits[row['split']] += 1
                    artists[artist.reason] += 1
                    dimensions[row['dimension_status']] += 1
                    if row['split'] == 'train':
                        for domain in MULTILABEL_DOMAINS:
                            counts[domain].update(row[domain + '_ids'])
                        if artist.valid:
                            counts['other_artist'][artist.target] += 1
        previous_shard, previous_end = None, 0
        for shard, offset, size in connection.execute('SELECT shard_id,offset,size FROM members ORDER BY shard_id,offset'):
            if shard == previous_shard and offset < previous_end:
                raise ValueError('Overlapping indexed TAR member extents')
            previous_shard, previous_end = shard, offset + size
        for domain, values in counts.items():
            connection.executemany('INSERT INTO support VALUES (?,?,?)',
                                   ((domain, label, total) for label, total in values.items()))
        stats = {'samples': count, 'planned_skipped': skipped, 'splits': dict(splits),
                 'artist_conversion': dict(artists), 'dimensions': dict(dimensions),
                 'support_basis': 'unique published train samples; Other Artist trusted CE only',
                 'handoff': HANDOFF_STATUS}
        metadata = {'version': LOCAL_INDEX_VERSION, 'identity': release.identity,
                    'stats': stats, 'complete': True}
        connection.executemany('INSERT INTO metadata VALUES (?,?)',
                               ((key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()))
        connection.commit()
        connection.close()
        os.replace(temporary, output)
    except BaseException:
        connection.close()
        Path(temporary).unlink(missing_ok=True)
        raise
    return SparseReleaseIndex(output, release=release)


class SparseReleaseIndex:
    """Lazy read-only SQLite lookup with independent per-process/thread handles."""
    def __init__(self, path, *, release):
        self.path = Path(path).resolve()
        self.release = release
        self._local = threading.local()
        metadata = dict(self._connection().execute('SELECT key,value FROM metadata'))
        self.metadata = {key: json.loads(value) for key, value in metadata.items()}
        if (self.metadata.get('version') != LOCAL_INDEX_VERSION or self.metadata.get('complete') is not True
                or self.metadata.get('identity') != release.identity):
            raise ValueError('Local index is partial or belongs to a different Release/Vocabulary')

    def _connection(self):
        if getattr(self._local, 'pid', None) != os.getpid():
            self._local.connection = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)
            self._local.pid = os.getpid()
        return self._local.connection

    def __getstate__(self):
        return {'path': self.path, 'release': self.release, 'metadata': self.metadata}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()

    def __len__(self):
        return self.metadata['stats']['samples']

    def __getitem__(self, index):
        row = self._connection().execute('SELECT record FROM samples WHERE ordinal=?', (int(index),)).fetchone()
        if row is None:
            raise IndexError(index)
        return json.loads(row[0])

    def iter_records(self, *, split=None, shard_id=None, batch_size=4096):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        clauses, values = [], []
        for name, value in (('split', split), ('shard_id', shard_id)):
            if value is not None:
                clauses.append(name + '=?')
                values.append(value)
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        cursor = self._connection().execute('SELECT ordinal,record FROM samples' + where + ' ORDER BY ordinal', values)
        while rows := cursor.fetchmany(batch_size):
            for ordinal, record in rows:
                yield ordinal, json.loads(record)

    def positive_support(self, domain):
        if domain not in LOGICAL_DOMAINS:
            raise ValueError('Support belongs to a new logical domain, never Original Artist')
        result = np.zeros(len(self.release.vocabulary.document['domains'][domain]), dtype=np.int64)
        for label, count in self._connection().execute('SELECT class_id,count FROM support WHERE domain=?', (domain,)):
            result[label] = count
        return result

    def close(self):
        if getattr(self._local, 'pid', None) == os.getpid():
            self._local.connection.close()
            self._local.pid = None
