"""Worker-safe TAR reads and full-frame decode with durable derived dimensions."""
from collections import defaultdict
from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3
import tarfile
import threading

from PIL import Image
import torch
from torch.utils.data import Dataset

from .preprocessing import NativePreprocessor
from .release import MAX_SHARD_BYTES, MULTILABEL_DOMAINS, relative_path

SAMPLE_VERSION = 'sakura-tagger-reference-sample-v1'


def pread_exact(fd, size, offset):
    """Short reads are retried; premature EOF never becomes a truncated image."""
    parts, remaining = [], size
    while remaining:
        chunk = os.pread(fd, remaining, offset)
        if not chunk:
            raise EOFError('Short TAR member read')
        parts.append(chunk)
        offset += len(chunk)
        remaining -= len(chunk)
    return b''.join(parts)


class TarReader:
    """No extraction and no shared seek cursor. This reader owns its file handle."""
    def __init__(self, path, *, expected_size=None, max_member_bytes=128 * 1024 * 1024):
        self.path = Path(path)
        if '.partial' in self.path.name:
            raise ValueError('Cannot read unpublished partial TAR files')
        self.size = self.path.stat().st_size
        if not 0 < self.size <= MAX_SHARD_BYTES or expected_size is not None and self.size != expected_size:
            raise ValueError('TAR size exceeds the exact 5 GiB bound or published inventory')
        if type(max_member_bytes) is not int or max_member_bytes < 1:
            raise ValueError('max_member_bytes must be positive')
        self.max_member_bytes = max_member_bytes
        self.fd = os.open(self.path, os.O_RDONLY)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def validate_member(self, name, offset_data, size):
        relative_path(name)
        if any(type(value) is not int for value in (offset_data, size)) or offset_data < 512 or offset_data % 512 or size < 1 or offset_data + size > self.size:
            raise ValueError('Invalid TAR member data range')
        if size > self.max_member_bytes:
            raise ValueError('Member exceeds configured read-memory bound')
        header = pread_exact(self.fd, 512, offset_data - 512)
        try:
            member = tarfile.TarInfo.frombuf(header, encoding='utf-8', errors='surrogateescape')
        except tarfile.HeaderError as error:
            raise ValueError('Invalid TAR member header/checksum') from error
        if not member.isreg() or member.name != name or member.size != size:
            raise ValueError('TAR member type/name/size does not match the published index')
        return member

    def read_member(self, name, offset_data, size):
        self.validate_member(name, offset_data, size)
        return pread_exact(self.fd, size, offset_data)

    def read_sample(self, row, *, include_json=True):
        image = self.read_member(row['image_member'], row['image_offset_data'], row['image_size'])
        metadata = None
        if include_json:
            metadata = json.loads(self.read_member(row['json_member'], row['json_offset_data'], row['json_size']))
        else:
            self.validate_member(row['json_member'], row['json_offset_data'], row['json_size'])
        return image, metadata

    def iter_members(self):
        """Sequential regular-member stream; uncompressed mode is explicit."""
        with tarfile.open(self.path, mode='r:') as archive:
            for member in archive:
                if not member.isreg():
                    raise ValueError('Reference TAR members must be regular files')
                yield member.name, self.read_member(member.name, member.offset_data, member.size)


def reference_sample_json(row, vocabulary, *, vocab_sha256=None):
    names = [domain + '_ids' for domain in (*MULTILABEL_DOMAINS, 'other_artist', 'original_artist')]
    names += [domain + '_' + suffix for domain in MULTILABEL_DOMAINS for suffix in ('negative_ids', 'forbidden_ids')]
    return {'schema_version': SAMPLE_VERSION, 'sample_id': row['sample_id'],
            'vocab_sha256': vocabulary.sha256 if vocab_sha256 is None else vocab_sha256,
            **{name: row.get(name) or [] for name in names},
            'artist_evidence': row.get('artist_evidence') or [], 'source_provenance': row.get('source_provenance') or [],
            'single_artist_confirmed': row.get('single_artist_confirmed')}


def validate_sample_json(metadata, row, release):
    if release.document.get('producer_format') == 'sakuratool-immutable-release-v1':
        from .sakuratool import validate_sample
        return validate_sample(metadata, row, release)
    expected = reference_sample_json(row, release.vocabulary, vocab_sha256=release.identity['vocab_sha256'])
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Reference TAR JSON and frozen index/Vocabulary disagree')


class DimensionCorrections:
    """Local derived metadata bound to a Release; released sample facts stay intact."""
    def __init__(self, path, release):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = release.identity
        self._local = threading.local()
        connection = self._connection()
        with connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('CREATE TABLE IF NOT EXISTS binding (id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS dimensions (sample_id TEXT PRIMARY KEY, document TEXT NOT NULL)')
            binding = json.dumps(self.identity, sort_keys=True)
            connection.execute('INSERT OR IGNORE INTO binding VALUES (1,?)', (binding,))
            existing = connection.execute('SELECT identity FROM binding WHERE id=1').fetchone()
            if existing[0] != binding:
                raise ValueError('Dimension corrections belong to a different Release')

    def _connection(self):
        if getattr(self._local, 'pid', None) != os.getpid():
            self._local.connection = sqlite3.connect(self.path, timeout=30)
            self._local.pid = os.getpid()
        return self._local.connection

    def __getstate__(self):
        return {'path': self.path, 'identity': self.identity}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()

    def get(self, sample_id):
        value = self._connection().execute('SELECT document FROM dimensions WHERE sample_id=?', (sample_id,)).fetchone()
        return None if value is None else json.loads(value[0])

    def put(self, sample_id, document):
        encoded = json.dumps(document, sort_keys=True)
        connection = self._connection()
        with connection:
            connection.execute('INSERT OR IGNORE INTO dimensions VALUES (?,?)', (sample_id, encoded))
            existing = connection.execute('SELECT document FROM dimensions WHERE sample_id=?', (sample_id,)).fetchone()[0]
            if existing != encoded:
                raise ValueError('Actual decoded dimensions changed inside one immutable Release')

    def close(self):
        if getattr(self._local, 'pid', None) == os.getpid():
            self._local.connection.close()
            self._local.pid = None


def decoder_capabilities():
    """Probe actual installed decoders with tiny locally encoded fixtures."""
    Image.init()
    result = {}
    for format in ('JPEG', 'PNG', 'WEBP', 'AVIF'):
        registered = format in Image.OPEN
        status = 'NOT_AVAILABLE'
        if registered:
            try:
                stream = BytesIO()
                Image.new('RGB', (16, 16), (32, 64, 96)).save(stream, format=format)
                with Image.open(BytesIO(stream.getvalue())) as decoded:
                    decoded.load()
                status = 'PASS'
            except (OSError, ValueError) as error:
                status = f'DECODER_REGISTERED_FIXTURE_NOT_VERIFIED: {type(error).__name__}'
        result[format] = {'decoder_registered': registered, 'local_encode_decode': status}
    return result


class TarDataset(Dataset):
    def __init__(self, index, cache, *, corrections, preprocessor=None, json_checks_per_shard=1,
                 max_member_bytes=128 * 1024 * 1024):
        if cache.release.identity != index.release.identity:
            raise ValueError('Dataset index and cache belong to different Releases')
        if corrections.identity != index.release.identity:
            raise ValueError('Dataset dimension corrections belong to another Release')
        if type(json_checks_per_shard) is not int or json_checks_per_shard < 1:
            raise ValueError('At least one lightweight JSON consistency check per shard is required')
        self.index, self.cache, self.corrections = index, cache, corrections
        self.preprocessor = preprocessor or NativePreprocessor()
        self.json_checks_per_shard, self.max_member_bytes = json_checks_per_shard, max_member_bytes
        self._checked = defaultdict(int)

    def __len__(self):
        return len(self.index)

    def _read_image(self, row):
        shard = row['shard_id']
        verify_json = self._checked[shard] < self.json_checks_per_shard
        with self.cache.lease(shard) as path:
            with TarReader(path, expected_size=self.index.release.shards[shard]['size_bytes'],
                           max_member_bytes=self.max_member_bytes) as reader:
                content, metadata = reader.read_sample(row, include_json=verify_json)
        if verify_json:
            validate_sample_json(metadata, row, self.index.release)
            self._checked[shard] += 1
        return content

    def _remember_dimensions(self, row, raw_size, effective, image_format, alpha):
        correction = {'raw_width': raw_size[0], 'raw_height': raw_size[1],
                      'effective_width': effective[0], 'effective_height': effective[1],
                      'raw_conflict': row.get('raw_width') is not None and raw_size != (row['raw_width'], row['raw_height']),
                      'effective_conflict': row.get('effective_width') is not None and effective != (row['effective_width'], row['effective_height']),
                      'format': image_format, 'alpha_composited_on_original_canvas': alpha}
        existing = self.corrections.get(row['sample_id'])
        if existing is None:
            self.corrections.put(row['sample_id'], correction)
        elif existing != correction:
            raise ValueError('Decoded dimensions changed within an immutable Release')
        return correction

    def inspect_dimensions(self, ordinal):
        """Actual first decode before scheduling; only one image lives at a time.

        Never trust the raw/P2 aspect to freeze a batch plan. Warm visits reuse the
        Release-bound derived record; cold planning costs one extra decode without
        resizing. Worker prefetch cannot subsequently change the planned geometry.
        """
        row = self.index[ordinal]
        existing = self.corrections.get(row['sample_id'])
        if existing is not None:
            return existing['effective_width'], existing['effective_height']
        with Image.open(BytesIO(self._read_image(row))) as image:
            image.load()
            raw_size = image.size
            # ImageOps.exif_transpose swaps axes for these four orientations.
            effective = raw_size[::-1] if image.getexif().get(274) in (5, 6, 7, 8) else raw_size
            self._remember_dimensions(row, raw_size, effective, image.format,
                                      image.mode in ('RGBA', 'LA') or 'transparency' in image.info)
        return effective

    def __getitem__(self, ordinal):
        occurrence = 0
        if isinstance(ordinal, tuple):
            ordinal, occurrence = ordinal
        row = self.index[ordinal]
        content = self._read_image(row)
        with Image.open(BytesIO(content)) as image:
            image.load()
            raw_size, image_format = image.size, image.format
            alpha = image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            prepared = self.preprocessor(image)
        effective = prepared.bucket.original_size
        correction = self._remember_dimensions(row, raw_size, effective, image_format, alpha)
        return {'images': prepared.tensor, 'record': row, 'image_id': row['sample_id'],
                'ordinal': int(ordinal), 'occurrence_id': int(occurrence), 'effective_size': effective,
                'dimension_correction': correction}

    def effective_size(self, ordinal):
        row = self.index[ordinal]
        actual = self.corrections.get(row['sample_id'])
        if actual is not None:
            return actual['effective_width'], actual['effective_height']
        if row['dimension_status'] != 'unverified' and row['effective_width'] is not None:
            return row['effective_width'], row['effective_height']
        return None


def collate_prepared(samples):
    if not samples or len({tuple(item['images'].shape) for item in samples}) != 1:
        raise ValueError('A real batch must contain one actual decoded bucket; rebucket before stacking')
    return {'images': torch.stack([item['images'] for item in samples]),
            'records': [item['record'] for item in samples], 'sample_ids': [item['image_id'] for item in samples],
            'ordinals': [item['ordinal'] for item in samples],
            'occurrence_ids': [item['occurrence_id'] for item in samples]}


def rebucket_prepared(samples, batch_size, *, max_open_buckets=128):
    """Bounded actual-shape grouping, including corrections, before torch.stack."""
    if type(batch_size) is not int or batch_size < 1 or max_open_buckets < 1:
        raise ValueError('Positive batch and bucket bounds are required')
    pending = {}
    for sample in samples:
        shape = tuple(sample['images'].shape)
        if shape not in pending and len(pending) >= max_open_buckets:
            oldest = next(iter(pending))
            yield collate_prepared(pending.pop(oldest))
        values = pending.setdefault(shape, [])
        values.append(sample)
        if len(values) == batch_size:
            yield collate_prepared(pending.pop(shape))
    for values in pending.values():
        yield collate_prepared(values)
