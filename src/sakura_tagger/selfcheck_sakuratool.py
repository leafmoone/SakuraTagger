"""Small offline producer/provider compatibility checks; never reads the network."""
from copy import deepcopy
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile

from .data.modelscope import dataset_file_url, ModelScopeShardProvider
from .data.sakuratool import PRODUCER_FORMAT, prepare_release, validate_sample
from .data.release import build_local_index, single_artist_target
from .selfcheck import synthetic_vocabulary


def check_sakuratool():
    import pyarrow as pa
    import pyarrow.parquet as pq
    repository, revision = 'fixture/dataset', '1' * 40
    manifest_path = 'metadata/releases/fixture/release_manifest.json'
    vocabulary = synthetic_vocabulary()
    files, vocab = {}, {}
    for domain, entries in {**vocabulary.document['domains'], 'artist_existing': vocabulary.document['original_artist']}.items():
        text = StringIO()
        writer = csv.writer(text)
        writer.writerow(['class_id', 'class_name'])
        writer.writerows((entry['class_id'], entry['name']) for entry in entries)
        path = f'vocab/{domain}.csv'
        files[path] = text.getvalue().encode()
        vocab[domain] = {'path': path, 'classes': len(entries), 'sha256': hashlib.sha256(files[path]).hexdigest()}
    rows = []
    for ordinal, reason in enumerate(('KNOWN_SINGLE_ARTIST', 'MULTIPLE_RELIABLE_ARTIST_IDENTITIES', 'NO_RELIABLE_OTHER_ARTIST')):
        positive = {'general': [0], 'character': [], 'copyright': [],
                    'other_artist': [1] if ordinal < 2 else [], 'artist_existing': []}
        scope = {domain: {'confirmed_negative': [], 'unlabeled': 'UNKNOWN_NOT_NEGATIVE'}
                 for domain in ('general', 'character', 'copyright')}
        scope['artist'] = {'other_artist': {'task': 'SINGLE_LABEL', 'unknown_is_not_negative': True,
                                          'reason': reason, 'status': 'KNOWN' if ordinal == 0 else 'UNKNOWN'}}
        rows.append({'sample_id': f'sample-{ordinal}', 'image_sha256': hashlib.sha256(str(ordinal).encode()).hexdigest(),
                     'split': 'train', 'shard_id': 'train-0', 'shard_path': 'shards/train-0.tar',
                     'image_member_name': f'{ordinal}.png', 'json_member_name': f'{ordinal}.json',
                     'image_offset': ordinal * 2048 + 512, 'image_size': 100,
                     'json_offset': ordinal * 2048 + 1536, 'json_size': 100,
                     'width': 32, 'height': 48, 'family_id': 'UNKNOWN',
                     'general_ids': [0], 'character_ids': [], 'copyright_ids': [],
                     'other_artist_id': 1 if ordinal == 0 else None,
                     'original_artist_id': None, 'original_artist_ids': [],
                     'frozen_raw_positive_ids': json.dumps(positive), 'supervision_scope': json.dumps(scope),
                     'source_provenance': json.dumps([{'source': 'fixture', 'raw_tags': []}]),
                     'tar_sha256': 'b' * 64, 'tar_size': 10240})
    table = pa.Table.from_pylist(rows).replace_schema_metadata({
        b'format': b'sakuratool-sample-index-v1', b'offset_semantics': b'absolute TAR data offset; not header'})
    stream = BytesIO()
    pq.write_table(table, stream)
    index_path = 'index/train.parquet'
    files[index_path] = stream.getvalue()
    manifest = {'format': PRODUCER_FORMAT, 'repository': repository, 'dataset_version': 'fixture',
                'planned_rows_trainable': False, 'vocabulary': vocab, 'vocab_hash': 'c' * 64,
                'sample_counts': 3, 'split_counts': {'train': 3},
                'shard_list': [{'shard_id': 'train-0', 'relative_path': 'shards/train-0.tar',
                                'actual_bytes': 10240, 'tar_sha256': 'b' * 64,
                                'state': 'REMOTE_VERIFIED', 'sample_count': 3, 'split': 'train'}],
                'index_list': [{'path': index_path, 'bytes': len(files[index_path]), 'rows': 3,
                                'sha256': hashlib.sha256(files[index_path]).hexdigest()}]}
    files[manifest_path] = json.dumps(manifest).encode()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        release = prepare_release(repository, revision, manifest_path, root / 'good', read_file=files.__getitem__)
        index = build_local_index(release, root / 'index.sqlite')
        try:
            normalized = [index[n] for n in range(3)]
            assert [single_artist_target(row, release.vocabulary).reason for row in normalized] == [
                'other_artist_valid', 'conflict', 'missing']
            assert all(row['dimension_status'] == 'unverified' and row['effective_width'] is None for row in normalized)
            assert all(row['general_negative_ids'] == [] for row in normalized)
            assert json.loads(normalized[1]['producer_record']) == rows[1]
            provider = ModelScopeShardProvider(release)
            assert revision in provider.url_for(release.shards['train-0'])
            def rejected(call):
                try:
                    call()
                except (ValueError, KeyError):
                    return
                raise AssertionError('Expected incompatible producer/provider input to be rejected')
            for raw, row in zip(rows, normalized):
                sample = {key: raw[key] for key in ('sample_id', 'image_sha256', 'split', 'width', 'height',
                                                    'family_id', 'other_artist_id', 'original_artist_id', 'original_artist_ids')}
                sample.update(format='sakuratool-tar-sample-v1', frozen_raw_positive_ids=json.loads(raw['frozen_raw_positive_ids']),
                              supervision_scope=json.loads(raw['supervision_scope']), source_provenance=json.loads(raw['source_provenance']))
                sample['positive_ids'] = {**sample['frozen_raw_positive_ids'],
                    'other_artist': [] if raw['other_artist_id'] is None else [raw['other_artist_id']]}
                validate_sample(sample, row, release)
                sample['source_provenance'] = []
                rejected(lambda: validate_sample(sample, row, release))
            rejected(lambda: dataset_file_url(repository, 'main', index_path))
            rejected(lambda: dataset_file_url(repository, revision, '../outside'))
            rejected(lambda: provider.url_for({'path': 'shards/unlisted.tar'}))
            rejected(lambda: prepare_release(repository, revision, manifest_path, root / 'good', read_file=files.__getitem__))
            for number, mutation in enumerate(('format', 'offset', 'sha')):
                bad_files = dict(files)
                bad_manifest = deepcopy(manifest)
                if mutation == 'format':
                    bad_manifest['format'] = 'unsupported'
                elif mutation == 'sha':
                    bad_files[index_path] += b'changed'
                else:
                    bad = table.replace_schema_metadata({b'format': b'sakuratool-sample-index-v1', b'offset_semantics': b'header'})
                    buffer = BytesIO()
                    pq.write_table(bad, buffer)
                    bad_files[index_path] = buffer.getvalue()
                    bad_manifest['index_list'][0].update(bytes=len(bad_files[index_path]), sha256=hashlib.sha256(bad_files[index_path]).hexdigest())
                bad_files[manifest_path] = json.dumps(bad_manifest).encode()
                target = root / f'bad-{number}'
                rejected(lambda: prepare_release(repository, revision, manifest_path, target, read_file=bad_files.__getitem__))
                assert not (target / 'release.json').exists()
        finally:
            index.close()
    return {'immutable_provider_and_inventory': 'PASS', 'producer_artist_conflict_and_unknown': 'PASS',
            'lossless_provenance_and_unverified_dimensions': 'PASS',
            'format_offset_and_metadata_failure_no_release': 'PASS', 'network_requests': 0,
            'full_real_release_reader': 'NOT_VERIFIED'}
