"""Durable synthetic Release/Parquet/CE contract acceptance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import tempfile

import torch

from .data.release import (DatasetRelease, RELEASE_VERSION, MAX_SHARD_BYTES, HANDOFF_STATUS,
                           artifact_descriptor, reference_index_schema, build_local_index,
                           validate_sparse_row, single_artist_target, collate_artist_targets)


def reference_row(index, **overrides):
    """Small metadata fixture; actual TAR bytes are exercised by the TAR suite."""
    row = {'sample_id': f'fixture_{index}', 'image_sha256': hashlib.sha256(f'image {index}'.encode()).hexdigest(),
           'split': 'train', 'status': 'published', 'shard_id': 'shard_0', 'file_path': 'shard_0.tar',
           'image_member': f'{index}.png', 'json_member': f'{index}.json',
           'image_offset_data': index * 2048 + 512, 'image_size': 100,
           'json_offset_data': index * 2048 + 1536, 'json_size': 100,
           'raw_width': 48, 'raw_height': 32, 'effective_width': 48, 'effective_height': 32,
           'dimension_status': 'confirmed', 'family_id': 'UNKNOWN', 'near_duplicate_status': 'UNKNOWN',
           'source_provenance': ['fixture_annotations'], 'single_artist_confirmed': None,
           'general_ids': [index % 3], 'character_ids': [0], 'copyright_ids': [],
           'other_artist_ids': [], 'original_artist_ids': [], 'artist_evidence': []}
    row.update(overrides)
    return row


def reference_release(root, rows, *, vocabulary=None, single_artist_confirmed=True):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .selfcheck import synthetic_vocabulary
    root = Path(root)
    vocabulary = vocabulary or synthetic_vocabulary()
    (root / 'vocabulary.json').write_text(json.dumps(vocabulary.document), encoding='utf-8')
    table = pa.Table.from_pylist(rows, schema=reference_index_schema(vocabulary, 'fixture_release'))
    pq.write_table(table, root / 'index.parquet')
    document = {'schema_version': RELEASE_VERSION, 'status': 'published', 'dataset_id': 'synthetic',
                'release_id': 'fixture_release', 'vocabulary_path': 'vocabulary.json',
                'vocab_sha256': vocabulary.sha256, 'vocabulary_csv_sha256': {'general': 'a' * 64},
                'supervision_assumptions': {'single_artist_confirmed': single_artist_confirmed},
                'source': {'provider': 'local'},
                'shards': [{'shard_id': 'shard_0', 'path': 'shard_0.tar', 'size_bytes': 65536,
                            'sha256': 'b' * 64, 'status': 'published'}],
                'indices': [{**artifact_descriptor(root / 'index.parquet', root), 'status': 'published'}]}
    (root / 'release.json').write_text(json.dumps(document), encoding='utf-8')
    return DatasetRelease.read(root / 'release.json')


def check_release():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        evidence = [{'source': 'one', 'domain': 'other_artist', 'class_id': 2, 'label': None},
                    {'source': 'two', 'domain': 'other_artist', 'class_id': 3, 'label': None}]
        rows = [reference_row(0, other_artist_ids=[1]),
                reference_row(1, original_artist_ids=[2]),
                reference_row(2),
                reference_row(3, artist_evidence=[{'source': 'one', 'domain': 'out_of_vocabulary',
                                                   'class_id': None, 'label': 'unknown_artist'}]),
                reference_row(4, artist_evidence=evidence),
                reference_row(5, split='val', general_ids=[10], other_artist_ids=[8]),
                reference_row(6, status='planned', file_path=None),
                reference_row(7, file_path=None),
                reference_row(8, raw_width=None, raw_height=None, effective_width=None,
                              effective_height=None, dimension_status='unverified'),
                reference_row(9, general_ids=[], other_artist_ids=[7], single_artist_confirmed=False)]
        release = reference_release(root, rows)
        index = build_local_index(release, root / 'samples.sqlite', batch_size=2)
        assert len(index) == 8
        assert index.metadata['stats']['planned_skipped'] == 2
        assert index.positive_support('general').tolist() == [2, 2, 2] + [0] * 8
        assert index.positive_support('other_artist').tolist() == [0, 1] + [0] * 7
        assert index.metadata['stats']['artist_conversion']['unconfirmed'] == 1
        assert single_artist_target(index[7], release.vocabulary, release_confirmed=True).reason == 'unconfirmed'
        assert index[4]['artist_evidence'] == evidence
        assert index[6]['raw_width'] is None and index[6]['effective_width'] is None
        assert index[0]['source_provenance'] == ['fixture_annotations']
        assert release.source_membership() is None  # Image source is never label membership.
        batch = collate_artist_targets([index[i] for i in range(5)], release)
        assert batch['targets'].dtype == torch.int64 and batch['valid'].dtype == torch.bool
        assert batch['targets'].tolist() == [1, -100, -100, -100, -100]
        assert batch['valid'].tolist() == [True, False, False, False, False]
        assert batch['original_ids'][1] == (2,)
        assert batch['status'] == ['other_artist_valid', 'original_only', 'missing', 'out_of_vocabulary', 'conflict']
        ce = torch.nn.functional.cross_entropy(torch.randn(5, 9)[batch['valid']], batch['targets'][batch['valid']])
        assert torch.isfinite(ce)
        assert single_artist_target(rows[0], release.vocabulary).reason == 'unconfirmed'
        assert single_artist_target({**rows[0], 'single_artist_confirmed': True}, release.vocabulary).valid
        assert not single_artist_target({**rows[0], 'original_artist_ids': [0]}, release.vocabulary,
                                        release_confirmed=True).valid
        restored = pickle.loads(pickle.dumps(index))
        assert restored[0] == index[0]
        assert build_local_index(release, root / 'samples.sqlite').metadata == index.metadata
        restored.close()
        index.close()
        rejects = 0
        def reject(call):
            nonlocal rejects
            try:
                call()
            except (ValueError, KeyError):
                rejects += 1
            else:
                raise AssertionError('Invalid Release/index contract accepted')
        for change in ({'status': 'draft'}, {'vocab_sha256': 'a' * 64}):
            reject(lambda change=change: DatasetRelease({**release.document, **change}, release.vocabulary))
        for change in ({'path': 'shard_0.tar.partial'}, {'status': 'writing'}, {'size_bytes': MAX_SHARD_BYTES + 1}):
            document = deepcopy(release.document)
            document['shards'][0].update(change)
            reject(lambda document=document: DatasetRelease(document, release.vocabulary))
        for change in ({'shard_id': 'unpublished'}, {'image_offset_data': 511},
                       {'image_size': 999999}, {'general_ids': [11]}, {'general_negative_ids': [0]}):
            reject(lambda change=change: validate_sparse_row({**rows[0], **change}, release))
        assert release.document['vocabulary_csv_sha256']['general'] != release.vocabulary.sha256
        boundary = deepcopy(release.document)
        boundary['shards'][0]['size_bytes'] = MAX_SHARD_BYTES
        large_release = DatasetRelease(boundary, release.vocabulary)
        large_row = {**rows[0], 'image_offset_data': 2**32 + 512}
        assert validate_sparse_row(large_row, large_release)['image_offset_data'] > 2**32
        # Duplicate SHA/sample and physical extents are checked once at construction.
        import sqlite3
        for name, duplicate in (('id', {**rows[1], 'sample_id': rows[0]['sample_id']}),
                                ('sha', {**rows[1], 'image_sha256': rows[0]['image_sha256']}),
                                ('overlap', {**rows[1], 'image_offset_data': 1024, 'image_size': 800})):
            subdir = root / name
            subdir.mkdir()
            bad = reference_release(subdir, [rows[0], duplicate])
            try:
                build_local_index(bad, subdir / 'bad.sqlite')
            except (ValueError, sqlite3.IntegrityError):
                assert not (subdir / 'bad.sqlite').exists()
                rejects += 1
            else:
                raise AssertionError('Duplicate or overlapping sample was accepted')
        return {'sparse_parquet_stream_and_lazy_index': 'PASS', 'trusted_CE_actual_loss': 'PASS',
                'artist_conversion': dict(zip(batch['status'], [1] * len(batch['status']))),
                'no_dense_mask_requirement': 'PASS', 'trusted_train_support_only': 'PASS',
                'sample_false_overrides_release_confirmation': 'PASS',
                'raw_effective_and_provenance_separate': 'PASS', 'planned_null_file_skipped': 2,
                'CSV_vs_canonical_SHA': 'PASS', 'uint64_extent_and_5GiB_metadata': 'PASS',
                'duplicate_boundary_checks': 'PASS', 'rejections': rejects,
                'worker_pickle_and_lazy_reload': 'PASS', 'handoff': HANDOFF_STATUS,
                'actual_TAR_bytes': 'NOT_RUN: this suite validates index metadata only'}
