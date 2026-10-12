"""Actual local/loopback TAR/cache acceptance, never production downloads."""
from copy import deepcopy
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import tarfile
import tempfile
import threading
import time
from unittest.mock import patch

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from .data import BucketConfig, BucketSelector, NativePreprocessor
from .data.release import (DatasetRelease, RELEASE_VERSION, MAX_SHARD_BYTES, artifact_descriptor,
                           reference_index_schema, build_local_index)
from .data.shard_cache import (ShardCache, LocalShardProvider, LoopbackShardProvider,
                               HuggingFaceShardProvider, CacheBackpressure)
from .data.tar_dataset import (TarReader, TarDataset, DimensionCorrections, reference_sample_json,
                               decoder_capabilities, rebucket_prepared)


def tiny_tar_release(root, *, count=4):
    """Actual images, paired JSON, TAR headers, Parquet and immutable inventory."""
    from .selfcheck import synthetic_vocabulary
    from .selfcheck_release import reference_row
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    vocabulary = synthetic_vocabulary()
    rows, shards = [], []
    for index in range(count):
        row = reference_row(index, shard_id=f'shard_{index}', file_path=f'shard_{index}.tar',
                            other_artist_ids=[index % 3], single_artist_confirmed=None)
        stream = BytesIO()
        if index % 4 == 0:
            image = Image.new('RGB', (48, 32), (30 + index, 80, 100))
            exif = Image.Exif()
            exif[274] = 6
            image.save(stream, format='JPEG', exif=exif)
            row.update(image_member=f'{index}.jpg', effective_width=48, effective_height=32)  # Deliberately stale P2 assumption.
        elif index % 4 == 1:
            Image.new('RGBA', (96, 64), (255, 10, 20, 100)).save(stream, format='PNG')
            row.update(raw_width=48, raw_height=32, effective_width=48, effective_height=32)
        elif index % 4 == 2:
            Image.new('RGB', (32, 48), (15, 40, 90)).save(stream, format='WEBP')
            row.update(image_member=f'{index}.webp', raw_width=32, raw_height=48, effective_width=32, effective_height=48)
        else:
            Image.new('RGB', (32, 32), (55, 60, 90)).save(stream, format='PNG')
            row.update(raw_width=32, raw_height=32, effective_width=32, effective_height=32)
        content = stream.getvalue()
        row['image_sha256'] = hashlib.sha256(content).hexdigest()
        metadata = json.dumps(reference_sample_json(row, vocabulary), sort_keys=True).encode()
        path = root / row['file_path']
        with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as archive:
            for prefix, body in (('image', content), ('json', metadata)):
                member = tarfile.TarInfo(row[prefix + '_member'])
                member.size = len(body)
                row[prefix + '_offset_data'], row[prefix + '_size'] = archive.offset + 512, len(body)
                archive.addfile(member, BytesIO(body))
        shards.append({**artifact_descriptor(path, root), 'shard_id': row['shard_id'], 'status': 'published'})
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows, schema=reference_index_schema(vocabulary, 'tiny_tar')), root / 'index.parquet')
    (root / 'vocabulary.json').write_text(json.dumps(vocabulary.document))
    document = {'schema_version': RELEASE_VERSION, 'status': 'published', 'dataset_id': 'synthetic_tar',
        'release_id': 'tiny_tar', 'vocabulary_path': 'vocabulary.json', 'vocab_sha256': vocabulary.sha256,
        'supervision_assumptions': {'single_artist_confirmed': True}, 'source': {'provider': 'local'},
        'shards': shards, 'indices': [{**artifact_descriptor(root / 'index.parquet', root), 'status': 'published'}]}
    (root / 'release.json').write_text(json.dumps(document))
    return DatasetRelease.read(root / 'release.json')


def _shared_cache_worker(release_path, cache_root, base_url, barrier, queue):
    try:
        release = DatasetRelease.read(release_path)
        cache = ShardCache(cache_root, release, LoopbackShardProvider(base_url), capacity_bytes=32768, job_id='shared')
        barrier.wait(timeout=15)
        with cache.lease('shard_0') as path:
            queue.put(('ok', path.stat().st_size))
            time.sleep(0.05)
        cache.close()
    except Exception as error:
        queue.put(('error', repr(error)))
        raise


def check_tar():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        release = tiny_tar_release(root / 'published')
        index = build_local_index(release, root / 'index.sqlite', batch_size=2)
        shard_size = release.shards['shard_0']['size_bytes']
        for bad_timeout in (float('nan'), float('inf'), 0, -1):
            for construct in (
                lambda: LocalShardProvider(release, timeout=bad_timeout),
                lambda: LoopbackShardProvider('http://127.0.0.1:1', read_timeout=bad_timeout),
                lambda: ShardCache(root / 'bad_time', release, LocalShardProvider(release),
                                   capacity_bytes=shard_size, wait_timeout=bad_timeout)):
                try:
                    construct()
                except ValueError:
                    pass
                else:
                    raise AssertionError('Nonfinite/nonpositive timeout accepted')
        with patch('sakura_tagger.data.shard_cache.fcntl', None):
            try:
                ShardCache(root / 'no_locks', release, LocalShardProvider(release), capacity_bytes=shard_size)
            except NotImplementedError:
                pass
            else:
                raise AssertionError('Unsupported shared locking did not fail explicitly')
        for _, row in index.iter_records():
            with TarReader(release.local_path(row['file_path'])) as reader:
                sequential = dict(reader.iter_members())
                image, metadata = reader.read_sample(row)
                assert image == sequential[row['image_member']]
                assert metadata == json.loads(sequential[row['json_member']])
                original_pread = os.pread
                with patch('sakura_tagger.data.tar_dataset.os.pread', side_effect=lambda fd, size, offset: original_pread(fd, min(size, 7), offset)) as short:
                    assert reader.read_member(row['image_member'], row['image_offset_data'], row['image_size']) == image
                    assert short.call_count > 1
        with TarReader(release.local_path('shard_0.tar')) as reader:
            row = index[0]
            for name, offset, size in ((row['image_member'], row['image_offset_data'] - 512, row['image_size']),
                                       ('wrong.png', row['image_offset_data'], row['image_size']),
                                       (row['image_member'], row['image_offset_data'], row['image_size'] + 1)):
                try:
                    reader.read_member(name, offset, size)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Wrong header/name/size accepted')
            with patch('sakura_tagger.data.tar_dataset.os.pread', return_value=b''):
                try:
                    reader.read_member(row['image_member'], row['image_offset_data'], row['image_size'])
                except EOFError:
                    pass
                else:
                    raise AssertionError('Premature EOF accepted')
        link_path = root / 'link.tar'
        member = tarfile.TarInfo('link.bin')
        member.type, member.linkname, member.size = tarfile.SYMTYPE, 'outside.bin', 1
        link_path.write_bytes(member.tobuf(format=tarfile.USTAR_FORMAT) + b'x' + b'\0' * 1023)
        with TarReader(link_path) as reader:
            try:
                reader.read_member('link.bin', 512, 1)
            except ValueError:
                pass
            else:
                raise AssertionError('Non-regular member accepted')
        sparse_path = root / 'sparse.tar'
        with sparse_path.open('wb') as stream:
            padding = tarfile.TarInfo('sparse_padding')
            padding.size = 2**32 - 512
            stream.write(padding.tobuf(format=tarfile.USTAR_FORMAT))
            stream.seek(2**32)
            member = tarfile.TarInfo('large_offset.bin')
            member.size = 8
            stream.write(member.tobuf(format=tarfile.USTAR_FORMAT))
            stream.write(b'12345678')
            stream.truncate(MAX_SHARD_BYTES)
        with TarReader(sparse_path) as reader:
            assert reader.read_member('large_offset.bin', 2**32 + 512, 8) == b'12345678'
        assert sparse_path.stat().st_blocks * 512 < 1024 * 1024
        with sparse_path.open('r+b') as stream:
            stream.truncate(MAX_SHARD_BYTES + 1)
        try:
            TarReader(sparse_path)
        except ValueError:
            pass
        else:
            raise AssertionError('Over-5GiB shard accepted')

        cache = ShardCache(root / 'cache', release, LocalShardProvider(release), capacity_bytes=shard_size,
                           job_id='local', wait_timeout=0.15, retries=0)
        with cache.lease('shard_0') as protected:
            try:
                with cache.lease('shard_1'):
                    pass
            except CacheBackpressure:
                pass
            else:
                raise AssertionError('Evicted leased data or exceeded capacity')
            assert protected.exists() and cache.stats()['leased_shards'] == 1
        with cache.lease('shard_1'):
            assert not protected.exists()
        with cache.lease('shard_0') as recovered:
            assert recovered.exists()
        recovered.unlink()  # Only this test job's unleased cache, not the source.
        with cache.lease('shard_0') as recovered:
            assert recovered.exists()
        cache.prefetch(['shard_2', 'shard_3'])
        cache.wait_prefetch()
        assert cache.stats()['reserved_bytes'] <= shard_size
        cache.close()

        started, proceed = threading.Event(), threading.Event()
        class PausedProvider(LocalShardProvider):
            def fetch(self, descriptor, destination, stop):
                Path(destination).write_bytes(b'x')
                started.set()
                if not proceed.wait(3):
                    raise TimeoutError('Fixture pause timed out')
                return super().fetch(descriptor, destination, stop)
        paused = ShardCache(root / 'cache', release, PausedProvider(release), capacity_bytes=shard_size,
                            job_id='partial', wait_timeout=0.15, retries=0)
        paused.prefetch(['shard_0'])
        assert started.wait(3)
        assert paused.stats()['partial_bytes'] == 1 and paused.stats()['reserved_bytes'] == shard_size
        try:
            with paused.lease('shard_1'):
                pass
        except CacheBackpressure:
            pass
        else:
            raise AssertionError('In-flight partial reservation was ignored')
        partial_path = paused._names('shard_0')[1]
        try:
            TarReader(partial_path)
        except ValueError:
            pass
        else:
            raise AssertionError('A partial TAR became readable')
        proceed.set()
        paused.wait_prefetch()
        assert paused.stats()['partial_bytes'] == 0
        paused.close()

        class CorruptProvider(LocalShardProvider):
            def fetch(self, descriptor, destination, stop):
                self.attempts = getattr(self, 'attempts', 0) + 1
                Path(destination).write_bytes(b'x' * descriptor['size_bytes'])
                return descriptor['size_bytes'], hashlib.sha256(Path(destination).read_bytes()).hexdigest()
        corrupt = ShardCache(root / 'cache', release, CorruptProvider(release), capacity_bytes=shard_size,
                             job_id='corrupt', wait_timeout=0.15, retries=1)
        corrupt.prefetch(['shard_0'])
        try:
            corrupt.wait_prefetch()
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupted download published')
        assert corrupt.stats()['reserved_bytes'] == 0 and not list(corrupt.root.glob('*.partial'))
        assert corrupt.provider.attempts == 2
        corrupt.close()

        cancelling = threading.Event()
        class CancelProvider(LocalShardProvider):
            def fetch(self, descriptor, destination, stop):
                cancelling.set()
                if not stop.wait(3):
                    raise TimeoutError('Cancellation fixture timeout')
                raise InterruptedError('Cancelled fixture download')
        cancelled = ShardCache(root / 'cache', release, CancelProvider(release), capacity_bytes=shard_size,
                              job_id='cancel', wait_timeout=0.15, retries=0)
        cancelled.prefetch(['shard_0'])
        assert cancelling.wait(3)
        cancelled.close()
        assert cancelled.stats()['reserved_bytes'] == 0

        requests, request_lock = [], threading.Lock()
        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                with request_lock:
                    requests.append(self.path)
                super().do_GET()
        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(release.root)))
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        base_url = f'http://127.0.0.1:{server.server_port}/'
        try:
            context = multiprocessing.get_context('spawn')
            barrier, queue = context.Barrier(2), context.Queue()
            children = [context.Process(target=_shared_cache_worker, args=(str(release.root / 'release.json'),
                        str(root / 'cache'), base_url, barrier, queue)) for _ in range(2)]
            for child in children:
                child.start()
            try:
                results = [queue.get(timeout=30) for _ in children]
                for child in children:
                    child.join(timeout=10)
                assert all(child.exitcode == 0 for child in children) and all(result == ('ok', shard_size) for result in results)
            finally:
                for child in children:
                    if child.is_alive():
                        child.terminate()
                        child.join(timeout=5)
            assert requests == ['/shard_0.tar']  # Shared workers, exactly one writer/download.
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=3)

        abandoned = ShardCache(root / 'cache', release, LocalShardProvider(release), capacity_bytes=shard_size,
                              job_id='abandoned', wait_timeout=0.2)
        with abandoned._state() as state:
            state['entries']['shard_0'] = {'size': shard_size, 'status': 'downloading', 'leases': {},
                                           'last': 0, 'writer': children[0].pid}
            abandoned._names('shard_0')[1].write_bytes(b'abandoned')
        with abandoned.lease('shard_1') as path:
            assert path.exists() and not abandoned._names('shard_0')[1].exists()
        try:
            ShardCache(root / 'cache', release, LocalShardProvider(release), capacity_bytes=shard_size * 2,
                       job_id='abandoned')
        except ValueError:
            pass
        else:
            raise AssertionError('Shared job accepted inconsistent capacity')
        abandoned.close()

        decode_cache = ShardCache(root / 'cache', release, LocalShardProvider(release), capacity_bytes=shard_size * 2,
                                  job_id='decode', wait_timeout=0.5)
        corrections = DimensionCorrections(root / 'dimensions.sqlite', release)
        preprocessing = NativePreprocessor(BucketSelector(BucketConfig(target_pixels=32 * 48,
                                                       max_pixels=48 * 48, max_side=96)))
        dataset = TarDataset(index, decode_cache, corrections=corrections, preprocessor=preprocessing)
        prepared = [dataset[i] for i in range(len(dataset))]
        assert prepared[0]['effective_size'] == (32, 48) and prepared[0]['dimension_correction']['effective_conflict']
        assert prepared[1]['effective_size'] == (96, 64) and prepared[1]['dimension_correction']['alpha_composited_on_original_canvas']
        assert prepared[1]['dimension_correction']['raw_conflict']
        assert dataset.effective_size(0) == (32, 48)
        assert corrections._connection() is corrections._connection()
        reopened = pickle.loads(pickle.dumps(corrections))
        assert reopened.get(prepared[0]['image_id']) == corrections.get(prepared[0]['image_id'])
        reopened.close()
        assert dataset[0]['effective_size'] == (32, 48)  # Persisted correction prevents recurring mismatch failure.
        batches = list(rebucket_prepared(prepared, batch_size=2))
        assert sum(len(batch['sample_ids']) for batch in batches) == 4
        assert all(batch['images'].ndim == 4 for batch in batches)
        decode_cache.close()
        corrections.close()
        index.close()
        hf_document = deepcopy(release.document)
        hf_document['source'] = {'provider': 'huggingface', 'repository': 'fixture/dataset', 'revision': 'a' * 40}
        hf = HuggingFaceShardProvider(DatasetRelease(hf_document, release.vocabulary))
        assert hf.url_for(release.shards['shard_0']) == 'https://huggingface.co/datasets/fixture/dataset/resolve/' + 'a' * 40 + '/shard_0.tar'
        hf_document['source']['revision'] = 'dev'
        try:
            DatasetRelease(hf_document, release.vocabulary)
        except ValueError:
            pass
        else:
            raise AssertionError('Mutable remote branch accepted')
        return {'actual_TAR_JSON_Parquet_sequential_offset': 'PASS', 'short_reads_header_name_type_ranges': 'PASS',
                'sparse_gt_2_32_offset_exact_5GiB_boundary': 'PASS', 'leased_protection_backpressure_eviction_redownload': 'PASS',
                'partial_reserved_accounting_corrupt_retry_cleanup': 'PASS', 'bounded_prefetch_epoch_tail': 'PASS',
                'cancel_dead_writer_reclaim_shared_capacity': 'PASS',
                'finite_timeouts_optional_POSIX_cache_lazy_dimensions': 'PASS',
                'loopback_two_process_single_writer': 'PASS', 'EXIF_RGBA_actual_bucket_corrections': 'PASS',
                'decoders': decoder_capabilities(), 'HuggingFace_commit_URL_construction': 'PASS',
                'remote_transfer': 'NOT_VERIFIED', 'real_5GiB_upload': 'NOT_VERIFIED',
                'producer_handoff': 'REAL_PRODUCER_HANDOFF_NOT_VERIFIED'}
