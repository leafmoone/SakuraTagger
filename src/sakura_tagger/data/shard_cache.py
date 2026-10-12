"""Small job-owned shared-worker shard cache with leases and bounded prefetch."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
try:
    import fcntl
except ImportError:  # Legacy readers/source filtering remain importable elsewhere.
    fcntl = None
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import quote, urlsplit
from urllib.request import urlopen
import uuid

from .release import relative_path


class CacheBackpressure(TimeoutError):
    pass


def _timeout(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Timeout must be finite and positive')
    return value


def _stream_copy(stream, destination, expected_size, stop, *, timeout=30):
    _timeout(timeout)
    digest, total, deadline = hashlib.sha256(), 0, time.monotonic() + timeout
    with Path(destination).open('wb') as output:
        while True:
            if stop.is_set():
                raise InterruptedError('Shard transfer cancelled')
            if time.monotonic() >= deadline:
                raise TimeoutError('Shard transfer deadline exceeded')
            chunk = stream.read(min(1024 * 1024, expected_size - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > expected_size:
                raise ValueError('Shard provider exceeded reserved size')
            digest.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    return total, digest.hexdigest()


class LocalShardProvider:
    """Copy from an already published immutable local inventory, never .partial."""
    def __init__(self, release, *, timeout=30):
        self.release, self.timeout = release, _timeout(timeout)

    def fetch(self, descriptor, destination, stop):
        path = self.release.local_path(descriptor['path'])
        with path.open('rb') as stream:
            return _stream_copy(stream, destination, descriptor['size_bytes'], stop, timeout=self.timeout)


class LoopbackShardProvider:
    """Only a local HTTP fixture; not a made-up production remote protocol."""
    def __init__(self, base_url, *, timeout=10, read_timeout=30):
        parsed = urlsplit(base_url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', '::1', 'localhost') or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Loopback fixture provider accepts local unauthenticated HTTP only')
        self.base_url, self.timeout = base_url.rstrip('/') + '/', _timeout(timeout)
        self.read_timeout = _timeout(read_timeout)

    def fetch(self, descriptor, destination, stop):
        path = relative_path(descriptor['path'])
        with urlopen(self.base_url + quote(path, safe='/'), timeout=self.read_timeout) as stream:
            return _stream_copy(stream, destination, descriptor['size_bytes'], stop, timeout=self.timeout)


class HuggingFaceShardProvider:
    """Public dataset files at full immutable Git commits; no credential lookup.

    URL construction uses the official SDK. Bytes stream into our sole bounded
    partial instead of an additional SDK cache. Real remote use is NOT_VERIFIED.
    """
    def __init__(self, release, *, timeout=30, read_timeout=30):
        source = release.source
        if source.get('provider') != 'huggingface' or not re.fullmatch('[0-9a-f]{40}', source.get('revision', '')):
            raise ValueError('Hugging Face requires an explicit full 40-hex dataset commit')
        self.repository, self.revision, self.timeout = source['repository'], source['revision'], _timeout(timeout)
        self.read_timeout = _timeout(read_timeout)

    def url_for(self, descriptor):
        from huggingface_hub import hf_hub_url
        return hf_hub_url(repo_id=self.repository, filename=relative_path(descriptor['path']),
                          repo_type='dataset', revision=self.revision, endpoint='https://huggingface.co')

    def fetch(self, descriptor, destination, stop):
        # urllib does not read Hugging Face token files or inject SDK credentials.
        with urlopen(self.url_for(descriptor), timeout=self.read_timeout) as stream:
            return _stream_copy(stream, destination, descriptor['size_bytes'], stop, timeout=self.timeout)


def provider_for_release(release, *, loopback_url=None, timeout=30, read_timeout=30):
    kind = release.source['provider']
    if kind == 'local':
        return LocalShardProvider(release, timeout=timeout)
    if kind == 'modelscope':
        from .modelscope import ModelScopeShardProvider
        return ModelScopeShardProvider(release, timeout=timeout, read_timeout=read_timeout)
    if kind == 'huggingface':
        return HuggingFaceShardProvider(release, timeout=timeout, read_timeout=read_timeout)
    if kind == 'loopback' and loopback_url:
        return LoopbackShardProvider(loopback_url, timeout=timeout, read_timeout=read_timeout)
    raise NotImplementedError('No verified immutable adapter for this provider; real producer handoff is required')


class ShardCache:
    """Leases and download reservations are shared via a small locked state file.

    Capacity covers reserved partial bytes plus ready/leased TARs. Lock/state
    metadata is small and separate. Only the selected job directory is managed.
    """
    def __init__(self, root, release, provider, *, capacity_bytes, job_id='run', high_watermark=0.9,
                 low_watermark=0.7, prefetch_count=2, workers=2, wait_timeout=30., retries=1):
        if fcntl is None:
            raise NotImplementedError('The shared-worker cache requires POSIX fcntl locks; legacy data APIs remain available')
        _timeout(wait_timeout)
        if type(capacity_bytes) is not int or capacity_bytes < 1:
            raise ValueError('Cache capacity_bytes must be positive')
        if not 0 <= low_watermark < high_watermark <= 1:
            raise ValueError('Require 0 <= low_watermark < high_watermark <= 1')
        if any(type(value) is not int or value < 1 for value in (prefetch_count, workers)) or wait_timeout <= 0 or type(retries) is not int or retries < 0:
            raise ValueError('Invalid prefetch, worker, timeout or retry settings')
        if not re.fullmatch('[A-Za-z0-9_-]+', job_id):
            raise ValueError('job_id must be one portable directory name')
        self.root = Path(root).resolve() / job_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.release, self.provider = release, provider
        self.capacity_bytes, self.high_watermark, self.low_watermark = capacity_bytes, high_watermark, low_watermark
        self.prefetch_count, self.workers, self.wait_timeout, self.retries = prefetch_count, workers, wait_timeout, retries
        self.job_id = job_id
        self._runtime_pid = os.getpid()
        self.stop = threading.Event()
        self._executor, self._pending, self._pending_lock = None, {}, threading.Lock()
        with self._state() as state:
            identity = {'release': release.identity, 'job_id': job_id, 'capacity_bytes': capacity_bytes,
                        'high_watermark': high_watermark, 'low_watermark': low_watermark}
            if state and state.get('owner') != identity:
                raise ValueError('Cache belongs to another job/Release; nothing was removed')
            if not state:
                # Never adopt an arbitrary directory containing unowned files.
                unexpected = set(path.name for path in self.root.iterdir()) - {'state.lock'}
                if unexpected:
                    raise ValueError('Cache job directory is not empty or recognized')
                state.update(owner=identity, entries={})
        (self.root / 'locks').mkdir(exist_ok=True)

    def __getstate__(self):
        return dict(root=str(self.root.parent), release=self.release, provider=self.provider,
                    capacity_bytes=self.capacity_bytes, job_id=self.job_id, high_watermark=self.high_watermark,
                    low_watermark=self.low_watermark, prefetch_count=self.prefetch_count, workers=self.workers,
                    wait_timeout=self.wait_timeout, retries=self.retries)

    def __setstate__(self, state):
        self.__init__(**state)

    def _process_runtime(self):
        # Forked DataLoader workers must not inherit a parent's thread executor,
        # futures or locks. Shared ownership remains in the on-disk state only.
        if self._runtime_pid != os.getpid():
            self._runtime_pid = os.getpid()
            self.stop = threading.Event()
            self._executor, self._pending, self._pending_lock = None, {}, threading.Lock()

    @contextmanager
    def _lock(self, path, *, timeout=None):
        deadline = time.monotonic() + (self.wait_timeout if timeout is None else timeout)
        with Path(path).open('a+b') as handle:
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise CacheBackpressure('Timed out waiting for cache ownership lock')
                    time.sleep(0.01)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @contextmanager
    def _state(self):
        with self._lock(self.root / 'state.lock'):
            path = self.root / 'state.json'
            state = json.loads(path.read_text()) if path.exists() else {}
            yield state
            temporary = self.root / 'state.json.tmp'
            temporary.write_text(json.dumps(state, separators=(',', ':')), encoding='utf-8')
            os.replace(temporary, path)

    def _names(self, shard_id):
        descriptor = self.release.shards[shard_id]
        name = hashlib.sha256((shard_id + descriptor['sha256']).encode()).hexdigest() + '.tar'
        return self.root / name, self.root / (name + '.partial')

    @staticmethod
    def _alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _clean_leases(self, state):
        for shard, item in list(state['entries'].items()):
            if item['status'] == 'downloading' and not self._alive(item['writer']):
                for path in self._names(shard):
                    path.unlink(missing_ok=True)
                del state['entries'][shard]
                continue
            item['leases'] = {token: pid for token, pid in item['leases'].items() if self._alive(pid)}

    def _drop(self, state, shard_id):
        item = state['entries'][shard_id]
        if item['leases'] or item['status'] == 'downloading':
            raise RuntimeError('Attempted to evict a leased or in-flight shard')
        for path in self._names(shard_id):
            path.unlink(missing_ok=True)
        del state['entries'][shard_id]

    def _room(self, state, size):
        self._clean_leases(state)
        used = sum(item['size'] for item in state['entries'].values())
        if used + size <= self.high_watermark * self.capacity_bytes:
            return True
        target = max(0, int(self.low_watermark * self.capacity_bytes) - size)
        evictable = sorted((item['last'], shard) for shard, item in state['entries'].items()
                           if item['status'] == 'ready' and not item['leases'])
        for _, shard in evictable:
            if used <= target:
                break
            used -= state['entries'][shard]['size']
            self._drop(state, shard)
        # One large shard may exceed the high watermark, but never hard capacity.
        return used + size <= self.capacity_bytes

    def _obtain(self, shard_id, token=None):
        self._process_runtime()
        if shard_id not in self.release.shards:
            raise KeyError(shard_id)
        descriptor = self.release.shards[shard_id]
        size = descriptor['size_bytes']
        if size > self.capacity_bytes:
            raise ValueError('A shard exceeds configured cache capacity')
        ready, partial = self._names(shard_id)
        deadline = time.monotonic() + self.wait_timeout
        with self._lock(self.root / 'locks' / (ready.name + '.lock')):
            while True:
                if self.stop.is_set():
                    raise InterruptedError('Shard cache is closed/cancelled')
                with self._state() as state:
                    self._clean_leases(state)
                    existing = state['entries'].get(shard_id)
                    if existing and existing['status'] == 'ready':
                        valid = ready.exists() and ready.stat().st_size == size and ready.stat().st_mtime_ns == existing['mtime_ns']
                        if valid:
                            existing['last'] = time.time_ns()
                            if token is not None:
                                existing['leases'][token] = os.getpid()
                            return ready
                        if existing['leases']:
                            raise ValueError('A leased shard changed or disappeared outside the cache')
                        self._drop(state, shard_id)
                    elif existing:
                        # The exclusive shard lock proves any earlier writer ended.
                        partial.unlink(missing_ok=True)
                        ready.unlink(missing_ok=True)
                        del state['entries'][shard_id]
                    if self._room(state, size):
                        state['entries'][shard_id] = {'size': size, 'status': 'downloading',
                            'leases': {}, 'last': time.time_ns(), 'writer': os.getpid()}
                        break
                if time.monotonic() >= deadline:
                    raise CacheBackpressure('Cache is full with leased/in-flight data; retry after consumption')
                self.stop.wait(0.01)
            try:
                for attempt in range(self.retries + 1):
                    try:
                        actual_size, digest = self.provider.fetch(descriptor, partial, self.stop)
                        if actual_size != size or digest != descriptor['sha256'] or partial.stat().st_size != size:
                            raise ValueError('Downloaded shard size/SHA mismatch')
                        break
                    except Exception:
                        partial.unlink(missing_ok=True)
                        if attempt == self.retries or self.stop.is_set():
                            raise
                with self._state() as state:
                    os.replace(partial, ready)
                    state['entries'][shard_id] = {'size': size, 'status': 'ready',
                        'leases': {} if token is None else {token: os.getpid()}, 'last': time.time_ns(),
                        'mtime_ns': ready.stat().st_mtime_ns}
                return ready
            except BaseException:
                with self._state() as state:
                    partial.unlink(missing_ok=True)
                    ready.unlink(missing_ok=True)
                    state['entries'].pop(shard_id, None)
                raise

    @contextmanager
    def lease(self, shard_id):
        token = uuid.uuid4().hex
        path = self._obtain(shard_id, token)
        try:
            yield path
        finally:
            with self._state() as state:
                if shard_id in state['entries']:
                    state['entries'][shard_id]['leases'].pop(token, None)

    def prefetch(self, shard_ids):
        self._process_runtime()
        if self.stop.is_set():
            raise InterruptedError('Shard cache is closed')
        with self._pending_lock:
            # Completed results are returned by wait_prefetch; do not silently
            # discard failures or grow an unbounded queue at an epoch boundary.
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='tagger-shards')
            for shard in shard_ids:
                if len(self._pending) >= self.prefetch_count:
                    break
                if shard not in self._pending:
                    self._pending[shard] = self._executor.submit(self._obtain, shard)
            return tuple(self._pending)

    def wait_prefetch(self):
        self._process_runtime()
        with self._pending_lock:
            pending, self._pending = self._pending, {}
        errors = []
        for future in pending.values():
            try:
                future.result()
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def stats(self):
        with self._state() as state:
            partial_bytes = 0
            for shard in state['entries']:
                try:
                    partial_bytes += self._names(shard)[1].stat().st_size
                except FileNotFoundError:
                    pass  # A failed transfer can remove its partial between observations.
            return {'reserved_bytes': sum(item['size'] for item in state['entries'].values()),
                    'partial_bytes': partial_bytes,
                    'ready_bytes': sum(item['size'] for item in state['entries'].values() if item['status'] == 'ready'),
                    'leased_shards': sum(bool(item['leases']) for item in state['entries'].values()),
                    'entries': len(state['entries']), 'capacity_bytes': self.capacity_bytes}

    def close(self):
        self._process_runtime()
        self.stop.set()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
