"""Unauthenticated ModelScope dataset files at an immutable Git revision."""
import re
from urllib.parse import quote, urlencode
from urllib.request import urlopen

from .release import relative_path
from .shard_cache import _stream_copy, _timeout


def dataset_file_url(repository, revision, path):
    if not isinstance(repository, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
        raise ValueError('Expected a ModelScope owner/dataset repository')
    if not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('ModelScope requires a full immutable 40-hex commit')
    path = relative_path(path)
    # Matches HubApi.get_dataset_file_url (repo/files may return LFS pointers).
    return ('https://www.modelscope.cn/api/v1/datasets/' + quote(repository, safe='/')
            + '/repo?' + urlencode({'Source': 'SDK', 'Revision': revision, 'FilePath': path, 'View': False}))


def read_metadata(repository, revision, path, *, max_bytes=64 * 1024**2, timeout=30):
    """Bound metadata reads; never load a TAR or consult local credentials."""
    with urlopen(dataset_file_url(repository, revision, path), timeout=_timeout(timeout)) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError('Metadata exceeds its read-memory bound')
    return data


class ModelScopeShardProvider:
    def __init__(self, release, *, timeout=3600, read_timeout=30):
        if release.source.get('provider') != 'modelscope':
            raise ValueError('Expected a ModelScope release')
        self.release = release
        self.timeout, self.read_timeout = _timeout(timeout), _timeout(read_timeout)
        dataset_file_url(release.source['repository'], release.source['revision'], 'identity')

    def url_for(self, descriptor):
        # The frozen Release is the only admissible inventory.
        if not any(dict(descriptor) == dict(item) for item in self.release.shards.values()):
            raise ValueError('Shard is outside the selected immutable Release')
        return dataset_file_url(self.release.source['repository'], self.release.source['revision'], descriptor['path'])

    def fetch(self, descriptor, destination, stop):
        with urlopen(self.url_for(descriptor), timeout=self.read_timeout) as response:
            return _stream_copy(response, destination, descriptor['size_bytes'], stop, timeout=self.timeout)
