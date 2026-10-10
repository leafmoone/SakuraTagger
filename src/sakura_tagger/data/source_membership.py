"""Vocabulary membership, never image provenance; an optional inference sidecar."""
from dataclasses import dataclass
from types import MappingProxyType
import re

from .contract import Vocabulary
from ..model.registry import LOGICAL_DOMAINS

SOURCE_MEMBERSHIP_VERSION = 'sakura-tagger-source-membership-v1'
MEMBERSHIP_DOMAINS = LOGICAL_DOMAINS + ('artist',)
_FIELDS = {'domain', 'class_id', 'source'}
_OPTIONAL_FIELDS = {'namespace', 'evidence_type'}
_SOURCE = re.compile(r'[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*')


@dataclass(frozen=True, init=False)
class SourceMembership:
    """Validated immutable index, constructed with from_records() or read().

    Metadata keys are schema_version, vocab_version and vocab_sha256. The SHA
    comes from Vocabulary.sha256 unchanged. IDs are domain-local, zero-based.
    Duplicate (domain, class_id, source) relationships are rejected, regardless
    of optional evidence fields. Unlisted labels have no inferred membership.
    """
    schema_version: str
    vocab_version: str
    vocab_sha256: str
    _names: object
    _allowed: object
    sources: tuple

    @classmethod
    def from_records(cls, records, vocabulary: Vocabulary, *, schema_version=SOURCE_MEMBERSHIP_VERSION,
                     vocab_version, vocab_sha256):
        if schema_version != SOURCE_MEMBERSHIP_VERSION:
            raise ValueError('Unsupported source membership schema version')
        if vocab_version != vocabulary.document['version']:
            raise ValueError('Source membership vocabulary version mismatch')
        if vocab_sha256 != vocabulary.sha256:
            raise ValueError('Source membership vocabulary SHA mismatch')
        names = {domain: tuple(entry['name'] for entry in entries)
                 for domain, entries in vocabulary.class_id_mapping.items()}
        relationships, index = set(), {}
        for row in records:
            if not isinstance(row, dict) or not _FIELDS <= row.keys() or row.keys() - (_FIELDS | _OPTIONAL_FIELDS):
                raise ValueError('Source membership rows require domain, class_id, source; optional namespace/evidence_type')
            domain, class_id, source = row['domain'], row['class_id'], row['source']
            if not isinstance(domain, str) or domain not in names:
                raise ValueError(f'Unknown membership domain: {domain!r}')
            if type(class_id) is not int or not 0 <= class_id < len(names[domain]):
                raise ValueError(f'{domain}: source membership class_id must be an in-range integer')
            if not isinstance(source, str) or not _SOURCE.fullmatch(source):
                raise ValueError('Source must be a canonical lowercase identifier')
            if any(row.get(key) is not None and not isinstance(row[key], str) for key in _OPTIONAL_FIELDS):
                raise ValueError('Optional namespace/evidence_type must be strings or null')
            relation = (domain, class_id, source)
            if relation in relationships:
                raise ValueError(f'Duplicate source membership: {relation}')
            relationships.add(relation)
            index.setdefault((domain, source), set()).add(class_id)
        result = object.__new__(cls)
        for key, value in {'schema_version': schema_version, 'vocab_version': vocab_version,
                           'vocab_sha256': vocab_sha256, '_names': MappingProxyType(names),
                           '_allowed': MappingProxyType({key: tuple(sorted(ids)) for key, ids in index.items()}),
                           'sources': tuple(sorted({source for _, source in index}))}.items():
            object.__setattr__(result, key, value)
        return result

    @classmethod
    def read(cls, path, vocabulary: Vocabulary):
        """Read Parquet once; PyArrow is imported only on this path."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError('Parquet membership requires sakura-tagger[parquet]') from error
        table = pq.read_table(path)
        fields = table.schema.names
        if len(set(fields)) != len(fields) or not _FIELDS <= set(fields) or set(fields) - (_FIELDS | _OPTIONAL_FIELDS):
            raise ValueError('Invalid source membership Parquet columns')
        for name in fields:
            kind = table.schema.field(name).type
            valid = pa.types.is_integer(kind) if name == 'class_id' else pa.types.is_string(kind) or pa.types.is_large_string(kind)
            if not valid:
                raise ValueError(f'Invalid Parquet type for {name}: {kind}')
        metadata = table.schema.metadata or {}
        try:
            binding = {name: metadata[name.encode()].decode('utf-8')
                       for name in ('schema_version', 'vocab_version', 'vocab_sha256')}
        except (KeyError, UnicodeDecodeError) as error:
            raise ValueError('Parquet requires UTF-8 schema_version, vocab_version, vocab_sha256 metadata') from error
        return cls.from_records(table.to_pylist(), vocabulary, **binding)

    def normalize_sources(self, source_filter):
        sources = [source_filter] if isinstance(source_filter, str) else source_filter
        if not isinstance(sources, (list, tuple)) or not sources or any(not isinstance(s, str) for s in sources):
            raise ValueError('source_filter must be a source string or a nonempty list of source strings')
        unknown = set(sources) - set(self.sources)
        if unknown:
            raise ValueError(f'Unknown membership sources: {sorted(unknown)}')
        return tuple(sorted(set(sources)))

    def allowed_ids(self, domain, source_filter):
        """Sorted original IDs for a source or deduplicated source union."""
        self._domain(domain)
        sources = self.normalize_sources(source_filter)
        if len(sources) == 1:
            return self._allowed.get((domain, sources[0]), ())
        return tuple(sorted({i for source in sources for i in self._allowed.get((domain, source), ())}))

    def _domain(self, domain):
        if domain not in self._names:
            raise ValueError(f'Unknown membership domain: {domain!r}')

    def class_name(self, domain, class_id):
        """Lookup without merging equal names across domains."""
        self._domain(domain)
        if type(class_id) is not int or not 0 <= class_id < len(self._names[domain]):
            raise ValueError(f'{domain}: invalid class_id')
        return self._names[domain][class_id]

    def validate_model(self, model):
        """Validate dimensions and the original Artist identity when binding."""
        if any(len(self._names[name]) != getattr(model.layout, name) for name in LOGICAL_DOMAINS):
            raise ValueError('Source membership vocabulary disagrees with model layout')
        artist = self._names['artist']
        if len(artist) != model.backbone.artist_classes or dict(enumerate(artist)) != getattr(model.backbone.legacy, 'class_mapping', None):
            raise ValueError('Source membership Original Artist vocabulary disagrees with loaded Kaloscope')
