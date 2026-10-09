"""Repeat-factor sampling statistics from the final deduplicated training split."""
from collections import Counter
import math

from .contract import validate_record
from ..model.registry import LOGICAL_DOMAINS


def repeat_factors(records, vocabulary, *, threshold=0.001, max_repeat_factor=3.0):
    if not math.isfinite(threshold) or threshold <= 0 or not math.isfinite(max_repeat_factor) or max_repeat_factor < 1:
        raise ValueError('RFS requires positive threshold and max_repeat_factor >= 1')
    training = [record for record in records if record.get('split') == 'train']
    seen_ids, seen_hashes, frequencies = set(), set(), Counter()
    for record in training:
        validate_record(record, vocabulary)
        if record['image_id'] in seen_ids or record['image_sha256'] in seen_hashes:
            raise ValueError('RFS frequencies require the final deduplicated training set')
        seen_ids.add(record['image_id'])
        seen_hashes.add(record['image_sha256'])
        frequencies.update((domain, label) for domain in LOGICAL_DOMAINS
                           for label in record['positive_labels'].get(domain, []))
    fractions = {key: count / len(training) for key, count in frequencies.items()}
    factors = [max((min(max_repeat_factor, max(1.0, math.sqrt(threshold / fractions[(domain, label)])))
                    for domain in LOGICAL_DOMAINS for label in record['positive_labels'].get(domain, [])), default=1.0)
               for record in training]
    return factors, fractions
