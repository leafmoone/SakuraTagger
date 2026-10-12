"""Non-parametric output policy; no change to classifier ID spaces or logits."""
import math

import torch


def normalize_thresholds(thresholds, counts):
    def probability(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Thresholds must be finite probabilities in [0,1]')
        return float(value)
    if isinstance(thresholds, (int, float)):
        thresholds = {domain: thresholds for domain in counts}
    if not isinstance(thresholds, dict) or set(thresholds) - set(counts):
        raise ValueError('Thresholds must declare known classification domains')
    result = {}
    for domain, setting in thresholds.items():
        if isinstance(setting, (int, float)):
            result[domain] = {'default': probability(setting), 'per_class': {}}
            continue
        if not isinstance(setting, dict) or set(setting) != {'default', 'per_class'} or not isinstance(setting['per_class'], dict):
            raise ValueError('Per-class thresholds require default and per_class mappings')
        values = {}
        for key, value in setting['per_class'].items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise ValueError('Threshold class IDs must be original integer IDs')
            try:
                class_id = int(key)
            except ValueError as error:
                raise ValueError('Invalid threshold class ID') from error
            if str(class_id) != str(key) or not 0 <= class_id < counts[domain] or str(class_id) in values:
                raise ValueError('Threshold class ID is duplicate or out of range')
            values[str(class_id)] = probability(value)
        result[domain] = {'default': probability(setting['default']), 'per_class': values}
    return result


def threshold_output(probabilities, original_ids, setting, max_tags):
    limits = torch.tensor([setting['per_class'].get(str(int(class_id)), setting['default'])
                           for class_id in original_ids.tolist()], device=probabilities.device)
    all_ids, all_scores = [], []
    for row in probabilities:
        selected = torch.nonzero(row >= limits, as_tuple=False).flatten()
        # Original IDs arrive sorted. Stable sorting fixes equal-score order.
        order = torch.argsort(row[selected], descending=True, stable=True)
        selected = selected[order[:max_tags] if max_tags is not None else order]
        all_ids.append(original_ids[selected]); all_scores.append(row[selected])
    return {'class_ids': all_ids, 'scores': all_scores}
