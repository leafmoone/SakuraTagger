"""Exact whole-cohort metrics with explicit incomplete-label protocols."""
import numpy as np
import torch

PROTOCOLS = ('observed_label_proxy', 'verified_scope')
BANDS = ('0', '1', '2-9', '10-99', '100-999', '1000+')


def exact_average_precision(labels, scores):
    """Non-interpolated AP over distinct score thresholds, including ties."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('AP requires aligned finite one-dimensional scores/labels')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('AP labels must be binary')
    positives = int(labels.sum())
    if not positives or positives == len(labels):
        raise ValueError('AP requires positive and comparable negative evidence')
    order = np.argsort(-scores, kind='stable')
    ordered = scores[order]
    true_positives = np.cumsum(labels[order], dtype=np.int64)
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]
    tp = true_positives[ends]
    precision = tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0, tp]) * precision) / positives)


def merge_score_partitions(partitions, *, expected_ids):
    """Merge raw scores by fixed cohort IDs before computing any global metric."""
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Expected cohort IDs must be unique')
    rows = {}
    for ids, scores in partitions:
        scores = np.asarray(scores)
        if scores.ndim != 2 or len(ids) != len(scores):
            raise ValueError('Invalid score partition')
        for identity, score in zip(ids, scores):
            if identity in rows:
                raise ValueError('Duplicate partition sample')
            rows[identity] = score
    if set(rows) != set(expected_ids):
        raise ValueError('Score partitions must exactly cover the declared cohort')
    return np.stack([rows[identity] for identity in expected_ids])


def labels_for_class(records, domain, class_id, protocol):
    if protocol not in PROTOCOLS:
        raise ValueError('Unknown evaluation protocol')
    positives = np.asarray([class_id in (row.get(domain + '_ids') or []) for row in records], dtype=bool)
    negatives = np.asarray([class_id in (row.get(domain + '_negative_ids') or []) for row in records], dtype=bool)
    if np.any(positives & negatives):
        raise ValueError('Conflicting positive/negative evaluation evidence')
    valid = np.ones(len(records), dtype=bool) if protocol == 'observed_label_proxy' else positives | negatives
    return positives, valid


def frequency_band(count):
    return '0' if count == 0 else '1' if count == 1 else '2-9' if count < 10 else '10-99' if count < 100 else '100-999' if count < 1000 else '1000+'


def multilabel_metrics(scores, class_ids, records, domain, train_support, *, protocol,
                       total_classes, min_positives=1, min_negatives=1, threshold=.5):
    """Scores are untruncated logits; micro globally orders all available pairs.

    The explicit micro class scope is the label-determined available class set.
    The same fixed cohort/protocol/support settings keep it checkpoint-invariant.
    """
    if scores.shape != (len(records), len(class_ids)):
        raise ValueError('Scores must cover the complete cohort and declared class scope')
    per_class, micro_y, micro_z = {}, [], []
    bands = {name: {'ap': [], 'tp': 0, 'positives': 0, 'classes': 0} for name in BANDS}
    for column, class_id in enumerate(class_ids):
        positive, known = labels_for_class(records, domain, int(class_id), protocol)
        y, z = positive[known], np.asarray(scores[:, column])[known]
        if not np.isfinite(z).all():
            raise ValueError('Nonfinite evaluated scores')
        p, n = int(y.sum()), int((~y).sum())
        reason = 'NO_POSITIVE' if not p else 'LOW_POSITIVE_SUPPORT' if p < min_positives else 'NO_COMPARABLE_NEGATIVE' if not n else 'LOW_NEGATIVE_SUPPORT' if n < min_negatives else None
        band = frequency_band(int(train_support[class_id]))
        value = {'positive_support': p, 'negative_support': n, 'known_pairs': len(y),
                 'train_support': int(train_support[class_id]), 'train_frequency_band': band,
                 'status': 'UNAVAILABLE' if reason else 'AVAILABLE', 'reason': reason, 'ap': None}
        # Recall is meaningful for observed positives, even without AP negatives.
        limit = threshold.get(int(class_id), .5) if isinstance(threshold, dict) else threshold
        if not 0 <= limit <= 1:
            raise ValueError('Recall threshold must be a probability')
        probabilities = torch.from_numpy(np.array(z, dtype=np.float32)).sigmoid().numpy()
        tp = int(np.sum((probabilities >= limit) & y))
        value['recall'] = tp / p if p else None
        value['recall_threshold'] = float(limit)
        bands[band]['tp'] += tp; bands[band]['positives'] += p
        if reason is None:
            value['ap'] = exact_average_precision(y, z)
            micro_y.append(y); micro_z.append(z)
            bands[band]['ap'].append(value['ap']); bands[band]['classes'] += 1
        per_class[str(class_id)] = value
    available = [int(key) for key, value in per_class.items() if value['status'] == 'AVAILABLE']
    macro = float(np.mean([per_class[str(key)]['ap'] for key in available])) if available else None
    micro = exact_average_precision(np.concatenate(micro_y), np.concatenate(micro_z)) if available else None
    return {'protocol': protocol, 'metric_name': 'observed-label AP' if protocol == 'observed_label_proxy' else 'verified-scope AP',
            'score_basis': 'complete raw logits, globally ordered; never truncated Top-K',
            'macro_ap': macro, 'micro_ap': micro, 'micro_scope': 'available_classes_all_protocol_known_pairs',
            'micro_class_ids': available, 'evaluated_classes': len(available), 'declared_classes': len(class_ids),
            'total_classes': total_classes, 'class_ids': list(map(int, class_ids)), 'per_class': per_class,
            'frequency_basis': 'independent unique published train samples before RFS',
            'frequency_bands': {key: {'macro_ap': float(np.mean(value['ap'])) if value['ap'] else None,
                                      'recall': value['tp'] / value['positives'] if value['positives'] else None,
                                      'available_ap_classes': value['classes'], 'positive_labels': value['positives']}
                                for key, value in bands.items()}}


def artist_metrics(top_ids, targets, *, total_classes):
    targets = np.asarray(targets, dtype=np.int64)
    if top_ids.ndim != 2 or len(top_ids) != len(targets):
        raise ValueError('Artist prediction and target counts differ')
    valid = targets >= 0
    if not valid.any():
        return {'status': 'UNAVAILABLE', 'reason': 'NO_TRUSTED_SINGLE_ARTIST', 'valid_samples': 0,
                'top1': None, 'top5': None, 'macro_top1': None, 'total_classes': total_classes}
    ids, truth = top_ids[valid], targets[valid]
    correct = ids[:, 0] == truth
    classes = np.unique(truth)
    return {'status': 'AVAILABLE', 'valid_samples': len(truth), 'top1': float(correct.mean()),
            'top5': float(np.any(ids[:, :5] == truth[:, None], axis=1).mean()),
            'macro_top1': float(np.mean([correct[truth == label].mean() for label in classes])),
            'evaluated_classes': len(classes), 'total_classes': total_classes}


def style_recall(query, query_ids, gallery=None, *, relevant_ids=None, ks=(1, 5), ram_bytes):
    if gallery is None:
        return {'status': 'NOT_VERIFIED', 'reason': 'NO_REFERENCE_GALLERY'}
    vectors, gallery_ids = gallery
    query, vectors = np.asarray(query), np.asarray(vectors)
    if (query.ndim != 2 or vectors.ndim != 2 or query.shape[1] != vectors.shape[1]
            or len(query_ids) != len(query) or len(gallery_ids) != len(vectors)
            or relevant_ids is None or len(relevant_ids) != len(query)):
        raise ValueError('Style retrieval requires aligned vectors, IDs and explicit relevance')
    if len(set(gallery_ids)) != len(gallery_ids) or not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError('Invalid gallery IDs or Recall@K')
    if 32 * len(vectors) + 8 * (query.size + vectors.size) > ram_bytes:
        raise ValueError('Style retrieval exceeds RAM budget')
    if not np.isfinite(query).all() or not np.isfinite(vectors).all():
        raise ValueError('Style vectors must be finite')
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    hits, count = {k: 0. for k in ks}, 0
    for q, identity, relevant in zip(query, query_ids, relevant_ids):
        eligible = np.asarray([value != identity for value in gallery_ids])
        truth = set(relevant) & {value for value, allowed in zip(gallery_ids, eligible) if allowed}
        if not truth:
            continue
        scores = vectors @ q
        scores[~eligible] = -np.inf
        order = np.argsort(-scores, kind='stable')[:int(eligible.sum())]
        for k in ks:
            hits[k] += len(truth & {gallery_ids[i] for i in order[:k]}) / len(truth)
        count += 1
    return {'status': 'AVAILABLE' if count else 'UNAVAILABLE', 'queries_with_gallery_positives': count,
            'metric': 'mean relevant-gallery recall; same sample ID excluded',
            'recall_at_k': {str(k): value / count if count else None for k, value in hits.items()}}
