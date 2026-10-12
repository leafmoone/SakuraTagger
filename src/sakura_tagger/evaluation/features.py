"""Checkpoint-bound compact task features and bounded terminal classifier blocks."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from ..model import artist_classifier_input
from ..training.window_checkpoint import state_digest

VERSION = 'sakura-tagger-task-features-v1'


def feature_binding(model, cohort, checkpoint_id, preprocessing):
    if not isinstance(checkpoint_id, str) or not checkpoint_id or not isinstance(preprocessing, dict) or not preprocessing:
        raise ValueError('Explicit checkpoint and preprocessing identities are required')
    return json.loads(json.dumps({'version': VERSION, 'checkpoint': checkpoint_id, 'model_sha256': state_digest(model),
            'cohort': cohort.identity, 'preprocessing': preprocessing, 'layout': model.layout.to_dict(),
            'other_artist_mode': model.other_artist_mode}))


class FeatureCache:
    def __init__(self, path, *, binding):
        self.path = Path(path)
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        if self.manifest.get('binding') != binding:
            raise ValueError('Feature cache checkpoint/cohort/preprocessing mismatch')
        self.binding = binding
        self.arrays = {}
        for name, shape in self.manifest['shapes'].items():
            if name not in ('general', 'identity', 'global', 'style'):
                raise ValueError('Unexpected cached feature domain')
            value = np.load(self.path / (name + '.npy'), mmap_mode='r', allow_pickle=False)
            if value.dtype != np.float32 or list(value.shape) != shape:
                raise ValueError('Invalid feature array shape/dtype')
            self.arrays[name] = value


@torch.no_grad()
def build_feature_cache(model, batches, cohort, path, *, checkpoint_id, preprocessing):
    cfg, n = cohort.config, len(cohort.records)
    dimensions = {'general': model.layout.general_dim, 'identity': model.layout.identity_dim,
                  'global': model.backbone.global_dim, 'style': model.backbone.style_dim}
    bytes_needed = 4 * n * sum(dimensions.values()) + 65536
    if bytes_needed > cfg.disk_bytes or 8 * cfg.row_batch * sum(dimensions.values()) > cfg.ram_bytes:
        raise ValueError('Feature cache exceeds declared disk/RAM budget')
    path = Path(path)
    if path.exists():
        raise ValueError('Feature cache output already exists; open with explicit binding instead')
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < bytes_needed:
        raise ValueError('Insufficient free disk for feature cache')
    binding = feature_binding(model, cohort, checkpoint_id, preprocessing)
    temporary = Path(tempfile.mkdtemp(prefix=path.name + '.partial.', dir=path.parent))
    arrays = {name: np.lib.format.open_memmap(temporary / (name + '.npy'), mode='w+', dtype=np.float32, shape=(n, width))
              for name, width in dimensions.items()}
    device, was_training, cursor = next(model.parameters()).device, model.training, 0
    model.eval()
    try:
        for batch in batches:
            images, ids = batch['images'], list(batch['sample_ids'])
            if (len(images) != len(ids) or not ids or len(ids) > cfg.row_batch
                    or ids != list(cohort.sample_ids[cursor:cursor + len(ids)])):
                raise ValueError('Feature batches must cover the fixed cohort exactly once in order, within row_batch')
            # One student backbone call; no full classification logits retained.
            bundle = model.backbone(images.to(device), include_artist=False, include_style=True)
            values = {'general': model.general.features(bundle.global_features, bundle.patch_tokens),
                      'identity': model.identity.features(bundle.global_features, bundle.patch_tokens),
                      'global': bundle.global_features, 'style': bundle.style_embedding}
            for name, value in values.items():
                value = value.float().cpu().numpy()
                if not np.isfinite(value).all():
                    raise ValueError('Nonfinite cached features')
                arrays[name][cursor:cursor + len(ids)] = value
            cursor += len(ids)
        if cursor != n:
            raise ValueError('Incomplete feature cohort')
        for value in arrays.values():
            value.flush()
        manifest = {'binding': binding, 'shapes': {name: [n, width] for name, width in dimensions.items()},
                    'feature_bytes': 4 * n * sum(dimensions.values()), 'sample_ids': list(cohort.sample_ids)}
        (temporary / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
        arrays.clear()
        temporary.rename(path)
    except BaseException:
        arrays.clear()
        shutil.rmtree(temporary)
        raise
    finally:
        model.train(was_training)
    return FeatureCache(path, binding=binding)


@torch.no_grad()
def classifier_blocks(model, cache, cohort, domain):
    """Yield all declared original IDs and untruncated [cohort,class-block] logits."""
    cfg = cohort.config
    n = len(cohort.records)
    if domain == 'general':
        features, classifier, offset = cache.arrays['general'], model.general.classifier, 0
    elif domain in ('character', 'copyright'):
        features, classifier = cache.arrays['identity'], model.identity.classifier
        offset = model.layout.character if domain == 'copyright' else 0
    else:
        features, offset = cache.arrays['global'], 0
        classifier = model.other_artist.classifier if domain == 'other_artist' else model.backbone.legacy.head
    if 64 * n * min(cfg.class_block, cohort.total_classes[domain]) + 8 * cfg.row_batch * features.shape[1] > cfg.ram_bytes:
        raise ValueError('Classifier blocks exceed RAM budget')
    device = classifier.weight.device
    wanted = set(cohort.class_ids[domain])
    for start in range(0, cohort.total_classes[domain], cfg.class_block):
        stop = min(start + cfg.class_block, cohort.total_classes[domain])
        ids = [value for value in range(start, stop) if value in wanted]
        if not ids:
            continue
        scores = np.empty((n, len(ids)), dtype=np.float32)
        selection = torch.tensor([value - start for value in ids], device=device)
        for row in range(0, n, cfg.row_batch):
            value = torch.from_numpy(np.array(features[row:row + cfg.row_batch], copy=True)).to(device)
            if domain in ('artist', 'other_artist'):
                value = artist_classifier_input(value)
            value = value.to(classifier.weight.dtype)
            logits = F.linear(value, classifier.weight[start + offset:stop + offset], classifier.bias[start + offset:stop + offset])
            scores[row:row + len(value)] = logits.index_select(1, selection).float().cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite classifier scores')
        yield ids, scores


def score_spool(model, cache, cohort, domain, path):
    """Bounded exact-small-cohort disk scores; callers unlink after this domain."""
    n, c, cfg = len(cohort.records), len(cohort.class_ids[domain]), cohort.config
    # Micro AP makes one true global ordering. Conservative allowance includes
    # labels, masks, concatenation, permutation and tie/cumsum sort temporaries.
    if 96 * n * c + 2048 * c + 64 * n * min(c, cfg.class_block) > cfg.ram_bytes:
        raise ValueError('Exact globally sorted Micro AP exceeds RAM budget; reduce fixed cohort/class scope')
    cache_bytes = cache.manifest['feature_bytes']
    if cache_bytes + n * c * 4 + 65536 > cfg.disk_bytes:
        raise ValueError('Features plus exact score spool exceed disk budget')
    if shutil.disk_usage(Path(path).parent).free < n * c * 4 + 65536:
        raise ValueError('Insufficient free disk for exact score spool')
    if c == 0:
        return np.empty((n, 0), dtype=np.float32)
    output = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n, c))
    column = 0
    for ids, scores in classifier_blocks(model, cache, cohort, domain):
        output[:, column:column + len(ids)] = scores
        column += len(ids)
    output.flush()
    return output
