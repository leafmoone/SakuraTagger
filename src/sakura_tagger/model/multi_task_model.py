"""Three physical modules, six selectable public output domains."""
from dataclasses import dataclass
import re

import torch
from torch import nn

from .feature_bundle import FeatureBundle
from .modules import GeneralModule, IdentityModule, OtherArtistModule
from .registry import HeadLayout, OUTPUT_NAMES, LOGICAL_DOMAINS
from .prediction import normalize_thresholds, threshold_output


@dataclass(frozen=True)
class MultiTaskOutput:
    features: FeatureBundle
    logits: dict[str, torch.Tensor]


class MultiTaskModel(nn.Module):
    def __init__(self, backbone, layout: HeadLayout, *, other_artist_mode="multilabel"):
        super().__init__()
        self.backbone, self.layout = backbone, layout
        self._source_membership = None
        self._vocabulary_sha256 = None
        self._untrained_labels = None
        self.set_other_artist_mode(other_artist_mode)
        self.general = GeneralModule(backbone.global_dim, backbone.patch_dim, layout)
        self.identity = IdentityModule(backbone.global_dim, layout, backbone.patch_dim)
        self.other_artist = OtherArtistModule(backbone.global_dim, layout.other_artist)

    def set_other_artist_mode(self, mode):
        """Match the effective LossConfig; this is non-parametric inference state."""
        if mode not in ('multilabel', 'single_label'):
            raise ValueError('Other Artist mode must be multilabel or single_label')
        self.other_artist_mode = mode
        return self

    def bind_vocabulary(self, vocabulary):
        """Bind the exact class names/order, not merely matching output dimensions."""
        vocabulary.validate_layout(self.layout)
        expected = {entry['class_id']: entry['name'] for entry in vocabulary.document['original_artist']}
        if expected != self.backbone.legacy.class_mapping:
            raise ValueError('Original Artist vocabulary identity mismatch')
        if self._vocabulary_sha256 not in (None, vocabulary.sha256):
            raise ValueError('Model weights already bind a different exact Vocabulary')
        if self._source_membership is not None and self._source_membership.vocab_sha256 != vocabulary.sha256:
            raise ValueError('Source membership uses a different exact Vocabulary')
        self._vocabulary_sha256 = vocabulary.sha256
        return self

    def set_untrained_labels(self, metadata):
        """Bind explicit new-head NO_TRAIN_POSITIVE IDs; never suppress old Artist."""
        if metadata is None:
            self._untrained_labels = None
            return self
        from ..data.contract import _ids
        if (metadata.get('version') != 'sakura-tagger-untrained-labels-v1'
                or self._vocabulary_sha256 is None or metadata.get('vocab_sha256') != self._vocabulary_sha256
                or not isinstance(metadata.get('release'), dict)
                or metadata['release'].get('vocab_sha256') != self._vocabulary_sha256
                or not isinstance(metadata['release'].get('release_sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', metadata['release']['release_sha256'])
                or metadata.get('reason') != 'NO_TRAIN_POSITIVE'
                or set(metadata.get('ids', {})) != set(LOGICAL_DOMAINS)):
            raise ValueError('Untrained-label metadata requires exact Vocabulary/Release and four new domains')
        ids = {domain: _ids(metadata['ids'][domain], getattr(self.layout, domain), domain) for domain in LOGICAL_DOMAINS}
        self._untrained_labels = {**metadata, 'ids': {name: tuple(sorted(values)) for name, values in ids.items()}}
        return self

    def set_source_membership(self, membership):
        """Bind a validated vocabulary sidecar once, or detach it with None."""
        from ..data.source_membership import SourceMembership
        if membership is not None:
            if not isinstance(membership, SourceMembership):
                raise TypeError('Expected a validated SourceMembership or None')
            membership.validate_model(self)
            if self._vocabulary_sha256 is not None and membership.vocab_sha256 != self._vocabulary_sha256:
                raise ValueError('Source membership exact Vocabulary mismatch')
        self._source_membership = membership
        return self

    def set_stage(self, stage):
        self.backbone.set_stage(stage)
        return self

    def _selection(self, head_selection):
        selected = set(OUTPUT_NAMES if head_selection is None else head_selection)
        unknown = selected - set(OUTPUT_NAMES)
        if unknown:
            raise ValueError(f'Unknown outputs: {sorted(unknown)}')
        return selected

    def forward(self, images, head_selection=None):
        selected = self._selection(head_selection)
        features = self.backbone.forward_features(images, include_artist='artist' in selected,
                                                   include_style='style_embedding' in selected)
        logits = {}
        if 'general' in selected:
            logits['general'] = self.general(features.global_features, features.patch_tokens)
        if {'character', 'copyright'} & selected:
            character, copyright = self.identity(features.global_features, features.patch_tokens)
            if 'character' in selected:
                logits['character'] = character
            if 'copyright' in selected:
                logits['copyright'] = copyright
        if 'other_artist' in selected:
            logits['other_artist'] = self.other_artist(features.global_features)
        if 'artist' in selected:
            logits['artist'] = features.artist_logits
        return MultiTaskOutput(features, logits)

    @torch.no_grad()
    def predict(self, images, head_selection=None, *, top_k=20, dense=False, source_filter=None,
                output_mode=None, thresholds=None, max_tags=None, suppress_untrained=True, class_scope=None):
        """Full-vocabulary probabilities, optionally restricted before Top-K.

        Without filtering, dense results remain [B,C] tensors. Filtered dense
        results are {class_ids: [B,M], scores: [B,M]}, ordered by original ID.
        Original Artist and single_label Other Artist use full softmax; other
        outputs use independent sigmoids. Artist domains are never merged.
        Threshold mode returns lists of IDs/scores per image (ragged output);
        thresholds are explicit and source filtering precedes the optional cap.
        Bound NO_TRAIN_POSITIVE metadata affects only new heads and can be
        explicitly disabled for diagnostics. Unbound old models are unchanged.
        Call eval() for inference. Style is returned unchanged.
        """
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        mode = ('dense' if dense else 'top_k') if output_mode is None else output_mode
        if mode not in ('top_k', 'threshold', 'dense') or dense and mode != 'dense':
            raise ValueError('Output mode must be top_k, threshold or dense without conflicting dense=True')
        if max_tags is not None and (type(max_tags) is not int or max_tags < 1 or mode != 'threshold'):
            raise ValueError('max_tags is a positive optional threshold-output cap')
        selected = self._selection(head_selection)
        counts = {name: getattr(self.layout, name) for name in LOGICAL_DOMAINS}
        counts['artist'] = self.backbone.artist_classes
        settings = normalize_thresholds(thresholds, counts) if mode == 'threshold' else None
        if settings is not None and (selected - {'style_embedding'}) - set(settings):
            raise ValueError('Thresholds must explicitly cover every requested classifier')
        candidates = None
        if source_filter is not None:
            if self._source_membership is None:
                raise ValueError('source_filter requires a validated SourceMembership')
            sources = self._source_membership.normalize_sources(source_filter)
            candidates = {name: self._source_membership.allowed_ids(name, sources)
                          for name in selected if name != 'style_embedding'}
        if class_scope is not None:
            from ..data.contract import _ids
            if not isinstance(class_scope, dict) or set(class_scope) - set(counts):
                raise ValueError('Class scope must declare known classification domains')
            scope = {name: set(_ids(list(ids), counts[name], name)) for name, ids in class_scope.items()}
            candidates = {name: tuple(class_id for class_id in (candidates[name] if candidates is not None else range(counts[name]))
                                     if name not in scope or class_id in scope[name])
                          for name in selected if name != 'style_embedding'}
        output = self(images, selected)
        result = {}
        for name, logits in output.logits.items():
            single_label = name == 'artist' or name == 'other_artist' and self.other_artist_mode == 'single_label'
            probabilities = logits.softmax(-1) if single_label else logits.sigmoid()
            original_ids = None
            excluded = () if not suppress_untrained or self._untrained_labels is None or name == 'artist' else self._untrained_labels['ids'][name]
            if candidates is not None:
                ids = candidates[name]
                if excluded:
                    blocked = set(excluded)
                    ids = [class_id for class_id in ids if class_id not in blocked]
                original_ids = torch.tensor(ids, dtype=torch.long, device=logits.device)
                probabilities = probabilities.index_select(-1, original_ids)
            elif excluded and mode != 'dense':
                blocked = set(excluded)
                original_ids = torch.tensor([class_id for class_id in range(logits.shape[-1]) if class_id not in blocked],
                                            dtype=torch.long, device=logits.device)
                probabilities = probabilities.index_select(-1, original_ids)
            if mode == 'dense' and candidates is None:
                # Preserve [B,C] and original ID positions. An explicit metadata
                # binding suppresses unsupported new labels with zero scores.
                if excluded:
                    probabilities[:, list(excluded)] = 0
                result[name] = probabilities
            elif mode == 'dense':
                result[name] = {'class_ids': original_ids.expand(probabilities.shape[0], -1), 'scores': probabilities}
            elif mode == 'threshold':
                if original_ids is None:
                    original_ids = torch.arange(logits.shape[-1], device=logits.device)
                result[name] = threshold_output(probabilities, original_ids, settings[name], max_tags)
            else:
                scores, ids = probabilities.topk(min(top_k, probabilities.shape[-1]), dim=-1)
                if original_ids is not None:
                    ids = original_ids[ids]
                result[name] = {'class_ids': ids, 'scores': scores}
        if output.features.style_embedding is not None:
            result['style_embedding'] = output.features.style_embedding
        return result

    def parameter_counts(self):
        return {name: {'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad),
                       'frozen': sum(p.numel() for p in module.parameters() if not p.requires_grad)}
                for name, module in self.named_children()}
