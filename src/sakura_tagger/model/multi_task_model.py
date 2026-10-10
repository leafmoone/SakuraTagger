"""Three physical modules, six selectable public output domains."""
from dataclasses import dataclass

import torch
from torch import nn

from .feature_bundle import FeatureBundle
from .modules import GeneralModule, IdentityModule, OtherArtistModule
from .registry import HeadLayout, OUTPUT_NAMES


@dataclass(frozen=True)
class MultiTaskOutput:
    features: FeatureBundle
    logits: dict[str, torch.Tensor]


class MultiTaskModel(nn.Module):
    def __init__(self, backbone, layout: HeadLayout, *, other_artist_mode="multilabel"):
        super().__init__()
        self.backbone, self.layout = backbone, layout
        self._source_membership = None
        self.set_other_artist_mode(other_artist_mode)
        self.general = GeneralModule(backbone.global_dim, backbone.patch_dim, layout)
        self.identity = IdentityModule(backbone.global_dim, layout)
        self.other_artist = OtherArtistModule(backbone.global_dim, layout.other_artist)

    def set_other_artist_mode(self, mode):
        """Match the effective LossConfig; this is non-parametric inference state."""
        if mode not in ('multilabel', 'single_label'):
            raise ValueError('Other Artist mode must be multilabel or single_label')
        self.other_artist_mode = mode
        return self

    def set_source_membership(self, membership):
        """Bind a validated vocabulary sidecar once, or detach it with None."""
        from ..data.source_membership import SourceMembership
        if membership is not None:
            if not isinstance(membership, SourceMembership):
                raise TypeError('Expected a validated SourceMembership or None')
            membership.validate_model(self)
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
            character, copyright = self.identity(features.global_features)
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
    def predict(self, images, head_selection=None, *, top_k=20, dense=False, source_filter=None):
        """Full-vocabulary probabilities, optionally restricted before Top-K.

        Without filtering, dense results remain [B,C] tensors. Filtered dense
        results are {class_ids: [B,M], scores: [B,M]}, ordered by original ID.
        Original Artist and single_label Other Artist use full softmax; other
        outputs use independent sigmoids. Artist domains are never merged.
        Call eval() for inference. Style is returned unchanged.
        """
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        selected = self._selection(head_selection)
        candidates = None
        if source_filter is not None:
            if self._source_membership is None:
                raise ValueError('source_filter requires a validated SourceMembership')
            sources = self._source_membership.normalize_sources(source_filter)
            candidates = {name: self._source_membership.allowed_ids(name, sources)
                          for name in selected if name != 'style_embedding'}
        output = self(images, selected)
        result = {}
        for name, logits in output.logits.items():
            single_label = name == 'artist' or name == 'other_artist' and self.other_artist_mode == 'single_label'
            probabilities = logits.softmax(-1) if single_label else logits.sigmoid()
            original_ids = None
            if candidates is not None:
                original_ids = torch.tensor(candidates[name], dtype=torch.long, device=logits.device)
                probabilities = probabilities.index_select(-1, original_ids)
            if dense and candidates is None:
                result[name] = probabilities
            elif dense:
                result[name] = {'class_ids': original_ids.expand(probabilities.shape[0], -1),
                                'scores': probabilities}
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
