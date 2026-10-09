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
    def __init__(self, backbone, layout: HeadLayout):
        super().__init__()
        self.backbone, self.layout = backbone, layout
        self.general = GeneralModule(backbone.global_dim, backbone.patch_dim, layout)
        self.identity = IdentityModule(backbone.global_dim, layout)
        self.other_artist = OtherArtistModule(backbone.global_dim, layout.other_artist)

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
    def predict(self, images, head_selection=None, *, top_k=20, dense=False):
        """Top-K IDs/scores by default. Artist domains are never merged/ranked together.

        Original Artist retains softmax semantics; the four new domains use
        independent sigmoid probabilities. Call eval() for inference.
        """
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError('top_k must be a positive integer')
        output = self(images, head_selection)
        result = {}
        for name, logits in output.logits.items():
            probabilities = logits.softmax(-1) if name == 'artist' else logits.sigmoid()
            if dense:
                result[name] = probabilities
            else:
                scores, ids = probabilities.topk(min(top_k, probabilities.shape[-1]), dim=-1)
                result[name] = {'class_ids': ids, 'scores': scores}
        if output.features.style_embedding is not None:
            result['style_embedding'] = output.features.style_embedding
        return result

    def parameter_counts(self):
        return {name: {'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad),
                       'frozen': sum(p.numel() for p in module.parameters() if not p.requires_grad)}
                for name, module in self.named_children()}
