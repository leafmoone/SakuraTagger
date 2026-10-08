# Upstream provenance and license boundaries

## Pinned source

- Repository: https://github.com/spawner1145/comfyui-kaloscope
- Submodule: `third_party/comfyui-kaloscope/`
- Commit: `1f1f12d4103fd10c6f890ecf197a0e280d1979d5`
- `model_loading.py` supplies the standalone DINO model construction, component
  loading, pooling, artist normalization, and style projector behavior.
- `kaloscope_dinov3/` supplies the original DINOv3 backbone. It is reused in place,
  not copied or replaced by another backbone. No submodule sources are edited.
- The plugin entry point and ComfyUI modules are not imported. Upstream's image
  center crop is not used by the SakuraTagger product input path.

## Official checkpoint

- Repository: https://huggingface.co/heathcliff01/Kaloscope3.0-preview
- Observed revision: `5e4bfa229619b7c459ec126283827373df9de564`
- Directory: `v1-artist-classifier/`
- `model.safetensors`: trained backbone, artist head, style projector, and
  auxiliary temperature/bias tensors. All are required and strictly loaded.
- `config.json`: architecture, feature, normalization, and class-count settings.
- `class_mapping.csv`: contiguous artist class ID/name mapping.
- Downloaded weights and verification results belong in ignored `checkpoints/`
  and `reports/`, never the source repository. Synthetic tests do not certify
  official checkpoint compatibility or classification quality.

## Licenses

The upstream plugin root contains [GNU GPL version 3](https://github.com/spawner1145/comfyui-kaloscope/blob/1f1f12d4103fd10c6f890ecf197a0e280d1979d5/LICENSE). The embedded
DINOv3 source has its own DINOv3 License Agreement in
`kaloscope_dinov3/LICENSE.md`, dated August 19, 2025, and retains its notices.
The checkpoint repository declares the DINOv3 license via its model card and
[LICENSE.md](https://huggingface.co/heathcliff01/Kaloscope3.0-preview/blob/5e4bfa229619b7c459ec126283827373df9de564/LICENSE.md).

These are distinct boundaries: the plugin license does not relicense Meta's
DINO materials, and checkpoint availability does not grant unrestricted use.
Review the original agreements, including redistribution and notice obligations,
before distributing code or weights. No claim of automatic compatibility or a
unified MIT/Apache license is made here. Use of DINO materials is subject to the
[DINOv3 agreement](https://github.com/spawner1145/comfyui-kaloscope/blob/1f1f12d4103fd10c6f890ecf197a0e280d1979d5/kaloscope_dinov3/LICENSE.md).
