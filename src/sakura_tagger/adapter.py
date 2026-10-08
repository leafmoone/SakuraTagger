"""Strict safe-tensor Kaloscope loading and explicitly synthetic CPU fixtures."""
import csv
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file

from .upstream import model_loading

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_keys(state):
    result = {}
    for original, value in state.items():
        key = original
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key.startswith("linear_head."):
            key = "head." + key.removeprefix("linear_head.")
        if key in result:
            raise ValueError(f"Checkpoint key collision after normalization: {original} -> {key}")
        result[key] = value
    return result


def read_mapping(path, count):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"class_id", "class_name"} <= set(reader.fieldnames):
            raise ValueError("class_mapping.csv requires class_id and class_name")
        rows = list(reader)
    mapping = {}
    for row in rows:
        index = int(row["class_id"])
        if index in mapping:
            raise ValueError(f"Duplicate class_id: {index}")
        mapping[index] = row["class_name"]
    if set(mapping) != set(range(count)):
        raise ValueError(f"Class mapping must contain exactly contiguous IDs 0..{count - 1}")
    return mapping


def model_options(config):
    selection = config.get("model")
    if isinstance(selection, dict):
        options = dict(selection)
    elif isinstance(selection, str):
        options = {key: config[key] for key in ("kwargs", "pooling", "feature_source", "classifier_input_normalization") if key in config}
        options["name"] = selection
    else:
        raise ValueError("config.json requires model name or model object")
    name = options.get("name")
    if not isinstance(name, str) or not (name.startswith("dinov3_vit") or name == "custom_vit"):
        raise ValueError("SakuraTagger requires a DINOv3 ViT architecture")
    return options


def load_state_strict(state, options):
    """Load every tensor; reject missing, unexpected, or colliding keys."""
    if not (options["name"].startswith("dinov3_vit") or options["name"] == "custom_vit"):
        raise ValueError("SakuraTagger requires a DINOv3 ViT architecture")
    state = normalize_keys(state)
    if not any(key.startswith("backbone.") for key in state):
        raise ValueError("Expected complete Kaloscope state with backbone.*, head.*, projector.*")
    for key in ("head.weight", "projector.0.weight", "projector.2.weight", "log_temperature", "bias"):
        if key not in state:
            raise ValueError(f"Missing checkpoint key: {key}")
    model, _ = model_loading()._load_dino(options["name"], state, options)
    for key in ("log_temperature", "bias"):
        model.register_buffer(key, torch.empty_like(state[key]))
    # A second strict pass checks the *entire* original state, including keys
    # that upstream component extraction otherwise silently discards.
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    model.register_forward_pre_hook(_validate_model_input, with_kwargs=True)
    return model


def load_kaloscope(model_dir, *, device="cpu"):
    """Load the real v1 artist package; never fall back to random weights."""
    directory = Path(model_dir)
    for name in ("model.safetensors", "config.json", "class_mapping.csv"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    config = model_loading().read_model_config(directory)
    options = model_options(config)
    if options["name"] != "dinov3_vitb16":
        raise ValueError("Official v1 requires dinov3_vitb16")
    required = {"pooling": "cls_mean", "feature_source": "projector", "classifier_input_normalization": "l2_sqrt_dim"}
    for key, value in required.items():
        if options.get(key) != value:
            raise ValueError(f"Official v1 requires {key}={value!r}")
    model = load_state_strict(load_file(str(directory / "model.safetensors"), device="cpu"), options)
    if (model.backbone.embed_dim, model.pooled_dim, model.feature_dim, model.head.out_features) != (768, 1536, 256, 44129):
        raise ValueError("Official v1 dimensions must be 768/1536/256/44129")
    declared_count = config.get("num_classes", options.get("num_classes", 44129))
    if declared_count != 44129:
        raise ValueError("Configured class count disagrees with official v1")
    model.class_mapping = read_mapping(directory / "class_mapping.csv", 44129)
    model.synthetic = False
    model.package_config = config
    return model.to(device=device)



def validate_images(images, patch_size=16):
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected images shaped [B,3,H,W]")
    if not images.is_floating_point():
        raise ValueError("Images must be floating-point ImageNet-normalized tensors")
    if images.shape[0] < 1 or any(size < patch_size or size % patch_size for size in images.shape[-2:]):
        raise ValueError(f"Image H/W must be positive multiples of patch size {patch_size}; crop/padding is prohibited")


def _validate_model_input(model, args, kwargs):
    images = args[0] if args else kwargs.get("images")
    validate_images(images, model.backbone.patch_size)
