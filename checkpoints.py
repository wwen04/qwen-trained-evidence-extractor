"""Selector-only checkpoints; safe tensor weights plus inspectable provenance."""
from pathlib import Path

from project_data import digest, file_hash, read_json, write_json

SCHEMA = "qwen-evidence-selector-v1"


def backbone_signature(config):
    text = config.text_config
    vision = config.vision_config
    return {"model_type": config.model_type,
            "text": {k: getattr(text, k) for k in ("hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads")},
            "vision": {k: getattr(vision, k) for k in ("hidden_size", "out_hidden_size", "depth", "patch_size", "spatial_merge_size")},
            "image_token_id": config.image_token_id}


def checkpoint_metadata(folder):
    folder = Path(folder).expanduser().resolve()
    metadata = read_json(folder / "metadata.json")
    if metadata.get("schema") != SCHEMA:
        raise ValueError("Unsupported selector checkpoint")
    if file_hash(folder / "selector.safetensors") != metadata["weights_sha256"]:
        raise ValueError("Selector weights do not match checkpoint metadata")
    return metadata


def save_checkpoint(folder, selector, provenance):
    from safetensors.torch import save_file
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "selector.safetensors.partial"
    save_file({k: v.detach().cpu().contiguous() for k, v in selector.state_dict().items()}, str(temporary))
    temporary.replace(folder / "selector.safetensors")
    metadata = {**provenance, "schema": SCHEMA, "selector_config": selector.config.to_dict(),
                "weights_sha256": file_hash(folder / "selector.safetensors")}
    write_json(folder / "metadata.json", metadata)
    return metadata


def load_checkpoint(folder, selector, backbone_config):
    from safetensors.torch import load_file
    metadata = checkpoint_metadata(folder)
    if metadata["backbone_signature"] != backbone_signature(backbone_config):
        raise ValueError("Selector was trained for a different Qwen architecture")
    if metadata["selector_config"] != selector.config.to_dict():
        raise ValueError("Selector architecture differs from checkpoint")
    if metadata.get("conditioning") != "public-question-context-options-v1":
        raise ValueError("Checkpoint conditioning protocol differs")
    selector.load_state_dict(load_file(str(Path(folder) / "selector.safetensors")), strict=True)
    if not bool(selector.null_initialized):
        raise ValueError("Checkpoint null token was never initialized")
    return metadata
