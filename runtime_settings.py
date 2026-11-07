"""Runtime selection and checkpoint validation without importing GPU libraries."""
from pathlib import Path

from checkpoints import checkpoint_metadata
from project_data import ROOT, read_json

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def read_runtime(path=None, **overrides):
    config = dict(path) if isinstance(path, dict) else read_json(path or ROOT / "configs/runtime.json")
    config.update({k:v for k,v in overrides.items() if v is not None})
    checkpoint = config.get("selector_checkpoint")
    if checkpoint:
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        config["selector_checkpoint"] = str(checkpoint.resolve())
        metadata = checkpoint_metadata(checkpoint)
        if metadata.get("training_updates", 0) < 1:
            raise ValueError("This checkpoint has no training updates; use identity mode without a checkpoint")
        if config.get("model_path", DEFAULT_MODEL) == DEFAULT_MODEL:
            config["model_path"] = metadata["backbone"]["model_path"]
        config["revision"] = metadata["backbone"].get("resolved_revision") or metadata["backbone"].get("revision")
        config.update(metadata["preprocessing"])
        config["precision"] = metadata.get("precision", config["precision"])
        config["quantization"] = metadata.get("quantization", config["quantization"])
        if config.get("rho") is None:
            config["rho"] = metadata["target_rho"]
    if config.get("gate_mode", "auto") == "auto":
        config["gate_mode"] = "soft" if checkpoint else "identity"
    if config["gate_mode"] not in {"identity", "soft", "hard"}:
        raise ValueError("Runtime gate_mode must be auto, identity, soft or hard")
    if not checkpoint and config["gate_mode"] != "identity":
        raise ValueError("Untrained UI inference must use identity mode")
    if config.get("rho") is None:
        config["rho"] = 1.0
    if not 0 <= config["rho"] <= 1:
        raise ValueError("rho must be in [0,1]")
    if not 0 < config["image_min_tokens"] <= config["image_max_tokens"]:
        raise ValueError("Invalid image token bounds")
    return config


def runtime_status(config):
    checkpoint = config.get("selector_checkpoint")
    return {"mode":config["gate_mode"], "checkpoint":checkpoint, "rho":config["rho"],
            "label": "Trained visual selector" if checkpoint and config["gate_mode"] != "identity"
            else "Untrained — original visual features" if not checkpoint else "Original visual features — selector bypassed"}
