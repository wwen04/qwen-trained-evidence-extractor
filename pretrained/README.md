# Bundled visual selector

`scienceqa-visual-1layer-seed42/` contains the one-layer trained visual selector used by the browser tutor. The two runtime files are `metadata.json` and `selector.safetensors` (about 21 MB). Qwen backbone weights and optimizer state are not included.

The selector was trained with the frozen `Qwen/Qwen2.5-VL-3B-Instruct` backbone. Its saved training settings, validation results, architecture and SHA-256 checksum are in `metadata.json`; runtime loading verifies the checksum and architecture. The saved checkpoint has 20 training updates and validation accuracy 0.5890 on 219 questions. This is answer accuracy, not a measurement of hint quality.

`configs/runtime.json` loads this directory using soft gates. No training step is needed to use it. The original run's output directory recorded in metadata is provenance, not a required runtime path.

ScienceQA-derived assets remain subject to the upstream terms and attribution described in the main README. Qwen use remains subject to its upstream model license.
