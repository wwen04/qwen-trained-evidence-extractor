# Science hint tutor with a trained visual selector

A local browser tutor using frozen Qwen2.5-VL-3B-Instruct and a trained, question-conditioned visual selector. The selector weights image features before Qwen generates a hint; it does not produce a written image description or verify the resulting hint.

The repository includes the trained one-layer selector (about 21 MB), the audited ScienceQA subset (613 training / 219 validation / 173 test questions and images), and the frozen 100-question AI2D test subset. You do not need to train a model or prepare datasets to use the tutor. The much larger Qwen backbone downloads from Hugging Face on the first generation request unless it is already cached.

## Requirements

- Python 3.12 (the locally tested version is 3.12.13).
- An NVIDIA CUDA GPU for the default 4-bit runtime. Locally tested on an RTX 4060 Ti with PyTorch 2.14.1+cu130 and CUDA runtime 13.0. Minimum VRAM has not been established.
- Internet access for dependency installation and the first Qwen download, plus disk space for the backbone and its cache.

The page opens on localhost; each person runs their own copy. A GitHub repository does not host the GPU model or provide a public tutoring service.

## Install and run

Clone the repository and enter its directory:

```sh
git clone https://github.com/wwen04/qwen-trained-evidence-extractor.git
cd qwen-trained-evidence-extractor
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or in Linux bash:

```sh
source .venv/bin/activate
```

Install the CUDA-enabled torch 2.14.1 and torchvision 0.29.1 wheels for your GPU/driver using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). The environment tested on this project used:

```sh
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip check
python check_setup.py
```

`check_setup.py` validates bundled question/image fingerprints and the selector weights without downloading Qwen. It also reports whether PyTorch can access CUDA. Resolve a `cuda_available: false` result before generating hints with the default configuration.

Start the tutor:

```sh
python ui_server.py
```

Open **http://127.0.0.1:8766** if the browser does not open automatically. The first hint takes longer while the backbone downloads and loads. The UI displays **Trained visual selector**.

Choose a question, then ask for a hint or select an answer and click **Check my answer**. An answer is required to submit; reasoning is optional. With no reasoning, the tutor checks only the selected choice and gives a hint when it is incorrect. Initial hints have three levels: notice a relevant detail, connect it to a science concept, or suggest a reasoning step.

For a full GPU smoke check:

```sh
python check_setup.py --generate
```

## Configuration and troubleshooting

`configs/runtime.json` points to `pretrained/scienceqa-visual-1layer-seed42`. Keep both `metadata.json` and `selector.safetensors` in that folder. Runtime loading checks the saved checksum and architecture. Checkpoint metadata records validation answer accuracy; it does not establish hint quality.

`gate_mode: auto` uses soft gates with a checkpoint. To bypass the selector while preserving the same preprocessing, run `python ui_server.py --gate-mode identity`. To use another trained checkpoint, pass `--selector-checkpoint PATH`. Restart the server after configuration changes.

- **CUDA unavailable:** make sure the environment has CUDA-enabled PyTorch and a compatible NVIDIA driver. The default 4-bit configuration does not run on CPU or Apple MPS.
- **Port already in use:** stop the existing server or run `python ui_server.py --port 8767`.
- **Old page after an update:** restart the server and refresh with Ctrl+F5.
- **Backbone download unavailable:** retry with internet access, or pass `--model-path PATH` pointing to a complete local copy of the same Qwen checkpoint.
- **Missing or changed bundled files:** restore them from Git. The application rejects invalid provenance rather than silently using a different dataset or checkpoint.

The pinned Transformers integration uses `mm_encoder_outputs`. Do not independently upgrade model dependencies without checking compatibility. Selector weights are float32; the backbone uses the precision and quantization saved in checkpoint metadata.

Hints can misread images, make incorrect claims, or disclose answers despite the prompts. Gate statistics describe internal numerical weights, not verified observations or confidence. This is a research prototype, not a validated assessment system.

## Verification

```sh
python -m pip check
python check_setup.py
python -m unittest discover -s tests -v
```

Tests cover selector tensors, checkpoint integrity, public/reference boundaries, data provenance, hint experiment packaging, and HTTP submission flows. They use fixtures and tiny random models instead of downloading the full backbone. The GitHub Actions workflow runs these checks on CPU; full hint generation is a separate GPU check.

## Research tools

The student UI contains question selection, hints and answer checking. Research comparisons remain available through the CLI and backend rather than a student-facing evaluation panel.

```sh
python train_selector.py --preflight
python evaluate_selector.py --mode identity --output-dir outputs/qa/baseline
python evaluate_selector.py --checkpoint pretrained/scienceqa-visual-1layer-seed42 --mode soft --output-dir outputs/qa/selector
python hint_experiments.py generate --condition E --checkpoint pretrained/scienceqa-visual-1layer-seed42 --output-dir outputs/hints/E
python hint_experiments.py package --run-dir outputs/hints/E --output-dir outputs/judging/E
```

Condition B uses unchanged visual features, D adds prompted textual evidence, and E uses trained feature selection. Judging is separate from generation; no judge is called automatically. The rubric is in `docs/judge_rubric.md`. Saved results, optimizer states, local archives and backbone weights are excluded from Git.

To rebuild the bundled data from the pinned upstream sources, run `prepare_data.py`, `finalize_data.py`, `prepare_ai2d.py`, then `prepare_selector_data.py`. This downloads the upstream datasets and recreates processed exports. The normal UI launch validates the bundled final exports and does not need these downloads. Original ScienceQA split membership is retained, including reported image overlaps. AI2D is test-only and comparisons on it are exploratory.

## Attribution and data terms

ScienceQA: Lu et al. (2022), *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering*, NeurIPS, https://scienceqa.github.io/. The included subset is derived from the pinned `derek-thomas/ScienceQA` revision in `configs/scienceqa.json`. Retain the upstream [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) terms and attribution. The inherited Hugging Face card has a conflicting top-level license tag; the upstream ScienceQA terms apply to this redistribution.

AI2D: Kembhavi et al. (2016), *A Diagram Is Worth A Dozen Images*, https://arxiv.org/abs/1603.07396. The included selection is recorded in `configs/ai2d.json` and `manifests/ai2d/`. Retain the original release's [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) terms and attribution.

Qwen backbone use is subject to the [upstream model license](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct). The repository does not redistribute Qwen backbone weights. Dataset terms are not a license grant for unrelated third-party material.
