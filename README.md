# Qwen trained visual evidence selector

Standalone implementation of a question-conditioned visual gate after Qwen2.5-VL's existing vision encoder and merger. The Qwen backbone is frozen. The selector learns from ScienceQA's existing answer labels; hint quality is evaluated separately.

The one-layer selector has been trained with multiple settings. The current best validation checkpoint is `outputs/training/selector-1layer-rho05-seed43/best` (65.9% accuracy on 179 ScienceQA validation questions). The browser tutor preserves the exact identity path unless a checkpoint is explicitly configured. The CLI and UI run independently; an optional Colab training/evaluation notebook is available under `notebook/`.

## What is here

- Independent copies of processed and final ScienceQA and AI2D data, including images, public questions and separate references. Selector training uses ScienceQA only; AI2D is test-only and is not used for training or checkpoint selection.
- Original dataset configurations, manifests, assisted visual reviews, audit reports and their preparation scripts.
- `selector.py`: cross-attention gates, shared null token, soft/hard/straight-through modes, text-only and free-transform controls.
- `model_runtime.py`: frozen Qwen, feature interception, answer scoring and hint generation.
- `train_selector.py`: preflight, model-interface check, selector-only training, validation, checkpoints and resume.
- `evaluate_selector.py`: QA baselines, option reshuffling and keep/remove/random feature controls.
- `hint_experiments.py`: B/D/E generation, blinded judging packages and returned-judgment summaries. It never calls a judge service.
- `ui_server.py`: the existing question/three-level-hint/answer-checking UI adapted to the new architecture.
- `docs/reference_plan.md`: the supplied plan, preserved as a reference. `docs/implementation.md` records implementation decisions and limitations.
- `docs/verification.md`: completed checks and the remaining full-model/GPU validation boundary.
- `notebook/README.md`: Colab setup, training, evaluation, checkpoint persistence and UI launch instructions. Upload bundles are included in that folder.

The original project is untouched. Model weights, download caches, environments and old experiment outputs were not duplicated. Initial copy hashes are in `docs/copied_files.json`. Copied source files subsequently adapted for this implementation naturally have different current hashes.

## Environment

Run these commands from this folder. Create an independent Python environment (Python 3.12+; the inherited pins were previously verified with Python 3.14):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the CUDA-enabled PyTorch/torchvision wheels matching the versions in `requirements-lock.txt` from the official PyTorch distribution, then:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

The model integration targets the pinned Transformers 5.21.0 `mm_encoder_outputs` interface. Do not silently change the dependency versions. Model loading downloads the public Qwen checkpoint on first use, unless `--model-path` points to an existing local copy. The UI and evaluation default to 4-bit Qwen; the selector runs in float32. Training defaults to bfloat16 computation with 4-bit frozen Qwen. Set `quantization` to `none` in a copied config if appropriate for your GPU.

The training memory requirement has **not** been measured on the full checkpoint. Frozen decoder weights still require backward activation memory. Start with the model check and a small pilot; reduce image token bounds or use a larger GPU if necessary. Keep preprocessing and precision matched across comparisons.

## Data and reproducibility

Copied data is ready to use. To independently rebuild from the pinned public sources:

```powershell
python prepare_data.py
python finalize_data.py
python prepare_ai2d.py
python prepare_selector_data.py
```

Preparation creates its caches when needed; the old project folder is not required. Hub/AI2D downloads use `data/cache/huggingface` in this folder. The inherited ScienceQA dataset cache uses `%LOCALAPPDATA%/IE4422/datasets` (or `~/.cache/IE4422/datasets`); it may reuse an existing pinned download. After a cache exists, the preparation scripts support `--offline`. `python finalize_data.py --check-only` validates the retained ScienceQA audit without rewriting outputs.

Training and evaluation use the complete original audited ScienceQA selection: **613 train / 219 validation / 173 test**. `prepare_selector_data.py` records these exact IDs and their order in `manifests/selector/splits.json`, together with image fingerprints, gold-letter counts and cross-split image overlaps. All original records are retained; image overlap is reported without additional exclusions. These are the original splits, which can share identical images.

AI2D remains the original frozen 100-question **test-only** sample; it is never used for training or hyperparameter selection. No cross-dataset contamination guarantee is made. The UI offers the full original ScienceQA selection for inspection; CLI training and evaluation use the same complete original selection.

## Browser tutor before and after training

Before training:

```powershell
python ui_server.py
```

The UI opens on `http://127.0.0.1:8766`. It displays **Untrained — original visual features**. The new model architecture is loaded lazily on the first request; its selector takes the identity path (`g=1`), preserving the original visual vectors exactly. An untrained random gate is never presented as a trained extractor.

After you train:

```powershell
python ui_server.py --selector-checkpoint outputs/training/selector-1layer-rho05-seed43/best
```

The same UI loads the frozen Qwen backbone plus the trained selector's separate weights. It displays **Trained visual selector**. The backbone weights are not overwritten by selector training. You can instead set `selector_checkpoint` in `configs/runtime.json`, leaving `gate_mode` as `auto`. Restart the server after changing checkpoints; there is no silent hot reload during a request. Missing, corrupt or incompatible checkpoints fail visibly.

With a trained checkpoint configured, the page offers **Compare all modes** on the same questions: identity, soft, hard, random, remove, remove-random, null and no-image. It displays accuracy, image-group bootstrap intervals, change and paired wins/losses versus identity. **Display all mode predictions** opens the per-question table. Validation is the default for mode exploration; reserve test for final reporting. The best observed checkpoint so far is the one-layer `rho=0.5`, seed 43 run at 65.9% validation accuracy; selecting among these modes on validation is exploratory and may overfit. Paired p-values are exploratory and not adjusted for multiple mode comparisons. Results and per-question predictions are saved under `outputs/comparisons/`.

For hard gates, add `--gate-mode hard`. For a bypass comparison with the same checkpoint/preprocessing, use `--gate-mode identity`.

Flow: question/image → frozen vision encoder and merger → identity or trained gate → frozen language model → requested hint. There is **one generation call per initial hint**, with no prompted textual evidence pass. The details panel shows selector status and gate statistics, not invented observations. Saved JSON includes gate values, image-grid shape and checkpoint hash. Checking an answer retains private reference-assisted explanation assessment, diagnosis when needed, and a hint. Selector conditioning always contains only the public question, context and options; it never receives references or the student's attempt. This answered branch is not part of the no-attempt hint experiment.

## Check without training

```powershell
python train_selector.py --preflight
python train_selector.py --check-model
python -m unittest discover -s tests -v
```

`--preflight` is the default: it checks data/configuration and does not load Qwen, create an optimizer or train. `--check-model` loads the actual checkpoint, compares stock/identity logits, verifies A-Z tokenization at the assistant boundary, and reports the first generated answer token. It makes no optimizer updates. Inspect its letter probe and baseline QA results before committing to training. Tests use fake UI responses and a tiny randomly initialized Qwen model; backward checks do not update any weights or download a pretrained model.

## Train when you choose

Edit or copy `configs/train_selector.json`. Training starts **only** with `--train`:

```powershell
python train_selector.py --train
```

For a one-layer evidence selector, use the dedicated config and output directory:

```powershell
python train_selector.py --config configs/train_selector_1layer.json --preflight
python train_selector.py --config configs/train_selector_1layer.json --check-model
python train_selector.py --config configs/train_selector_1layer.json --train
```

The model check downloads/loads Qwen if needed but does not update weights. Run it before training to verify model compatibility and that it loads on your GPU.

Each epoch reshuffles option order and remaps the gold index. Training uses restricted-letter CE plus a mean-gate budget penalty. The budget moves from 1.0 to the target during the first 20% of optimizer updates. A mean visual token is computed from training images only to initialize the shared null vector. Qwen parameters are frozen; gradients flow through its decoder into the selector. Validation runs periodically and at every epoch end. Best checkpoint selection uses validation accuracy, then CE; it never uses test labels.

The supplied `configs/pilot.json` disables the budget penalty and sets `target_rho=1`. Start with 64 examples:

```powershell
python train_selector.py --config configs/pilot.json --train --limit 64 --output-dir outputs/training/pilot
```

For the budget sweep and multiple seeds, use distinct output directories:

```powershell
python train_selector.py --train --rho 0.5 --seed 42 --output-dir outputs/training/selector-rho05-seed42
python train_selector.py --train --rho 0.1 --seed 43 --output-dir outputs/training/selector-rho01-seed43
```

To continue an interrupted run, use the **same configuration and CLI overrides** plus `--train --resume`. Checkpoints save at optimizer-update boundaries; interrupted partial accumulation is discarded and repeated. The original resolved Hub revision is reused. A partially written selector/optimizer pair is rejected, not silently resumed. Existing output directories are not overwritten for new runs.

`best/` and `last/` contain `selector.safetensors` plus `metadata.json`. `last/optimizer.pt` is only for resuming your own run. The metadata records architecture, resolved backbone revision when available, preprocessing, training configuration, split fingerprint, update count and validation results. When using local backbone paths, keep the exact same original weights; structural compatibility alone cannot prove identical local weights.

Supplied control configs are `configs/train_text_only.json` and `configs/train_free_transform.json`; pass either with `--config`. They have distinct output directories. Free-transform requires `budget_weight=0` and `straight_through_fraction=0`. Use `--no-option-shuffle` for the B5 control. Null types are `learned`, `mean` and `zero`. `straight_through_fraction=0.2` optionally enables hard-forward/soft-backward top-k during the final 20% of updates. B6 LoRA/merger fine-tuning is not implemented.

## QA evaluation and feature faithfulness

```powershell
python evaluate_selector.py --mode identity --output-dir outputs/qa/baseline --preflight
python evaluate_selector.py --mode identity --output-dir outputs/qa/baseline
python evaluate_selector.py --checkpoint outputs/training/selector-rho03-seed42/best --mode soft --output-dir outputs/qa/selector --compare outputs/qa/baseline/predictions.jsonl
python evaluate_selector.py --checkpoint outputs/training/selector-rho03-seed42/best --faithfulness --output-dir outputs/qa/faithfulness
```

Defaults use ScienceQA validation. Use `--split test` only after fixing configuration; for AI2D add `--dataset ai2d --split test`. `--limit` creates a pilot. `--mode no_image` tests the text-only backbone. `--shuffle-options` tests option-order robustness. `--mismatched-question` changes only selector conditioning while retaining the original question for Qwen; it uses the next question in the evaluation list, which may come from a different source diagram.

Faithfulness mode runs keep-only (`hard`), random keep, remove-selected and random removal at fractions 0.1/0.3/0.5/0.7/1.0. Results include gate vectors and per-condition accuracy, bootstrap intervals grouped by identical RGB images, and letter histograms. Paired comparisons require matching inputs, backbone revision, preprocessing, precision and QA protocol, and report exact McNemar results. Use the same checkpoint with `--mode identity` to produce a baseline matched to its settings. Failed generations remain incorrect outcomes in the denominator. No localization IoU is claimed because the imported data has no region annotations.

**A gate budget is not a speedup:** replacing features with a null token preserves sequence length and attention work. Soft mean gate 0.3 is not the same as retaining exactly 30% of tokens. Learned gates or attention maps alone do not prove faithful evidence; answer improvement does not imply better hints.

## B/D/E hint experiments and judging

Use the same held-out question IDs, prompts and image settings across conditions. B uses identity visual features, D adds the original prompted textual evidence, and E uses the trained selector without a textual evidence pass. E rejects an identity/bypassed selector. Pass the same `--checkpoint` for B and D when you need to inherit E's exact preprocessing and precision; B and D still force identity features.

```powershell
python hint_experiments.py generate --condition B --output-dir outputs/hints/B --preflight
python hint_experiments.py generate --condition B --output-dir outputs/hints/B
python hint_experiments.py generate --condition D --output-dir outputs/hints/D
python hint_experiments.py generate --condition E --checkpoint outputs/training/selector-rho03-seed42/best --output-dir outputs/hints/E
python hint_experiments.py package --run-dir outputs/hints/E --output-dir outputs/judging/E
```

ScienceQA test generation covers all **173 test questions / 519 hints per condition**. Test judge packages automatically select the exact **30 questions / 90 planned hints per condition** in `manifests/scienceqa/test_judge_selection.json`, copied from the supplied original-project manifest. The same IDs and all three tiers are reused across B/D/E; source configuration, parent test IDs, public-input hashes and image hashes are verified. ScienceQA validation packages continue to include all generated validation questions (219 questions / 657 planned hints per condition). AI2D generation defaults to **all 100 test questions / 300 hints per condition**. Its package automatically selects the **same existing 30 questions / 90 planned hints per condition** from `manifests/ai2d/test_judge_selection.json` across B/D/E. The package seed changes only case order. Selected failures stay in the 90-outcome denominator and are never replaced. A pilot missing any of the fixed 30 IDs cannot produce the main AI2D package.

```powershell
python hint_experiments.py generate --dataset ai2d --split test --condition E --checkpoint outputs/training/selector-rho03-seed42/best --output-dir outputs/hints/ai2d/E
python hint_experiments.py package --run-dir outputs/hints/ai2d/E --output-dir outputs/judging/ai2d/E
```

For the ScienceQA test package, generate with `--dataset scienceqa --split test`, then package the completed run:

```powershell
python hint_experiments.py generate --dataset scienceqa --split test --condition E --checkpoint outputs/training/selector-rho03-seed42/best --output-dir outputs/hints/scienceqa-test/E
python hint_experiments.py package --run-dir outputs/hints/scienceqa-test/E --output-dir outputs/judging/scienceqa-test/E
```

Both datasets use their fixed test judging IDs regardless of the package seed; the seed only shuffles case order. All 30 selected IDs must be present in the completed run. Selected failures remain in the 90-outcome denominator without replacement.

Share only `outputs/judging/E/attachments` with the judge. Keep the private record map outside its input. The same seven-field Part A rubric is used; package files include the image, question, choices, tier, hint and reference answer. No judge is called automatically. Save returned files as `judgment_r_000001.json`, etc., then:

```powershell
python judge_validate.py outputs/returned/E
python hint_experiments.py summarize --package-dir outputs/judging/E --judgments-dir outputs/returned/E --output outputs/judging/E/metrics.json
```

Generate and package B/D similarly. The summaries report missing/invalid judgments rather than silently treating them as passes or failures. Tier-progression and pairwise Parts B/D of the inherited rubric remain optional manual analyses; the automated summary covers pointwise Part A. These are rubric-based hint assessments, not measured student learning outcomes.

## Attribution

ScienceQA: Lu et al. (2022), *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering*, NeurIPS, https://scienceqa.github.io/. Retain the upstream CC BY-NC-SA 4.0 terms and attribution; the inherited Hugging Face card has a conflicting top-level license tag, as recorded in the original project.

AI2D: Kembhavi et al. (2016), *A Diagram Is Worth A Dozen Images*, https://arxiv.org/abs/1603.07396. The original AI2D release is distributed under CC BY-SA. Source revisions and the exact prepared subset are recorded in the copied configurations and manifests. Model use is subject to the selected Qwen checkpoint's license.
