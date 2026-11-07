# Implementation verification — 2026-11-07

The one-layer selector training run and paired held-out QA evaluation have now been run locally. The initial implementation checks below were recorded before training; run-specific results and the later UI comparison verification are recorded separately.

The local Python 3.12 environment used the pinned project dependencies with CUDA PyTorch on an NVIDIA GeForce RTX 4060 Ti (16 GiB).

## Completed checks

- The initial implementation's **42 tests** passed in a full `python -m unittest discover -s tests -v` run.
- Tiny randomly initialized Qwen: stock and identity-feature logits match; cached generation matches; gradients reach the selector through the frozen decoder with non-reentrant checkpointing. No backbone parameter receives gradients, and no weights are updated.
- Selector gates, feature budgets, option remapping, checkpoint save/load integrity, and checkpoint-driven runtime settings passed their checks.
- Public question inputs remain separate from reference answers. Initial hint generation uses one call without a textual evidence extraction prompt.
- Judging packages exclude condition labels; altered cases and unrelated judgments are rejected. Missing judgments remain pending, and failed generations remain in the all-planned denominator.
- Paired QA comparisons reject duplicate records, mismatched inputs and incompatible runtime provenance. Bootstrap groups use decoded RGB pixels, including images with different file encodings.
- Local HTTP test: status, page/assets, a fixture hint job, polling and saved-result download all passed.
- Real copied-data HTTP check: 219 original validation questions were listed; a question, its image and its audit inspection page were served successfully. The model remained unloaded, and the temporary server was stopped.
- UI startup finalization revalidated all **2,075** ScienceQA reviews. The original final counts remain **613 / 219 / 173**.
- One-layer ScienceQA training completed **60 optimizer updates** over **456 training questions**, with **179 validation questions** and **0 test questions loaded by training**. The best validation checkpoint was update 45: accuracy **63.7%**, mean CE **0.894**, mean gate **0.250**. Validation was used for checkpoint selection.
- A paired evaluation of the same **173 held-out ScienceQA test questions** scored **50.9%** with original visual features and **57.8%** with the trained selector (**+6.9 percentage points**; 26 wins, 14 losses; exact McNemar p = **0.0807**). Image-group bootstrap 95% intervals were 42.9–58.4% and 50.3–65.1%. This is suggestive, not statistically conclusive at p < 0.05, and does not measure hint quality or establish visual faithfulness.
- The separate AI2D selection contains **100 test questions** and is not part of selector training.
- Original tracked project files remain unchanged; the only new project-root Git entry is `qwen-trained-evidence-extractor/`.

Derived split manifest SHA-256:

`c70fba99cb53bccb2449a96c1f866c3ab9b198da5c23ce89a6b3e777fb7ad7bd`

## Training and UI follow-up

The pinned-model check passed: identity-path maximum logit difference was 0, and the first generated answer token was a valid option letter. A one-update pilot completed with selector gradients. The full one-layer run and test comparison reports are under `outputs/training/selector-1layer-rho03-seed42/` and `outputs/comparisons/`.

The UI paired-prediction table and API tests passed; the browser displayed all 173 per-question pairs. `node --check ui/app.js`, Python compilation and `git diff --check` passed. A later full test-suite run had 41/43 passing; two hint-packaging tests failed with `Image differs from generation input`. The focused UI tests both pass.

The evaluation does not establish hint quality, causal visual faithfulness, or improvement on another dataset. The paired test result is modest and not statistically conclusive.

## All-mode validation comparison

A second one-layer candidate (`rho=0.5`, seed 43) completed 60 optimizer updates; its best validation checkpoint was update 15 at **65.9%**. The all-mode comparison then scored that checkpoint on the same **179 ScienceQA validation questions** across all eight supported modes. The UI lists each mode, image-group bootstrap interval, change and paired wins/losses versus identity, plus its per-question predictions.

| Mode | Accuracy | Change vs identity | Wins / losses |
|---|---:|---:|---:|
| identity | 54.7% | 0.0 pp | 0 / 0 |
| soft | **65.9%** | **+11.2 pp** | 30 / 10 |
| hard | 61.5% | +6.7 pp | 20 / 8 |
| random | 52.5% | -2.2 pp | 15 / 19 |
| remove | 41.3% | -13.4 pp | 14 / 38 |
| remove_random | 52.5% | -2.2 pp | 12 / 16 |
| null | 41.3% | -13.4 pp | 14 / 38 |
| no_image | 41.3% | -13.4 pp | 16 / 40 |

Soft was best on validation, followed by hard gating. This is exploratory: validation informed checkpoint and mode selection, the random control used one fixed seed, and the reported paired p-values are not adjusted for seven mode comparisons. Do not treat these validation results as an unbiased final estimate or select further using the test split. The saved per-mode predictions and complete metrics are in `outputs/comparisons/selector_comparison_3e518a1ccdde4942a6258c30a9905e22.json`.

The browser comparison displayed all eight rows. Expanding **Display all mode predictions** showed **179 rows** with the correct answer and each mode's answer. Focused UI integration tests, JavaScript syntax checking, Python compilation and `git diff --check` passed.
