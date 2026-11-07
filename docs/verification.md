# Implementation verification — 2026-11-07

No training or optimizer updates were run. No pretrained checkpoint was downloaded for these checks. No notebooks or trained selector weight files are included.

Verification used the existing parent Python environment, `C:/AI/IE4422/.venv/Scripts/python.exe`, with this new folder as the working directory. A fresh environment installation and full pretrained GPU execution have not been verified.

## Completed checks

- All **42 tests** passed in a full `python -m unittest discover -s tests -v` run.
- Tiny randomly initialized Qwen: stock and identity-feature logits match; cached generation matches; gradients reach the selector through the frozen decoder with non-reentrant checkpointing. No backbone parameter receives gradients, and no weights are updated.
- Selector gates, feature budgets, option remapping, checkpoint save/load integrity, and checkpoint-driven runtime settings passed their checks.
- Public question inputs remain separate from reference answers. Initial hint generation uses one call without a textual evidence extraction prompt.
- Judging packages exclude condition labels; altered cases and unrelated judgments are rejected. Missing judgments remain pending, and failed generations remain in the all-planned denominator.
- Paired QA comparisons reject duplicate records, mismatched inputs and incompatible runtime provenance. Bootstrap groups use decoded RGB pixels, including images with different file encodings.
- Local HTTP test: status, page/assets, a fixture hint job, polling and saved-result download all passed.
- Real copied-data HTTP check: 219 original validation questions were listed; a question, its image and its audit inspection page were served successfully. The model remained unloaded, and the temporary server was stopped.
- UI startup finalization revalidated all **2,075** ScienceQA reviews. The original final counts remain **613 / 219 / 173**.
- Training preflight verified **456 train / 179 validation**, with **0 test questions loaded**. The 60 planned optimizer updates were not executed.
- The separate AI2D selection contains **100 test questions**. All four supplied training/pilot/control configurations passed validation.
- Original tracked project files remain unchanged; the only new project-root Git entry is `qwen-trained-evidence-extractor/`.

Derived split manifest SHA-256:

`c70fba99cb53bccb2449a96c1f866c3ab9b198da5c23ce89a6b3e777fb7ad7bd`

## Checks for the user before training

After setting up the environment described in the README, run `python train_selector.py --check-model`. This loads the full pretrained model, checks the identity path and answer-letter boundary, and makes no training updates. Then establish the baseline and run the small pilot when ready.

Full-model inference, quantized backward execution on the target GPU, peak GPU memory, training convergence, answer accuracy, visual faithfulness and hint quality remain unmeasured. Passing the implementation tests does not establish those experimental outcomes.
