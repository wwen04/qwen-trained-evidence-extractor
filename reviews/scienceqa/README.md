# Visual-need audit and question inspection

The starting point is the 2,075 existing natural-science Grade 7-8 questions in `data/processed/scienceqa/`. All source preparation outputs remain separate from this audit.

`annotations.jsonl` holds one saved classification per item ID, a short rationale, necessary visual evidence for retained questions, and fingerprints of the public record and image. `audit_manifest.json` records the rubric, method, source release, file fingerprints and limitations. These are Codex-assisted decisions, without independent human adjudication. They were not selected using Qwen outputs or held-out answer references.

Categories:

- `required`: the intended task needs a value, configuration or relationship supplied by the image.
- `helpful_optional`: the image supports the task but the text or ordinary relevant knowledge supplies a sufficient route.
- `incidental`: the image illustrates the setting rather than contributing necessary evidence.
- `unclear`: necessity is ambiguous, including strong answer-choice or everyday-knowledge shortcuts.

Only `required` is retained. Categories are not numerical confidence scores. Repeated templates can share a rationale; every original ID retains its own decision and source fingerprints. The audit is about eligibility, not whether a language model can guess an answer without seeing an image.

## Inspect an audited ID

From the repository root:

```powershell
.\.venv\Scripts\python.exe inspect_question.py validation_000008 --open
```

This generates/replaces `outputs/inspection/current_question.md` and a matching `.html` page. `--open` opens the HTML page in your browser. The page shows question, supplied context, ordered choices, full-size image, metadata, saved audit decision and source check. Reference answer, lecture and solution are in a separate expandable section in HTML. These reference fields are for your inspection and are not sent to the tutor.

Omit the item ID to be prompted. List available IDs with:

```powershell
.\.venv\Scripts\python.exe inspect_question.py --list validation
```

Keep multiple inspection pages by choosing a distinct Markdown filename:

```powershell
.\.venv\Scripts\python.exe inspect_question.py validation_000008 --output outputs/inspection/planet.md --open
```

The inspector uses Python's standard library and needs neither model weights nor a GPU. It defaults to final questions and final images, and checks the final manifest. Excluded IDs remain inspectable through an explicit audit-only option:

```powershell
.\.venv\Scripts\python.exe inspect_question.py validation_000003 --source processed --open
```

Use `--source processed --list validation` to list candidates instead of final IDs. Generated pages are disposable; replacing them never changes saved decisions or tutor inputs.

## Reproduce the final selection

```powershell
.\.venv\Scripts\python.exe finalize_data.py --check-only
.\.venv\Scripts\python.exe finalize_data.py
```

The first command checks complete coverage, unique IDs, source fingerprints, splits, references and image fingerprints without writing exports. Text-file fingerprints are normalized across LF/CRLF line endings; image fingerprints remain byte-exact. Preparation-generated candidate/valid manifests and the integrity report are checked against the prepared questions, review provenance and split counts rather than trusted by their file hashes. The second command applies the saved decisions and writes:

- `data/final/scienceqa/questions/<split>.jsonl`
- `data/final/scienceqa/references/<split>.jsonl`
- `data/final/scienceqa/images/<item_id>.png` (byte-identical copies)
- `data/final/scienceqa/selection.json`
- `manifests/scienceqa/<split>_final_ids.json`
- `reports/scienceqa/visual_audit.json` and `.md`

Answers and audit rationale are not added to the final public question records. Original IDs, choices and official splits are preserved. Final image paths point to the final copies. The report includes retained cross-split overlap groups from the existing integrity audit; no duplicate-removal policy is silently applied. The command-line runner, Python helper, browser tutor, inspector defaults and Colab demo use the final set. They reject missing final data and excluded IDs instead of falling back to candidates. The browser startup and Colab preparation cell run finalization automatically; for the standalone CLI, run it after preparation.

Rerunning `prepare_data.py` on the same source and filters should reproduce the audited inputs. If source bytes, records, filters or manifests change, finalization refuses to silently reuse stale decisions. Review the differences and update the audit before rebuilding. Final manifests and exported questions define membership; leftover images from previous exports do not.

Commit this review directory, the final manifests, reports and both scripts. Like the existing processed dataset, `data/final/` is ignored by Git and is recreated locally using the saved decisions.
