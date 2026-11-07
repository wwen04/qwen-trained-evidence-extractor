# ScienceQA visual-need audit

Reviewed: 2075 questions. Reviewer: Codex assisted review. Date: 2026-10-29.

Keep required only; exclude helpful_optional, incidental and unclear.

| Official split | Required / retained | Helpful but optional | Incidental | Unclear |
|---|---:|---:|---:|---:|
| train | 613 | 214 | 434 | 11 |
| validation | 219 | 64 | 134 | 4 |
| test | 173 | 69 | 137 | 3 |

Total retained: 1005.

The original cache, processed exports, candidate/valid manifests and preparation reports are retained. Final questions and references are separate, and final image paths refer to copies under data/final/scienceqa/images/.

## Method

Assessed the prepared question, supplied context, ordered choices and corresponding image for visual necessity. Image contact sheets were grouped by skill for presentation; identical image bytes were shown once and linked back to their rows. Repeated passages were compared for differences, including option sets, so names and option-order changes did not require repeated image viewing. Complex food webs were also opened at original size. Decisions are saved per item ID with row and image fingerprints; recurring patterns share rationale wording. No Qwen outputs, QA success rates, answer keys, lectures or solutions were used to choose rows. The inspection tool separately reveals references for the user.

## Boundaries

- This is a single AI-assisted curation pass, not independently adjudicated human ground truth. The labels are eligibility judgements, not measurements of Qwen accuracy or hint quality.
- Required means required for the intended evidence-based task, not impossible for any model to guess or recall. No answerable-without-image experiment has been conducted.
- A conservative policy excludes named-substance classifications and general life-cycle recall even when their diagrams support a useful explanation. Numerical planet comparisons remain eligible; basic named-planet ordering/composition claims are excluded.
- Thermal-transfer items with explicit cooling or strong everyday heating/cooling cues are optional; pizza-delivery cases with a strong but unstated cooling assumption are unclear.
- Some food-web choices allow role/diet-based elimination. Obvious producer/fungus cases are optional; ambiguous shortcut cases are unclear.
- Selection is restricted to the existing natural-science Grade 7-8 candidates. It does not implement the all-grade scope of the proposal.
- Shared diagrams and repeated questions across official splits are reported rather than removed. This visual-need audit does not resolve independence of train/test examples or near duplicates.

## Cross-split overlap

Official splits preserved; no automatic removal of shared images or repeated public inputs.

- cross_split_identical_pixels: 60 groups in the retained selection.
- cross_split_identical_problem_text: 96 groups in the retained selection.
- cross_split_identical_public_input: 0 groups in the retained selection.

## Rebuild

Run `python finalize_data.py --check-only` to check coverage and source fingerprints. Run `python finalize_data.py` to regenerate final exports, final manifests and this report from the saved decisions. Source changes require a fresh review. Files left in final/images/ from an earlier selection do not define membership; final manifests and question exports do.

The browser tutor, inspector defaults and Colab UI use data/final and final-ID manifests. Only explicit candidate inspection (--source processed) reads excluded rows; it does not change tutor inputs.
