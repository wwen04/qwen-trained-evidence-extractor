# AI2D preparation and frozen evaluation sample

Source: `lmms-lab-encoder/ai2d` at `c83a9b9692933aff8349157c88a413df9d02c4e5` (test split).

Source questions: 3088; valid: 3088; excluded: 0.
Image groups: 814. Selected: 100 questions from 100 different groups, seed 42.

Groups share an original image filename or identical decoded pixels. Uniformly sample groups without replacement, then uniformly select one question within each sampled group. Keep source row order in the exports.

Option IDs and the answer key are formatted; diagram labels, images, choice contents and choice order are unchanged. Public questions and answer references are separate. Final manifests and exports define membership; leftover image files do not.

## Scope and limitations

- Source question-image pairings are preserved; no label-by-label inspection or image relabelling was performed.
- No visual-necessity, curriculum, scientific-correctness or answer-key review was performed. Valid means structurally usable, not independently verified.
- Image groups join matching source image filenames or identical decoded RGBA pixels. Near duplicates with different filenames and pixels may remain.
- No cross-dataset overlap check was performed. AI2D and ScienceQA selections and results remain separate.
- The source supplies no per-question grade, lecture or solution. Empty reference explanations mean unavailable.
- This is a diagram-balanced sample of the source test split, not a uniform random sample of question rows. It is reserved for evaluation without student attempts.

## Reproduce

Run `python prepare_ai2d.py` to download and prepare, or `python prepare_ai2d.py --offline` to rebuild from the pinned local cache. Existing frozen selection fingerprints must match. No experiments, judge calls or UI integration are performed.
