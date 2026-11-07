"""Rebuild the final ScienceQA selection from saved visual-need decisions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
LABELS = ("required", "helpful_optional", "incidental", "unclear")
DERIVED_SOURCE_FILES = {
    "reports/scienceqa/integrity_audit.json",
    *(f"manifests/scienceqa/{split}_{kind}_ids.json"
      for split in SPLITS for kind in ("candidate", "valid")),
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    content = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".jsonl", ".md", ".py", ".txt"}:
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def row_digest(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def unique_index(rows, description):
    index = {}
    for row in rows:
        key = row["item_id"]
        if key in index:
            raise ValueError(f"Duplicate ID in {description}: {key}")
        index[key] = row
    return index


def finalize(root=ROOT, check_only=False):
    root = root.resolve()
    review_root = root / "reviews/scienceqa"
    annotation_path = review_root / "annotations.jsonl"
    provenance = json.loads((review_root / "audit_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in provenance["source_files_sha256"].items():
        if relative in DERIVED_SOURCE_FILES:
            continue
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or sha256(source) != expected:
            raise ValueError(f"Audited source changed: {relative}. Review it before rebuilding.")
    integrity_path = root / "reports/scienceqa/integrity_audit.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if integrity.get("source") != provenance["source"]:
        raise ValueError("Preparation report source differs from the reviewed dataset.")
    annotations = unique_index(read_jsonl(annotation_path), "annotations")
    questions, references = {}, {}
    all_ids = set()
    for split in SPLITS:
        base = root / "data/processed/scienceqa"
        questions[split] = read_jsonl(base / "questions" / f"{split}.jsonl")
        public_index = unique_index(questions[split], f"{split} questions")
        references[split] = unique_index(read_jsonl(base / "references" / f"{split}.jsonl"), f"{split} references")
        valid = json.loads((root / f"manifests/scienceqa/{split}_valid_ids.json").read_text(encoding="utf-8"))
        if valid != list(public_index):
            raise ValueError(f"Prepared questions and valid manifest differ for {split}")
        candidate_path = root / f"manifests/scienceqa/{split}_candidate_ids.json"
        candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
        if (len(candidates) != len(set(candidates))
                or not set(valid).issubset(candidates)):
            raise ValueError(f"Candidate manifest is inconsistent for {split}")
        if set(public_index) != set(references[split]):
            raise ValueError(f"Question/reference IDs differ for {split}")
        if all_ids.intersection(public_index):
            raise ValueError("An item ID occurs in more than one split")
        all_ids.update(public_index)
    valid_counts = {split: len(rows) for split, rows in questions.items()}
    if integrity.get("valid_counts") != valid_counts:
        raise ValueError("Preparation report counts differ from the prepared questions.")
    if set(annotations) != all_ids:
        missing = sorted(all_ids - set(annotations))
        extra = sorted(set(annotations) - all_ids)
        raise ValueError(f"Audit coverage mismatch: {len(missing)} missing, {len(extra)} extra IDs")
    for split in SPLITS:
        for row in questions[split]:
            annotation = annotations[row["item_id"]]
            if row["split"] != split or annotation["split"] != split:
                raise ValueError(f"Split mismatch: {row['item_id']}")
            if annotation["visual_need"] not in LABELS or not annotation.get("rationale", "").strip():
                raise ValueError(f"Invalid review decision: {row['item_id']}")
            if annotation["visual_need"] == "required" and not annotation.get("necessary_evidence", "").strip():
                raise ValueError(f"Missing necessary evidence: {row['item_id']}")
            image = (root / row["image_path"]).resolve()
            if not image.is_relative_to(base.resolve() / "images"):
                raise ValueError(f"Invalid prepared image path: {row['item_id']}")
            if row_digest(row) != annotation["source_row_sha256"] or sha256(image) != annotation["image_sha256"]:
                raise ValueError(f"Item changed since review: {row['item_id']}")
    selected = {split: [r for r in questions[split] if annotations[r["item_id"]]["visual_need"] == "required"] for split in SPLITS}
    final_ids = {split: [r["item_id"] for r in selected[split]] for split in SPLITS}
    counts = dict(Counter(a["visual_need"] for a in annotations.values()))
    report = {
        "status": "assisted_audit_complete",
        "reviewed_date": provenance["reviewed_date"],
        "reviewer": provenance["reviewer"],
        "source": provenance["source"],
        "selection_rule": "Keep required only; exclude helpful_optional, incidental and unclear.",
        "reviewed_count": len(annotations),
        "category_counts": {label: counts.get(label, 0) for label in LABELS},
        "by_split": {split: dict(Counter(annotations[r["item_id"]]["visual_need"] for r in questions[split])) for split in SPLITS},
        "final_counts": {split: len(ids) for split, ids in final_ids.items()},
        "final_by_skill": dict(sorted(Counter(r["metadata"]["skill"] for rows in selected.values() for r in rows).items())),
        "final_by_grade": dict(sorted(Counter(r["metadata"]["grade"] for rows in selected.values() for r in rows).items())),
        "annotations_sha256": sha256(annotation_path),
        "source_files_sha256": provenance["source_files_sha256"],
        "duplicate_policy": "Official splits preserved; no automatic removal of shared images or repeated public inputs.",
        "limitations": provenance["limitations"],
    }
    # Reuse the existing pixel/text audit without treating shared diagrams as identical questions.
    keep = {key for ids in final_ids.values() for key in ids}
    for key in ("cross_split_identical_pixels", "cross_split_identical_problem_text", "cross_split_identical_public_input"):
        groups = [[member for member in group if member["item_id"] in keep] for group in integrity.get(key, [])]
        report[key] = [group for group in groups if len({m["split"] for m in group}) > 1]
    if check_only:
        return report
    destination = root / "data/final/scienceqa"
    for split in SPLITS:
        public_rows, reference_rows = [], []
        for row in selected[split]:
            image = destination / "images" / f"{row['item_id']}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / row["image_path"], image)
            public_rows.append({**row, "image_path": image.relative_to(root).as_posix()})
            reference_rows.append(references[split][row["item_id"]])
        write_jsonl(destination / "questions" / f"{split}.jsonl", public_rows)
        write_jsonl(destination / "references" / f"{split}.jsonl", reference_rows)
        write_json(root / f"manifests/scienceqa/{split}_final_ids.json", final_ids[split])
    write_json(destination / "selection.json", {
        "source": provenance["source"], "selection_rule": report["selection_rule"],
        "annotations_sha256": report["annotations_sha256"], "final_counts": report["final_counts"],
    })
    write_json(root / "reports/scienceqa/visual_audit.json", report)
    lines = ["# ScienceQA visual-need audit", "", f"Reviewed: {len(annotations)} questions. Reviewer: {provenance['reviewer']}. Date: {provenance['reviewed_date']}.", "", report["selection_rule"], "", "| Official split | Required / retained | Helpful but optional | Incidental | Unclear |", "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        tally = report["by_split"][split]
        lines.append(f"| {split} | " + " | ".join(str(tally.get(label, 0)) for label in LABELS) + " |")
    lines.extend(["", f"Total retained: {sum(report['final_counts'].values())}.", "", "The original cache, processed exports, candidate/valid manifests and preparation reports are retained. Final questions and references are separate, and final image paths refer to copies under data/final/scienceqa/images/.", "", "## Method", "", provenance["method"], "", "## Boundaries", "", *[f"- {item}" for item in provenance["limitations"]], "", "## Cross-split overlap", "", report["duplicate_policy"], ""])
    for key in ("cross_split_identical_pixels", "cross_split_identical_problem_text", "cross_split_identical_public_input"):
        lines.append(f"- {key}: {len(report[key])} groups in the retained selection.")
    lines.extend(["", "## Rebuild", "", "Run `python finalize_data.py --check-only` to check coverage and source fingerprints. Run `python finalize_data.py` to regenerate final exports, final manifests and this report from the saved decisions. Source changes require a fresh review. Files left in final/images/ from an earlier selection do not define membership; final manifests and question exports do.", "", "The browser tutor, inspector defaults and Colab UI use data/final and final-ID manifests. Only explicit candidate inspection (--source processed) reads excluded rows; it does not change tutor inputs.", ""])
    (root / "reports/scienceqa/visual_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate audit coverage and fingerprints without writing exports")
    args = parser.parse_args()
    try:
        result = finalize(check_only=args.check_only)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Final selection failed: {error}\n")
    print(json.dumps({"reviewed": result["reviewed_count"], "categories": result["category_counts"], "final": result["final_counts"]}, indent=2))


if __name__ == "__main__":
    main()
