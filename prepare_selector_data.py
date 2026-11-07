"""Audit the original ScienceQA splits, retaining every finalized question."""
import argparse
from collections import Counter
from pathlib import Path

from project_data import ROOT, digest, file_hash, pixel_hash, read_json, read_jsonl, write_json


def prepare(root=ROOT):
    root = Path(root)
    rows, source_files, ids = [], {}, {}
    for split in ("train", "validation", "test"):
        for kind in ("questions", "references"):
            rel = f"data/final/scienceqa/{kind}/{split}.jsonl"
            source_files[rel] = file_hash(root / rel)
        manifest = f"manifests/scienceqa/{split}_final_ids.json"
        source_files[manifest] = file_hash(root / manifest)
        items = read_jsonl(root / f"data/final/scienceqa/questions/{split}.jsonl")
        ids[split] = [item["item_id"] for item in items]
        if (ids[split] != read_json(root / manifest) or len(set(ids[split])) != len(ids[split])
                or any(item["split"] != split for item in items)):
            raise ValueError("Finalized questions differ from the original audited split IDs")
        rows.extend(items)
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("Question IDs repeat across original splits")
    seen_pixels, groups, repeated_text, fingerprints = {}, {}, {}, {}
    split_for = {row["item_id"]: row["split"] for row in rows}
    for row in rows:
        key = row["item_id"]
        image_path = root / row["image_path"]
        pixels = pixel_hash(image_path)
        text = digest([row["question"], row.get("context", ""), sorted(c["text"] for c in row["choices"])])
        group = seen_pixels.setdefault(pixels, key)
        groups.setdefault(group, []).append(key)
        repeated_text.setdefault(text, []).append(key)
        fingerprints[key] = {"pixel_sha256": pixels, "problem_sha256": text, "image_sha256": file_hash(image_path)}
    cross_split = [{"group": group, "item_ids": members, "splits": sorted({split_for[key] for key in members})}
                   for group, members in groups.items() if len({split_for[key] for key in members}) > 1]
    refs = {}
    for split in ids:
        refs.update({r["item_id"]: r["gold_option_id"] for r in read_jsonl(root / f"data/final/scienceqa/references/{split}.jsonl")})
    result = {"schema": "selector-splits-v2", "selection_policy": "original-audited-splits-v1",
              "policy": "Retain all original finalized ScienceQA train/validation/test IDs and their order. Identical-image overlaps are reported without exclusions or reassignment.",
              "source_files": source_files, "ids": ids, "counts": {s: len(v) for s,v in ids.items()},
              "excluded": [], "groups": groups, "cross_split_image_groups": cross_split, "fingerprints": fingerprints,
              "repeated_problem_text": [v for v in repeated_text.values() if len(v)>1],
              "gold_letter_counts": {s: dict(Counter(refs[k] for k in keys)) for s,keys in ids.items()},
              "limitations": ["Original splits can share identical images; this policy retains them and is not image-disjoint.",
                              "Near duplicates and pretrained-model contamination are not detected.",
                              "AI2D remains test-only; no cross-dataset near-duplicate guarantee."]}
    write_json(root / "manifests/selector/splits.json", result)
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = prepare()
    print({"counts": result["counts"], "excluded": len(result["excluded"]),
           "cross_split_image_groups_reported": len(result["cross_split_image_groups"])})
