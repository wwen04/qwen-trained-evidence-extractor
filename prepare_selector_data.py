"""Derive leakage-resistant splits without changing any original datasets or audits."""
import argparse
from collections import Counter
from pathlib import Path

from project_data import ROOT, digest, file_hash, pixel_hash, read_jsonl, write_json


def prepare(root=ROOT):
    root = Path(root)
    rows, source_files, parent = [], {}, {}
    for split in ("train", "validation", "test"):
        for kind in ("questions", "references"):
            rel = f"data/final/scienceqa/{kind}/{split}.jsonl"
            source_files[rel] = file_hash(root / rel)
        rows.extend(read_jsonl(root / f"data/final/scienceqa/questions/{split}.jsonl"))
    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]
    def union(a, b):
        parent[find(a)] = find(b)
    seen_pixels, repeated_text, fingerprints = {}, {}, {}
    for row in rows:
        key = row["item_id"]
        image_path = root / row["image_path"]
        pixels = pixel_hash(image_path)
        text = digest([row["question"], row.get("context", ""), sorted(c["text"] for c in row["choices"])])
        find(key)
        if pixels in seen_pixels:
            union(key, seen_pixels[pixels])
        else:
            seen_pixels[pixels] = key
        repeated_text.setdefault(text, []).append(key)
        fingerprints[key] = {"pixel_sha256": pixels, "problem_sha256": text, "image_sha256": file_hash(image_path)}
    priority = {"train": 0, "validation": 1, "test": 2}
    group_priority = {}
    for r in rows:
        group = find(r["item_id"])
        group_priority[group] = max(group_priority.get(group, 0), priority[r["split"]])
    ids = {s: [] for s in priority}
    excluded = []
    groups = {}
    for r in rows:
        key = r["item_id"]
        group = find(key)
        groups.setdefault(group, []).append(key)
        if priority[r["split"]] == group_priority[group]:
            ids[r["split"]].append(key)
        else:
            excluded.append({"item_id": key, "split": r["split"], "reason": "Shares identical RGB pixels with a higher-priority held-out split", "group": group})
    refs = {}
    for split in priority:
        refs.update({r["item_id"]: r["gold_option_id"] for r in read_jsonl(root / f"data/final/scienceqa/references/{split}.jsonl")})
    result = {"schema": "selector-splits-v1", "policy": "Identical RGB image groups; test > validation > train. Exclude lower-priority members; never reassign original splits. Repeated question templates with different diagrams are reported but retained.",
              "source_files": source_files, "ids": ids, "counts": {s: len(v) for s,v in ids.items()},
              "excluded": excluded, "groups": groups, "fingerprints": fingerprints,
              "repeated_problem_text": [v for v in repeated_text.values() if len(v)>1],
              "gold_letter_counts": {s: dict(Counter(refs[k] for k in keys)) for s,keys in ids.items()},
              "limitations": ["Near duplicates and pretrained-model contamination are not detected.", "AI2D remains a separate test-only dataset; no cross-dataset near-duplicate guarantee."]}
    write_json(root / "manifests/selector/splits.json", result)
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = prepare()
    print({"counts": result["counts"], "excluded": len(result["excluded"])})
