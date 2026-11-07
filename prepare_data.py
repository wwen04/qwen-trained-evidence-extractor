"""Prepare ScienceQA candidates from one pinned Hugging Face release."""

import argparse
from collections import Counter, defaultdict
import csv
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import platform

from PIL import Image


ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
TEXT_FIELDS = ("question", "hint", "task", "grade", "subject", "topic",
               "category", "skill", "lecture", "solution")
REVIEW_FIELDS = ("item_id", "split", "image_path", "visual_need", "visual_type",
                 "reviewer", "rationale")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def item_id(split, source_index):
    """Use the original row position, before filtering or validation."""
    return f"{split}_{source_index:06d}"


def datasets_cache_dir():
    configured = os.environ.get("HF_DATASETS_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    return cache_root / "IE4422" / "datasets"


def load_scienceqa(config, offline):
    os.environ["HF_HOME"] = str(ROOT / "data/cache/huggingface")
    cache_dir = datasets_cache_dir()
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    # Import after setting cache/offline options: libraries read environment settings.
    from datasets import Image as DatasetImage, load_dataset

    datasets = load_dataset(
        config["dataset_id"],
        revision=config["dataset_revision"],
        cache_dir=str(cache_dir),
    )
    if set(datasets) != set(SPLITS):
        raise ValueError(f"Expected splits {SPLITS}; received {tuple(datasets)}")
    required_fields = set(TEXT_FIELDS) | {"image", "choices", "answer"}
    for split in SPLITS:
        dataset = datasets[split]
        missing = required_fields - set(dataset.column_names)
        if missing:
            raise ValueError(f"Missing columns in {split}: {sorted(missing)}")
        # Offline loading can use the latest cached release; require our pin.
        if not dataset.cache_files or not all(
                config["dataset_revision"] in Path(entry["filename"]).parts
                for entry in dataset.cache_files):
            raise ValueError("Cached dataset revision differs from the pinned revision")
        datasets[split] = dataset.cast_column("image", DatasetImage(decode=False))
    return datasets


def candidate_rejection(row, config):
    """The first failed condition defines mutually exclusive audit counts."""
    if not row.get("image"):
        return "no_image"
    if row.get("subject") != config["subject"]:
        return "other_subject"
    if row.get("grade") not in config["grades"]:
        return "other_grade"
    return None


def audit_dataset(datasets, config):
    counts, candidates = {}, {}
    for split in SPLITS:
        tally = Counter()
        candidates[split] = []
        for source_index, row in enumerate(datasets[split]):
            reason = candidate_rejection(row, config)
            tally[reason or "candidate"] += 1
            if reason is None:
                candidates[split].append(source_index)
        counts[split] = {"total": len(datasets[split]), **dict(sorted(tally.items()))}
    return counts, candidates


def validate_problem(row):
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError("blank_question")
    choices = row.get("choices")
    if (not isinstance(choices, list) or not 2 <= len(choices) <= 26
            or any(not isinstance(choice, str) or not choice.strip() for choice in choices)):
        raise ValueError("invalid_choices")
    answer = row.get("answer")
    if type(answer) is not int or not 0 <= answer < len(choices):
        raise ValueError("invalid_answer_index")
    for field in TEXT_FIELDS:
        if not isinstance(row.get(field), str):
            raise ValueError(f"invalid_{field}")


def separate_records(question_id, split, row, image_path):
    """Only the public record may be loaded by future generation code."""
    validate_problem(row)
    public = {
        "item_id": question_id,
        "split": split,
        "question": row["question"],
        # ScienceQA's 'hint' is supplied context, not a generated tutor hint.
        "context": row["hint"],
        "choices": [{"id": chr(65 + index), "text": text}
                    for index, text in enumerate(row["choices"])],
        "image_path": image_path,
        "metadata": {field: row[field] for field in
                     ("task", "grade", "subject", "topic", "category", "skill")},
    }
    reference = {
        "item_id": question_id,
        "gold_option_id": chr(65 + row["answer"]),
        "lecture": row["lecture"],
        "solution": row["solution"],
    }
    return public, reference


def inspect_image(content):
    with Image.open(BytesIO(content)) as image:
        image.verify()
    with Image.open(BytesIO(content)) as image:
        image.load()  # verify() alone may miss truncated pixel data.
        return {"width": image.width, "height": image.height}


def text_key(record):
    return (record["question"], record["context"],
            tuple((choice["id"], choice["text"]) for choice in record["choices"]))


def cross_split_groups(groups):
    result = []
    for members in groups:
        if len({row["split"] for row in members}) > 1:
            result.append(sorted(
                ({"item_id": row["item_id"], "split": row["split"]} for row in members),
                key=lambda row: row["item_id"],
            ))
    return sorted(result, key=lambda members: tuple(row["item_id"] for row in members))


def find_overlaps(records):
    """Compare actual text and pixels, retaining representatives one size at a time."""
    text_groups, dimension_groups = defaultdict(list), defaultdict(list)
    for record in records:
        text_groups[text_key(record)].append(record)
        dimension_groups[(record["width"], record["height"])].append(record)
    image_matches, combined_matches = [], []
    for dimensions in sorted(dimension_groups):
        representatives = []
        for record in dimension_groups[dimensions]:
            with Image.open(ROOT / record["image_path"]) as image:
                pixels = image.convert("RGB").tobytes()
            for existing_pixels, members in representatives:
                if pixels == existing_pixels:
                    members.append(record)
                    break
            else:
                representatives.append((pixels, [record]))
        image_matches.extend(cross_split_groups(members for _, members in representatives))
        for _, members in representatives:
            same_text = defaultdict(list)
            for record in members:
                same_text[text_key(record)].append(record)
            combined_matches.extend(cross_split_groups(same_text.values()))
    return {
        "cross_split_identical_pixels": cross_split_groups(image_matches),
        "cross_split_identical_problem_text": cross_split_groups(text_groups.values()),
        "cross_split_identical_public_input": cross_split_groups(combined_matches),
    }


def prepare(datasets, candidates):
    output = ROOT / "data/processed/scienceqa"
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    exclusions, audit_records, valid_ids, review_rows = [], [], {}, []
    for split in SPLITS:
        questions, references = [], []
        for source_index in candidates[split]:
            row = datasets[split][source_index]
            question_id = item_id(split, source_index)
            try:
                validate_problem(row)
                image = row["image"]
                if (not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes)
                        or not image["bytes"]):
                    raise ValueError("missing_image_bytes")
            except ValueError as error:
                exclusions.append({"item_id": question_id, "split": split, "reason": str(error)})
                continue
            content = image["bytes"]
            try:
                details = inspect_image(content)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                exclusions.append({"item_id": question_id, "split": split,
                                   "reason": "unreadable_image", "detail": str(error)})
                continue
            image_path = image_dir / f"{question_id}.png"
            image_path.write_bytes(content)
            public, reference = separate_records(question_id, split, row,
                                                 image_path.relative_to(ROOT).as_posix())
            questions.append(public)
            references.append(reference)
            audit_records.append({**public, **details})
            review_rows.append({"item_id": question_id, "split": split, "image_path": public["image_path"],
                                "visual_need": "", "visual_type": "", "reviewer": "", "rationale": ""})
        valid_ids[split] = [row["item_id"] for row in questions]
        save_jsonl(output / "questions" / f"{split}.jsonl", questions)
        save_jsonl(output / "references" / f"{split}.jsonl", references)
        print(f"{split}: {len(questions)} validated candidates", flush=True)
    with (output / "visual_review_template.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review_rows)
    return {
        "valid_counts": {split: len(ids) for split, ids in valid_ids.items()},
        "exclusions": exclusions,
        **find_overlaps(audit_records),
        "duplicate_policy": "Report only; preserve splits. Resolve overlap policy before final sampling.",
    }, valid_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true",
                        help="Audit filters/counts without decoding or exporting images; may download the full release")
    parser.add_argument("--offline", action="store_true", help="Require the pinned local Hugging Face cache")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/scienceqa.json").read_text(encoding="utf-8"))
    datasets = load_scienceqa(config, args.offline)
    counts, candidates = audit_dataset(datasets, config)
    base_report = {
        "source": config,
        "id_convention": "<split>_<zero-based source row index, padded to six digits>; assigned before filtering",
        "filter_counts": counts,
        "selection_status": "Candidates only; visual review and final sampling pending.",
    }
    for split, indices in candidates.items():
        save_json(ROOT / "manifests/scienceqa" / f"{split}_candidate_ids.json",
                  [item_id(split, index) for index in indices])
    save_json(ROOT / "reports/scienceqa/metadata_audit.json", base_report)
    print(json.dumps({split: len(indices) for split, indices in candidates.items()}), flush=True)
    if args.metadata_only:
        return
    image_report, valid_ids = prepare(datasets, candidates)
    for split, ids in valid_ids.items():
        save_json(ROOT / "manifests/scienceqa" / f"{split}_valid_ids.json", ids)
    save_json(ROOT / "reports/scienceqa/integrity_audit.json", {**base_report, **image_report})
    save_json(ROOT / "data/processed/scienceqa/environment.json", {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("datasets", "Pillow", "pyarrow", "huggingface-hub")},
    })


if __name__ == "__main__":
    main()
