"""Prepare a reproducible, no-attempt AI2D evaluation sample (no model calls)."""

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import random
import shutil

from PIL import Image


ROOT = Path(__file__).resolve().parent
METHOD = "random_image_groups_then_one_random_question_per_group"
LIMITATIONS = [
    "Source question-image pairings are preserved; no label-by-label inspection or image relabelling was performed.",
    "No visual-necessity, curriculum, scientific-correctness or answer-key review was performed. Valid means structurally usable, not independently verified.",
    "Image groups join matching source image filenames or identical decoded RGBA pixels. Near duplicates with different filenames and pixels may remain.",
    "No cross-dataset overlap check was performed. AI2D and ScienceQA selections and results remain separate.",
    "The source supplies no per-question grade, lecture or solution. Empty reference explanations mean unavailable.",
    "This is a diagram-balanced sample of the source test split, not a uniform random sample of question rows. It is reserved for evaluation without student attempts.",
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_problem(row):
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError("blank_question")
    options = row.get("options")
    if (not isinstance(options, list) or not 2 <= len(options) <= 26
            or any(not isinstance(text, str) or not text.strip() for text in options)):
        raise ValueError("invalid_choices")
    answer = row.get("answer")
    # The pinned release stores zero-based answer positions as strings, not letters.
    if isinstance(answer, str) and answer.isascii() and answer.isdecimal():
        answer = int(answer)
    if type(answer) is not int or not 0 <= answer < len(options):
        raise ValueError("invalid_answer_index")
    return answer


def image_details(content):
    with Image.open(BytesIO(content)) as image:
        image.verify()
    with Image.open(BytesIO(content)) as image:
        image.load()
        header = f"RGBA:{image.width}:{image.height}:".encode("ascii")
        pixels = hashlib.sha256(header + image.convert("RGBA").tobytes()).hexdigest()
        extensions = {"PNG": ".png", "JPEG": ".jpg", "GIF": ".gif", "WEBP": ".webp", "TIFF": ".tiff", "BMP": ".bmp"}
        return {"width": image.width, "height": image.height, "pixel_sha256": pixels,
                "extension": extensions.get(image.format, ".img")}


def group_questions(questions):
    """Join source-diagram variants and exact pixel duplicates, transitively."""
    parents = list(range(len(questions)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(questions):
        metadata = row["metadata"]
        keys = [("pixels", metadata["pixel_sha256"])]
        if metadata["source_image_name"]:
            keys.append(("source_name", metadata["source_image_name"]))
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    components = defaultdict(list)
    for index, row in enumerate(questions):
        components[find(index)].append(row)
    groups = {}
    for members in components.values():
        group_id = "ai2d_image_" + min(row["metadata"]["pixel_sha256"] for row in members)
        groups[group_id] = members
        for row in members:
            row["metadata"]["image_group_id"] = group_id
    return groups


def select_questions(groups, selection):
    size, seed = selection["size"], selection["seed"]
    if selection["method"] != METHOD:
        raise ValueError("Unknown selection method")
    if type(size) is not int or not 1 <= size <= len(groups) or type(seed) is not int:
        raise ValueError(f"Sample size must be between 1 and {len(groups)} image groups, with an integer seed")
    rng = random.Random(seed)
    chosen = [rng.choice(sorted(groups[key], key=lambda row: row["item_id"]))
              for key in rng.sample(sorted(groups), size)]
    return sorted(chosen, key=lambda row: row["item_id"])


def prepare_rows(root, config, rows, source_files):
    """Validate all rows, then freeze one question per sampled image group."""
    root = Path(root).resolve()
    if config["split"] != "test":
        raise ValueError("AI2D preparation preserves the source test split")
    frozen_path = root / "manifests/ai2d/test_selection.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else None
    if frozen is not None and frozen["configuration"] != config:
        raise ValueError("Configuration differs from the frozen AI2D selection; archive that selection before changing it")
    processed = root / "data/processed/ai2d"
    questions, references, exclusions = [], {}, []
    images = {}
    source_count = 0
    for source_index, row in enumerate(rows):
        source_count += 1
        item_id = f"ai2d_test_{source_index:06d}"
        try:
            answer = validate_problem(row)
            source_image = row.get("image")
            if (not isinstance(source_image, dict) or not isinstance(source_image.get("bytes"), bytes)
                    or not source_image["bytes"]):
                raise ValueError("missing_image_bytes")
            content = source_image["bytes"]
            byte_hash = hashlib.sha256(content).hexdigest()
            if byte_hash not in images:
                details = image_details(content)
                path = processed / "images" / (byte_hash + details["extension"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                images[byte_hash] = {**details, "path": path.relative_to(root).as_posix()}
            details = images[byte_hash]
            source_path = source_image.get("path")
            image_name = PurePosixPath(source_path.replace("\\", "/")).name if isinstance(source_path, str) else None
        except (ValueError, OSError, Image.DecompressionBombError) as error:
            exclusions.append({"item_id": item_id, "reason": str(error)})
            continue
        questions.append({
            "item_id": item_id, "split": "test", "question": row["question"], "context": "",
            "choices": [{"id": chr(65 + index), "text": text} for index, text in enumerate(row["options"])],
            "image_path": details["path"],
            "metadata": {"dataset": "ai2d", "source_row_index": source_index,
                         "source_image_name": image_name, "image_sha256": byte_hash,
                         "pixel_sha256": details["pixel_sha256"], "width": details["width"], "height": details["height"]},
        })
        references[item_id] = {"item_id": item_id, "gold_option_id": chr(65 + answer),
                               "lecture": "", "solution": "", "reference_explanation_available": False}
    if source_count != config["expected_rows"]:
        raise ValueError(f"Expected {config['expected_rows']} source rows, received {source_count}")
    groups = group_questions(questions)
    selected = select_questions(groups, config["selection"])
    selection = {
        "configuration": config, "source_files_sha256": source_files,
        "purpose": "Frozen AI2D evaluation sample without student attempts",
        "grouping": "Join equal source image filenames or equal decoded RGBA pixel hashes, transitively",
        "items": [{"item_id": row["item_id"], "image_group_id": row["metadata"]["image_group_id"],
                   "image_sha256": row["metadata"]["image_sha256"], "public_row_sha256": digest(row)} for row in selected],
    }
    if frozen is not None and frozen != selection:
        raise ValueError("Inputs differ from the frozen AI2D selection; existing final exports were not replaced")
    final = root / "data/final/ai2d"
    final_questions = []
    for row in selected:
        image = final / "images" / Path(row["image_path"]).name
        image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / row["image_path"], image)
        final_questions.append({**row, "image_path": image.relative_to(root).as_posix()})
    selected_ids = [row["item_id"] for row in selected]
    artifacts = {
        processed / "questions/test.jsonl": questions,
        processed / "references/test.jsonl": list(references.values()),
        final / "questions/test.jsonl": final_questions,
        final / "references/test.jsonl": [references[key] for key in selected_ids],
    }
    for path, records in artifacts.items():
        write_jsonl(path, records)
    write_json(root / "manifests/ai2d/test_valid_ids.json", [row["item_id"] for row in questions])
    write_json(root / "manifests/ai2d/test_final_ids.json", selected_ids)
    write_json(frozen_path, selection)
    write_json(final / "selection.json", selection)
    metadata = {
        "source": config, "source_files_sha256": source_files, "source_count": source_count,
        "valid_count": len(questions), "choice_count_distribution": dict(sorted(Counter(len(row["choices"]) for row in questions).items())),
        "id_convention": "ai2d_test_<zero-based row position across configured Parquet files, padded to six digits>; assigned before validation",
        "formatting": "Preserve question, option text/order and image bytes; assign A-Z option IDs and map the zero-based answer index to its option ID",
        "grade_filter": None, "reference_explanations_available": False,
    }
    report = {
        **metadata, "status": "prepared_and_sample_frozen", "exclusions": exclusions,
        "unique_image_bytes": len(images), "unique_decoded_images": len({row["metadata"]["pixel_sha256"] for row in questions}),
        "image_group_count": len(groups),
        "questions_per_group_distribution": dict(sorted(Counter(len(members) for members in groups.values()).items())),
        "selected_count": len(selected), "selected_image_groups": len({row["metadata"]["image_group_id"] for row in selected}),
        "selection_sha256": digest(selection), "visual_review_performed": False,
        "artifact_files_sha256": {path.relative_to(root).as_posix(): file_digest(path) for path in artifacts},
        "limitations": LIMITATIONS,
    }
    write_json(root / "reports/ai2d/metadata_audit.json", metadata)
    write_json(root / "reports/ai2d/integrity_audit.json", report)
    summary = ["# AI2D preparation and frozen evaluation sample", "",
               f"Source: `{config['dataset_id']}` at `{config['dataset_revision']}` (test split).", "",
               f"Source questions: {source_count}; valid: {len(questions)}; excluded: {len(exclusions)}.",
               f"Image groups: {len(groups)}. Selected: {len(selected)} questions from {len(selected)} different groups, seed {config['selection']['seed']}.", "",
               "Groups share an original image filename or identical decoded pixels. Uniformly sample groups without replacement, then uniformly select one question within each sampled group. Keep source row order in the exports.", "",
               "Option IDs and the answer key are formatted; diagram labels, images, choice contents and choice order are unchanged. Public questions and answer references are separate. Final manifests and exports define membership; leftover image files do not.", "",
               "## Scope and limitations", "", *[f"- {item}" for item in LIMITATIONS], "",
               "## Reproduce", "", "Run `python prepare_ai2d.py` to download and prepare, or `python prepare_ai2d.py --offline` to rebuild from the pinned local cache. Existing frozen selection fingerprints must match. No experiments, judge calls or UI integration are performed.", ""]
    (root / "reports/ai2d/preparation.md").write_text("\n".join(summary), encoding="utf-8")
    return report


def load_source(root, config, offline=False):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    paths, fingerprints = [], {}
    for filename in ["README.md", *config["data_files"]]:
        path = Path(hf_hub_download(repo_id=config["dataset_id"], repo_type="dataset",
                    revision=config["dataset_revision"], filename=filename,
                    cache_dir=str(root / "data/cache/huggingface"), token=False, local_files_only=offline))
        if config["dataset_revision"] not in path.parts:
            raise ValueError("Cached AI2D source differs from the pinned revision")
        fingerprints[filename] = file_digest(path)
        if filename in config["data_files"]:
            paths.append(path)
    for path in paths:
        missing = {"question", "options", "answer", "image"} - set(pq.ParquetFile(path).schema_arrow.names)
        if missing:
            raise ValueError(f"Missing AI2D source columns: {sorted(missing)}")

    def rows():
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                yield from batch.to_pylist()

    return rows(), fingerprints


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use only the pinned local download cache")
    args = parser.parse_args(argv)
    try:
        config = json.loads((ROOT / "configs/ai2d.json").read_text(encoding="utf-8"))
        rows, fingerprints = load_source(ROOT, config, args.offline)
        report = prepare_rows(ROOT, config, rows, fingerprints)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"AI2D preparation failed: {error}\n")
    print(json.dumps({key: report[key] for key in ("source_count", "valid_count", "image_group_count", "selected_count", "selected_image_groups")}, indent=2))


if __name__ == "__main__":
    main()
