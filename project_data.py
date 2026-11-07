"""Local, independently runnable data access with explicit public/reference boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pixel_hash(path):
    """Group identical RGB images even when their encodings differ."""
    from PIL import Image
    with Image.open(path) as source:
        image = source.convert("RGB")
        return hashlib.sha256(f"{image.size}:".encode() + image.tobytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def public_text(item):
    parts = ["Question: " + item["question"]]
    if item.get("context", "").strip():
        parts.append("Supplied context: " + item["context"])
    parts.append("Options:\n" + "\n".join(f'{c["id"]}. {c["text"]}' for c in item["choices"]))
    return "\n\n".join(parts)


def load_questions(dataset="scienceqa", split="validation", *, root=ROOT, clean=True):
    root = Path(root).resolve()
    if dataset not in {"scienceqa", "ai2d"} or split not in {"train", "validation", "test"}:
        raise ValueError("Unknown dataset or split")
    if dataset == "ai2d" and split != "test":
        raise ValueError("The imported AI2D release is test-only, never training data")
    base = root / "data/final" / dataset
    selection = None
    rows = read_jsonl(base / "questions" / f"{split}.jsonl")
    manifest = read_json(root / "manifests" / dataset / f"{split}_final_ids.json")
    if [r["item_id"] for r in rows] != manifest:
        raise ValueError("Question exports differ from their original audit manifest")
    if dataset == "scienceqa" and clean:
        selection = read_json(root / "manifests/selector/splits.json")
        if (selection.get("selection_policy") != "original-audited-splits-v1"
                or selection["ids"][split] != [r["item_id"] for r in rows]):
            raise ValueError("Selector audit must retain the original split IDs; rerun prepare_selector_data.py")
        for rel, expected in selection["source_files"].items():
            if file_hash(root / rel) != expected:
                raise ValueError("Selector split provenance changed; rerun prepare_selector_data.py")
    result = []
    for row in rows:
        image = (root / row["image_path"]).resolve()
        if not image.is_relative_to(base.resolve() / "images") or not image.is_file():
            raise ValueError("Question image is missing or outside the dataset image folder")
        if selection and file_hash(image) != selection["fingerprints"][row["item_id"]]["image_sha256"]:
            raise ValueError("Image changed since the selector split audit; rerun prepare_selector_data.py")
        choices = [{"id": c["id"], "text": c["text"]} for c in row["choices"]]
        if not 2 <= len(choices) <= 26 or [c["id"] for c in choices] != [chr(65+i) for i in range(len(choices))]:
            raise ValueError("Options must use consecutive A-Z identifiers")
        result.append({"item_id": row["item_id"], "split": split, "question": row["question"],
                       "context": row.get("context", ""), "choices": choices, "image_path": str(image)})
    return result


def load_references(dataset, split, *, root=ROOT):
    if dataset not in {"scienceqa", "ai2d"} or split not in {"train", "validation", "test"}:
        raise ValueError("Unknown dataset or split")
    return {r["item_id"]: r for r in read_jsonl(Path(root) / "data/final" / dataset / "references" / f"{split}.jsonl")}


def shuffle_options(item, gold_id, seed):
    result = copy.deepcopy(item)
    choices = list(result["choices"])
    random.Random(seed).shuffle(choices)
    gold_index = next(i for i, c in enumerate(choices) if c["id"] == gold_id)
    result["choices"] = [{"id": chr(65+i), "text": c["text"]} for i, c in enumerate(choices)]
    return result, gold_index


def stable_seed(seed, *parts):
    return int(hashlib.sha256((str(seed) + ":" + ":".join(map(str, parts))).encode()).hexdigest()[:15], 16)
