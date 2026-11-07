from contextlib import redirect_stdout
import csv
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from prepare_data import (ROOT, SPLITS, TEXT_FIELDS, audit_dataset, candidate_rejection,
                          find_overlaps, inspect_image, item_id, load_scienceqa, main,
                          prepare, separate_records, validate_problem)


def image_bytes(color="red", annotated=False):
    stream = BytesIO()
    info = PngInfo()
    if annotated:
        info.add_text("description", "Different encoding, same pixels")
    Image.new("RGB", (3, 2), color=color).save(
        stream, format="PNG", pnginfo=info, compress_level=0 if annotated else 9)
    return stream.getvalue()


def example():
    return {
        "question": "Which arrow points toward the Sun?", "choices": ["left", "right"],
        "answer": 1, "hint": "Use the diagram.",
        "image": {"path": "image.png", "bytes": image_bytes()},
        "task": "closed choice", "grade": "grade7", "subject": "natural science",
        "topic": "physics", "category": "Motion", "skill": "Read arrows",
        "lecture": "REFERENCE LECTURE", "solution": "REFERENCE SOLUTION",
    }


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.config = {"subject": "natural science", "grades": ["grade7", "grade8"]}

    def test_filter_excludes_non_image_non_science_and_other_grades(self):
        row = example()
        self.assertIsNone(candidate_rejection(row, self.config))
        for field, value, reason in [("image", None, "no_image"),
                                     ("subject", "social science", "other_subject"),
                                     ("grade", "grade6", "other_grade")]:
            with self.subTest(field=field):
                self.assertEqual(candidate_rejection({**row, field: value}, self.config), reason)

    def test_ids_preserve_source_positions_and_splits_when_filter_changes(self):
        datasets = {
            "train": [{**example(), "image": None},
                      {**example(), "grade": "grade6"}, example()],
            "validation": [example()], "test": [example()],
        }
        counts, candidates = audit_dataset(datasets, self.config)
        self.assertEqual(candidates, {"train": [2], "validation": [0], "test": [0]})
        self.assertEqual(counts["train"], {"total": 3, "candidate": 1, "no_image": 1, "other_grade": 1})
        broader = {**self.config, "grades": ["grade6", "grade7", "grade8"]}
        _, new_candidates = audit_dataset(datasets, broader)
        self.assertEqual(new_candidates["train"], [1, 2])
        self.assertEqual(item_id("train", candidates["train"][0]), "train_000002")
        self.assertEqual(item_id("train", new_candidates["train"][1]), "train_000002")
        ids = [item_id(split, index) for split in SPLITS for index in candidates[split]]
        self.assertEqual(len(set(ids)), 3)

    def test_references_are_separate_and_choice_order_is_preserved(self):
        row = example()
        public, reference = separate_records("train_000042", "train", row, "data/image.png")
        self.assertEqual(public["choices"], [{"id": "A", "text": "left"}, {"id": "B", "text": "right"}])
        self.assertEqual(public["context"], row["hint"])
        self.assertEqual(reference["gold_option_id"], "B")
        self.assertEqual(set(public), {"item_id", "split", "question", "context", "choices",
                                       "image_path", "metadata"})
        self.assertNotIn("REFERENCE", str(public))
        self.assertEqual(public["item_id"], reference["item_id"])

    def test_invalid_answers_and_fields_fail_visibly(self):
        for changes in [{"answer": -1}, {"answer": 2}, {"answer": True},
                        {"question": " "}, {"choices": ["one", ""]}, {"lecture": None}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_problem({**example(), **changes})

    def test_image_validation(self):
        self.assertEqual(inspect_image(image_bytes()), {"width": 3, "height": 2})
        with self.assertRaises(OSError):
            inspect_image(b"corrupt pixels")

    def test_overlaps_compare_pixels_and_text_including_different_encodings(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, patch("prepare_data.ROOT", Path(directory)):
            root = Path(directory)
            records = []
            inputs = [("train", 0, "red", False, "Question"),
                      ("validation", 0, "red", True, "Question"),
                      ("test", 0, "red", False, "Different question"),
                      ("train", 1, "blue", False, "Question")]
            for split, index, color, annotated, question in inputs:
                question_id = item_id(split, index)
                content = image_bytes(color, annotated)
                (root / f"{question_id}.png").write_bytes(content)
                public, _ = separate_records(question_id, split, {**example(), "question": question},
                                             f"{question_id}.png")
                records.append({**public, **inspect_image(content)})
            self.assertNotEqual(image_bytes("red"), image_bytes("red", True))
            report = find_overlaps(records)
            self.assertEqual(len(report["cross_split_identical_pixels"]), 1)
            self.assertEqual(len(report["cross_split_identical_problem_text"]), 1)
            combined = report["cross_split_identical_public_input"]
            self.assertEqual(len(combined), 1)
            self.assertEqual({row["item_id"] for row in combined[0]},
                             {"train_000000", "validation_000000"})
            self.assertEqual(len(report["cross_split_identical_pixels"][0]), 3)

    def test_prepare_records_failures_and_keeps_ids_and_image_bytes(self):
        datasets = {
            "train": [{**example(), "image": None}, {**example(), "answer": 8},
                      {**example(), "image": {"bytes": b"bad image"}},
                      {**example(), "image": {"bytes": None}}, example()],
            "validation": [example()], "test": [],
        }
        _, candidates = audit_dataset(datasets, self.config)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, patch("prepare_data.ROOT", Path(directory)), redirect_stdout(StringIO()):
            report, valid_ids = prepare(datasets, candidates)
            self.assertEqual(valid_ids, {"train": ["train_000004"], "validation": ["validation_000000"], "test": []})
            self.assertEqual([row["reason"] for row in report["exclusions"]],
                             ["invalid_answer_index", "unreadable_image", "missing_image_bytes"])
            output = Path(directory) / "data/processed/scienceqa"
            self.assertEqual((output / "images/train_000004.png").read_bytes(), image_bytes())
            question = json.loads((output / "questions/train.jsonl").read_text(encoding="utf-8"))
            reference = json.loads((output / "references/train.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(question["item_id"], reference["item_id"])
            self.assertEqual((output / "questions/test.jsonl").read_bytes(), b"")

    def test_no_candidates_produces_empty_exports_and_review_header(self):
        datasets = {split: [] for split in SPLITS}
        _, candidates = audit_dataset(datasets, self.config)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, patch("prepare_data.ROOT", Path(directory)), redirect_stdout(StringIO()):
            report, ids = prepare(datasets, candidates)
            self.assertEqual(report["exclusions"], [])
            self.assertEqual(ids, {split: [] for split in SPLITS})
            path = Path(directory) / "data/processed/scienceqa/visual_review_template.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertIn("item_id", reader.fieldnames)
                self.assertEqual(list(reader), [])

    def test_metadata_only_writes_candidates_without_decoding_or_full_exports(self):
        datasets = {split: [example()] for split in SPLITS}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, patch("prepare_data.ROOT", Path(directory)), redirect_stdout(StringIO()):
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs/scienceqa.json").write_text(json.dumps(self.config), encoding="utf-8")
            with patch("prepare_data.load_scienceqa", return_value=datasets), patch("prepare_data.prepare") as full_prepare, patch("sys.argv", ["prepare_data.py", "--metadata-only", "--offline"]):
                main()
                full_prepare.assert_not_called()
            ids = json.loads((root / "manifests/scienceqa/train_candidate_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(ids, ["train_000000"])
            self.assertFalse((root / "data/processed").exists())

    def test_loader_rejects_wrong_cached_revision_missing_splits_and_columns(self):
        config = {**self.config, "dataset_id": "test/dataset", "dataset_revision": "pinned-release"}
        datasets = {}
        for split in SPLITS:
            dataset = Mock()
            dataset.column_names = list(TEXT_FIELDS) + ["image", "answer", "choices"]
            dataset.cache_files = [{"filename": str(ROOT / "cache/pinned-release/data.arrow")}]
            dataset.cast_column.return_value = dataset
            datasets[split] = dataset
        local_app_data = ROOT / "short-user-cache"
        with patch.dict(os.environ, {"LOCALAPPDATA": str(local_app_data)}):
            os.environ.pop("HF_DATASETS_CACHE", None)
            with patch("datasets.load_dataset", return_value=datasets) as load_dataset:
                self.assertEqual(set(load_scienceqa(config, offline=True)), set(SPLITS))
                cache_dir = local_app_data / "IE4422" / "datasets"
                self.assertEqual(os.environ["HF_DATASETS_CACHE"], str(cache_dir))
                load_dataset.assert_called_once_with(
                    config["dataset_id"],
                    revision=config["dataset_revision"],
                    cache_dir=str(cache_dir),
                )
                datasets["train"].cache_files = [{"filename": str(ROOT / "cache/wrong-release/data.arrow")}]
                with self.assertRaisesRegex(ValueError, "revision differs"):
                    load_scienceqa(config, offline=True)
                datasets["train"].cache_files = [{"filename": str(ROOT / "cache/pinned-release/data.arrow")}]
                datasets["train"].column_names.remove("image")
                with self.assertRaisesRegex(ValueError, "Missing columns"):
                    load_scienceqa(config, offline=True)
                del datasets["test"]
                with self.assertRaisesRegex(ValueError, "Expected splits"):
                    load_scienceqa(config, offline=True)


if __name__ == "__main__":
    unittest.main()
