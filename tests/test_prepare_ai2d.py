from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from prepare_ai2d import METHOD, ROOT, group_questions, prepare_rows, validate_problem


def example(name="4352.png", color="red", annotated=False):
    stream = BytesIO()
    info = PngInfo()
    if annotated:
        info.add_text("note", "Same diagram pixels, different encoding")
    Image.new("RGB", (4, 3), color).save(stream, format="PNG", pnginfo=info)
    return {"question": "Which label identifies the object?", "options": ["c", "D", "b", "a"],
            "answer": "1", "image": {"path": name, "bytes": stream.getvalue()}}


def config(size=2, count=3):
    return {"dataset_id": "fixture/ai2d", "dataset_revision": "fixture-revision", "split": "test",
            "data_files": ["fixture.parquet"], "expected_rows": count,
            "selection": {"size": size, "seed": 42, "method": METHOD}}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class AI2DPreparationTests(unittest.TestCase):
    def test_letter_contents_are_not_confused_with_option_ids_or_answers(self):
        rows = [example(), example("4352.png"), example("273.png", "blue")]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            report = prepare_rows(root, config(), iter(rows), {"fixture.parquet": "source-hash"})
            public = read_rows(root / "data/final/ai2d/questions/test.jsonl")
            references = read_rows(root / "data/final/ai2d/references/test.jsonl")
            self.assertEqual(report["selected_count"], 2)
            for row, ref in zip(public, references):
                self.assertEqual(row["choices"], [{"id": "A", "text": "c"}, {"id": "B", "text": "D"},
                                                   {"id": "C", "text": "b"}, {"id": "D", "text": "a"}])
                self.assertEqual(ref["gold_option_id"], "B")
                self.assertEqual(row["context"], "")
                self.assertEqual(row["split"], "test")
                self.assertNotIn("gold_option_id", row)
                self.assertNotIn("lecture", row)
                self.assertNotIn("solution", row)
                self.assertEqual(ref["lecture"], "")
                self.assertFalse(ref["reference_explanation_available"])
                self.assertEqual((root / row["image_path"]).read_bytes(), rows[row["metadata"]["source_row_index"]]["image"]["bytes"])
            self.assertFalse(report["visual_review_performed"])

    def test_image_variants_and_pixel_duplicates_share_a_sampling_group(self):
        # Equal source names join variants; equal pixels join differently named copies.
        questions = [{"item_id": str(i), "metadata": {"source_image_name": name, "pixel_sha256": pixels}}
                     for i, (name, pixels) in enumerate([("one.png", "red"), ("one.png", "blue"),
                                                        ("copy.png", "blue"), ("two.png", "green")])]
        groups = group_questions(questions)
        self.assertEqual(sorted(len(members) for members in groups.values()), [1, 3])
        self.assertEqual(len({row["metadata"]["image_group_id"] for row in questions[:3]}), 1)

    def test_repeat_run_is_identical_and_changed_inputs_cannot_replace_frozen_selection(self):
        rows = [example(), example("copy.png", annotated=True), example("273.png", "blue")]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            prepare_rows(root, config(), iter(rows), {"fixture.parquet": "source-hash"})
            paths = [root / "manifests/ai2d/test_selection.json", root / "data/final/ai2d/questions/test.jsonl",
                     root / "data/final/ai2d/references/test.jsonl", root / "reports/ai2d/integrity_audit.json"]
            before = [path.read_bytes() for path in paths]
            prepare_rows(root, config(), iter(rows), {"fixture.parquet": "source-hash"})
            self.assertEqual(before, [path.read_bytes() for path in paths])
            with self.assertRaisesRegex(ValueError, "frozen"):
                prepare_rows(root, config(), iter(rows), {"fixture.parquet": "changed-hash"})
            self.assertEqual(before, [path.read_bytes() for path in paths])
            changed = config()
            changed["selection"]["seed"] = 43
            with self.assertRaisesRegex(ValueError, "frozen"):
                prepare_rows(root, changed, iter(rows), {"fixture.parquet": "source-hash"})

    def test_invalid_rows_do_not_renumber_source_ids(self):
        rows = [example(), example(), example("273.png", "blue")]
        rows[0]["answer"] = "D"  # This source field is a position, not a diagram/option letter.
        rows[1]["options"] = ["w", "x", "y"]
        rows[1]["answer"] = "2"
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            report = prepare_rows(root, config(), iter(rows), {})
            ids = json.loads((root / "manifests/ai2d/test_final_ids.json").read_text())
            self.assertEqual(ids, ["ai2d_test_000001", "ai2d_test_000002"])
            self.assertEqual(report["exclusions"], [{"item_id": "ai2d_test_000000", "reason": "invalid_answer_index"}])
            refs = read_rows(root / "data/final/ai2d/references/test.jsonl")
            self.assertEqual(refs[0]["gold_option_id"], "C")

    def test_invalid_indices_and_empty_options_are_rejected_but_four_is_not_required(self):
        row = example()
        row["options"] = ["w", "x"]
        self.assertEqual(validate_problem(row), 1)
        for changes in [{"answer": True}, {"answer": "-1"}, {"answer": "4"},
                        {"answer": "B"}, {"options": ["w", " "]}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_problem({**row, **changes})

    def test_source_count_and_insufficient_diagram_count_fail_before_final_exports(self):
        for cfg in (config(count=4), config(size=3)):
            with self.subTest(config=cfg), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                root = Path(directory)
                with self.assertRaises(ValueError):
                    prepare_rows(root, deepcopy(cfg), iter([example(), example(), example("273.png", "blue")]), {})
                self.assertFalse((root / "data/final/ai2d/questions/test.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
