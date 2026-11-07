"""Original split membership is retained, including repeated images; no training."""
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from prepare_selector_data import prepare
from project_data import ROOT, load_questions, read_json, write_json
from train_selector import preflight


class OriginalSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = self.root / "data/final/scienceqa/images/shared.png"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (4, 4), "white").save(image)
        import json
        for split, count in (("train", 2), ("validation", 1), ("test", 1)):
            rows = [{"item_id": f"{split}_{index}", "split": split, "question": "Which direction?", "context": "",
                     "choices": [{"id": "A", "text": "Left"}, {"id": "B", "text": "Right"}],
                     "image_path": "data/final/scienceqa/images/shared.png"} for index in range(count)]
            for kind, values in (("questions", rows), ("references", [{"item_id": row["item_id"], "gold_option_id": "A"} for row in rows])):
                path = self.root / f"data/final/scienceqa/{kind}/{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
            write_json(self.root / f"manifests/scienceqa/{split}_final_ids.json", [row["item_id"] for row in rows])

    def test_repeated_images_are_reported_without_excluding_original_questions(self):
        audit = prepare(self.root)
        self.assertEqual(audit["counts"], {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(audit["excluded"], [])
        self.assertEqual(len(audit["cross_split_image_groups"]), 1)
        self.assertEqual(audit["selection_policy"], "original-audited-splits-v1")
        for split in audit["ids"]:
            self.assertEqual([row["item_id"] for row in load_questions(split=split, root=self.root)], audit["ids"][split])
        self.assertEqual(prepare(self.root), audit)

    def test_stale_filtered_manifest_cannot_silently_reduce_training_data(self):
        audit = prepare(self.root)
        audit["ids"]["train"] = audit["ids"]["train"][:1]
        write_json(self.root / "manifests/selector/splits.json", audit)
        with self.assertRaisesRegex(ValueError, "original split IDs"):
            load_questions(split="train", root=self.root)
        prepare(self.root)
        self.assertEqual(len(load_questions(split="train", root=self.root)), 2)

    def test_original_manifest_mismatch_fails_before_rebuilding(self):
        write_json(self.root / "manifests/scienceqa/train_final_ids.json", ["different"])
        with self.assertRaisesRegex(ValueError, "original audited split"):
            prepare(self.root)

    def test_real_data_preflight_uses_613_train_and_219_validation(self):
        config = read_json(ROOT / "configs/train_selector.json")
        train, validation, report = preflight(config)
        self.assertEqual(len(train), 613)
        self.assertEqual(len(validation), 219)
        self.assertEqual(report["test_questions_loaded"], 0)
        self.assertEqual(report["optimizer_updates_planned"], 80)
        for split in ("train", "validation", "test"):
            original = read_json(ROOT / f"manifests/scienceqa/{split}_final_ids.json")
            self.assertEqual([row["item_id"] for row in load_questions(split=split)], original)


if __name__ == "__main__":
    unittest.main()
