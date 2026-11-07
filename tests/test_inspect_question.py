import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from finalize_data import row_digest
from inspect_question import ROOT, load_item, render


class QuestionInspectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.retained = "validation_000008"
        self.excluded = "validation_000003"
        annotations = []
        for stage in ("processed", "final"):
            base = self.root / f"data/{stage}/scienceqa"
            for folder in ("questions", "references", "images"):
                (base / folder).mkdir(parents=True)
            rows = []
            for key in ([self.retained, self.excluded] if stage == "processed" else [self.retained]):
                image = base / f"images/{key}.png"
                image.write_bytes(b"reviewed image")
                row = {
                    "item_id": key, "split": "validation", "question": "Read the diagram.",
                    "context": "Student passage.", "choices": [{"id": "A", "text": "Choice"}],
                    "image_path": image.relative_to(self.root).as_posix(),
                }
                rows.append(row)
                if stage == "processed":
                    annotations.append({
                        "item_id": key, "visual_need": "required" if key == self.retained else "incidental",
                        "source_row_sha256": row_digest(row),
                        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    })
            (base / "questions/validation.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (base / "references/validation.jsonl").write_text("", encoding="utf-8")
        manifest = self.root / "manifests/scienceqa/validation_final_ids.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps([self.retained]), encoding="utf-8")
        audit = self.root / "reviews/scienceqa/annotations.jsonl"
        audit.parent.mkdir(parents=True)
        audit.write_text("".join(json.dumps(row) + "\n" for row in annotations), encoding="utf-8")

    def test_default_inspection_uses_final_image_and_matches_audit_fingerprints(self):
        # A changed candidate copy must not replace the audited image in inspection.
        (self.root / f"data/processed/scienceqa/images/{self.retained}.png").write_bytes(b"changed")
        item, reference, audit, image = load_item(self.retained, self.root)
        self.assertEqual(image.read_bytes(), b"reviewed image")
        self.assertEqual(audit["source_status"], "matches reviewed source")
        output = self.root / "outputs/inspection/question.md"
        page = render(item, reference, audit, image, output)
        self.assertIn("data/final/scienceqa/images/", page.read_text(encoding="utf-8"))
        self.assertIn("Student passage.", output.read_text(encoding="utf-8"))

    def test_excluded_id_requires_explicit_candidate_inspection(self):
        with self.assertRaisesRegex(ValueError, "--source processed"):
            load_item(self.excluded, self.root)
        _, _, audit, image = load_item(self.excluded, self.root, data_source="processed")
        self.assertEqual(audit["visual_need"], "incidental")
        self.assertTrue(image.is_relative_to(self.root / "data/processed"))
        (self.root / "data/final/scienceqa/questions/validation.jsonl").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "finalize_data.py"):
            load_item(self.retained, self.root)


if __name__ == "__main__":
    unittest.main()
