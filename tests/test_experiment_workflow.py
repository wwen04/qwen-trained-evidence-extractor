"""Experiment accounting, blinding and provenance checks without model loading."""
import argparse
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import hint_experiments as hints
from evaluate_selector import bootstrap_accuracy, comparison_fingerprint, paired_comparison, summarize
from project_data import ROOT, digest, file_hash, pixel_hash, read_json, read_jsonl, write_json


def good_judgment():
    return {
        "tier_compliance": {"verdict": "yes", "reason": "Matches this tier."},
        "visual_grounding": {"verdict": "yes", "reason": "Uses the arrow."},
        "unsupported_visual_claims": {"count": 0, "unsupported_claims": []},
        "scientific_correctness": {"verdict": "yes", "reason": "Correct relationship."},
        "steers_toward_wrong_answer": {"verdict": "no", "reason": "Leaves the choice open."},
        "answer_disclosure": {"score": 0, "reason": "Does not name the answer."},
        "actionability": {"score": 4, "reason": "Directs attention to an observable detail."},
    }


class HintExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = self.root / "data/final/scienceqa/images/one.png"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (4, 4), "white").save(image)
        (self.root / "docs").mkdir()
        (self.root / "docs/judge_rubric.md").write_text("Fixture rubric.", encoding="utf-8")
        public = {"item_id": "one", "question": "Which direction?", "context": "",
                  "choices": [{"id": "A", "text": "Left"}, {"id": "B", "text": "Right"}],
                  "image_path": "data/final/scienceqa/images/one.png"}
        self.run_dir = self.root / "run"
        write_json(self.run_dir / "run.json", {"schema": "selector-hints-v1", "status": "complete",
                   "item_ids": ["one"], "dataset": "scienceqa", "split": "validation", "condition": "E"})
        rows = [{"item_id": "one", "tier": tier, "status": "ok" if tier < 3 else "failed",
                 "hint": "Look at the arrow." if tier < 3 else None, "public_inputs": public,
                 "input_sha256": digest(public), "image_sha256": file_hash(image)} for tier in (1, 2, 3)]
        (self.run_dir / "hints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.package_dir = self.root / "package"
        self.judgments_dir = self.root / "returned"
        self.judgments_dir.mkdir()
        self.root_patch = patch.object(hints, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def make_package(self):
        args = argparse.Namespace(run_dir=self.run_dir, output_dir=self.package_dir, seed=42)
        with patch.object(hints, "load_references", return_value={"one": {"gold_option_id": "B", "solution": "PRIVATE"}}), redirect_stdout(io.StringIO()):
            hints.package(args)
        return read_jsonl(self.package_dir / "record_map_PRIVATE.jsonl")

    def summarize(self, name="metrics.json"):
        output = self.root / name
        with redirect_stdout(io.StringIO()):
            hints.summarize(argparse.Namespace(package_dir=self.package_dir, judgments_dir=self.judgments_dir, output=output))
        return read_json(output)

    def test_blinding_missing_judgments_and_failed_generation_denominator(self):
        mapping = self.make_package()
        successful = [row for row in mapping if row["status"] == "ok"]
        self.assertEqual(len(successful), 2)
        for row in successful:
            case = read_json(self.package_dir / "attachments/cases" / row["record_id"] / "case.json")
            self.assertNotIn("condition", case)
            self.assertNotIn("item_id", case)
            self.assertEqual(case["gold_reference"]["solution"], "PRIVATE")
        write_json(self.judgments_dir / successful[0]["expected_file"], good_judgment())
        partial = self.summarize("partial.json")
        self.assertEqual(len(partial["pending"]), 1)
        self.assertIsNone(partial["good_hint_rate_all_planned"])
        write_json(self.judgments_dir / successful[1]["expected_file"], good_judgment())
        report = self.summarize()
        self.assertEqual(report["generation_failures"], 1)
        self.assertEqual(report["overall"]["good_hint_rate"], 1.0)
        self.assertAlmostEqual(report["good_hint_rate_all_planned"], 2 / 3)

    def test_tampered_case_and_unrelated_judgments_are_rejected(self):
        mapping = self.make_package()
        write_json(self.judgments_dir / "judgment_unknown.json", good_judgment())
        with self.assertRaisesRegex(ValueError, "do not belong"):
            self.summarize()
        row = next(row for row in mapping if row["status"] == "ok")
        case = self.package_dir / "attachments/cases" / row["record_id"] / "case.json"
        case.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "modified"):
            self.summarize()

    def test_incomplete_run_records_are_rejected(self):
        path = self.run_dir / "hints.jsonl"
        path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "complete run selection"):
            self.make_package()

    def test_e_cannot_silently_bypass_selector(self):
        args = argparse.Namespace(dataset="scienceqa", split="validation", limit=None, preflight=False,
                                  runtime_config=None, model_path=None, checkpoint=None, condition="E")
        with patch.object(hints, "load_questions", return_value=[{}]), patch("runtime_settings.read_runtime",
                return_value={"selector_checkpoint": "fixture", "gate_mode": "identity"}):
            with self.assertRaisesRegex(ValueError, "active selector"):
                hints.generate(args)


class EvaluationTests(unittest.TestCase):
    def rows(self):
        return [{"item_id": str(i), "correct": i < 2, "status": "ok" if i < 2 else "failed",
                 "predicted": "A" if i < 2 else None, "gold": "A", "input_sha256": str(i),
                 "image_sha256": str(i), "pixel_sha256": "shared-image", "comparison_sha256": "matched"}
                for i in range(3)]

    def test_bootstrap_clusters_pixels_and_counts_failures(self):
        rows = self.rows()
        interval = bootstrap_accuracy(rows, repetitions=50)
        self.assertEqual(interval, [2 / 3, 2 / 3])
        report = summarize(rows)
        self.assertEqual(report["failures"], 1)
        self.assertEqual(report["accuracy"], 2 / 3)

    def test_pairing_rejects_duplicates_mismatched_settings_and_inputs(self):
        rows = self.rows()
        baseline = copy.deepcopy(rows)
        baseline[0]["correct"] = False
        result = paired_comparison(rows, baseline)
        self.assertEqual((result["won"], result["lost"]), (1, 0))
        self.assertEqual(result["mcnemar_exact_p"], 1.0)
        with self.assertRaises(ValueError):
            paired_comparison(rows, baseline + baseline[:1])
        for field in ("gold", "input_sha256", "image_sha256", "comparison_sha256"):
            changed = copy.deepcopy(baseline)
            changed[0][field] = "different"
            with self.assertRaises(ValueError):
                paired_comparison(rows, changed)

    def test_comparison_fingerprint_uses_resolved_revision_and_preprocessing(self):
        provenance = {"backbone": {"model_path": "fixture", "revision": None, "resolved_revision": "abc"},
                      "backbone_signature": {}, "preprocessing": {"image_max_tokens": 512}, "precision": "bfloat16",
                      "quantization": "4bit", "letter_token_ids": [1, 2], "conditioning": "public",
                      "answer_instruction_sha256": "prompt"}
        expected = comparison_fingerprint(provenance)
        provenance["backbone"]["revision"] = "main"
        self.assertEqual(comparison_fingerprint(provenance), expected)
        provenance["preprocessing"]["image_max_tokens"] = 256
        self.assertNotEqual(comparison_fingerprint(provenance), expected)

    def test_pixel_grouping_matches_copied_audit_and_different_encodings(self):
        audit = read_json(ROOT / "manifests/selector/splits.json")
        key = audit["ids"]["validation"][0]
        rows = read_jsonl(ROOT / "data/final/scienceqa/questions/validation.jsonl")
        item = next(row for row in rows if row["item_id"] == key)
        self.assertEqual(pixel_hash(ROOT / item["image_path"]), audit["fingerprints"][key]["pixel_sha256"])
        with tempfile.TemporaryDirectory() as folder:
            png, bmp = Path(folder) / "one.png", Path(folder) / "two.bmp"
            Image.new("RGB", (4, 4), "white").save(png)
            Image.new("RGB", (4, 4), "white").save(bmp)
            self.assertNotEqual(file_hash(png), file_hash(bmp))
            self.assertEqual(pixel_hash(png), pixel_hash(bmp))


if __name__ == "__main__":
    unittest.main()
