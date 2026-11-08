"""HTTP and saved-result checks using a fake generator; no GPU or training."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from PIL import Image
import torch

from ui_server import TutorWebApp


class FakeTutor:
    last_selection = {"label":"Untrained — original visual features", "mode":"identity", "visual_tokens":4, "mean_gate":1.0}
    def ask(self, image, instruction, max_new_tokens=160, *, evidence_inputs=None):
        assert "gold_option_id" not in evidence_inputs
        return "Look at the arrowhead."

class FakeComparisonRuntime:
    letter_ids = [0, 1]
    metadata = {
        "target_rho": 0.3,
        "weights_sha256": "fixture-checkpoint-hash",
        "selector_config": {"layers": 1, "variant": "visual"},
        "train_config": {"seed": 42},
        "validation": {"questions": 179, "accuracy": 0.65},
    }
    config = {"selector_checkpoint": "fixture-checkpoint"}

    def provenance(self):
        return {
            "backbone": {"model_path": "fixture", "revision": None, "resolved_revision": "fixture"},
            "backbone_signature": {},
            "preprocessing": {},
            "precision": "float32",
            "quantization": "none",
            "letter_token_ids": self.letter_ids,
            "conditioning": "public-question-context-options-v1",
            "answer_instruction_sha256": "fixture-prompt",
        }

    def answer_forward(self, item, *, mode, rho, seed):
        predicts_b = mode in {"soft", "hard"} and item["item_id"] != "test_000001"
        logits = torch.tensor([[-2.0, 2.0] if predicts_b else [2.0, -2.0]])
        return logits, torch.ones(1)


class UIServerTests(unittest.TestCase):
    def test_submission_requires_answer_but_reasoning_is_optional(self):
        app = TutorWebApp()
        item = {"choices": [{"id": "A", "text": "First"}]}
        payload = {"mode": "check", "split": "validation", "item_id": "validation_000001"}
        with patch('ui_server.runner.load_question', return_value=item), \
                patch('ui_server.runner.load_reference', return_value={}), \
                patch('ui_server.threading.Thread') as worker:
            for answer in (None, '', 'B'):
                with self.subTest(answer=answer), self.assertRaises(ValueError):
                    app.start_job({**payload, 'student_answer': answer})
            worker.assert_not_called()
            for reasoning in (None, '', '   '):
                with self.subTest(reasoning=reasoning):
                    job_id = app.start_job({**payload, 'student_answer': 'A', 'explanation': reasoning})
                    self.assertIn(job_id, app.jobs)
            self.assertEqual(worker.call_count, 3)

    def test_ai2d_comparison_rejects_non_test_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            app = TutorWebApp(root=Path(folder), splits=("validation", "test"))
            with self.assertRaisesRegex(ValueError, "AI2D is available only as a test split"):
                app.start_comparison({"dataset": "ai2d", "split": "validation"})

    def test_identity_status_hint_job_and_download(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'data/final/scienceqa/questions').mkdir(parents=True)
            (root/'data/final/scienceqa/images').mkdir()
            (root/'manifests/scienceqa').mkdir(parents=True)
            image = root/'data/final/scienceqa/images/validation_000001.png'
            image.write_bytes(b'fixture-only')
            item = {"item_id":"validation_000001","split":"validation","question":"Which direction?",
                    "context":"","choices":[{"id":"A","text":"Left"},{"id":"B","text":"Right"}],
                    "image_path":"data/final/scienceqa/images/validation_000001.png"}
            (root/'data/final/scienceqa/questions/validation.jsonl').write_text(json.dumps(item)+'\n')
            (root/'manifests/scienceqa/validation_final_ids.json').write_text(json.dumps([item['item_id']]))
            from project_data import ROOT, read_json
            baseline = {**read_json(ROOT/'configs/runtime.json'), 'selector_checkpoint': None}
            app = TutorWebApp(root=root,splits=('validation',),runtime_config=baseline)
            app.tutor = FakeTutor()
            server = ThreadingHTTPServer(('127.0.0.1',0),app.handler())
            thread = threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            def get(path):
                with urlopen(base+path,timeout=5) as response:
                    return response.read()
            try:
                status = json.loads(get('/api/status'))
                self.assertEqual(status['selector']['mode'],'identity')
                self.assertFalse(status['comparison_available'])
                self.assertFalse(status['comparison_datasets']['ai2d'])
                self.assertIn(b'id="model-status"',get('/'))
                self.assertIn(b'status.selector',get('/assets/app.js'))
                self.assertNotIn(b'id="comparison-button"',get('/'))
                self.assertNotIn(b'id="comparison-dataset"',get('/'))
                self.assertNotIn(b'comparisonButton',get('/assets/app.js'))
                page = get('/')
                self.assertNotIn(b'id="research-results"', page)
                self.assertIn('Your reasoning <span class="optional-label">· optional</span>'.encode(), page)
                self.assertIn('Your answer <span class="optional-label">· required to submit</span>'.encode(), page)
                self.assertIn('Explain your thinking, or leave this blank…'.encode(), page)
                payload = {"mode":"hint","split":"validation","item_id":item['item_id'],"hint_level":2}
                req = Request(base+'/api/jobs',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
                with urlopen(req,timeout=5) as response:
                    self.assertEqual(response.status,202)
                    job_id = json.loads(response.read())['job_id']
                deadline = time.monotonic()+5
                while time.monotonic()<deadline:
                    job = json.loads(get('/api/jobs/'+job_id))
                    if job['status']!='running': break
                    time.sleep(0.02)
                self.assertEqual(job['status'],'complete',job)
                saved = json.loads(get('/api/results/'+job_id))
                self.assertEqual(saved['hint'],'Look at the arrowhead.')
                self.assertEqual(saved['selection']['mode'],'identity')
                self.assertNotIn('gold_option_id',saved)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

    def test_all_mode_comparison_reports_metrics_and_saves_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data/final/ai2d/questions").mkdir(parents=True)
            (root / "data/final/ai2d/references").mkdir()
            (root / "manifests/ai2d").mkdir(parents=True)
            (root / "data/final/ai2d/questions/test.jsonl").write_text("")
            (root / "data/final/ai2d/references/test.jsonl").write_text("")
            (root / "manifests/ai2d/test_final_ids.json").write_text("[]")
            image_path = root / "diagram.png"
            Image.new("RGB", (4, 4), "white").save(image_path)
            items = [
                {"item_id": f"test_{index:06d}", "question": "Which direction?",
                 "context": "", "choices": [{"id": "A", "text": "Left"},
                                             {"id": "B", "text": "Right"}],
                 "image_path": str(image_path)}
                for index in range(1, 4)
            ]
            references = {
                items[0]["item_id"]: {"gold_option_id": "A"},
                items[1]["item_id"]: {"gold_option_id": "B"},
                items[2]["item_id"]: {"gold_option_id": "B"},
            }
            app = TutorWebApp(root=root, splits=("test",))
            app.runtime_config["selector_checkpoint"] = "fixture-checkpoint"
            app.tutor = SimpleNamespace(runtime=FakeComparisonRuntime())
            server = ThreadingHTTPServer(("127.0.0.1", 0), app.handler())
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"

            def get(path):
                with urlopen(base + path, timeout=5) as response:
                    return json.loads(response.read())

            with patch("project_data.load_questions", return_value=items) as load_items, \
                    patch("project_data.load_references", return_value=references):
                try:
                    status = get("/api/status")
                    self.assertTrue(status["comparison_available"])
                    self.assertTrue(status["comparison_datasets"]["ai2d"])
                    request = Request(
                        base + "/api/comparisons",
                        data=json.dumps({"dataset": "ai2d", "split": "test"}).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(request, timeout=5) as response:
                        self.assertEqual(response.status, 202)
                        job_id = json.loads(response.read())["job_id"]
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        job = get(f"/api/comparisons/{job_id}")
                        if job["status"] != "running":
                            break
                        time.sleep(0.02)
                    self.assertEqual(job["status"], "complete", job)
                    report = job["result"]
                    self.assertEqual(report["dataset"], "ai2d")
                    self.assertEqual(load_items.call_args.args[:2], ("ai2d", "test"))
                    self.assertEqual(
                        report["modes"],
                        ["identity", "soft", "hard", "random", "remove",
                         "remove_random", "null", "no_image"],
                    )
                    self.assertEqual(report["results"]["identity"]["accuracy"], 1 / 3)
                    self.assertEqual(report["results"]["soft"]["accuracy"], 1.0)
                    self.assertEqual(report["results"]["soft"]["paired_vs_identity"]["won"], 2)
                    self.assertEqual(report["results"]["soft"]["paired_vs_identity"]["lost"], 0)
                    self.assertEqual(report["best_mode"], "soft")
                    self.assertTrue(all(len(rows) == 3 for rows in report["predictions"].values()))
                    saved = get(f"/api/comparisons/{job_id}/result")
                    self.assertEqual(saved["checkpoint_weights_sha256"], "fixture-checkpoint-hash")
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=3)

    def test_free_transform_comparison_uses_only_supported_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data/final/ai2d/questions").mkdir(parents=True)
            (root / "data/final/ai2d/references").mkdir()
            (root / "manifests/ai2d").mkdir(parents=True)
            image_path = root / "diagram.png"
            Image.new("RGB", (4, 4), "white").save(image_path)
            items = [{
                "item_id": "test_000001",
                "question": "Which direction?",
                "context": "",
                "choices": [{"id": "A", "text": "Left"}, {"id": "B", "text": "Right"}],
                "image_path": str(image_path),
            }]
            references = {"test_000001": {"gold_option_id": "A"}}
            app = TutorWebApp(root=root, splits=("test",))
            app.tutor = SimpleNamespace(runtime=FakeComparisonRuntime())
            app.comparison_jobs["fixture-job"] = {"status": "running"}

            with patch.dict(FakeComparisonRuntime.metadata["selector_config"], {
                "variant": "free_transform",
            }), patch("project_data.load_questions", return_value=items), \
                    patch("project_data.load_references", return_value=references):
                report, _ = app.compare_selector("ai2d", "test", "fixture-job")

            self.assertEqual(report["modes"], ["identity", "soft", "no_image"])
            self.assertEqual(set(report["predictions"]), {"identity", "soft", "no_image"})
            self.assertIsNone(report["rho"])
            self.assertIn("without a token budget", report["note"])

if __name__=='__main__': unittest.main()
