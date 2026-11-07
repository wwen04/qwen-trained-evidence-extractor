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
        "selector_config": {"layers": 1},
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
            app = TutorWebApp(root=root,splits=('validation',))
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
                self.assertIn(b'id="model-status"',get('/'))
                self.assertIn(b'status.selector',get('/assets/app.js'))
                self.assertIn(b'id="comparison-button"',get('/'))
                self.assertIn(b'comparisonAvailable',get('/assets/app.js'))
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

            with patch("project_data.load_questions", return_value=items), \
                    patch("project_data.load_references", return_value=references):
                try:
                    status = get("/api/status")
                    self.assertTrue(status["comparison_available"])
                    request = Request(
                        base + "/api/comparisons",
                        data=json.dumps({"split": "test"}).encode(),
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

if __name__=='__main__': unittest.main()
