"""HTTP and saved-result checks using a fake generator; no GPU or training."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen

from ui_server import TutorWebApp


class FakeTutor:
    last_selection = {"label":"Untrained — original visual features", "mode":"identity", "visual_tokens":4, "mean_gate":1.0}
    def ask(self, image, instruction, max_new_tokens=160, *, evidence_inputs=None):
        assert "gold_option_id" not in evidence_inputs
        return "Look at the arrowhead."


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
                self.assertIn(b'id="model-status"',get('/'))
                self.assertIn(b'status.selector',get('/assets/app.js'))
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


if __name__=='__main__': unittest.main()
