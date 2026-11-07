import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from project_data import public_text, shuffle_options
from runtime_settings import read_runtime
from tutor_pipeline import run_initial_hint, run_answered_pipeline
from train_selector import validate_config


class FakeTutor:
    def __init__(self,responses):
        self.responses = iter(responses)
        self.calls = []
        self.last_selection = {"label":"Untrained — original visual features","mode":"identity","visual_tokens":4,"mean_gate":1.0}
    def ask(self,image,instruction,max_new_tokens=160,*,evidence_inputs=None):
        self.calls.append((instruction,copy.deepcopy(evidence_inputs)))
        return next(self.responses)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.item = {"question":"Which object?","context":"Look carefully.","choices":[{"id":"A","text":"First"},{"id":"B","text":"Second"}],"image_path":"example.png"}

    def test_public_conditioning_and_shuffle(self):
        dirty = {**self.item,"gold_option_id":"SECRET","solution":"SECRET"}
        self.assertNotIn("SECRET",public_text(dirty))
        shuffled,gold = shuffle_options(self.item,"B",7)
        self.assertEqual(shuffled['choices'][gold]['text'],'Second')
        self.assertEqual(self.item['choices'][0]['id'],'A')

    def test_hint_one_call_and_no_textual_evidence(self):
        tutor = FakeTutor(['Notice the arrow.'])
        result = run_initial_hint(tutor,'example.png',public_text(self.item),1,evidence_inputs=self.item)
        self.assertEqual(len(tutor.calls),1)
        self.assertEqual(result['hint'],'Notice the arrow.')
        self.assertNotIn('Visual evidence (unverified)',tutor.calls[0][0])
        self.assertEqual(result['selection']['mode'],'identity')

    def test_answer_check_reference_is_not_selector_conditioning(self):
        tutor = FakeTutor(['NEEDS_WORK','{"category":"insufficient_evidence"}','Inspect the labels.'])
        result = run_answered_pipeline(tutor,'example.png',public_text(self.item),'B','Guess','Second','A','PRIVATE REFERENCE',2,evidence_inputs=self.item)
        self.assertEqual(len(tutor.calls),3)
        self.assertIn('PRIVATE REFERENCE',tutor.calls[0][0])
        for prompt,condition in tutor.calls:
            self.assertNotIn('PRIVATE REFERENCE',json.dumps(condition))
        for prompt,_ in tutor.calls[1:]:
            self.assertNotIn('PRIVATE REFERENCE',prompt)
        self.assertEqual(result['hint'],'Inspect the labels.')

    def test_untrained_default_identity_and_bad_checkpoint_rejected(self):
        config = read_runtime()
        self.assertEqual(config['gate_mode'],'identity')
        with self.assertRaises(ValueError):
            read_runtime(gate_mode='soft')
        with self.assertRaises(FileNotFoundError):
            read_runtime(selector_checkpoint='does-not-exist')

    def test_invalid_training_settings(self):
        from project_data import ROOT, read_json
        config = read_json(ROOT/'configs/train_selector.json')
        validate_config(config)
        config['gradient_accumulation']=0
        with self.assertRaises(ValueError): validate_config(config)

    def test_checkpoint_runtime_uses_saved_settings_and_rejects_untrained(self):
        from checkpoints import SCHEMA
        from project_data import file_hash, write_json
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)
            # Metadata fixture only; these are not model weights and are never loaded.
            (checkpoint / 'selector.safetensors').write_bytes(b'metadata-check-fixture')
            metadata = {'schema': SCHEMA, 'weights_sha256': file_hash(checkpoint / 'selector.safetensors'),
                        'training_updates': 1, 'backbone': {'model_path': 'fixture-backbone', 'resolved_revision': 'fixed'},
                        'preprocessing': {'image_min_tokens': 64, 'image_max_tokens': 128},
                        'precision': 'float32', 'quantization': 'none', 'target_rho': 0.3}
            write_json(checkpoint / 'metadata.json', metadata)
            config = read_runtime(selector_checkpoint=str(checkpoint))
            self.assertEqual(config['gate_mode'], 'soft')
            self.assertEqual(config['model_path'], 'fixture-backbone')
            self.assertEqual(config['revision'], 'fixed')
            self.assertEqual(config['rho'], 0.3)
            self.assertEqual(config['image_max_tokens'], 128)
            self.assertEqual(config['precision'], 'float32')
            self.assertEqual(config['quantization'], 'none')
            metadata['training_updates'] = 0
            write_json(checkpoint / 'metadata.json', metadata)
            with self.assertRaisesRegex(ValueError, 'no training updates'):
                read_runtime(selector_checkpoint=str(checkpoint))


if __name__ == '__main__':
    unittest.main()
