"""Validate bundled data and selector weights; optionally generate a real hint."""
import argparse
import json

from project_data import ROOT, load_questions
from runtime_settings import read_runtime, runtime_status
from ui_server import ensure_data_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true', help='Load Qwen and generate one hint (requires CUDA by default).')
    args = parser.parse_args()
    ensure_data_ready()
    config = read_runtime()
    import torch
    from checkpoints import checkpoint_metadata
    from safetensors.torch import load_file
    from selector import EvidenceSelector, SelectorConfig
    metadata = checkpoint_metadata(config['selector_checkpoint'])
    selector = EvidenceSelector(SelectorConfig(**metadata['selector_config']))
    selector.load_state_dict(load_file(str(ROOT / config['selector_checkpoint'] / 'selector.safetensors')), strict=True)
    if not bool(selector.null_initialized):
        raise ValueError('Checkpoint null token was not initialized')
    counts = {split: len(load_questions(split=split)) for split in ('train', 'validation', 'test')}
    print(json.dumps({'selector': runtime_status(config), 'questions': counts,
                      'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()}, indent=2))
    if not torch.cuda.is_available() and config['device'] == 'cuda':
        print('Hint generation requires a CUDA-capable NVIDIA GPU and CUDA-enabled PyTorch. See README.md.')
    if args.generate:
        from qwen_engine import QwenTutor
        from scienceqa_tutor import load_question, run_hint
        item_id = load_questions(split='validation')[0]['item_id']
        item = load_question(ROOT, 'validation', item_id)
        result = run_hint(QwenTutor(), item)
        if not result['hint'] or result['selection']['mode'] != 'soft':
            raise RuntimeError('Expected a nonempty hint generated with the trained soft selector')
        print(json.dumps({'item_id': item_id, 'hint': result['hint'], 'selection': result['selection']}, indent=2))


if __name__ == '__main__':
    main()
