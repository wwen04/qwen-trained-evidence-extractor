"""Lazy UI adapter for the frozen Qwen + visual-selector architecture."""
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class QwenTutor:
    def __init__(self, model_path=None, *, runtime_config=None, selector_checkpoint=None, gate_mode=None):
        from model_runtime import QwenSelectorRuntime
        from runtime_settings import read_runtime
        self.config = read_runtime(runtime_config, model_path=model_path, selector_checkpoint=selector_checkpoint, gate_mode=gate_mode)
        self.runtime = QwenSelectorRuntime(self.config)
        self.last_selection = None

    def ask(self, image_path, instruction, max_new_tokens=160, *, evidence_inputs=None, mode=None):
        if not isinstance(evidence_inputs, dict):
            raise ValueError("Public question/context/choices are required for selector conditioning")
        item = {"image_path": str(Path(image_path)), "question": evidence_inputs["question"],
                "context": evidence_inputs.get("context", ""), "choices": evidence_inputs["choices"]}
        output = self.runtime.generate(item, instruction, max_new_tokens=max_new_tokens, mode=mode)
        self.last_selection = self.runtime.last_selection
        return output
