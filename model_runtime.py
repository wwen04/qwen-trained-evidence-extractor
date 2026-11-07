"""Frozen Qwen with a differentiable visual-selector boundary.

Uses the installed Transformers precomputed multimodal feature interface. The
vision output is restored to native image order by Qwen before selection.
"""
import inspect

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling

from checkpoints import backbone_signature, checkpoint_metadata, load_checkpoint
from project_data import public_text, digest
from selector import EvidenceSelector, SelectorConfig
from runtime_settings import read_runtime, runtime_status

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
CONDITIONING = "public-question-context-options-v1"
ANSWER_INSTRUCTION = "\n\nAnswer with the option's uppercase letter only, with no leading space or explanation."


def selected_encoder_output(visual):
    return {"image": BaseModelOutputWithPooling(pooler_output=(visual,))}


class QwenSelectorRuntime:
    def __init__(self, config, *, training=False):
        self.config = dict(config)
        self.training = training
        if not torch.cuda.is_available() and config.get("device", "cuda") != "cpu":
            raise RuntimeError("A CUDA GPU is required. CPU is available only when explicitly configured.")
        self.device = torch.device(config.get("device", "cuda"))
        precision = config.get("precision", "auto")
        if precision == "auto":
            precision = "bfloat16" if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32" if self.device.type == "cpu" else "float16"
        if training and precision == "float16":
            raise ValueError("Use bfloat16 or float32 for selector training; float16 needs a separate loss-scaling path")
        dtype = getattr(torch, precision)
        self.config["precision"] = precision
        kwargs = {"dtype": dtype, "device_map": {"": str(self.device)}, "revision": config.get("revision")}
        if config.get("quantization") == "4bit":
            if self.device.type != "cuda":
                raise ValueError("4-bit execution requires CUDA")
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        elif config.get("quantization", "none") != "none":
            raise ValueError("quantization must be none or 4bit")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.get("model_path", DEFAULT_MODEL), **kwargs)
        if "mm_encoder_outputs" not in inspect.signature(self.model.forward).parameters:
            raise RuntimeError("This Transformers version lacks precomputed multimodal inputs; install this project's pinned requirements")
        self.model.requires_grad_(False)
        self.model.eval()
        if training and config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            # Checkpointing blocks use their training flag; do not enable gradients on weights.
            self.model.model.language_model.train()
        self.processor = AutoProcessor.from_pretrained(config.get("model_path", DEFAULT_MODEL), revision=config.get("revision"))
        checkpoint = config.get("selector_checkpoint")
        if checkpoint:
            selector_config = SelectorConfig(**checkpoint_metadata(checkpoint)["selector_config"])
        else:
            selector_config = SelectorConfig(hidden_size=self.model.config.text_config.hidden_size, **config.get("selector", {}))
        self.selector = EvidenceSelector(selector_config).to(device=self.device, dtype=torch.float32)
        self.metadata = load_checkpoint(checkpoint, self.selector, self.model.config) if checkpoint else None
        self.selector.train(training)
        self.letter_ids = self.verify_letters()
        self.last_selection = None

    def verify_letters(self):
        tokenizer = self.processor.tokenizer
        ids = []
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            tokens = tokenizer.encode(letter, add_special_tokens=False)
            if len(tokens) != 1 or tokenizer.decode(tokens) != letter:
                raise ValueError(f"Answer letter {letter} is not exactly one round-tripping token")
            ids.append(tokens[0])
        if len(set(ids)) != 26:
            raise ValueError("Answer-letter token IDs are not unique")
        # Check the actual assistant boundary, not just isolated letters.
        prompt = self.processor.apply_chat_template([{"role": "user", "content": "Answer with A."}], tokenize=False, add_generation_prompt=True)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        for letter, token_id in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", ids):
            appended = tokenizer.encode(prompt + letter, add_special_tokens=False)
            if appended != prefix + [token_id]:
                raise ValueError("Answer tokenization changes at the assistant boundary")
        return ids

    def prepare(self, item, instruction, *, include_image=True):
        content = []
        image = None
        if include_image:
            with Image.open(item["image_path"]) as source:
                image = source.convert("RGB")
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": instruction})
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        kwargs = {"text": [prompt], "padding": True, "return_tensors": "pt"}
        if image is not None:
            kwargs.update(images=[image], images_kwargs={"size": {
                "shortest_edge": self.config.get("image_min_tokens", 256) * 28 * 28,
                "longest_edge": self.config.get("image_max_tokens", 512) * 28 * 28}})
        inputs = self.processor(**kwargs).to(self.device)
        # These inputs deliberately exclude task instructions, attempts, reference
        # answers and hints: the same conditioning is used during QA and tutoring.
        text_ids = self.processor.tokenizer(public_text(item), add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        with torch.no_grad():
            text = self.model.get_input_embeddings()(text_ids)[0]
        return inputs, text

    @torch.no_grad()
    def visual_features(self, inputs):
        output = self.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
        if len(output.pooler_output) != 1:
            raise ValueError("Exactly one image per microbatch is required")
        visual = output.pooler_output[0]
        count = (inputs["input_ids"] == self.model.config.image_token_id).sum().item()
        if count != len(visual):
            raise ValueError("Visual-token and image-placeholder counts differ")
        return visual

    def select(self, inputs, text, *, mode=None, rho=None, seed=0, visual=None):
        visual = self.visual_features(inputs) if visual is None else visual
        mode = mode or self.config.get("gate_mode", "identity")
        rho = self.config.get("rho", 1.0) if rho is None else rho
        modified, gates = self.selector(visual, text, mode=mode, rho=rho, random_seed=seed)
        features = {k: v for k,v in inputs.items() if k not in {"pixel_values", "pixel_values_videos"}}
        features["mm_encoder_outputs"] = selected_encoder_output(modified)
        return features, gates

    def answer_forward(self, item, *, mode=None, rho=None, seed=0, conditioning_item=None, visual=None):
        instruction = public_text(item) + ANSWER_INSTRUCTION
        include_image = mode != "no_image"
        inputs, text = self.prepare(item, instruction, include_image=include_image)
        if conditioning_item is not None:
            ids = self.processor.tokenizer(public_text(conditioning_item), add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
            with torch.no_grad():
                text = self.model.get_input_embeddings()(ids)[0]
        if include_image:
            features, gates = self.select(inputs, text, mode=mode, rho=rho, seed=seed, visual=visual)
        else:
            features, gates = dict(inputs), torch.empty(0, device=self.device)
        # Do not use no_grad/inference_mode here: the frozen decoder must transmit
        # loss gradients to the selected image embeddings.
        output = self.model(**features, use_cache=False, logits_to_keep=1)
        return output.logits[:, -1, :], gates

    @torch.no_grad()
    def letter_probe(self, item):
        instruction = public_text(item) + ANSWER_INSTRUCTION
        inputs, _ = self.prepare(item, instruction)
        was_training = self.model.model.language_model.training
        self.model.eval()
        generated = self.model.generate(**inputs, max_new_tokens=1, do_sample=False)
        token = generated[0, inputs['input_ids'].shape[1]].item()
        if was_training:
            self.model.model.language_model.train()
        return {"first_generated_token":token, "decoded":self.processor.tokenizer.decode([token]),
                "is_valid_option_letter":token in self.letter_ids[:len(item['choices'])],
                "note":"A diagnostic on one item, not an accuracy measurement. Restricted-letter scores are reported separately."}

    @torch.no_grad()
    def generate(self, item, instruction, *, max_new_tokens=160, mode=None, rho=None):
        inputs, text = self.prepare(item, instruction)
        features, gates = self.select(inputs, text, mode=mode, rho=rho)
        result = self.model.generate(**features, max_new_tokens=max_new_tokens, do_sample=False)
        text_result = self.processor.batch_decode(result[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)[0].strip()
        effective_mode = mode or self.config.get("gate_mode", "identity")
        self.last_selection = {**runtime_status({**self.config, "gate_mode":effective_mode}), "mode": effective_mode, "visual_tokens": len(gates),
            "mean_gate": float(gates.mean()), "gates": gates.float().cpu().tolist(),
            "image_grid_thw": inputs["image_grid_thw"].cpu().tolist(),
            "spatial_merge_size": self.model.config.vision_config.spatial_merge_size,
            "weights_sha256": self.metadata["weights_sha256"] if self.metadata else None,
            "note": "Gates are model feature weights, not verified observations or proof of causal evidence. Sequence length is unchanged."}
        return text_result

    @torch.no_grad()
    def identity_check(self, item):
        inputs, text = self.prepare(item, public_text(item) + "\nAnswer with the option letter only.")
        self.model.eval()
        original = self.model(**inputs, use_cache=False, logits_to_keep=1).logits
        modified, _ = self.select(inputs, text, mode="identity")
        new = self.model(**modified, use_cache=False, logits_to_keep=1).logits
        difference = float((original.float() - new.float()).abs().max())
        ok = torch.allclose(original.float(), new.float(), atol=0.02, rtol=0.01)
        if self.training and self.config.get("gradient_checkpointing", True):
            self.model.model.language_model.train()
        if not ok:
            raise RuntimeError(f"Identity path differs from stock Qwen: max difference {difference}")
        return {"passed": True, "max_logit_difference": difference}

    def provenance(self):
        return {"backbone": {"model_path": self.config.get("model_path", DEFAULT_MODEL),
                             "revision": self.config.get("revision"),
                             "resolved_revision": getattr(self.model.config, "_commit_hash", None)},
                "backbone_signature": backbone_signature(self.model.config),
                "conditioning": CONDITIONING,
                "answer_instruction_sha256": digest(ANSWER_INSTRUCTION),
                "preprocessing": {k: self.config[k] for k in ("image_min_tokens", "image_max_tokens")},
                "letter_token_ids": self.letter_ids, "precision": self.config.get("precision"),
                "quantization": self.config.get("quantization")}
