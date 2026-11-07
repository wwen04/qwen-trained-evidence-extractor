"""Pure tensor and tiny random-model checks; no optimizer steps or pretrained downloads."""
import tempfile
from pathlib import Path
import unittest

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from checkpoints import backbone_signature, load_checkpoint, save_checkpoint
from model_runtime import selected_encoder_output
from selector import EvidenceSelector, SelectorConfig, answer_loss, budget_target, hard_mask


def tiny_model():
    config = Qwen2_5_VLConfig(
        text_config={"vocab_size":80,"hidden_size":48,"intermediate_size":96,"num_hidden_layers":2,
                     "num_attention_heads":4,"num_key_value_heads":2,"max_position_embeddings":128,
                     "bos_token_id":0,"eos_token_id":2,"pad_token_id":1,
                     "rope_parameters":{"rope_type":"default","rope_theta":10000.0,"mrope_section":[2,2,2]}},
        vision_config={"depth":1,"hidden_size":32,"intermediate_size":64,"num_heads":4,"out_hidden_size":48,
                       "patch_size":14,"spatial_merge_size":2,"fullatt_block_indexes":[0]},
        image_token_id=60,video_token_id=61,vision_start_token_id=62,vision_end_token_id=63)
    config._attn_implementation = "eager"
    model = Qwen2_5_VLForConditionalGeneration(config).eval().requires_grad_(False)
    ids = torch.tensor([[10,62,60,60,60,60,63,11,12,13]])
    inputs = {"input_ids":ids,"attention_mask":torch.ones_like(ids),"image_grid_thw":torch.tensor([[1,4,4]]),
              "mm_token_type_ids":(ids == 60).long(),"pixel_values":torch.randn(16,3*2*14*14)}
    return model, inputs


class SelectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.selector = EvidenceSelector(SelectorConfig(48,width=16,heads=4,layers=1))
        self.visual, self.text = torch.randn(8,48), torch.randn(5,48)

    def test_identity_before_null_initialization(self):
        output,gates = self.selector(self.visual,self.text,mode="identity")
        self.assertTrue(torch.equal(output,self.visual))
        self.assertTrue(torch.equal(gates,torch.ones(8)))
        with self.assertRaises(ValueError):
            self.selector(self.visual,self.text)

    def test_keep_remove_and_random_budgets(self):
        self.selector.initialize_null(self.visual.mean(0))
        kept,gates = self.selector(self.visual,self.text,mode="hard",rho=0.25)
        removed,other = self.selector(self.visual,self.text,mode="remove",rho=0.25)
        self.assertEqual(gates.sum().item(),2)
        self.assertTrue(torch.equal(other,1-gates))
        self.assertTrue(torch.equal(kept[gates.bool()],self.visual[gates.bool()]))
        self.assertTrue(torch.equal(removed[~gates.bool()],self.visual[~gates.bool()]))
        self.assertTrue(torch.equal(hard_mask(gates,0.25,random_seed=42),hard_mask(gates,0.25,random_seed=42)))

    def test_text_only_scores_ignore_visual_content(self):
        module = EvidenceSelector(SelectorConfig(48,width=16,heads=4,layers=1,variant="text_only",null_type="zero"))
        _,a = module(self.visual,self.text)
        _,b = module(self.visual+10,self.text)
        self.assertTrue(torch.equal(a,b))

    def test_straight_through_gradient_no_weight_update(self):
        self.selector.initialize_null(self.visual.mean(0))
        before = {k:v.clone() for k,v in self.selector.state_dict().items()}
        output,gates = self.selector(self.visual,self.text,mode="straight_through",rho=0.25)
        (output.square().sum()+gates.sum()).backward()
        self.assertGreater(self.selector.score.weight.grad.abs().sum().item(),0)
        for key,value in before.items():
            self.assertTrue(torch.equal(value,self.selector.state_dict()[key]))

    def test_loss_masks_options_by_gather_and_schedule(self):
        logits = torch.tensor([[0.,1.,3.,999.]])
        ce,z = answer_loss(logits,[0,1,2],2)
        self.assertEqual(z.shape,(1,3))
        self.assertLess(ce.item(),0.2)
        self.assertEqual(budget_target(0,100,0.3),1.0)
        self.assertAlmostEqual(budget_target(20,100,0.3),0.3)

    def test_checkpoint_roundtrip_and_tampering(self):
        model,_ = tiny_model()
        self.selector.initialize_null(self.visual.mean(0))
        with tempfile.TemporaryDirectory() as folder:
            provenance = {"backbone_signature":backbone_signature(model.config),"conditioning":"public-question-context-options-v1"}
            save_checkpoint(folder,self.selector,provenance)
            target = EvidenceSelector(self.selector.config)
            load_checkpoint(folder,target,model.config)
            for key,value in self.selector.state_dict().items():
                self.assertTrue(torch.equal(value,target.state_dict()[key]))
            with (Path(folder)/"selector.safetensors").open("ab") as stream:
                stream.write(b"bad")
            with self.assertRaises(ValueError):
                load_checkpoint(folder,target,model.config)


class QwenBoundaryTests(unittest.TestCase):
    def test_identity_forward_generation_and_gradient(self):
        torch.manual_seed(4)
        model,inputs = tiny_model()
        selector = EvidenceSelector(SelectorConfig(48,width=16,heads=4,layers=1))
        with torch.no_grad():
            visual = model.get_image_features(inputs['pixel_values'],inputs['image_grid_thw'],return_dict=True).pooler_output[0]
            text = model.get_input_embeddings()(inputs['input_ids'][:,-3:])[0]
            original = model(**inputs,use_cache=False).logits
        selected,_ = selector(visual,text,mode="identity")
        kwargs = {k:v for k,v in inputs.items() if k!='pixel_values'}
        kwargs['mm_encoder_outputs'] = selected_encoder_output(selected)
        actual = model(**kwargs,use_cache=False).logits
        self.assertTrue(torch.allclose(original,actual,atol=1e-6,rtol=1e-6))
        with torch.no_grad():
            raw_generated = model.generate(**inputs,max_new_tokens=3,do_sample=False)
            selected_generated = model.generate(**kwargs,max_new_tokens=3,do_sample=False)
        self.assertTrue(torch.equal(raw_generated,selected_generated))
        selector.initialize_null(visual.mean(0))
        selected,_ = selector(visual,text)
        kwargs['mm_encoder_outputs'] = selected_encoder_output(selected)
        before = {k:v.clone() for k,v in selector.state_dict().items()}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.model.language_model.train()
        loss,_ = answer_loss(model(**kwargs,use_cache=False,logits_to_keep=1).logits[:,-1], [5,6,7], 1)
        loss.backward()
        self.assertGreater(selector.score.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(v,selector.state_dict()[k]) for k,v in before.items()))


if __name__ == '__main__':
    unittest.main()
