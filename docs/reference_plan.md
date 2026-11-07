# Plan: Question-Conditioned Visual Evidence Selector on Frozen Qwen2.5-VL

## 1. Goal and hypothesis

**Goal.** Train a small module that sits between the (frozen) vision encoder + merger and the (frozen) LLM. It selects which visual tokens matter for a given question and options, using only the multiple-choice answer as supervision.

**Hypothesis.** A question-conditioned *selector* (it can only re-weight existing visual tokens, not invent new content) will improve or at least preserve accuracy relative to zero-shot Qwen, and the tokens it keeps will be causally relevant to the answer.

**What would count as success.**
1. Accuracy >= zero-shot baseline at a reduced token budget (e.g. keeping 30% of tokens).
2. The selector beats the controls in section 6, especially the text-only gate and the free-transform adapter.
3. Faithfulness tests pass (section 7).

Gains in accuracy alone are not evidence of "evidence extraction"; the controls are what establish that.

---

## 2. Architecture

```
image -> ViT (frozen) -> merger (frozen) -> V [N, D] --+
                                                       +--> Selector (trainable) -> V' [N, D]
question + options -> embed_tokens (frozen) -> T [L, D] +                            |
                                                                                     v
                       prompt embeds with <|image_pad|> positions replaced by V' -> LLM (frozen) -> LM head (frozen)
```

**Insertion point.** After the merger, so tokens are already in LLM hidden space (dimension D).

**Inputs to the selector.**
- `V`: merged visual tokens, `[N, D]`.
- `T`: embeddings of the *question + options + instruction* span only (from the frozen `embed_tokens`). Use the tokens after `<|vision_end|>` up to the end of the user turn. Exclude the system prompt.

**Why the selector needs the text.** The image comes before the question in the prompt, and the LLM is causal, so visual tokens cannot "see" the question inside the LLM. The selector is the only place where image and question interact before the LLM reads the image.

**Module design (v1).**
- Project `V` and `T` to a small width `d = 512` (linear layers).
- 2 blocks of: cross-attention (queries = visual tokens, keys/values = text tokens) -> MLP, with pre-LayerNorm and residuals. Optional: add a visual self-attention layer so a token's score can depend on its neighbors (cost O(N^2), fine for N up to ~1000).
- Linear head -> one scalar logit per visual token -> `g = sigmoid(logit)` in [0, 1].
- Output: `V' = g * V + (1 - g) * null`, where `null` is a learnable D-vector (initialize to the mean visual token from a few training images, not zeros, so the LLM input stays in-distribution).

**Why a gate rather than a free transform.** A free transform can output arbitrary vectors and can learn answer-letter priors or generic "answer better" shifts. The gate can only blend existing image tokens with one shared null vector, so the channel for shortcuts is much narrower (not zero: it can still pick tokens that correlate with dataset artifacts).

**Initialization.** Head weights ~ N(0, 0.02), bias = +3, so g ~ 0.95 at the start (close to baseline but with usable sigmoid gradients). Do **not** use a huge bias such as +8; the sigmoid gradient would be ~0 and training stalls. For plumbing verification (Phase 0) force `g = 1` exactly.

**What stays fixed.** Sequence length, number of `<|image_pad|>` placeholders, attention mask, and M-RoPE position ids all stay the same, because tokens are replaced in place and never deleted. (Physically deleting tokens would break position ids.)

---

## 3. Loss

**Answer loss (restricted-letter cross entropy).** Take the LM logits at the last prompt position (the one that predicts the first answer token), gather logits for the letter tokens A-E, mask letters the question doesn't have, and apply cross entropy. This handles 3/4/5 options.

**Budget loss.** Without a budget the trivial solution is g = 1 everywhere. Use a hinge so it only penalizes exceeding the budget:

`L_budget = relu(mean(g) - rho)`, averaged over the batch.

**Total.** `L = L_CE + lambda_b * L_budget`, with `lambda_b` starting at 2.0 (tune on validation).

**Budget schedule.** Anneal `rho` from 1.0 down to the target (e.g. 0.3) over the first 20% of steps, then hold. Run the sweep rho in {1.0, 0.5, 0.3, 0.1} as separate runs.

**Train/test mismatch.** Training uses soft gates; you may want hard selection at test time. Plan: evaluate both soft gates and hard top-k (keep the top `rho*N` tokens by score, others replaced by `null`). If hard top-k loses accuracy vs soft, do a short final stage (last ~20% of steps) with straight-through top-k.

---

## 4. Data handling

- Fixed prompt template via the chat template with `add_generation_prompt=True`, ending with an instruction like: *"Answer with the option's letter from the given choices directly."* The last position then predicts the first answer token.
- **Letter token ids:** verify with `tokenizer.encode("A", add_special_tokens=False)` that each letter is a single token and that it matches what the model emits right after `assistant\n` ("A" vs " A" differ).
- **Shuffle option order every epoch** (and remap the gold index). This removes letter-position bias, so the module cannot profit from "B is common".
- Report the gold-letter distribution of train/val/test, and track the predicted-letter distribution during eval.
- Cap image size (`max_pixels`) so N stays bounded, e.g. 256-1024 merged tokens. Keep this identical across all runs and baselines.
- Split into train/val/test. Choose all hyperparameters (rho, lambda_b, lr) on val only. Touch test once at the end.

---

## 5. Training setup

| Item | Setting (starting point) |
|---|---|
| Backbone | Qwen2.5-VL-3B-Instruct for iteration; 7B for final numbers if memory allows |
| Trainable params | Selector only (a few million params). All Qwen params `requires_grad=False` |
| Precision | bf16 for Qwen, selector in bf16 or fp32; **loss computed in fp32** |
| Batch | Start with batch size 1 and gradient accumulation to effective 32 (avoids splitting variable-length visual tokens across a batch) |
| Optimizer | AdamW, lr 1e-4 (try 3e-4), weight decay 0.01, 5% warmup, cosine decay |
| Epochs | 3-5, evaluate on val every ~500 steps |
| Memory | Gradients flow back *through* the frozen LLM, so activations are stored. Use gradient checkpointing. As a rough guide, 3B fits on a 24 GB GPU at a few hundred tokens; 7B typically wants 40-80 GB |
| Seeds | 3 seeds for the main configuration |

Notes:
- Run the ViT + merger under `torch.no_grad()` (frozen and upstream of the selector); it saves memory.
- Since the selector output feeds `inputs_embeds` and carries grad, you do not need `enable_input_require_grads()`.

---

## 6. Baselines and controls

| ID | Run | What it tells you |
|---|---|---|
| B0 | Zero-shot Qwen, same prompt and restricted-letter scoring | The reference accuracy |
| B1 | Blank or no image | How much the questions can be answered from text alone |
| B2 | Random gate at the same budget rho | Whether learned selection beats chance selection |
| B3 | **Text-only gate:** same module, but visual *content* is replaced by position embeddings only (still sees the text) | If this matches the real selector, it is learning priors, not evidence |
| B4 | **Free-transform adapter** (residual cross-attention, no gating, similar parameter count) | Shows whether unconstrained modules gain via shortcuts and how the gate compares |
| B5 | Real selector with option order *not* shuffled | Quantifies letter-prior exploitation |
| B6 (optional) | LoRA on the LLM, or unfreezing the merger | Upper-bound reference for how much the frozen setup leaves on the table |

**Ablations on the main model:** budget rho; null token type (learned vs mean vs zero); with vs without question conditioning (at test time, feed a *mismatched* question to the selector and measure the drop).

---

## 7. Evaluation

**Accuracy.**
- Accuracy on test with 95% bootstrap CIs; paired comparison against B0 (McNemar).
- Accuracy under *re-shuffled* option order at test time (should not change much).
- Predicted-letter histogram vs gold-letter histogram.

**Faithfulness (does the selector pick evidence?).**
1. *Keep-only:* keep only the selected tokens (others -> null). Accuracy should stay close to the full-image result.
2. *Remove-selected:* replace the top-k selected tokens with null. Accuracy should drop clearly more than removing k *random* tokens.
3. *Deletion/insertion curves:* sweep k and plot accuracy for selector order vs random order.
4. *Question sensitivity:* run the selector with a different question on the same image; gate maps should differ (report mean overlap between the two keep-sets).
5. *Localization (if the dataset has region or box annotations):* project gates onto the image grid (the merged grid is `(h/2) x (w/2)` from `image_grid_thw`) and report IoU / pointing-game accuracy against the annotation.

---

## 8. Phases and exit criteria

**Phase 0: Plumbing correctness (do this first).**
- Build the custom forward (see section 10): ViT -> merger -> in-place scatter -> explicit `position_ids` -> LLM.
- Exit: with `g = 1` forced, logits match the stock `model(**inputs)` forward (`torch.allclose` with a bf16-appropriate tolerance). Verify letter-token ids and zero-shot accuracy (B0) are sensible.

**Phase 1: Overfit sanity check.**
- Train on ~64 examples with no budget loss.
- Exit: train CE falls near 0 and the gates move off their initial value. If not, debug gradient flow (section 9).

**Phase 2: Main training.**
- Full train with the rho sweep, on val for selection. Exit: a configuration that matches or beats B0 at rho < 1.

**Phase 3: Controls (B1-B5).**
- Exit: the real selector beats B3 and B4 in the faithfulness tests, not just in accuracy.

**Phase 4: Faithfulness analysis and test-set report.**
- Run section 7 once on test, with 3 seeds for the main config.

**Phase 5 (optional upgrades).**
- Contextual text features (a short text-only pass through early LLM layers instead of raw embeddings), straight-through top-k stage, or comparison against B6.

---

## 9. Failure modes and what to do

| Symptom | Likely cause | Action |
|---|---|---|
| Gates stay ~1 | Budget too weak | Raise `lambda_b`, check `rho` schedule is actually decreasing |
| Gates collapse to ~0, accuracy tanks | Budget too strong or rho too low | Lower `lambda_b`, raise rho, lengthen the anneal |
| Accuracy below B0 even at rho = 1 | Plumbing bug (positions, placeholder count, null token) | Re-run the Phase 0 equivalence test |
| Loss does not move / grad is None | Graph broken | Confirm `V'` requires grad, Qwen params frozen but not detached, no `torch.no_grad()` around the LLM |
| Accuracy up but B3 or B4 match it | Learning priors or shortcuts, not evidence | Report honestly; tighten the design (stronger budget, shuffle options, drop free-form capacity) |
| Hard top-k much worse than soft | Train/test mismatch | Add the straight-through top-k stage |
| Little or no gain over B0 | Frozen LLM already near its ceiling on this data | Report as a finding; compare to B6 to see where headroom exists |

---

## 10. Code skeleton (verify against your installed `transformers` version)

```python
import torch, torch.nn as nn, torch.nn.functional as F

class CrossBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, h, t, t_pad):
        a, _ = self.attn(self.n1(h), t, t, key_padding_mask=t_pad)
        h = h + a
        return h + self.mlp(self.n2(h))

class EvidenceSelector(nn.Module):
    def __init__(self, D, d=512, heads=8, layers=2, init_bias=3.0):
        super().__init__()
        self.v_in, self.t_in = nn.Linear(D, d), nn.Linear(D, d)
        self.blocks = nn.ModuleList([CrossBlock(d, heads) for _ in range(layers)])
        self.score = nn.Linear(d, 1)
        nn.init.normal_(self.score.weight, std=0.02)
        nn.init.constant_(self.score.bias, init_bias)
        self.null = nn.Parameter(torch.zeros(D))  # set to mean visual token before training

    def forward(self, V, T, t_pad=None, force_identity=False):
        # V: [B,N,D], T: [B,L,D], t_pad: [B,L] True = padding
        if force_identity:
            return V, torch.ones(V.shape[:2], device=V.device)
        h, t = self.v_in(V), self.t_in(T)
        for blk in self.blocks:
            h = blk(h, t, t_pad)
        g = torch.sigmoid(self.score(h)).squeeze(-1)             # [B,N]
        Vp = g.unsqueeze(-1) * V + (1 - g.unsqueeze(-1)) * self.null
        return Vp, g

def answer_loss(logits_last, letter_ids, n_opts, gold):
    z = logits_last[:, letter_ids].float()                        # [B,5]
    valid = torch.arange(len(letter_ids), device=z.device)[None] < n_opts[:, None]
    z = z.masked_fill(~valid, float("-inf"))
    return F.cross_entropy(z, gold), z

def forward_with_selector(model, selector, batch, rho, lam_b, letter_ids, force_identity=False):
    ids, attn = batch["input_ids"], batch["attention_mask"]
    with torch.no_grad():
        V = model.visual(batch["pixel_values"], grid_thw=batch["image_grid_thw"])  # [N,D], merger included
    embeds = model.get_input_embeddings()(ids)                                       # [1,S,D] (batch size 1)
    img_mask = ids == model.config.image_token_id
    T = embeds[batch["text_span_mask"]][None]                                        # question+options span
    Vp, g = selector(V[None].to(embeds.dtype), T, None, force_identity)
    embeds = embeds.masked_scatter(img_mask.unsqueeze(-1), Vp[0].to(embeds.dtype))
    pos, _ = model.get_rope_index(input_ids=ids, image_grid_thw=batch["image_grid_thw"],
                                  attention_mask=attn)  # may live at model.model.get_rope_index in newer versions
    out = model(inputs_embeds=embeds, position_ids=pos, attention_mask=attn)
    ce, _ = answer_loss(out.logits[:, -1], letter_ids, batch["n_opts"], batch["gold"])
    budget = F.relu(g.mean(dim=1) - rho).mean()
    return ce + lam_b * budget, ce, g
```

---

## 11. Open items to settle before starting

1. **Model size** (3B vs 7B) and GPU memory available.
2. **Dataset** (size, image resolution, whether region annotations exist for the localization test).
3. The "three-level hints" evaluation shown in your diagram is not defined in this plan because I don't know what it refers to. Tell me what it is and I'll add it to section 7.
