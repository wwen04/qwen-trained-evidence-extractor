"""Question-conditioned visual gating. No backbone weights are owned here."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class SelectorConfig:
    hidden_size: int
    width: int = 512
    heads: int = 8
    layers: int = 2
    init_bias: float = 3.0
    variant: str = "visual"
    null_type: str = "learned"

    def __post_init__(self):
        if min(self.hidden_size, self.width, self.heads, self.layers) < 1 or self.width % self.heads:
            raise ValueError("Positive dimensions and width divisible by heads are required")
        if self.variant not in {"visual", "text_only", "free_transform"}:
            raise ValueError("Unknown selector variant")
        if self.null_type not in {"learned", "mean", "zero"}:
            raise ValueError("Unknown null-token type")

    def to_dict(self):
        return asdict(self)


class CrossBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.visual_norm = nn.LayerNorm(width)
        self.text_norm = nn.LayerNorm(width)
        self.output_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, visual, text):
        text = self.text_norm(text)
        attended, _ = self.attention(self.visual_norm(visual), text, text, need_weights=False)
        visual = visual + attended
        return visual + self.mlp(self.output_norm(visual))


def keep_count(count, rho):
    if not 0 <= rho <= 1:
        raise ValueError("rho must be between 0 and 1")
    return min(count, max(0, math.ceil(count * rho)))


def hard_mask(scores, rho, *, random_seed=None):
    count = scores.shape[-1]
    k = keep_count(count, rho)
    mask = torch.zeros_like(scores)
    if random_seed is None:
        indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]
    else:
        generator = torch.Generator(device="cpu").manual_seed(random_seed)
        indices = torch.randperm(count, generator=generator)[:k].to(scores.device)
        indices = indices.expand(*scores.shape[:-1], k)
    return mask.scatter(-1, indices, 1)


class EvidenceSelector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual_projection = nn.Linear(config.hidden_size, config.width)
        self.text_projection = nn.Linear(config.hidden_size, config.width)
        self.blocks = nn.ModuleList([CrossBlock(config.width, config.heads) for _ in range(config.layers)])
        self.score = nn.Linear(config.width, 1)
        nn.init.normal_(self.score.weight, std=0.02)
        nn.init.constant_(self.score.bias, config.init_bias)
        self.null = nn.Parameter(torch.zeros(config.hidden_size), requires_grad=config.null_type == "learned")
        self.register_buffer("null_initialized", torch.tensor(config.null_type == "zero"))
        if config.variant == "free_transform":
            self.delta = nn.Linear(config.width, config.hidden_size)
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)

    @torch.no_grad()
    def initialize_null(self, mean):
        if mean.shape != self.null.shape or not torch.isfinite(mean).all():
            raise ValueError("Invalid mean visual token")
        if self.config.null_type != "zero":
            self.null.copy_(mean.to(self.null))
        self.null_initialized.fill_(True)

    def forward(self, visual, text, *, mode="soft", rho=1.0, random_seed=0):
        # One image per forward; variable image and text lengths need no padding.
        if visual.ndim != 2 or text.ndim != 2 or not len(visual) or not len(text):
            raise ValueError("Expected nonempty [tokens, hidden_size] visual and text tensors")
        if visual.shape[-1] != self.config.hidden_size or text.shape[-1] != self.config.hidden_size:
            raise ValueError("Feature width differs from selector configuration")
        if mode == "identity":
            return visual, torch.ones(len(visual), device=visual.device)
        if mode not in {"soft", "hard", "straight_through", "remove", "random", "remove_random", "null"}:
            raise ValueError("Unknown gate mode")
        if not bool(self.null_initialized):
            raise ValueError("Initialize the null token from TRAIN images before using gates")
        original_dtype = visual.dtype
        v, t = visual.to(self.null), text.to(self.null)
        if self.config.variant == "text_only":
            # Scores depend on text and token position, never visual content.
            position = torch.arange(len(v), device=v.device, dtype=v.dtype)[:, None]
            frequency = torch.exp(torch.arange(v.shape[-1], device=v.device, dtype=v.dtype)
                                  * (-math.log(10000.0) / v.shape[-1]))[None]
            queries = torch.sin(position * frequency)
        else:
            queries = v
        hidden, encoded_text = self.visual_projection(queries)[None], self.text_projection(t)[None]
        for block in self.blocks:
            hidden = block(hidden, encoded_text)
        scores = torch.sigmoid(self.score(hidden)[0, :, 0])
        if self.config.variant == "free_transform":
            if mode != "soft":
                raise ValueError("The free-transform control supports soft mode only; it has no token budget")
            return (v + self.delta(hidden)[0]).to(original_dtype), torch.ones_like(scores)
        if mode in {"hard", "remove", "straight_through"}:
            selected = hard_mask(scores, rho)
            gates = selected + scores - scores.detach() if mode == "straight_through" else selected
            if mode == "remove":
                gates = 1 - gates
        elif mode in {"random", "remove_random"}:
            gates = hard_mask(scores, rho, random_seed=random_seed)
            if mode == "remove_random":
                gates = 1 - gates
        elif mode == "null":
            gates = torch.zeros_like(scores)
        else:
            gates = scores
        output = gates[:, None] * v + (1 - gates[:, None]) * self.null
        # Return soft scores for a differentiable budget penalty during ST training.
        return output.to(original_dtype), scores if mode == "straight_through" else gates


def answer_loss(logits_last, letter_ids, gold_index):
    """Restricted-option CE; options have already been validated and remapped."""
    if logits_last.ndim != 2 or logits_last.shape[0] != 1:
        raise ValueError("Microbatch size must be one")
    if not 0 <= gold_index < len(letter_ids):
        raise ValueError("Gold option index out of range")
    restricted = logits_last[:, letter_ids].float()
    return nn.functional.cross_entropy(restricted, torch.tensor([gold_index], device=restricted.device)), restricted


def budget_target(step, total_steps, target, anneal_fraction=0.2):
    if total_steps < 1 or not 0 < target <= 1 or not 0 <= anneal_fraction <= 1:
        raise ValueError("Invalid budget schedule")
    if anneal_fraction == 0:
        return target
    progress = min(1.0, step / max(1, math.ceil(total_steps * anneal_fraction)))
    return 1.0 + (target - 1.0) * progress
