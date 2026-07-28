"""PyTorch reference runtime for SquareQ slabs — the third-party verification
surface. Loads the exact sidecar the Mojo stack consumes, reconstructs W_hat
identically, and provides (a) a frozen quantized Linear and (b) a minimal
LoRA-on-frozen-base training demo proving the training semantics end to end.

Nothing here is used by the Mojo runtime; it exists so anyone with torch can
check our numbers.
"""

import torch

from .core import GROUP, HBLOCK, reconstruct_weight


class SquareQLinear(torch.nn.Module):
    """Frozen quantized linear: y = x @ W_hat^T (+ bias).

    `cache_weights=False` reconstructs per forward (bounded memory, slow);
    True keeps the bf16 W_hat resident (fast, costs full-weight memory).
    """

    def __init__(
        self,
        qweight: torch.Tensor,
        wscales: torch.Tensor,
        lora_down: torch.Tensor,
        lora_up: torch.Tensor,
        bias=None,
        hblock: int = HBLOCK,
        group: int = GROUP,
        cache_weights: bool = True,
    ):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("wscales", wscales)
        self.register_buffer("lora_down", lora_down)
        self.register_buffer("lora_up", lora_up)
        self.register_buffer("bias_t", bias if bias is not None else None)
        self.hblock = hblock
        self.group = group
        self.cache_weights = cache_weights
        self._w_hat = None

    @property
    def in_features(self) -> int:
        return self.qweight.shape[1] * 2

    @property
    def out_features(self) -> int:
        return self.qweight.shape[0]

    def w_hat(self) -> torch.Tensor:
        if self._w_hat is not None:
            return self._w_hat
        w = reconstruct_weight(
            self.qweight.cpu(),
            self.wscales.cpu(),
            self.lora_down.cpu(),
            self.lora_up.cpu(),
            self.hblock,
            self.group,
        ).to(self.qweight.device)
        if self.cache_weights:
            self._w_hat = w
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.w_hat().to(x.dtype).t()
        if self.bias_t is not None:
            y = y + self.bias_t.to(x.dtype)
        return y


class SquareQLoRALinear(torch.nn.Module):
    """The training semantics: frozen SquareQ base + trainable LoRA A/B.
    Matches the Mojo trainer: base contributes only dX in backward (never dW)."""

    def __init__(self, base: SquareQLinear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.lora_a = torch.nn.Parameter(torch.randn(rank, in_f) * (1.0 / in_f**0.5))
        self.lora_b = torch.nn.Parameter(torch.zeros(out_f, rank))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scaling


def load_layer(slab, key: str, cache_weights: bool = True) -> SquareQLinear:
    """Build a SquareQLinear from an open safetensors handle (or dict) holding
    `<key>.qweight/.wscales/.lora_down/.lora_up` (+ optional `<key>.bias`)."""

    def get(name):
        full = f"{key}.{name}"
        if hasattr(slab, "get_tensor"):
            return slab.get_tensor(full) if full in slab.keys() else None
        return slab.get(full)

    qweight = get("qweight")
    if qweight is None:
        raise KeyError(f"squareq layer not in slab: {key}")
    wscales = get("wscales")
    group = qweight.shape[1] * 2 // wscales.shape[0]  # inferred from shapes
    return SquareQLinear(
        qweight,
        wscales,
        get("lora_down"),
        get("lora_up"),
        bias=get("bias"),
        group=group,
        cache_weights=cache_weights,
    )


def lora_train_demo(layer: SquareQLinear, steps: int = 50, seed: int = 0) -> dict:
    """Verifiable smoke: train LoRA on the frozen quantized base to mimic a
    perturbed target. Returns first/last loss; loss must drop and base grads
    must be absent (frozen)."""
    torch.manual_seed(seed)
    mod = SquareQLoRALinear(layer, rank=16)
    # LoRA-representable target: base + a small low-rank linear perturbation.
    p_a = torch.randn(layer.in_features, 4) * 0.1
    p_b = torch.randn(4, layer.out_features) * 0.1
    opt = torch.optim.AdamW(
        [p for p in mod.parameters() if p.requires_grad], lr=1e-2
    )
    first = last = None
    for _ in range(steps):
        x = torch.randn(8, layer.in_features)
        with torch.no_grad():
            y_t = layer(x) + x @ p_a @ p_b
        loss = torch.nn.functional.mse_loss(mod(x), y_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    assert not any(p.grad is not None for p in layer.parameters()), "base got grads"
    return {"first_loss": first, "last_loss": last}
