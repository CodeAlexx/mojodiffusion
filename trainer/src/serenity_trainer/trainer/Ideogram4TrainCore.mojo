# Ideogram4TrainCore.mojo — first pure-Mojo Ideogram4 training path.
#
# This is actual optimizer/backward code, not a UI contract:
#   safetensors cache -> LoRA forward on final_layer.linear -> MSE flow loss
#   -> tape backward -> BF16 AdamW parameter update.
#
# Current trainable slice:
#   diffusion_model.final_layer.linear / transformer.final_layer.linear
#
# Ideogram4's full transformer forward in mojodiffusion is inference-only today.
# Until a full hand-written backward is ported for the 34-layer DiT, this trains
# the final projection LoRA from cached pre-final hidden states. The data cache
# uses:
#   hidden.<i>  [tokens, 4608] or [1, tokens, 4608]
#   target.<i>  [tokens, 128]  or [1, tokens, 128]   (noise - clean)
# Single-sample aliases "hidden" / "target" are also accepted.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.autograd import Tape, backward
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.fp8 import load_fp8_dequant
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import (
    add_scalar,
    reshape,
    sub,
    zeros_device,
)
from serenitymojo.models.dit.ideogram4_dit import ideogram4_forward_prefinal_hidden
from serenity_trainer.modelSampler.Ideogram4Sampler import (
    IDEOGRAM4_HIDDEN,
    IDEOGRAM4_PACKED_CHANNELS,
)
from serenity_trainer.module.LoRAModule import LoraAdapter, make_lora_adapter
from serenity_trainer.util.config.TrainConfig import TrainConfig
from serenity_trainer.util.optimizer.adamw_extensions import adamw_step

comptime TArc = ArcPointer[Tensor]
comptime IDEOGRAM4_FINAL_LINEAR_PREFIX = "transformer.final_layer.linear"
comptime IDEOGRAM4_FINAL_LINEAR_DIFFUSION_PREFIX = "diffusion_model.final_layer.linear"


@fieldwise_init
struct Ideogram4FinalLinearTrainResult(Copyable, Movable):
    var loss: Float32
    var b_l1: Float32
    var did_update: Bool


@fieldwise_init
struct Ideogram4FinalLinearTrainSummary(Copyable, Movable):
    var steps: Int
    var samples: Int
    var last_loss: Float32
    var adapter_b_l1: Float32


struct Ideogram4FinalLinearBaseWeights(Movable):
    var weight: Tensor
    var bias: Tensor

    def __init__(out self, var weight: Tensor, var bias: Tensor):
        self.weight = weight^
        self.bias = bias^


struct Ideogram4FinalLinearSample(Copyable, Movable):
    var hidden: TArc
    var target: TArc
    var index: Int

    def __init__(out self, var hidden: TArc, var target: TArc, index: Int):
        self.hidden = hidden^
        self.target = target^
        self.index = index


struct Ideogram4FinalLinearCache(Movable):
    var src: ShardedSafeTensors
    var hidden_keys: List[String]
    var target_keys: List[String]

    def __init__(
        out self,
        var src: ShardedSafeTensors,
        var hidden_keys: List[String],
        var target_keys: List[String],
    ):
        self.src = src^
        self.hidden_keys = hidden_keys^
        self.target_keys = target_keys^

    @staticmethod
    def open(path: String) raises -> Ideogram4FinalLinearCache:
        var src = ShardedSafeTensors.open(path)
        var hk = List[String]()
        var tk = List[String]()
        _discover_final_linear_cache(src, hk, tk)
        if len(hk) == 0:
            raise Error(
                String("Ideogram4FinalLinearCache: no samples in ") + path
                + String(" — expected hidden.<i>/target.<i> or hidden/target")
            )
        if len(hk) != len(tk):
            raise Error("Ideogram4FinalLinearCache: hidden/target key count mismatch")
        return Ideogram4FinalLinearCache(src^, hk^, tk^)

    def len(self) -> Int:
        return len(self.hidden_keys)

    def sample[Hidden: Int, Out: Int](
        self, index: Int, ctx: DeviceContext
    ) raises -> Ideogram4FinalLinearSample:
        if index < 0 or index >= self.len():
            raise Error(
                String("Ideogram4FinalLinearCache.sample: index ") + String(index)
                + String(" out of range")
            )

        var h = cast_tensor(
            Tensor.from_view(self.src.tensor_view(self.hidden_keys[index]), ctx),
            STDtype.BF16,
            ctx,
        )
        var t = cast_tensor(
            Tensor.from_view(self.src.tensor_view(self.target_keys[index]), ctx),
            STDtype.BF16,
            ctx,
        )
        var hf = _as_token_matrix[Hidden](h^, ctx)
        var tf = _as_token_matrix[Out](t^, ctx)
        if hf.shape()[0] != tf.shape()[0]:
            raise Error("Ideogram4FinalLinearCache.sample: token count mismatch")
        return Ideogram4FinalLinearSample(TArc(hf^), TArc(tf^), index)


struct Ideogram4FinalLinearLoRA[Hidden: Int, Out: Int](Movable):
    var adapter: LoraAdapter
    var m_a: Tensor
    var v_a: Tensor
    var m_b: Tensor
    var v_b: Tensor

    def __init__(
        out self,
        var adapter: LoraAdapter,
        var m_a: Tensor,
        var v_a: Tensor,
        var m_b: Tensor,
        var v_b: Tensor,
    ):
        self.adapter = adapter^
        self.m_a = m_a^
        self.v_a = v_a^
        self.m_b = m_b^
        self.v_b = v_b^

    @staticmethod
    def new(
        rank: Int,
        alpha: Float32,
        seed: UInt64,
        ctx: DeviceContext,
    ) raises -> Ideogram4FinalLinearLoRA[Self.Hidden, Self.Out]:
        var adapter = make_lora_adapter(
            Self.Hidden, Self.Out, rank, alpha, seed, ctx
        )
        var m_a = _zeros_like(adapter.a, ctx)
        var v_a = _zeros_like(adapter.a, ctx)
        var m_b = _zeros_like(adapter.b, ctx)
        var v_b = _zeros_like(adapter.b, ctx)
        return Ideogram4FinalLinearLoRA[Self.Hidden, Self.Out](
            adapter^, m_a^, v_a^, m_b^, v_b^
        )

    def b_l1(self, ctx: DeviceContext) raises -> Float32:
        return _l1_sum(self.adapter.b, ctx)

    def train_step(
        mut self,
        base_w: Tensor,
        base_b: Tensor,
        hidden_tokens: Tensor,
        target_velocity: Tensor,
        optimizer_step: Int,
        cfg: TrainConfig,
        ctx: DeviceContext,
    ) raises -> Ideogram4FinalLinearTrainResult:
        if optimizer_step < 1:
            raise Error("Ideogram4FinalLinearLoRA.train_step: optimizer_step must be >= 1")
        _validate_final_shapes[Self.Hidden, Self.Out](
            base_w, base_b, hidden_tokens, target_velocity
        )

        var tape = Tape()
        self.adapter.a.set_id(0)
        self.adapter.b.set_id(0)
        self.adapter.track(tape)

        var raw = _final_linear_lora_forward(
            tape, hidden_tokens, base_w, base_b, self.adapter, ctx
        )
        # ai-toolkit Ideogram4 returns negative velocity from predict_velocity;
        # training target is noise - clean. Match that sign here.
        var predicted = tape.record_mul(raw, _const_like(raw, Float32(-1.0), ctx), ctx)
        var loss = tape.mse_loss(predicted, target_velocity, ctx)
        var host_loss = loss.to_host(ctx)[0]
        var gmap = backward(tape, loss, ctx)

        if gmap.__contains__(self.adapter.a.id):
            adamw_step(
                self.adapter.a,
                self.m_a,
                self.v_a,
                gmap[self.adapter.a.id][],
                optimizer_step,
                cfg.learning_rate,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.weight_decay,
                cfg.stochastic_rounding,
                cfg.seed + UInt32(optimizer_step),
                ctx,
            )
        else:
            var za = _zeros_like(self.adapter.a, ctx)
            adamw_step(
                self.adapter.a,
                self.m_a,
                self.v_a,
                za,
                optimizer_step,
                cfg.learning_rate,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.weight_decay,
                cfg.stochastic_rounding,
                cfg.seed + UInt32(optimizer_step),
                ctx,
            )

        if gmap.__contains__(self.adapter.b.id):
            adamw_step(
                self.adapter.b,
                self.m_b,
                self.v_b,
                gmap[self.adapter.b.id][],
                optimizer_step,
                cfg.learning_rate,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.weight_decay,
                cfg.stochastic_rounding,
                cfg.seed + UInt32(optimizer_step + 7919),
                ctx,
            )
        else:
            var zb = _zeros_like(self.adapter.b, ctx)
            adamw_step(
                self.adapter.b,
                self.m_b,
                self.v_b,
                zb,
                optimizer_step,
                cfg.learning_rate,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.weight_decay,
                cfg.stochastic_rounding,
                cfg.seed + UInt32(optimizer_step + 7919),
                ctx,
            )

        return Ideogram4FinalLinearTrainResult(host_loss, self.b_l1(ctx), True)

    def train_step_from_frozen_trunk[S: Int](
        mut self,
        transformer_weights: ShardedSafeTensors,
        base_w: Tensor,
        base_b: Tensor,
        x_in: Tensor,
        llm_in: Tensor,
        t_in: Tensor,
        indicator: Tensor,
        cosf: Tensor,
        sinf: Tensor,
        target_velocity: Tensor,
        optimizer_step: Int,
        cfg: TrainConfig,
        ctx: DeviceContext,
        num_layers: Int = 34,
        num_heads: Int = 18,
        head_dim: Int = 256,
        hidden_dim: Int = 4608,
    ) raises -> Ideogram4FinalLinearTrainResult:
        var hn3 = ideogram4_forward_prefinal_hidden[S](
            transformer_weights,
            x_in,
            llm_in,
            t_in,
            indicator,
            cosf,
            sinf,
            num_layers,
            num_heads,
            head_dim,
            hidden_dim,
            ctx,
        )
        var hn = _as_token_matrix[Self.Hidden](hn3^, ctx)
        var target = _as_token_matrix[Self.Out](target_velocity.clone(ctx), ctx)
        return self.train_step(
            base_w, base_b, hn^, target^, optimizer_step, cfg, ctx
        )


def make_ideogram4_final_linear_lora(
    rank: Int, alpha: Float32, seed: UInt64, ctx: DeviceContext
) raises -> Ideogram4FinalLinearLoRA[IDEOGRAM4_HIDDEN, IDEOGRAM4_PACKED_CHANNELS]:
    return Ideogram4FinalLinearLoRA[
        IDEOGRAM4_HIDDEN, IDEOGRAM4_PACKED_CHANNELS
    ].new(rank, alpha, seed, ctx)


def load_ideogram4_final_linear_base_weights(
    transformer_dir_or_file: String, ctx: DeviceContext
) raises -> Ideogram4FinalLinearBaseWeights:
    var st = ShardedSafeTensors.open(transformer_dir_or_file)
    var w = load_fp8_dequant(st, String("final_layer.linear.weight"), ctx)
    var b = cast_tensor(
        Tensor.from_view(st.tensor_view(String("final_layer.linear.bias")), ctx),
        STDtype.BF16,
        ctx,
    )
    return Ideogram4FinalLinearBaseWeights(w^, b^)


def train_ideogram4_final_linear_lora_cache[Hidden: Int, Out: Int](
    mut state: Ideogram4FinalLinearLoRA[Hidden, Out],
    base_w: Tensor,
    base_b: Tensor,
    cache_path: String,
    steps: Int,
    cfg: TrainConfig,
    ctx: DeviceContext,
) raises -> Ideogram4FinalLinearTrainSummary:
    if steps < 1:
        raise Error("train_ideogram4_final_linear_lora_cache: steps must be >= 1")
    var cache = Ideogram4FinalLinearCache.open(cache_path)
    var last_loss = Float32(0.0)
    for step in range(1, steps + 1):
        var sample_index = (step - 1) % cache.len()
        var sample = cache.sample[Hidden, Out](sample_index, ctx)
        var result = state.train_step(
            base_w,
            base_b,
            sample.hidden[],
            sample.target[],
            step,
            cfg,
            ctx,
        )
        last_loss = result.loss
    return Ideogram4FinalLinearTrainSummary(
        steps, cache.len(), last_loss, state.b_l1(ctx)
    )


def ideogram4_flow_target_from_noise_and_clean(
    noise_tokens: Tensor, clean_tokens: Tensor, ctx: DeviceContext
) raises -> Tensor:
    return sub(noise_tokens, clean_tokens, ctx)


def _final_linear_lora_forward(
    mut tape: Tape,
    x: Tensor,
    base_w: Tensor,
    base_b: Tensor,
    adapter: LoraAdapter,
    ctx: DeviceContext,
) raises -> Tensor:
    var base = linear(x, base_w, Optional[Tensor](base_b.clone(ctx)), ctx)

    var down_bias_shape = List[Int]()
    down_bias_shape.append(adapter.rank)
    var down_bias = zeros_device(down_bias_shape^, STDtype.BF16, ctx)
    var down = tape.record_linear(x, adapter.a, down_bias, ctx)

    var up_bias_shape = List[Int]()
    up_bias_shape.append(base_w.shape()[0])
    var up_bias = zeros_device(up_bias_shape^, STDtype.BF16, ctx)
    var up = tape.record_linear(down, adapter.b, up_bias, ctx)
    var scaled = tape.record_mul(up, _const_like(up, adapter.scale(), ctx), ctx)
    return tape.record_add(base, scaled, ctx)


def _as_token_matrix[Features: Int](var x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var sh = x.shape()
    if len(sh) == 2:
        if sh[1] != Features:
            raise Error(
                String("Ideogram4 cache tensor feature dim mismatch: got ")
                + String(sh[1]) + String(" expected ") + String(Features)
            )
        return x^
    if len(sh) == 3:
        if sh[0] != 1:
            raise Error("Ideogram4 cache tensor has batched dim > 1")
        if sh[2] != Features:
            raise Error(
                String("Ideogram4 cache tensor feature dim mismatch: got ")
                + String(sh[2]) + String(" expected ") + String(Features)
            )
        var ns = List[Int]()
        ns.append(sh[1])
        ns.append(Features)
        return reshape(x^, ns^, ctx)
    raise Error("Ideogram4 cache tensor must be [tokens,features] or [1,tokens,features]")


def _validate_final_shapes[Hidden: Int, Out: Int](
    base_w: Tensor,
    base_b: Tensor,
    hidden_tokens: Tensor,
    target_velocity: Tensor,
) raises:
    if len(base_w.shape()) != 2 or base_w.shape()[0] != Out or base_w.shape()[1] != Hidden:
        raise Error("Ideogram4 final base weight shape mismatch")
    if len(base_b.shape()) != 1 or base_b.shape()[0] != Out:
        raise Error("Ideogram4 final base bias shape mismatch")
    if len(hidden_tokens.shape()) != 2 or hidden_tokens.shape()[1] != Hidden:
        raise Error("Ideogram4 hidden token shape mismatch")
    if len(target_velocity.shape()) != 2 or target_velocity.shape()[1] != Out:
        raise Error("Ideogram4 target token shape mismatch")
    if hidden_tokens.shape()[0] != target_velocity.shape()[0]:
        raise Error("Ideogram4 hidden/target token count mismatch")


def _zeros_like(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    return zeros_device(x.shape(), x.dtype(), ctx)


def _const_like(t: Tensor, val: Float32, ctx: DeviceContext) raises -> Tensor:
    var z = zeros_device(t.shape(), STDtype.BF16, ctx)
    return add_scalar(z, val, ctx)


def _l1_sum(x: Tensor, ctx: DeviceContext) raises -> Float32:
    var host = x.to_host(ctx)
    var s = Float32(0.0)
    for i in range(len(host)):
        var v = host[i]
        if v < Float32(0.0):
            s -= v
        else:
            s += v
    return s


def _discover_final_linear_cache(
    src: ShardedSafeTensors, mut hk: List[String], mut tk: List[String]
) raises:
    var i = 0
    while True:
        var hkey = String("hidden.") + String(i)
        var tkey = String("target.") + String(i)
        if hkey in src.name_to_shard and tkey in src.name_to_shard:
            hk.append(hkey)
            tk.append(tkey)
            i += 1
        else:
            break
    if len(hk) > 0:
        return

    if String("hidden") in src.name_to_shard and String("target") in src.name_to_shard:
        hk.append(String("hidden"))
        tk.append(String("target"))
