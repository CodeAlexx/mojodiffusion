# LensModelLoader.mojo — 1:1 port of Serenity
#   modules/modelLoader/LensModelLoader.py  (pr-1510)
# structurally mirrored on modelLoader/ZImageModelLoader.mojo (the existing
# Serenity Trainer loader template).
#
# ─────────────────────────────────────────────────────────────────────────────
# Serenity SOURCE (modules/modelLoader/LensModelLoader.py, __load_diffusers):
#
#   transformer = LensTransformer2DModel.from_pretrained(
#       base_model_name, "transformer", torch_dtype=bf16 ...)
#   tokenizer   = PreTrainedTokenizerFast.from_pretrained(base, "tokenizer")
#   selected_layer_index = transformer.config.selected_layer_index   # [5,11,17,23]
#   def load_text_encoder():
#       te = LensGptOssEncoder.from_pretrained(base, "text_encoder")
#       te.set_selected_layers(selected_layer_index)
#       return te
#   text_encoder = OnDemandModule(load_text_encoder) if on_demand else load_text_encoder()
#   noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base, "scheduler")
#   vae = AutoencoderKLFlux2.from_pretrained(vae_name or base, "vae")
#   model.{tokenizer,noise_scheduler,text_encoder,vae,transformer} = ...
#   model.text_encoder_hidden_size = LensGptOssEncoder.config_class
#       .from_pretrained(base, "text_encoder").hidden_size            # 2880
#
# The Mojo port is data-oriented (no torch nn.Module / no DiffusionPipeline). The
# loader's job is to materialize the FROZEN transformer weight store the block
# fwd/bwd consume, and to expose the checkpoint paths for the on-demand encoder /
# scheduler / VAE so the sampler+dataloader can read them. Component objects:
#   • transformer weights  → LensWeights (this file; BF16 store, name→Tensor)
#   • text_encoder         → models/text_encoder/gpt_oss_encoder.mojo (on-demand,
#                            loaded by the dataLoader/sampler, NOT here — matches
#                            OnDemandModule(load_text_encoder))
#   • scheduler            → sampling/lens_flowmatch.mojo (host scalar schedule)
#   • vae                  → model/LensVAE (AutoencoderKLFlux2 = Flux2 VAE)
# Checkpoint layout (LensModelLoader base_model_name):
#   /home/alex/.serenity/models/microsoft_lens/{transformer,vae,text_encoder,
#                                                scheduler,tokenizer}/
#
# BORROW boundary: import serenitymojo {tensor, io}. The on-disk dtype (BF16) is
# preserved by Tensor.from_view. All base weights are FROZEN (LoRA training): they
# are loaded but never tracked on a tape and never receive gradients — only their
# LoRA d_A/d_B do (the double-stream adapter overlay, model/LensModel.mojo).

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor


comptime TArc = ArcPointer[Tensor]

# ── Checkpoint layout (LensModelLoader base_model_name subfolders) ─────────────
comptime LENS_CKPT_DIR        = "/home/alex/.serenity/models/microsoft_lens"
comptime LENS_TRANSFORMER_DIR = "/home/alex/.serenity/models/microsoft_lens/transformer"
comptime LENS_VAE_DIR         = "/home/alex/.serenity/models/microsoft_lens/vae"
comptime LENS_TEXT_ENC_DIR    = "/home/alex/.serenity/models/microsoft_lens/text_encoder"
comptime LENS_SCHEDULER_DIR   = "/home/alex/.serenity/models/microsoft_lens/scheduler"
comptime LENS_TOKENIZER_DIR   = "/home/alex/.serenity/models/microsoft_lens/tokenizer"

# ── Lens transformer config (config.json; cited in SLICE prompt + meta.json) ───
comptime LENS_NUM_LAYERS   = 48
comptime LENS_INNER_DIM    = 1536
comptime LENS_NUM_HEADS    = 24
comptime LENS_HEAD_DIM     = 64    # inner_dim / num_heads
comptime LENS_ENC_DIM      = 2880  # enc_hidden_dim (per GPT-OSS layer)
comptime LENS_TXT_LAYERS   = 4     # len(selected_layer_index)
comptime LENS_TXT_IN       = 11520 # enc_dim * txt_layers (concat dim for txt_in)
comptime LENS_IN_CH        = 128   # in_channels (patchified latent channels)
comptime LENS_OUT_CH       = 32    # out_channels (proj_out per unpatchify -> 32)
comptime LENS_PATCH        = 2


# A frozen weight store: name → device Tensor (BF16). Tensor is move-only, so we
# box each in ArcPointer and key by name through an Int index (List/Dict can't
# hold a bare Tensor). Identical mechanics to ZImageWeights.
struct LensWeights(Movable):
    var weights: List[TArc]
    var name_to_idx: Dict[String, Int]

    def __init__(out self, var weights: List[TArc], var name_to_idx: Dict[String, Int]):
        self.weights = weights^
        self.name_to_idx = name_to_idx^

    # Load every transformer tensor from a sharded safetensors dir (H2D copy,
    # dtype preserved). Mirrors LensTransformer2DModel.from_pretrained(subfolder=
    # "transformer") weight materialization; here only the raw state dict is read
    # (the module graph is reconstructed by model/LensDiT.mojo's forward).
    @staticmethod
    def load(dir: String, ctx: DeviceContext) raises -> LensWeights:
        var sharded = ShardedSafeTensors.open(dir)
        var weights = List[TArc]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            var tv = sharded.tensor_view(nm)
            # Store BF16: both the infer (LensDiT) and LoRA (lens_stack_lora) forwards
            # cast every weight to BF16 before the GEMM, so a BF16 store is numerically
            # transparent to the forward and lets the hand-chained backward consume the
            # SAME BF16 base weights (no F32/BF16 dtype mismatch, half the VRAM).
            var t = cast_tensor(Tensor.from_view(tv, ctx), STDtype.BF16, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return LensWeights(weights^, name_to_idx^)

    # Reference to a weight by full name (raises if missing).
    def get(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("LensWeights: missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def has(self, name: String) -> Bool:
        return name in self.name_to_idx

    def count(self) -> Int:
        return len(self.weights)


# ── per-block key helpers (double-stream MM-DiT) ──────────────────────────────
# Lens transformer block keys (diffusers LensTransformer2DModel state dict; the
# SAME keys the borrowed forward reads — see
# serenitymojo/pipeline/lens_pipeline_1024_multistep.mojo::lens_block_forward).
# For one block `transformer_blocks.<i>`:
#   img_mod.1.{weight,bias}                 modulation (6 chunks) for image stream
#   txt_mod.1.{weight,bias}                 modulation (6 chunks) for text  stream
#   img_norm1.weight / img_norm2.weight     RMSNorm (image)
#   txt_norm1.weight / txt_norm2.weight     RMSNorm (text)
#   attn.img_qkv.{weight,bias}              fused QKV (image)  [3*dim, dim]
#   attn.txt_qkv.{weight,bias}              fused QKV (text)   [3*dim, dim]
#   attn.norm_q/norm_k.weight               per-head QK RMSNorm (image)
#   attn.norm_added_q/norm_added_k.weight   per-head QK RMSNorm (text)
#   attn.to_out.0.{weight,bias}             image attn output proj
#   attn.to_add_out.{weight,bias}           text  attn output proj
#   img_mlp.w1/w2/w3.weight                 SwiGLU MLP (image)
#   txt_mlp.w1/w2/w3.weight                 SwiGLU MLP (text)
def lens_block_prefix(block_idx: Int) -> String:
    return String("transformer_blocks.") + String(block_idx)


@fieldwise_init
struct LensBlockKeys(Copyable, Movable):
    var prefix: String

    @staticmethod
    def for_block(block_idx: Int) -> LensBlockKeys:
        return LensBlockKeys(lens_block_prefix(block_idx))

    def img_mod_w(self) -> String:     return self.prefix + String(".img_mod.1.weight")
    def img_mod_b(self) -> String:     return self.prefix + String(".img_mod.1.bias")
    def txt_mod_w(self) -> String:     return self.prefix + String(".txt_mod.1.weight")
    def txt_mod_b(self) -> String:     return self.prefix + String(".txt_mod.1.bias")
    def img_norm1(self) -> String:     return self.prefix + String(".img_norm1.weight")
    def img_norm2(self) -> String:     return self.prefix + String(".img_norm2.weight")
    def txt_norm1(self) -> String:     return self.prefix + String(".txt_norm1.weight")
    def txt_norm2(self) -> String:     return self.prefix + String(".txt_norm2.weight")
    def img_qkv_w(self) -> String:     return self.prefix + String(".attn.img_qkv.weight")
    def img_qkv_b(self) -> String:     return self.prefix + String(".attn.img_qkv.bias")
    def txt_qkv_w(self) -> String:     return self.prefix + String(".attn.txt_qkv.weight")
    def txt_qkv_b(self) -> String:     return self.prefix + String(".attn.txt_qkv.bias")
    def norm_q(self) -> String:        return self.prefix + String(".attn.norm_q.weight")
    def norm_k(self) -> String:        return self.prefix + String(".attn.norm_k.weight")
    def norm_added_q(self) -> String:  return self.prefix + String(".attn.norm_added_q.weight")
    def norm_added_k(self) -> String:  return self.prefix + String(".attn.norm_added_k.weight")
    def to_out_w(self) -> String:      return self.prefix + String(".attn.to_out.0.weight")
    def to_out_b(self) -> String:      return self.prefix + String(".attn.to_out.0.bias")
    def to_add_out_w(self) -> String:  return self.prefix + String(".attn.to_add_out.weight")
    def to_add_out_b(self) -> String:  return self.prefix + String(".attn.to_add_out.bias")
    def img_mlp_w1(self) -> String:    return self.prefix + String(".img_mlp.w1.weight")
    def img_mlp_w2(self) -> String:    return self.prefix + String(".img_mlp.w2.weight")
    def img_mlp_w3(self) -> String:    return self.prefix + String(".img_mlp.w3.weight")
    def txt_mlp_w1(self) -> String:    return self.prefix + String(".txt_mlp.w1.weight")
    def txt_mlp_w2(self) -> String:    return self.prefix + String(".txt_mlp.w2.weight")
    def txt_mlp_w3(self) -> String:    return self.prefix + String(".txt_mlp.w3.weight")


# ── resident (non-block) keys ─────────────────────────────────────────────────
# Mirrors serenitymojo LensResident.load: img_in / txt_in / txt_norm.{0..3} /
# time_text_embed.timestep_embedder.linear_{1,2} / norm_out.linear / proj_out.
@fieldwise_init
struct LensResidentKeys(Copyable, Movable):
    @staticmethod
    def img_in_w() -> String:      return String("img_in.weight")
    @staticmethod
    def img_in_b() -> String:      return String("img_in.bias")
    @staticmethod
    def txt_in_w() -> String:      return String("txt_in.weight")
    @staticmethod
    def txt_in_b() -> String:      return String("txt_in.bias")
    @staticmethod
    def txt_norm(i: Int) -> String: return String("txt_norm.") + String(i) + String(".weight")
    @staticmethod
    def temb_l1_w() -> String:     return String("time_text_embed.timestep_embedder.linear_1.weight")
    @staticmethod
    def temb_l1_b() -> String:     return String("time_text_embed.timestep_embedder.linear_1.bias")
    @staticmethod
    def temb_l2_w() -> String:     return String("time_text_embed.timestep_embedder.linear_2.weight")
    @staticmethod
    def temb_l2_b() -> String:     return String("time_text_embed.timestep_embedder.linear_2.bias")
    @staticmethod
    def norm_out_w() -> String:    return String("norm_out.linear.weight")
    @staticmethod
    def norm_out_b() -> String:    return String("norm_out.linear.bias")
    @staticmethod
    def proj_out_w() -> String:    return String("proj_out.weight")
    @staticmethod
    def proj_out_b() -> String:    return String("proj_out.bias")


# ══════════════════════════════════════════════════════════════════════════════
# LoRA reload — the inverse of modelSaver/lens/LensLoRASaver.save_lens_lora.
#
# Mirrors Serenity LensLoRASetup.setup_model (LensLoRASetup.py):
#   if model.lora_state_dict:
#       model.transformer_lora.load_state_dict(model.lora_state_dict)
# i.e. a previously-saved adapter checkpoint is read back so training can RESUME
# from it (vs the default A~randn / B=0 cold start). We read the saved key pair
#   "transformer.<module>.lora_down.weight" → A [rank, in]
#   "transformer.<module>.lora_up.weight"   → B [out, rank]
# for each (block, slot) prefix (and the optional ".alpha" scalar) and return a
# flat (A,B,alpha) bundle the block fwd/bwd consume.  Dtype preserved (BF16).
#
# `prefixes` is the block-major/slot-minor module list produced by
# modelSetup/lensLoraTargets.lens_lora_target_prefixes (SLICE B). Each prefix is
# of the form "transformer.transformer_blocks.<b>.<module>".
struct LensLoraReload(Movable):
    var a: List[TArc]        # [n_layers*slots] each [rank, in]
    var b: List[TArc]        # [n_layers*slots] each [out, rank]
    var alpha: List[Float32] # per-adapter alpha (rank if absent)
    var rank: Int

    def __init__(out self, var a: List[TArc], var b: List[TArc], var alpha: List[Float32], rank: Int):
        self.a = a^
        self.b = b^
        self.alpha = alpha^
        self.rank = rank


def load_lens_lora(
    path: String,
    prefixes: List[String],
    ctx: DeviceContext,
) raises -> LensLoraReload:
    var sharded = ShardedSafeTensors.open(path)
    var have = Dict[String, Int]()
    for ref nm in sharded.names():
        have[nm] = 1
    var a = List[TArc]()
    var b = List[TArc]()
    var alpha = List[Float32]()
    var rank = -1
    for i in range(len(prefixes)):
        var pre = prefixes[i]
        var ak = pre + String(".lora_down.weight")
        var bk = pre + String(".lora_up.weight")
        var at = Tensor.from_view(sharded.tensor_view(ak), ctx)
        var bt = Tensor.from_view(sharded.tensor_view(bk), ctx)
        if rank < 0:
            rank = at.shape()[0]            # A is [rank, in]
        var ah = pre + String(".alpha")
        var al = Float32(rank)
        if ah in have:
            var alt = Tensor.from_view(sharded.tensor_view(ah), ctx)
            al = alt.to_host(ctx)[0]
        a.append(TArc(at^))
        b.append(TArc(bt^))
        alpha.append(al)
    return LensLoraReload(a^, b^, alpha^, rank)
