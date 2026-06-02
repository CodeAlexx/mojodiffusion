# cfg_kv_cache.mojo - Lance T2V variable-length CFG/KV-cache planning.
#
# This module is intentionally metadata-only. It captures the row/index contract
# used by the upstream `validation_gen_KVcache` path before the Mojo model grows
# a cached query forward. Pipeline smokes should keep using the dense padded CFG
# path until the cached forward lands.

from serenitymojo.models.lance.lance_t2v import LanceT2VConfig, LanceT2VInput


comptime LANCE_T2V_CFG_BRANCH_COND = 0
comptime LANCE_T2V_CFG_BRANCH_TEXT_UNCOND = 1


@fieldwise_init
struct LanceT2VKVBranchPlan(Copyable, Movable, ImplicitlyCopyable):
    var prefix_len: Int
    var query_start: Int
    var query_len: Int
    var query_gen_start: Int
    var query_gen_len: Int
    var packed_index_shift: Int
    var prefill_updates_cache: Bool
    var prefill_is_causal: Bool

    def query_end(self) -> Int:
        return self.query_start + self.query_len

    def query_gen_end(self) -> Int:
        return self.query_gen_start + self.query_gen_len

    def kv_elements_per_layer(self, cfg: LanceT2VConfig) -> Int:
        # K and V are cached post-mRoPE, pre-repeat-kv:
        # [B=1, num_kv_heads, prefix_len, head_dim] each.
        return self.prefix_len * cfg.num_kv_heads * cfg.head_dim * 2

    def kv_bytes_per_layer(self, cfg: LanceT2VConfig, dtype_bytes: Int) -> Int:
        return self.kv_elements_per_layer(cfg) * dtype_bytes


@fieldwise_init
struct LanceT2VKVLayerCall(Copyable, Movable, ImplicitlyCopyable):
    var branch_id: Int
    var layer_idx: Int
    var source_query_start: Int
    var query_len: Int
    var packed_query_start: Int
    var local_gen_start: Int
    var gen_len: Int
    var prefix_cache_len: Int
    var query_kv_start: Int
    var attention_kv_len: Int
    var prefix_cache_write_len: Int
    var prefix_cache_is_causal: Bool

    def source_query_end(self) -> Int:
        return self.source_query_start + self.query_len

    def packed_query_end(self) -> Int:
        return self.packed_query_start + self.query_len

    def local_gen_end(self) -> Int:
        return self.local_gen_start + self.gen_len

    def query_kv_end(self) -> Int:
        return self.query_kv_start + self.query_len

    def query_kv_elements_per_layer(self, cfg: LanceT2VConfig) -> Int:
        return self.query_len * cfg.num_kv_heads * cfg.head_dim * 2

    def attention_score_elements_per_layer(self, cfg: LanceT2VConfig) -> Int:
        return self.query_len * self.attention_kv_len * cfg.num_heads


@fieldwise_init
struct LanceT2VCfgKVPlan(Copyable, Movable, ImplicitlyCopyable):
    var cond: LanceT2VKVBranchPlan
    var text_uncond: LanceT2VKVBranchPlan
    var dropped_text_len: Int
    var gen_len: Int
    var cond_total_len: Int
    var uncond_query_len: Int

    def total_kv_elements_per_layer(self, cfg: LanceT2VConfig) -> Int:
        return (
            self.cond.kv_elements_per_layer(cfg)
            + self.text_uncond.kv_elements_per_layer(cfg)
        )

    def total_kv_bytes_all_layers(self, cfg: LanceT2VConfig, dtype_bytes: Int) -> Int:
        return self.total_kv_elements_per_layer(cfg) * dtype_bytes * cfg.num_layers


def build_lance_t2v_kv_layer_call(
    branch_id: Int, layer_idx: Int, branch: LanceT2VKVBranchPlan
) raises -> LanceT2VKVLayerCall:
    """Map a branch span plan into one cached layer call contract.

    The cached forward will receive only the visual query span. `source_*` keeps
    the original full conditional row mapping, while `packed_*` is the row range
    after dropping any text prefix for the text-uncond branch.
    """
    if (
        branch_id != LANCE_T2V_CFG_BRANCH_COND
        and branch_id != LANCE_T2V_CFG_BRANCH_TEXT_UNCOND
    ):
        raise Error(String("Lance KV layer call: invalid branch id ") + String(branch_id))
    if layer_idx < 0:
        raise Error("Lance KV layer call: layer_idx must be non-negative")
    if branch.prefix_len < 0:
        raise Error("Lance KV layer call: prefix_len must be non-negative")
    if branch.query_start < 0 or branch.query_len <= 0:
        raise Error("Lance KV layer call: query span must be positive")
    if branch.packed_index_shift < 0:
        raise Error("Lance KV layer call: packed_index_shift must be non-negative")
    if branch.query_start < branch.packed_index_shift:
        raise Error("Lance KV layer call: packed query start would be negative")
    if branch.query_gen_start < 0 or branch.query_gen_len < 0:
        raise Error("Lance KV layer call: generated span must be non-negative")
    if branch.query_gen_end() > branch.query_len:
        raise Error("Lance KV layer call: generated span exceeds query span")

    var packed_query_start = branch.query_start - branch.packed_index_shift
    var prefix_write_len = 0
    if branch.prefill_updates_cache:
        prefix_write_len = branch.prefix_len
    return LanceT2VKVLayerCall(
        branch_id,
        layer_idx,
        branch.query_start,
        branch.query_len,
        packed_query_start,
        branch.query_gen_start,
        branch.query_gen_len,
        branch.prefix_len,
        branch.prefix_len,
        branch.prefix_len + branch.query_len,
        prefix_write_len,
        branch.prefill_is_causal,
    )


def build_lance_t2v_cfg_kv_layer_call(
    plan: LanceT2VCfgKVPlan, branch_id: Int, layer_idx: Int
) raises -> LanceT2VKVLayerCall:
    if branch_id == LANCE_T2V_CFG_BRANCH_COND:
        return build_lance_t2v_kv_layer_call(branch_id, layer_idx, plan.cond)
    if branch_id == LANCE_T2V_CFG_BRANCH_TEXT_UNCOND:
        return build_lance_t2v_kv_layer_call(branch_id, layer_idx, plan.text_uncond)
    raise Error(String("Lance CFG KV plan: invalid branch id ") + String(branch_id))


def validate_lance_t2v_cfg_kv_plan_for_latents(
    input: LanceT2VInput,
    plan: LanceT2VCfgKVPlan,
    cfg: LanceT2VConfig,
    latent_t: Int,
    latent_h: Int,
    latent_w: Int,
) raises:
    validate_lance_t2v_input_for_kv_cache(String("cond"), input)
    if latent_t <= 0 or latent_h <= 0 or latent_w <= 0:
        raise Error("Lance CFG KV latent dimensions must be positive")
    if cfg.num_layers <= 0:
        raise Error("Lance CFG KV model depth must be positive")
    var expected_gen_len = latent_t * latent_h * latent_w
    if input.gen_len != expected_gen_len or plan.gen_len != expected_gen_len:
        raise Error("Lance CFG KV gen_len does not match latent dimensions")
    if plan.cond_total_len != len(input.full_ids):
        raise Error("Lance CFG KV cond_total_len does not match input length")
    if plan.dropped_text_len != input.text_split_len:
        raise Error("Lance CFG KV dropped text length must equal text prefix")
    if plan.cond.prefix_len != input.text_split_len:
        raise Error("Lance CFG KV cond prefix must equal text_split_len")
    if plan.cond.query_start != input.text_split_len:
        raise Error("Lance CFG KV cond query must start at visual start token")
    if plan.cond.query_len != expected_gen_len + 2:
        raise Error("Lance CFG KV cond query must cover visual start/gen/end")
    if plan.cond.query_end() != len(input.full_ids):
        raise Error("Lance CFG KV cond query must end at full sequence length")
    if plan.cond.packed_index_shift != 0:
        raise Error("Lance CFG KV cond packed shift must be zero")
    if plan.text_uncond.query_len != plan.cond.query_len:
        raise Error("Lance CFG KV uncond query length must match cond query length")
    if plan.uncond_query_len != plan.cond.query_len:
        raise Error("Lance CFG KV uncond packed length must match query length")
    if plan.text_uncond.prefix_len != 0:
        raise Error("Lance CFG KV uncond prefix cache must be empty")
    if plan.text_uncond.query_start != plan.cond.query_start:
        raise Error("Lance CFG KV uncond source query must match cond query")
    if plan.text_uncond.packed_index_shift != input.text_split_len:
        raise Error("Lance CFG KV uncond packed shift must drop text prefix")

    var cond_call = build_lance_t2v_cfg_kv_layer_call(
        plan, LANCE_T2V_CFG_BRANCH_COND, cfg.num_layers - 1
    )
    var uncond_call = build_lance_t2v_cfg_kv_layer_call(
        plan, LANCE_T2V_CFG_BRANCH_TEXT_UNCOND, cfg.num_layers - 1
    )
    if (
        cond_call.layer_idx >= cfg.num_layers
        or uncond_call.layer_idx >= cfg.num_layers
    ):
        raise Error("Lance CFG KV layer call exceeds model depth")
    if cond_call.attention_kv_len != len(input.full_ids):
        raise Error("Lance CFG KV cond attention KV length must equal full sequence")
    if uncond_call.attention_kv_len != plan.uncond_query_len:
        raise Error("Lance CFG KV uncond attention KV length must equal query sequence")
    if cond_call.query_kv_start != input.text_split_len:
        raise Error("Lance CFG KV cond query KV must append after cached text prefix")
    if uncond_call.query_kv_start != 0:
        raise Error("Lance CFG KV uncond query KV must start at packed row zero")
    if cond_call.packed_query_end() != len(input.full_ids):
        raise Error("Lance CFG KV cond packed query must end at full sequence length")
    if uncond_call.packed_query_end() != plan.uncond_query_len:
        raise Error("Lance CFG KV uncond packed query must end at packed length")
    if (
        cond_call.local_gen_start != 1
        or uncond_call.local_gen_start != 1
        or cond_call.local_gen_end() != expected_gen_len + 1
        or uncond_call.local_gen_end() != expected_gen_len + 1
    ):
        raise Error("Lance CFG KV generated span must skip visual start/end tokens")
    if cond_call.prefix_cache_write_len != input.text_split_len:
        raise Error("Lance CFG KV cond prefill must write text prefix cache")
    if uncond_call.prefix_cache_write_len != 0:
        raise Error("Lance CFG KV uncond prefill must not write a prefix cache")


def validate_lance_t2v_input_for_kv_cache(
    name: String, input: LanceT2VInput
) raises:
    var total = len(input.full_ids)
    if total != len(input.t_pos) or total != len(input.h_pos) or total != len(input.w_pos):
        raise Error(name + ": token ids and mRoPE position ids must have equal length")
    if input.text_split_len <= 0:
        raise Error(name + ": text_split_len must be positive")
    if input.gen_len <= 0:
        raise Error(name + ": gen_len must be positive")
    if input.gen_start != input.text_split_len + 1:
        raise Error(name + ": gen_start must point after the visual start token")
    if total != input.text_split_len + 1 + input.gen_len + 1:
        raise Error(name + ": expected [text prefix, visual start, gen tokens, visual end]")
    if len(input.latent_pos_ids) != input.gen_len:
        raise Error(name + ": latent_pos_ids length must equal gen_len")


def build_lance_t2v_text_drop_cfg_kv_plan(
    cond: LanceT2VInput
) raises -> LanceT2VCfgKVPlan:
    """Plan the production text-uncond CFG cache layout for simple T2V.

    Conditional branch:
      - prefill caches rows [0, text_split_len) causally;
      - denoise queries rows [text_split_len, total), i.e. visual start,
        latent tokens, visual end;
      - local query gen rows are [1, 1 + gen_len).

    Text-uncond branch:
      - drops the text prefix instead of padding it;
      - has an empty prefix cache;
      - reuses the same visual query rows, but packed indexes are shifted down
        by dropped_text_len to match the shorter sequence.
    """
    validate_lance_t2v_input_for_kv_cache(String("cond"), cond)

    var query_len = cond.gen_len + 2
    var cond_branch = LanceT2VKVBranchPlan(
        cond.text_split_len,
        cond.text_split_len,
        query_len,
        1,
        cond.gen_len,
        0,
        True,
        True,
    )
    var uncond_branch = LanceT2VKVBranchPlan(
        0,
        cond.text_split_len,
        query_len,
        1,
        cond.gen_len,
        cond.text_split_len,
        False,
        False,
    )
    return LanceT2VCfgKVPlan(
        cond_branch,
        uncond_branch,
        cond.text_split_len,
        cond.gen_len,
        len(cond.full_ids),
        query_len,
    )
