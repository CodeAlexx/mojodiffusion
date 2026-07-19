# cfg_kv_cache_smoke.mojo - compile/run gate for Lance CFG KV-cache metadata.

from serenitymojo.models.lance.cfg_kv_cache import (
    LANCE_T2V_CFG_BRANCH_COND,
    LANCE_T2V_CFG_BRANCH_TEXT_UNCOND,
    build_lance_t2v_cfg_kv_layer_call,
    build_lance_t2v_text_drop_cfg_kv_plan,
    validate_lance_t2v_cfg_kv_plan_for_latents,
)
from serenitymojo.models.lance.lance_t2v import (
    LanceT2VConfig,
    build_lance_t2v_input_from_text_ids,
)


def _check(name: String, got: Int, expected: Int) raises:
    print("[lance-cfg-kv]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("lance cfg kv mismatch: ") + name)


def main() raises:
    var text = List[Int]()
    text.append(1234)
    text.append(5678)
    var input = build_lance_t2v_input_from_text_ids(text^, 3, 2, 1)
    var plan = build_lance_t2v_text_drop_cfg_kv_plan(input)
    var cfg = LanceT2VConfig.lance_3b_video()

    _check(String("cond total len"), plan.cond_total_len, 12)
    _check(String("gen len"), plan.gen_len, 6)
    _check(String("cond prefix len"), plan.cond.prefix_len, 4)
    _check(String("cond query start"), plan.cond.query_start, 4)
    _check(String("cond query len"), plan.cond.query_len, 8)
    _check(String("cond query gen start"), plan.cond.query_gen_start, 1)
    _check(String("cond query gen end"), plan.cond.query_gen_end(), 7)
    _check(String("uncond prefix len"), plan.text_uncond.prefix_len, 0)
    _check(String("uncond query start"), plan.text_uncond.query_start, 4)
    _check(String("uncond packed shift"), plan.text_uncond.packed_index_shift, 4)
    _check(String("cond kv elems/layer"), plan.cond.kv_elements_per_layer(cfg), 2048)
    _check(String("all cache bytes"), plan.total_kv_bytes_all_layers(cfg, 2), 147456)

    var cond_call = build_lance_t2v_cfg_kv_layer_call(
        plan, LANCE_T2V_CFG_BRANCH_COND, 0
    )
    var uncond_call = build_lance_t2v_cfg_kv_layer_call(
        plan, LANCE_T2V_CFG_BRANCH_TEXT_UNCOND, 0
    )
    _check(String("cond call packed start"), cond_call.packed_query_start, 4)
    _check(String("cond call query kv start"), cond_call.query_kv_start, 4)
    _check(String("cond call attention kv len"), cond_call.attention_kv_len, 12)
    _check(String("uncond call packed start"), uncond_call.packed_query_start, 0)
    _check(String("uncond call query kv start"), uncond_call.query_kv_start, 0)
    _check(String("uncond call attention kv len"), uncond_call.attention_kv_len, 8)
    _check(
        String("cond call attention scores"),
        cond_call.attention_score_elements_per_layer(cfg),
        1536,
    )

    var prod_text = List[Int]()
    prod_text.append(1234)
    prod_text.append(5678)
    prod_text.append(9012)
    prod_text.append(3456)
    prod_text.append(7890)
    var prod = build_lance_t2v_input_from_text_ids(prod_text^, 3, 16, 16)
    var prod_plan = build_lance_t2v_text_drop_cfg_kv_plan(prod)
    validate_lance_t2v_cfg_kv_plan_for_latents(prod, prod_plan, cfg, 3, 16, 16)
    var prod_cond_call = build_lance_t2v_cfg_kv_layer_call(
        prod_plan, LANCE_T2V_CFG_BRANCH_COND, cfg.num_layers - 1
    )
    var prod_uncond_call = build_lance_t2v_cfg_kv_layer_call(
        prod_plan, LANCE_T2V_CFG_BRANCH_TEXT_UNCOND, cfg.num_layers - 1
    )
    _check(String("prod gen len"), prod_plan.gen_len, 768)
    _check(String("prod cond total len"), prod_plan.cond_total_len, 777)
    _check(String("prod cond packed start"), prod_cond_call.packed_query_start, 7)
    _check(String("prod uncond packed start"), prod_uncond_call.packed_query_start, 0)
    _check(String("prod cond packed end"), prod_cond_call.packed_query_end(), 777)
    _check(String("prod uncond packed end"), prod_uncond_call.packed_query_end(), 770)
    _check(String("prod cond attention kv len"), prod_cond_call.attention_kv_len, 777)
    _check(String("prod uncond attention kv len"), prod_uncond_call.attention_kv_len, 770)
    _check(String("prod cond prefix write"), prod_cond_call.prefix_cache_write_len, 7)
    _check(String("prod uncond prefix write"), prod_uncond_call.prefix_cache_write_len, 0)
    _check(String("prod cond local gen end"), prod_cond_call.local_gen_end(), 769)
    _check(String("prod uncond local gen end"), prod_uncond_call.local_gen_end(), 769)
    _check(
        String("prod cond query kv elems"),
        prod_cond_call.query_kv_elements_per_layer(cfg),
        394240,
    )
    _check(
        String("prod uncond query kv elems"),
        prod_uncond_call.query_kv_elements_per_layer(cfg),
        394240,
    )
    _check(
        String("prod cond attention scores"),
        prod_cond_call.attention_score_elements_per_layer(cfg),
        9572640,
    )
    _check(
        String("prod uncond attention scores"),
        prod_uncond_call.attention_score_elements_per_layer(cfg),
        9486400,
    )
