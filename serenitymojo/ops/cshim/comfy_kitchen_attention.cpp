// Direct C ABI bridge to Comfy Kitchen's exported Sage INT8 CUDA launchers.
//
// The upstream wheel exposes the CUDA launchers as extern "C" symbols, but
// packages them inside a Python extension.  Loading libpython globally only
// satisfies that extension's unused binding symbols; no Python API is called
// on the attention path.  Q/K/V, scratch, output, and the MAX CUDA stream are
// all owned by the Mojo caller.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <mutex>
#include <string>

namespace {

using QuantQK = void (*)(
    const void *, void *, void *, const void *, void *, void *,
    int, int, int, int, int, int, int, int, int, int,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, void *, void *);

using QuantV = void (*)(
    const void *, void *, void *, int, int, int, int, int,
    int64_t, int64_t, int64_t, int, void *);

using Attend = void (*)(
    const void *, const void *, const void *, void *, const void *,
    const void *, const void *, const void *,
    int64_t, int64_t, int64_t, int64_t, int, int, int, int, int, int, int,
    int, int, int, int, int, int, int, int, int, int, int, int, int,
    float, int, void *);

struct Api {
  void *handle = nullptr;
  QuantQK quant_qk = nullptr;
  QuantV quant_v = nullptr;
  Attend attend = nullptr;
  std::string error;
};

Api api;
std::once_flag api_once;
thread_local std::string call_error;

void init_api() {
  const char *configured = std::getenv("SERENITY_COMFY_KITCHEN_CUDA");
  // Prefer the attention-only sm_86 library. Besides being self-contained,
  // it avoids registering 160+ MiB of unrelated extension fatbins at every
  // fresh H3 worker startup.
  const char *minimal_candidates[] = {
      configured,
      "output/lib/libserenity_ck_attention.so",
      "libserenity_ck_attention.so",
      nullptr};
  for (const char **path = minimal_candidates;
       *path || path == minimal_candidates; ++path) {
    if (!*path || !**path) continue;
    api.handle = dlopen(*path, RTLD_NOW | RTLD_LOCAL);
    if (api.handle) break;
  }

  if (!api.handle) {
    // The fallback wheel is a Python extension. CPython symbols are referenced
    // by nanobind but never called through this raw-launcher bridge.
    const char *python_sonames[] = {
        "libpython3.12.so.1.0", "libpython3.12.so", nullptr};
    for (const char **name = python_sonames; *name; ++name) {
      if (dlopen(*name, RTLD_NOW | RTLD_GLOBAL)) break;
    }
  }
  const char *extension_candidates[] = {
      configured,
      "_C.abi3.so",
      nullptr};
  if (!api.handle) {
    for (const char **path = extension_candidates;
         *path || path == extension_candidates; ++path) {
      if (!*path || !**path) continue;
      api.handle = dlopen(*path, RTLD_NOW | RTLD_LOCAL);
      if (api.handle) break;
    }
  }
  if (!api.handle) {
    const char *why = dlerror();
    api.error = std::string("unable to load Comfy Kitchen CUDA extension: ") +
                (why ? why : "unknown dlopen error");
    return;
  }

  api.quant_qk = reinterpret_cast<QuantQK>(
      dlsym(api.handle, "launch_quant_qk_per_thread_int8"));
  api.quant_v = reinterpret_cast<QuantV>(
      dlsym(api.handle, "launch_quant_v_int8_kernel"));
  api.attend = reinterpret_cast<Attend>(
      dlsym(api.handle, "launch_sage_attn_kernel"));
  if (!api.quant_qk || !api.quant_v || !api.attend) {
    const char *why = dlerror();
    api.error = std::string("Comfy Kitchen launcher symbol missing: ") +
                (why ? why : "unknown dlsym error");
    api.quant_qk = nullptr;
    api.quant_v = nullptr;
    api.attend = nullptr;
  }
}

}  // namespace

extern "C" const char *serenity_comfy_kitchen_last_error() {
  if (!call_error.empty()) return call_error.c_str();
  std::call_once(api_once, init_api);
  return api.error.c_str();
}

extern "C" int serenity_comfy_kitchen_available() {
  std::call_once(api_once, init_api);
  return api.quant_qk && api.quant_v && api.attend ? 1 : 0;
}

extern "C" int serenity_comfy_kitchen_sage_bf16(
    const void *q, const void *k, const void *v, void *out,
    void *q_int8, void *q_scale, void *k_int8, void *k_scale,
    void *v_int8, void *v_scale, void *anchor_indices,
    int batch, int seq, int heads, int head_dim, float sm_scale,
    void *stream) {
  call_error.clear();
  std::call_once(api_once, init_api);
  if (!api.quant_qk || !api.quant_v || !api.attend) return -1;
  if (!q || !k || !v || !out || !q_int8 || !q_scale || !k_int8 ||
      !k_scale || !v_int8 || !v_scale || !anchor_indices || batch <= 0 ||
      seq <= 0 || heads <= 0 || head_dim != 128) {
    call_error = "invalid Comfy Kitchen Sage BF16 argument";
    return -2;
  }

  try {
    constexpr int cta_q = 128;
    constexpr int warp_q = 32;
    constexpr int cta_k = 128;
    constexpr int warp_k = 128;
    constexpr int bf16_code = 2;
    const int padded_seq = (seq + cta_k - 1) / cta_k * cta_k;

    // Mojo tensors are [B,S,H,D]. Describe those same bytes logically as
    // [B,H,S,D]; quantized Q/K are emitted contiguous [B,H,S,D].
    const int64_t in_sb = int64_t(seq) * heads * head_dim;
    const int64_t in_sh = head_dim;
    const int64_t in_sn = int64_t(heads) * head_dim;
    api.quant_qk(
        q, q_int8, q_scale, k, k_int8, k_scale,
        batch, heads, seq, heads, seq, head_dim,
        cta_q, warp_q, cta_k, warp_k,
        in_sb, in_sh, in_sn, in_sb, in_sh, in_sn,
        bf16_code, anchor_indices, stream);
    api.quant_v(
        v, v_int8, v_scale, batch, heads, seq, head_dim, padded_seq,
        in_sb, in_sh, in_sn, bf16_code, stream);

    const int q_b = heads * seq * head_dim;
    const int q_n = head_dim;
    const int q_h = seq * head_dim;
    const int v_b = heads * head_dim * padded_seq;
    const int v_h = head_dim * padded_seq;
    const int v_d = padded_seq;
    const int o_b = seq * heads * head_dim;
    const int o_n = heads * head_dim;
    const int o_h = head_dim;
    api.attend(
        q_int8, k_int8, v_int8, out, q_scale, k_scale, v_scale,
        nullptr, 0, 0, 0, 0, 0,
        cta_k, batch, seq, seq, heads, heads, head_dim,
        q_b, q_n, q_h, q_b, q_n, q_h,
        v_b, v_h, v_d, o_b, o_n, o_h,
        sm_scale, bf16_code, stream);
    return 0;
  } catch (const std::exception &e) {
    call_error = e.what();
    return -3;
  } catch (...) {
    call_error = "unknown exception from Comfy Kitchen CUDA launcher";
    return -4;
  }
}
