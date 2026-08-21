// Direct C ABI bridge to Comfy Kitchen's exported Sage INT8 CUDA launchers.
//
// The upstream sources expose the CUDA launchers as extern "C" symbols. We
// compile only those launchers into an architecture-tagged DSO; no Python API
// runs on the attention path. Q/K/V, scratch, output, and the MAX CUDA stream
// are all owned by the Mojo caller.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <exception>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

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

using MetadataInt = int (*)();

struct Api {
  void *handle = nullptr;
  QuantQK quant_qk = nullptr;
  QuantV quant_v = nullptr;
  Attend attend = nullptr;
  int current_sm = 0;
  int target_sm = 0;
  std::string error;
};

Api api;
std::once_flag api_once;
thread_local std::string call_error;

bool truthy_env(const char *name) {
  const char *value = std::getenv(name);
  return value && (!std::strcmp(value, "1") || !std::strcmp(value, "true") ||
                   !std::strcmp(value, "yes"));
}

bool load_candidate(const std::string &path, bool allow_untagged,
                    std::string &why) {
  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    const char *dl_why = dlerror();
    why = path + ": " + (dl_why ? dl_why : "unknown dlopen error");
    return false;
  }

  auto abi = reinterpret_cast<MetadataInt>(
      dlsym(handle, "serenity_ck_attention_abi_version"));
  auto target = reinterpret_cast<MetadataInt>(
      dlsym(handle, "serenity_ck_attention_target_sm"));
  if (!abi || !target) {
    if (!allow_untagged) {
      why = path + ": missing CK architecture metadata";
      dlclose(handle);
      return false;
    }
    api.target_sm = api.current_sm;
  } else {
    const int abi_version = abi();
    api.target_sm = target();
    if (abi_version != 1) {
      why = path + ": unsupported CK attention ABI " +
            std::to_string(abi_version);
      dlclose(handle);
      return false;
    }
    if (api.target_sm != api.current_sm) {
      why = path + ": compiled for sm_" + std::to_string(api.target_sm) +
            " but active CUDA device is sm_" +
            std::to_string(api.current_sm);
      dlclose(handle);
      return false;
    }
  }
  api.handle = handle;
  return true;
}

void init_api() {
  int device = 0;
  int major = 0;
  int minor = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status == cudaSuccess)
    status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                    device);
  if (status == cudaSuccess)
    status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                    device);
  if (status != cudaSuccess) {
    api.error = std::string("unable to identify active CUDA device: ") +
                cudaGetErrorString(status);
    return;
  }
  api.current_sm = major * 10 + minor;

  const char *configured = std::getenv("SERENITY_COMFY_KITCHEN_CUDA");
  const bool allow_untagged = truthy_env("SERENITY_CK_ALLOW_UNTAGGED");
  std::vector<std::string> candidates;
  if (configured && *configured) {
    // An explicit path is authoritative. A mismatch must fail closed rather
    // than silently falling through to a different implementation.
    candidates.emplace_back(configured);
  } else {
    candidates.emplace_back("output/lib/ck/sm" +
                            std::to_string(api.current_sm) +
                            "/libserenity_ck_attention.so");
    candidates.emplace_back("libserenity_ck_attention_sm" +
                            std::to_string(api.current_sm) + ".so");
  }

  std::ostringstream errors;
  for (const std::string &path : candidates) {
    std::string why;
    if (load_candidate(path, allow_untagged, why)) break;
    if (errors.tellp() > 0) errors << "; ";
    errors << why;
  }
  if (!api.handle) {
    api.error = "no CK attention build admitted for active sm_" +
                std::to_string(api.current_sm) + ": " + errors.str() +
                ". Use cU-DNN or build this exact target with "
                "SERENITY_CK_CUDA_ARCH=sm_" +
                std::to_string(api.current_sm) +
                " scripts/build_h3_ck_attention.sh";
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

extern "C" int serenity_comfy_kitchen_current_sm() {
  std::call_once(api_once, init_api);
  return api.current_sm;
}

extern "C" int serenity_comfy_kitchen_target_sm() {
  std::call_once(api_once, init_api);
  return api.target_sm;
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
