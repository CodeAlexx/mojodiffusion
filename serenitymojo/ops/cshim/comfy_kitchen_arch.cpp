// Metadata compiled into each architecture-specific CK attention DSO.
//
// The runtime bridge refuses to launch a DSO unless this target matches the
// active CUDA device. This prevents an architecture build (or an untagged
// wheel extension) from being presented as the measured CK backend on a
// different GPU.

#ifndef SERENITY_CK_TARGET_SM
#error "SERENITY_CK_TARGET_SM must be defined by build_h3_ck_attention.sh"
#endif

extern "C" int serenity_ck_attention_abi_version() { return 1; }

extern "C" int serenity_ck_attention_target_sm() {
  return SERENITY_CK_TARGET_SM;
}
