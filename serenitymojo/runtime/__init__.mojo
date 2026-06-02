# Runtime configuration primitives for modular pipeline entry points.
#
# Runtime modules:
#   model_manifest    - static model/checkpoint metadata.
#   execution_config  - precision/offload/runtime knobs.
#   request           - user-facing generation request metadata.
#   shape_profile     - geometry profiles for static model targets.
#   static_dispatch   - finite comptime specialization registry.
#   static_entrypoints - compile-only family entrypoint contracts.
