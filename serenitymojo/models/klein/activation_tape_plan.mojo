# Klein activation tape byte plan for OneTrainer CPU_OFFLOADED work.
#
# This is scalar accounting only. It does not move tensors or prove runtime
# offload parity. The goal is to pin the exact boundary activations retained by
# the current Klein LoRA offload-turbo tape so the CPU_OFFLOADED branch has a
# measured BF16/F16 target and cannot quietly become a Float32 host carrier.

from serenitymojo.io.dtype import STDtype


@fieldwise_init
struct KleinActivationTapePlan(Copyable, Movable, ImplicitlyCopyable):
    var num_double: Int
    var num_single: Int
    var n_img: Int
    var n_txt: Int
    var dim: Int
    var dtype: STDtype
    var double_save_tail: Int
    var single_save_tail: Int

    def seq_len(self) -> Int:
        return self.n_img + self.n_txt

    def bytes_per_elem(self) -> Int:
        return self.dtype.byte_size()

    def stream_boundary_elems(self) -> Int:
        # One image stream plus one text stream, or the concatenated single stream.
        return self.seq_len() * self.dim

    def input_projection_boundary_elems(self) -> Int:
        # img_in_act + txt_in_act retained by the current KleinStackForward.
        # The LoRA backward path does not currently consume them.
        return self.stream_boundary_elems()

    def double_input_boundary_elems(self) -> Int:
        # Per double block: dbl_img_in + dbl_txt_in.
        return self.num_double * self.stream_boundary_elems()

    def single_input_boundary_elems(self) -> Int:
        # Per single block: sgl_x_in.
        return self.num_single * self.stream_boundary_elems()

    def final_boundary_elems(self) -> Int:
        # img_out + ln_img_out are consumed by final-layer backward.
        return 2 * self.n_img * self.dim

    def current_boundary_elems(self) -> Int:
        return (
            self.input_projection_boundary_elems()
            + self.double_input_boundary_elems()
            + self.single_input_boundary_elems()
            + self.final_boundary_elems()
        )

    def live_backward_boundary_elems(self) -> Int:
        # The minimal LoRA CPU_OFFLOADED tape needs only block inputs plus
        # final-layer inputs consumed by backward.
        return (
            self.double_input_boundary_elems()
            + self.single_input_boundary_elems()
            + self.final_boundary_elems()
        )

    def unused_input_projection_boundary_elems(self) -> Int:
        return self.input_projection_boundary_elems()

    def current_boundary_bytes(self) -> Int:
        return self.current_boundary_elems() * self.bytes_per_elem()

    def live_backward_boundary_bytes(self) -> Int:
        return self.live_backward_boundary_elems() * self.bytes_per_elem()

    def unused_input_projection_boundary_bytes(self) -> Int:
        return self.unused_input_projection_boundary_elems() * self.bytes_per_elem()

    def current_boundary_f32_bytes(self) -> Int:
        return self.current_boundary_elems() * STDtype.F32.byte_size()

    def live_backward_boundary_f32_bytes(self) -> Int:
        return self.live_backward_boundary_elems() * STDtype.F32.byte_size()

    def current_boundary_f32_over_storage_bytes(self) -> Int:
        return self.current_boundary_f32_bytes() - self.current_boundary_bytes()

    def live_backward_f32_over_storage_bytes(self) -> Int:
        return self.live_backward_boundary_f32_bytes() - self.live_backward_boundary_bytes()

    def internal_tail_is_recompute_only(self) -> Bool:
        return self.double_save_tail == 0 and self.single_save_tail == 0


def klein9b_lora_training_activation_tape_plan() -> KleinActivationTapePlan:
    return KleinActivationTapePlan(
        8,       # double blocks
        24,      # single blocks
        1024,    # 32x32 latent image tokens
        512,     # OneTrainer/Klein text tokens
        4096,    # Klein 9B inner dim
        STDtype.BF16,
        0,       # DBL_SAVE_TAIL in klein_stack_lora.mojo
        0,       # SGL_SAVE_TAIL in klein_stack_lora.mojo
    )
