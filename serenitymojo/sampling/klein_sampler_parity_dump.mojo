# sampling/klein_sampler_parity_dump.mojo -- Mojo-side Klein sampler artifact
# producer for paired OneTrainer sampler parity.
#
# This is a parity/debug producer, not a production sampler entry. It consumes a
# post-patch/post-pack initial-noise sidecar, runs the existing Klein denoise
# math shape, and writes raw tensor-bin artifacts for the Mojo side of
# scripts/check_klein_sampler_artifact_manifest.py. The JSON manifest it emits
# is intentionally partial: it never marks comparisons accepted and leaves
# OneTrainer-side artifacts as blockers unless a paired OT producer supplies
# them.

from std.collections import List
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import ArcPointer, alloc
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.cap_cache import save_tensor_bin
from serenitymojo.io.ffi import (
    BytePtr,
    O_CREAT,
    O_TRUNC,
    O_WRONLY,
    sys_close,
    sys_open,
    sys_pwrite,
    sys_system,
)
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.training.train_config import TrainConfig
from serenitymojo.training.sample_prompt_config import SamplePrompt
from serenitymojo.models.klein.klein_stack_lora import (
    KleinLoraSet,
    build_klein_lora_set,
    klein_lora_set_to_device,
    load_klein_lora_resume,
    scale_klein_lora_set,
    klein_stack_lora_predict_device_offload_turbo_moddev_rope_scratch,
    klein_stack_lora_predict_cfg_offload_turbo_moddev_rope_scratch,
)
from serenitymojo.models.klein.weights import (
    build_klein_step_mods_device_cached,
    build_klein_vec_silu,
    load_klein_stack_base,
    load_klein_step_mod_weights,
)
from serenitymojo.models.vae.klein_decoder import (
    KleinVaeDecoder,
    _inverse_bn,
    _unpatchify_packed,
)
from serenitymojo.offload.plan import OffloadConfig, build_klein_block_plan
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.ops.tensor_algebra import add, mul_scalar, reshape
from serenitymojo.sampling.base_sampler import save_image, tokens_to_packed_nchw
from serenitymojo.sampling.flux2_klein import (
    build_flux2_sigma_schedule,
    compute_empirical_mu,
    flux2_cfg,
    flux2_scheduler_timestep_from_sigma,
)
from serenitymojo.sampling.klein_sampler import (
    _initial_noise_tokens_from_sidecar,
    _rope_host,
)
from serenitymojo.training.progress_display import (
    print_sample_saved,
    print_sample_setup,
    print_sample_step,
)


comptime TArc = ArcPointer[Tensor]
comptime SAMPLE_SCREEN_EVERY = 5


@fieldwise_init
struct KleinSamplerParityDenoiseResult(Movable):
    var final_latent_tokens: Tensor
    var latent_trajectory: Tensor
    var denoise_seconds_per_step: Float64
    var peak_vram_mib: Float64


@fieldwise_init
struct KleinSamplerParityDumpResult(Copyable, Movable):
    var artifact_dir: String
    var manifest_path: String
    var png_path: String
    var denoise_seconds_per_step: Float64
    var vae_decode_seconds: Float64
    var peak_vram_mib: Float64


def _mkdir_p(path: String) raises:
    var rc = sys_system(String("mkdir -p ") + path)
    if rc != 0:
        raise Error(String("klein parity dump: mkdir failed for ") + path)


def _join(dir: String, name: String) -> String:
    if dir.endswith(String("/")):
        return dir + name
    return dir + String("/") + name


def _write_text(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("klein parity dump: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("klein parity dump: short write to ") + path)


def _json_byte(c: Int) raises -> String:
    var b = List[UInt8]()
    b.append(UInt8(c))
    return String(from_utf8=b)


def _json_str(s: String) raises -> String:
    var out = String("\"")
    var bytes = s.as_bytes()
    for i in range(s.byte_length()):
        var c = Int(bytes[i])
        if c == 0x22:
            out += String("\\\"")
        elif c == 0x5C:
            out += String("\\\\")
        elif c == 0x0A:
            out += String("\\n")
        elif c == 0x0D:
            out += String("\\r")
        elif c == 0x09:
            out += String("\\t")
        else:
            out += _json_byte(c)
    out += String("\"")
    return out^


def _json_bool(value: Bool) -> String:
    return String("true") if value else String("false")


def _shape_json(shape: List[Int]) -> String:
    var out = String("[")
    for i in range(len(shape)):
        if i != 0:
            out += String(",")
        out += String(shape[i])
    out += String("]")
    return out^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    return out^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    return out^


def _tensor_bin_size(t: Tensor) -> Int:
    return 24 + len(t.shape()) * 8 + t.nbytes()


def _tensor_artifact_json(
    path: String, dtype: String, shape: List[Int], byte_size: Int = -1
) raises -> String:
    var out = String("{\"path\":")
    out += _json_str(path)
    out += String(",\"dtype\":")
    out += _json_str(dtype)
    out += String(",\"shape\":")
    out += _shape_json(shape)
    if byte_size >= 0:
        out += String(",\"byte_size\":")
        out += String(byte_size)
    out += String("}")
    return out^


def _png_artifact_json(path: String, width: Int, height: Int) raises -> String:
    return (
        String("{\"path\":")
        + _json_str(path)
        + String(",\"width\":")
        + String(width)
        + String(",\"height\":")
        + String(height)
        + String("}")
    )


def _scheduler_json(num_steps: Int, image_seq_len: Int) raises -> String:
    var sigmas = build_flux2_sigma_schedule(num_steps, image_seq_len)
    var mu = compute_empirical_mu(image_seq_len, num_steps)
    var out = String("{\"name\":\"FlowMatchEuler\",\"sigmas\":[")
    for i in range(len(sigmas)):
        if i != 0:
            out += String(",")
        out += String(sigmas[i])
    out += String("],\"timesteps\":[")
    for i in range(num_steps):
        if i != 0:
            out += String(",")
        out += String(flux2_scheduler_timestep_from_sigma(sigmas[i]))
    out += String("],\"mu\":")
    out += String(mu)
    out += String(",\"step_trace\":[")
    for i in range(num_steps):
        if i != 0:
            out += String(",")
        var sigma = sigmas[i]
        var sigma_next = sigmas[i + 1]
        out += String("{\"index\":")
        out += String(i)
        out += String(",\"sigma\":")
        out += String(sigma)
        out += String(",\"sigma_next\":")
        out += String(sigma_next)
        out += String(",\"dt\":")
        out += String(sigma_next - sigma)
        out += String(",\"timestep\":")
        out += String(flux2_scheduler_timestep_from_sigma(sigma))
        out += String("}")
    out += String("]}")
    return out^


def _support_evidence_json() -> String:
    return String(
        "{"
        + "\"status\":\"support_only_not_sampler_parity\","
        + "\"sampler_parity_accepted\":false,"
        + "\"smoke_images_accepted_parity\":false,"
        + "\"note\":\"Support-only smoke/artifact records; not sampler parity.\","
        + "\"records\":["
        + "{\"id\":\"resume10_lora_identity\",\"kind\":\"lora_artifact_identity\",\"evidence_scope\":\"artifact_identity_not_sampler_parity\",\"parity_accepted\":false,\"config\":\"serenitymojo/configs/klein9b_cpu_offloaded_resume10_smoke.json\",\"lora_artifact\":\"/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors\",\"state_artifact\":\"/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors.state.safetensors\",\"lora_tensor_count\":432,\"lora_dtype\":\"BF16\",\"state_adapter_tensor_count\":288,\"state_adapter_dtype\":\"BF16\",\"state_moment_tensor_count\":576,\"state_moment_dtype\":\"F32\",\"guard\":\"scripts/check_klein_product_smoke_artifacts.py\"},"
        + "{\"id\":\"resume20_lora_identity\",\"kind\":\"lora_artifact_identity\",\"evidence_scope\":\"artifact_identity_not_sampler_parity\",\"parity_accepted\":false,\"config\":\"serenitymojo/configs/klein9b_cpu_offloaded_resume20_smoke.json\",\"lora_artifact\":\"/tmp/klein9b_cpu_offloaded_resume20_smoke.safetensors\",\"state_artifact\":\"/tmp/klein9b_cpu_offloaded_resume20_smoke.safetensors.state.safetensors\",\"lora_tensor_count\":432,\"lora_dtype\":\"BF16\",\"state_adapter_tensor_count\":288,\"state_adapter_dtype\":\"BF16\",\"state_moment_tensor_count\":576,\"state_moment_dtype\":\"F32\",\"guard\":\"scripts/check_klein_product_smoke_artifacts.py\"},"
        + "{\"id\":\"resume10_fast512_cfg1_smoke\",\"kind\":\"sampler_smoke_image\",\"evidence_scope\":\"fast cfg=1 512 smoke; not accepted parity\",\"parity_accepted\":false,\"quality_accepted\":false,\"speed_parity_accepted\":false,\"config\":\"serenitymojo/configs/klein9b_alina_samples_fast512.json\",\"lora_artifact\":\"/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors\",\"image_path\":\"output/alina_train/klein_lora_resume10_fast512_cfg1.png\",\"width\":512,\"height\":512,\"steps\":1,\"cfg_scale\":1.0},"
        + "{\"id\":\"resume10_guided512_cfg4_smoke\",\"kind\":\"sampler_smoke_image\",\"evidence_scope\":\"guided cfg=4 smoke; not accepted parity\",\"parity_accepted\":false,\"quality_accepted\":false,\"speed_parity_accepted\":false,\"config\":\"serenitymojo/configs/klein9b_alina_samples_fast512.json\",\"lora_artifact\":\"/tmp/klein9b_cpu_offloaded_resume10_smoke.safetensors\",\"image_path\":\"output/alina_train/klein_lora_resume10_fast512.png\",\"width\":512,\"height\":512,\"steps\":1,\"cfg_scale\":4.0},"
        + "{\"id\":\"resume20_fast512_cfg1_smoke\",\"kind\":\"sampler_smoke_image\",\"evidence_scope\":\"fast cfg=1 512 smoke; not accepted parity\",\"parity_accepted\":false,\"quality_accepted\":false,\"speed_parity_accepted\":false,\"config\":\"serenitymojo/configs/klein9b_alina_samples_fast512.json\",\"lora_artifact\":\"/tmp/klein9b_cpu_offloaded_resume20_smoke.safetensors\",\"image_path\":\"output/alina_train/klein_lora_resume20_fast512_cfg1.png\",\"width\":512,\"height\":512,\"steps\":1,\"cfg_scale\":1.0}"
        + "],"
        + "\"remaining_sampler_parity_requirements\":["
        + "\"real paired OneTrainer/Mojo sampler manifest\","
        + "\"shared prompt, seed, resolution, steps, cfg, scheduler, and dtype identity\","
        + "\"OneTrainer-equivalent raw/post-patch/post-pack initial-noise artifacts\","
        + "\"paired latent trajectory and final latent numeric comparisons\","
        + "\"paired VAE tensor and final PNG numeric comparisons\","
        + "\"matched denoise/VAE timing and peak VRAM evidence\""
        + "]}"
    )


def _update_min_free(ctx: DeviceContext, min_free: Int) raises -> Int:
    var mem = ctx.get_memory_info()
    var free_now = Int(mem[0])
    if free_now < min_free:
        return free_now
    return min_free


# Traced variant of klein_sampler._denoise_lora_from_initial. The denoise math
# and loading sequence are intentionally the same; this parity/debug wrapper adds
# only raw-byte trajectory capture and timing/VRAM measurement.
def _denoise_lora_from_initial_with_trajectory[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, LH: Int, LW: Int
](
    cfg: TrainConfig,
    lora_path: String,
    pos_txt: Tensor,
    neg_txt: Tensor,
    cfg_scale: Float32,
    num_steps: Int,
    var x: Tensor,
    ctx: DeviceContext,
    lora_multiplier: Float32 = Float32(1.0),
) raises -> KleinSamplerParityDenoiseResult:
    var mem0 = ctx.get_memory_info()
    var total_vram = Int(mem0[1])
    var min_free = Int(mem0[0])

    var step_bytes = x.nbytes()
    var traj_bytes = (num_steps + 1) * step_bytes
    var traj_buf = ctx.enqueue_create_buffer[DType.uint8](traj_bytes)
    var dst0 = traj_buf.create_sub_buffer[DType.uint8](0, step_bytes)
    ctx.enqueue_copy(dst_buf=dst0, src_buf=x.buf)
    ctx.synchronize()
    min_free = _update_min_free(ctx, min_free)

    var st = SafeTensors.open(cfg.checkpoint)
    var seed_ts = Tensor.from_host([Float32(500.0)], [1], STDtype.F32, ctx)
    var seed_vec_silu = build_klein_vec_silu(st, seed_ts, cfg.timestep_dim, cfg.d_model, ctx)
    var base = load_klein_stack_base(st, seed_vec_silu, cfg.d_model, ctx)
    var mod_weights = load_klein_step_mod_weights(st, cfg.d_model, ctx)
    var plan = build_klein_block_plan(cfg.num_double, cfg.num_single)
    var loader = TurboPlannedLoader.open(
        cfg.checkpoint, plan^, OffloadConfig.synchronous_cfg_paired(), ctx
    )
    min_free = _update_min_free(ctx, min_free)

    var lora: KleinLoraSet
    if lora_path == String(""):
        lora = build_klein_lora_set(
            cfg.num_double, cfg.num_single, cfg.d_model, cfg.mlp_hidden,
            cfg.lora_rank, cfg.lora_alpha
        )
    else:
        lora = load_klein_lora_resume(
            cfg.num_double, cfg.num_single, cfg.lora_rank, cfg.lora_alpha, lora_path, ctx
        )
    scale_klein_lora_set(lora, lora_multiplier)
    var lora_dev = klein_lora_set_to_device(lora, ctx)
    var rope = _rope_host[N_IMG, N_TXT, S, H]()
    var cos_dev = Tensor.from_host(rope[0].copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_dev = Tensor.from_host(rope[1].copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    var txt_tokens_t = TArc(pos_txt.clone(ctx))
    var neg_tokens_t = TArc(neg_txt.clone(ctx))
    var sigmas = build_flux2_sigma_schedule(num_steps, N_IMG)
    var scratch = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)
    print_sample_setup(String("Klein-parity-dump"), cfg.name, num_steps, cfg_scale, N_IMG, cfg.n_layers())

    var denoise_t0 = perf_counter_ns()
    for i in range(num_steps):
        scratch.reset()
        var sigma = sigmas[i]
        var dt = sigmas[i + 1] - sigma
        var t_step0 = perf_counter_ns()
        var mods = build_klein_step_mods_device_cached(
            mod_weights, sigma, cfg.timestep_dim, cfg.d_model, ctx
        )
        var img_mod = mods[0].copy()
        var txt_mod = mods[1].copy()
        var single_mod = mods[2].copy()
        base.final_shift = mods[3].copy()
        base.final_scale = mods[4].copy()

        var v_dev: Tensor
        if cfg_scale == Float32(1.0):
            v_dev = klein_stack_lora_predict_device_offload_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                TArc(x.clone(ctx)), txt_tokens_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev, sin_dev,
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch,
            )
        else:
            var preds = klein_stack_lora_predict_cfg_offload_turbo_moddev_rope_scratch[H, Dh, N_IMG, N_TXT, S](
                TArc(x.clone(ctx)), txt_tokens_t, neg_tokens_t, base,
                loader, lora_dev, img_mod, txt_mod, single_mod, cos_dev, sin_dev,
                cfg.d_model, cfg.mlp_hidden, cfg.in_channels, cfg.joint_attention_dim,
                cfg.out_channels, cfg.eps, ctx, scratch,
            )
            v_dev = flux2_cfg(preds.pos, preds.neg, cfg_scale, ctx)

        x = add(x, mul_scalar(v_dev, dt, ctx), ctx)
        var dst = traj_buf.create_sub_buffer[DType.uint8]((i + 1) * step_bytes, step_bytes)
        ctx.enqueue_copy(dst_buf=dst, src_buf=x.buf)
        ctx.synchronize()
        min_free = _update_min_free(ctx, min_free)

        var t_step1 = perf_counter_ns()
        var secs = Float64(t_step1 - t_step0) / 1.0e9
        var speed = Float64(1.0) / secs if secs > 0.0 else Float64(0.0)
        var step = i + 1
        if step == 1 or step == num_steps or step % SAMPLE_SCREEN_EVERY == 0:
            print_sample_step(String("Klein-parity-dump"), step, num_steps, sigma, secs, speed)

    var denoise_t1 = perf_counter_ns()
    var traj_shape = _shape3(num_steps + 1, N_IMG, cfg.in_channels)
    var denoise_seconds = Float64(denoise_t1 - denoise_t0) / 1.0e9
    var peak_mib = Float64(total_vram - min_free) / 1048576.0
    return KleinSamplerParityDenoiseResult(
        x.clone(ctx),
        Tensor(traj_buf^, traj_shape^, x.dtype()),
        denoise_seconds / Float64(num_steps),
        peak_mib,
    )


def _build_manifest_json(
    prompt: SamplePrompt,
    initial_noise_path: String,
    initial_noise_dtype: String,
    initial_noise_shape: List[Int],
    trajectory: Tensor,
    final_packed: Tensor,
    final_unscaled: Tensor,
    decoded: Tensor,
    out_dir: String,
    manifest_path: String,
    png_path: String,
    denoise_seconds_per_step: Float64,
    vae_decode_seconds: Float64,
    peak_vram_mib: Float64,
) raises -> String:
    var raw_shape = _shape4(1, 32, prompt.height // 8, prompt.width // 8)
    var post_patch_shape = _shape4(1, 128, prompt.height // 16, prompt.width // 16)
    var decoded_shape = _shape4(1, 3, prompt.height, prompt.width)
    var missing = String("MISSING_")

    var traj_path = _join(out_dir, String("mojo_latent_trajectory.bin"))
    var packed_path = _join(out_dir, String("mojo_final_packed_latent.bin"))
    var unpacked_path = _join(out_dir, String("mojo_final_unpacked_latent.bin"))
    var unscaled_path = _join(out_dir, String("mojo_final_unscaled_unpatchified_latent.bin"))
    var decoded_path = _join(out_dir, String("mojo_vae_decoded_tensor.bin"))

    var out = String("{\n")
    out += String("  \"schema_version\":1,\n")
    out += String("  \"producer\":\"serenitymojo/sampling/klein_sampler_parity_dump_cli.mojo\",\n")
    out += String("  \"scope\":\"Mojo-side Klein sampler artifact dump; partial paired manifest; no numeric parity accepted\",\n")
    out += String("  \"target_manifest\":") + _json_str(manifest_path) + String(",\n")
    out += String("  \"model_type\":\"FLUX_2\",\n")
    out += String("  \"parity_claimed\":false,\n")
    out += String("  \"parity_note\":\"Mojo artifacts are real; OneTrainer artifacts and numeric comparisons must be supplied by a paired OT run before strict acceptance.\",\n")
    out += String("  \"debug_parity_output\":true,\n")
    out += String("  \"dtype_note\":\"Tensor-bin artifacts preserve Mojo tensor storage bytes; PNG/JSON are parity/debug file-format outputs.\",\n")
    out += String("  \"prompt\":{")
    out += String("\"id\":") + _json_str(prompt.label)
    out += String(",\"positive\":") + _json_str(prompt.prompt)
    out += String(",\"negative\":") + _json_str(prompt.negative)
    out += String(",\"seed\":") + String(prompt.seed)
    out += String(",\"width\":") + String(prompt.width)
    out += String(",\"height\":") + String(prompt.height)
    out += String(",\"steps\":") + String(prompt.steps)
    out += String(",\"random_seed\":") + _json_bool(prompt.random_seed)
    out += String(",\"cfg_scale\":") + String(prompt.cfg)
    out += String("},\n")
    out += String("  \"scheduler\":") + _scheduler_json(prompt.steps, (prompt.height // 16) * (prompt.width // 16)) + String(",\n")
    out += String("  \"artifact_groups\":{")
    out += String("\"onetrainer_seed_replay_inputs\":[\"onetrainer_initial_noise_raw_nchw\",\"onetrainer_initial_noise_post_patch_nchw\",\"onetrainer_initial_noise_post_pack\"],")
    out += String("\"paired_latent_trajectory\":[{\"onetrainer\":\"onetrainer_latent_trajectory\",\"mojo\":\"mojo_latent_trajectory\"},{\"onetrainer\":\"onetrainer_final_packed_latent\",\"mojo\":\"mojo_final_packed_latent\"},{\"onetrainer\":\"onetrainer_final_unpacked_latent\",\"mojo\":\"mojo_final_unpacked_latent\"},{\"onetrainer\":\"onetrainer_final_unscaled_unpatchified_latent\",\"mojo\":\"mojo_final_unscaled_unpatchified_latent\"}],")
    out += String("\"paired_decode_and_png\":[{\"onetrainer\":\"onetrainer_vae_decoded_tensor\",\"mojo\":\"mojo_vae_decoded_tensor\"},{\"onetrainer\":\"onetrainer_png\",\"mojo\":\"mojo_png\"}]},\n")
    out += String("  \"artifacts\":{\n")
    out += String("    \"onetrainer_initial_noise_raw_nchw\":")
    out += _tensor_artifact_json(_join(out_dir, missing + String("onetrainer_initial_noise_raw_nchw.bin")), String("F32"), raw_shape)
    out += String(",\n    \"onetrainer_initial_noise_post_patch_nchw\":")
    out += _tensor_artifact_json(_join(out_dir, missing + String("onetrainer_initial_noise_post_patch_nchw.bin")), String("F32"), post_patch_shape)
    out += String(",\n    \"onetrainer_initial_noise_post_pack\":")
    out += _tensor_artifact_json(initial_noise_path, initial_noise_dtype, initial_noise_shape)
    out += String(",\n    \"onetrainer_latent_trajectory\":")
    out += _tensor_artifact_json(_join(out_dir, missing + String("onetrainer_latent_trajectory.bin")), String("F32"), trajectory.shape())
    out += String(",\n    \"mojo_latent_trajectory\":")
    out += _tensor_artifact_json(traj_path, trajectory.dtype().name(), trajectory.shape(), _tensor_bin_size(trajectory))
    out += String(",\n    \"onetrainer_final_packed_latent\":")
    out += _tensor_artifact_json(_join(out_dir, missing + String("onetrainer_final_packed_latent.bin")), String("F32"), final_packed.shape())
    out += String(",\n    \"mojo_final_packed_latent\":")
    out += _tensor_artifact_json(packed_path, final_packed.dtype().name(), final_packed.shape(), _tensor_bin_size(final_packed))
    out += String(",\n    \"onetrainer_final_unpacked_latent\":")
    out += _tensor_artifact_json(_join(out_dir, missing + String("onetrainer_final_unpacked_latent.bin")), String("F32"), final_packed.shape())
    out += String(",\n    \"mojo_final_unpacked_latent\":")
    out += _tensor_artifact_json(unpacked_path, final_packed.dtype().name(), final_packed.shape(), _tensor_bin_size(final_packed))
    out += String(",\n    \"onetrainer_final_unscaled_unpatchified_latent\":")
    out += _tensor_artifact_json(_join(out_dir, missing + String("onetrainer_final_unscaled_unpatchified_latent.bin")), String("F32"), final_unscaled.shape())
    out += String(",\n    \"mojo_final_unscaled_unpatchified_latent\":")
    out += _tensor_artifact_json(unscaled_path, final_unscaled.dtype().name(), final_unscaled.shape(), _tensor_bin_size(final_unscaled))
    out += String(",\n    \"onetrainer_vae_decoded_tensor\":")
    out += _tensor_artifact_json(_join(out_dir, missing + String("onetrainer_vae_decoded_tensor.bin")), String("F32"), decoded_shape)
    out += String(",\n    \"mojo_vae_decoded_tensor\":")
    out += _tensor_artifact_json(decoded_path, decoded.dtype().name(), decoded.shape(), _tensor_bin_size(decoded))
    out += String(",\n    \"onetrainer_png\":")
    out += _png_artifact_json(_join(out_dir, String("MISSING_onetrainer_png.png")), prompt.width, prompt.height)
    out += String(",\n    \"mojo_png\":")
    out += _png_artifact_json(png_path, prompt.width, prompt.height)
    out += String("\n  },\n")
    out += String("  \"metrics\":{\"onetrainer\":{\"denoise_seconds_per_step\":0.0,\"vae_decode_seconds\":0.0,\"peak_vram_mib\":0.0},\"mojo\":{")
    out += String("\"denoise_seconds_per_step\":") + String(denoise_seconds_per_step)
    out += String(",\"vae_decode_seconds\":") + String(vae_decode_seconds)
    out += String(",\"peak_vram_mib\":") + String(peak_vram_mib)
    out += String("}},\n")
    out += String("  \"comparisons\":{")
    out += String("\"trajectory\":{\"accepted\":false,\"max_abs\":null,\"tolerance\":null,\"note\":\"No paired OT comparison has been run.\"},")
    out += String("\"final_latent\":{\"accepted\":false,\"max_abs\":null,\"tolerance\":null,\"note\":\"No paired OT comparison has been run.\"},")
    out += String("\"vae_png\":{\"accepted\":false,\"max_abs\":null,\"tolerance\":null,\"note\":\"No paired OT comparison has been run.\"}},\n")
    out += String("  \"current_split_validation_evidence\":") + _support_evidence_json() + String("\n")
    out += String("}\n")
    return out^


def dump_klein_sampler_parity_artifacts[
    N_IMG: Int, N_TXT: Int, S: Int, LH: Int, LW: Int, H: Int, Dh: Int
](
    cfg: TrainConfig,
    lora_path: String,
    prompt: SamplePrompt,
    pos_txt: Tensor,
    neg_txt: Tensor,
    var initial_noise: Tensor,
    initial_noise_path: String,
    out_dir: String,
    manifest_path_in: String,
    ctx: DeviceContext,
    lora_multiplier: Float32 = Float32(1.0),
) raises -> KleinSamplerParityDumpResult:
    if cfg.n_heads != H:
        raise Error(String("klein parity dump: cfg.n_heads ") + String(cfg.n_heads) + " != comptime H " + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("klein parity dump: cfg.head_dim ") + String(cfg.head_dim) + " != comptime Dh " + String(Dh))
    if prompt.steps <= 0:
        raise Error("klein parity dump: prompt.steps must be positive")

    _mkdir_p(out_dir)
    var manifest_path = manifest_path_in
    if manifest_path == String(""):
        manifest_path = _join(out_dir, String("klein_sampler_mojo_manifest.json"))

    var initial_noise_dtype = initial_noise.dtype().name()
    var initial_noise_shape = initial_noise.shape()
    var x0 = _initial_noise_tokens_from_sidecar[N_IMG, LH, LW](
        initial_noise^, cfg.in_channels, ctx
    )
    var denoised = _denoise_lora_from_initial_with_trajectory[H, Dh, N_IMG, N_TXT, S, LH, LW](
        cfg, lora_path, pos_txt, neg_txt, prompt.cfg, prompt.steps, x0^, ctx,
        lora_multiplier,
    )

    var traj_path = _join(out_dir, String("mojo_latent_trajectory.bin"))
    save_tensor_bin(denoised.latent_trajectory, traj_path, ctx)

    var packed = tokens_to_packed_nchw[LH, LW](denoised.final_latent_tokens, ctx)
    var packed_path = _join(out_dir, String("mojo_final_packed_latent.bin"))
    var unpacked_path = _join(out_dir, String("mojo_final_unpacked_latent.bin"))
    save_tensor_bin(packed, packed_path, ctx)
    save_tensor_bin(packed, unpacked_path, ctx)

    var vae_t0 = perf_counter_ns()
    var vae = KleinVaeDecoder[LH, LW].load(cfg.vae, ctx)
    var z_bn = _inverse_bn(packed, vae.bn_scale, vae.bn_bias, ctx)
    var final_unscaled = _unpatchify_packed(z_bn, ctx)
    var unscaled_path = _join(out_dir, String("mojo_final_unscaled_unpatchified_latent.bin"))
    save_tensor_bin(final_unscaled, unscaled_path, ctx)
    var img = vae.decode(packed, ctx)
    ctx.synchronize()
    var vae_t1 = perf_counter_ns()
    var vae_decode_seconds = Float64(vae_t1 - vae_t0) / 1.0e9

    var decoded_path = _join(out_dir, String("mojo_vae_decoded_tensor.bin"))
    save_tensor_bin(img, decoded_path, ctx)
    var png_path = _join(out_dir, String("mojo_png.png"))
    save_image(img, png_path, ctx)
    print_sample_saved(String("Klein-parity-dump"), png_path)

    var mem_end = ctx.get_memory_info()
    var end_peak = Float64(Int(mem_end[1]) - Int(mem_end[0])) / 1048576.0
    var peak_vram_mib = denoised.peak_vram_mib
    if end_peak > peak_vram_mib:
        peak_vram_mib = end_peak

    var manifest = _build_manifest_json(
        prompt,
        initial_noise_path,
        initial_noise_dtype,
        initial_noise_shape,
        denoised.latent_trajectory,
        packed,
        final_unscaled,
        img,
        out_dir,
        manifest_path,
        png_path,
        denoised.denoise_seconds_per_step,
        vae_decode_seconds,
        peak_vram_mib,
    )
    _write_text(manifest_path, manifest)
    print("[klein-parity-dump] manifest:", manifest_path)
    print("[klein-parity-dump] parity_claimed: false")

    return KleinSamplerParityDumpResult(
        out_dir,
        manifest_path,
        png_path,
        denoised.denoise_seconds_per_step,
        vae_decode_seconds,
        peak_vram_mib,
    )
