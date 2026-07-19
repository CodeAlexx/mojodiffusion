"""Frame-driven trainer runtime bridge for the native Serenity Mojo UI."""

from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from std.builtin.type_aliases import MutExternalOrigin
from std.time import perf_counter
from mojoui.render.backend import Backend

from serenity_trainer.ui.TrainerConfigModel import (
    SERENITY_TRAINER_OUTPUT_DIR,
    TrainerUIConfig,
    trainer_ui_config_json_snapshot,
    trainer_ui_ideogram4_levers_path_or_skip,
    trainer_ui_ideogram4_levers_set,
    trainer_ui_ignored_lever_summary,
    trainer_ui_json_escape,
    trainer_ui_runner_train_config_json,
    trainer_ui_total_steps,
    trainer_ui_validate,
)


comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]
comptime IDEOGRAM4_LIVE_RUNNER = "target/serenity_ideogram4_live_trainer"
comptime TERMINAL_LAUNCHER = "scripts/serenity_terminal_launcher.sh"
comptime IDEOGRAM4_LIVE_LOG = "/tmp/serenity_ideogram4_live_trainer.log"
comptime IDEOGRAM4_LIVE_PID = "target/serenity_ideogram4_live_trainer.pid"


struct TrainerUILiveStats(Copyable, Movable):
    var step: Int32
    var total_steps: Int32
    var epoch: Int32
    var total_epochs: Int32
    var global_step: Int32
    var loss: Float32
    var smooth_loss: Float32
    var grad_norm: Float32
    var learning_rate: Float32
    var speed_it_s: Float32
    var eta_secs: Int32
    var gpu_util: Float32
    var vram_gb: Float32
    var vram_total_gb: Float32
    var temp_c: Int32
    var cpu_util: Float32
    var ram_gb: Float32

    def __init__(out self):
        self.step = 0
        self.total_steps = 0
        self.epoch = 0
        self.total_epochs = 0
        self.global_step = 0
        self.loss = 0.0
        self.smooth_loss = 0.0
        self.grad_norm = 0.0
        self.learning_rate = 0.0
        self.speed_it_s = 0.0
        self.eta_secs = 0
        self.gpu_util = 0.0
        self.vram_gb = 0.0
        self.vram_total_gb = 0.0
        self.temp_c = 0
        self.cpu_util = 0.0
        self.ram_gb = 0.0


struct TrainerUIRuntime(Movable):
    var backend_label: String
    var backend_target: String
    var has_running: Bool
    var paused: Bool
    var frame_counter: Int32
    var run_id: UInt64
    var progress_file_path: String
    var command_file_path: String
    var progress_file_bytes: Int
    var live_launch_enabled: Bool
    var using_live_progress: Bool
    var using_callback_progress: Bool
    var start_time: Float64
    var last_progress_time: Float64
    var last_progress_step: Int32
    var status_text: String
    var last_validation_summary: String
    var last_command: String
    var gpu_name: String
    var gpu_driver: String
    var cpu_name: String
    var ram_total_gb: Float32
    var live: TrainerUILiveStats
    var samples: List[String]
    var checkpoints: List[String]
    var runs: List[String]
    var logs: List[String]

    def __init__(out self):
        self.backend_label = String("Klein 9B")
        self.backend_target = String("klein")
        self.has_running = False
        self.paused = False
        self.frame_counter = 59
        self.run_id = 0
        self.progress_file_path = String("target/serenity_trainer_progress.log")
        self.command_file_path = String("target/serenity_trainer_commands.jsonl")
        self.progress_file_bytes = 0
        self.live_launch_enabled = True
        self.using_live_progress = False
        self.using_callback_progress = False
        self.start_time = 0.0
        self.last_progress_time = 0.0
        self.last_progress_step = 0
        self.status_text = String("Idle")
        self.last_validation_summary = String("Ready")
        self.last_command = String("idle")
        self.gpu_name = String("")
        self.gpu_driver = String("")
        self.cpu_name = String("")
        self.ram_total_gb = 0.0
        self.live = TrainerUILiveStats()
        self.samples = List[String]()
        self.checkpoints = List[String]()
        self.runs = List[String]()
        self.logs = List[String]()


def _append_runtime_log(mut rt: TrainerUIRuntime, text: String):
    rt.logs.append(text.copy())
    while len(rt.logs) > 512:
        _ = rt.logs.pop(0)


def _append_command_event(mut rt: TrainerUIRuntime, action: String, config_path: String = String("")) -> Bool:
    try:
        var f = open(rt.command_file_path.copy(), "a")
        f.write(
            String("{\"schema\":\"serenity.trainer_command.v1\",")
            + String("\"action\":\"") + trainer_ui_json_escape(action) + String("\",")
            + String("\"run_id\":") + String(rt.run_id) + String(",")
            + String("\"backend\":\"") + trainer_ui_json_escape(rt.backend_target) + String("\",")
            + String("\"progress_file\":\"") + trainer_ui_json_escape(rt.progress_file_path) + String("\",")
            + String("\"config_file\":\"") + trainer_ui_json_escape(config_path) + String("\"}")
        )
        f.write("\n")
        f.close()
        return True
    except e:
        _append_runtime_log(rt, String("command bridge write failed: ") + String(e))
        return False


def _sys_system(command: String) -> Int:
    var n = command.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = command.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var cstr = BytePtr(unsafe_from_address=Int(buf))
    var status = Int(external_call["system", Int32](cstr))
    buf.free()
    return status


def _shell_quote(value: String) -> String:
    # POSIX single-quote: safe for every byte except the single quote itself,
    # which is closed-escaped-reopened ('\''). Byte-accumulated so multi-byte
    # UTF-8 in paths survives intact. The old version documented the apostrophe
    # hole instead of handling it (UI audit 2026-07-03, bug 2 class).
    var out = List[UInt8]()
    out.append(0x27)  # opening '
    var vb = value.as_bytes()
    for i in range(value.byte_length()):
        if vb[i] == 0x27:  # ' -> '\''
            out.append(0x27)
            out.append(0x5C)
            out.append(0x27)
            out.append(0x27)
        else:
            out.append(vb[i])
    out.append(0x27)  # closing '
    try:
        return String(from_utf8=out)
    except:
        # unreachable for valid input strings; fall back to the raw wrap
        return String("'") + value.copy() + String("'")


def _ideogram4_transformer_path(cfg: TrainerUIConfig) -> String:
    return (
        cfg.base_model_name.copy()
        + String("/transformer/diffusion_pytorch_model.safetensors")
    )


def _is_config_runner_target(target: String) -> Bool:
    # Config-driven runners (serenitymojo train_<m>_real.mojo built into
    # target/serenity_<m>_live_trainer). They take `<train_config.json> <steps>`
    # and stream the shared print_trainer_progress line shape on stdout.
    # klein joined 2026-06-12 (UI wave 2): train_klein_real argv is
    # `<config.json> [run_steps]` and it CONSUMES the T1 levers
    # (loss/optimizer/EMA/caption-dropout) from the config JSON — the old
    # positional KleinLiveTrainer port carried no levers at all.
    return (
        target == String("chroma")
        or target == String("ernie")
        or target == String("anima")
        or target == String("sdxl")
        or target == String("zimage")
        or target == String("l2p")
        or target == String("klein")
    )


def _is_hidream_target(target: String) -> Bool:
    # HiDream-O1: serenitymojo train_hidream_o1_real built into
    # target/serenity_hidream_live_trainer. Positional argv (NOT the
    # `<config.json> <steps>` config-runner shape):
    #   <stage_dir> <steps> [lr] [rank] [out_dir] [ema_decay] [config.json]
    # Progress streams on stdout (tee'd into the progress file like the
    # config runners).
    return target == String("hidream")


def _is_krea2_target(target: String) -> Bool:
    # Krea-2 Raw: mojodiffusion train_krea2.mojo built into
    # target/serenity_krea2_live_trainer. Positional argv:
    #   <cache.safetensors> <steps> <train_config.json>
    # Progress streams on stdout like the config-driven runners.
    return target == String("krea2")


def _backend_label(target: String) -> String:
    if target == String("klein"):
        return String("Klein 9B")
    if target == String("ideogram4"):
        return String("Ideogram4")
    if _is_hidream_target(target):
        return String("HiDream O1")
    if _is_krea2_target(target):
        return String("Krea 2")
    if target == String("chroma"):
        return String("Chroma1 HD")
    if target == String("ernie"):
        return String("Ernie Image")
    if target == String("anima"):
        return String("Anima")
    if target == String("sdxl"):
        return String("SDXL 1.0")
    if target == String("zimage"):
        return String("Z-Image")
    if target == String("l2p"):
        return String("Z-Image L2P")
    return target.copy()


def _runner_path(target: String) -> String:
    if _is_config_runner_target(target) or _is_hidream_target(target) or _is_krea2_target(target):
        return String("target/serenity_") + target.copy() + String("_live_trainer")
    return String(IDEOGRAM4_LIVE_RUNNER)


def _pid_path(target: String) -> String:
    if _is_config_runner_target(target) or _is_hidream_target(target) or _is_krea2_target(target):
        return String("target/serenity_") + target.copy() + String("_live_trainer.pid")
    return String(IDEOGRAM4_LIVE_PID)


def _log_path(target: String) -> String:
    if _is_config_runner_target(target) or _is_hidream_target(target) or _is_krea2_target(target):
        # Config-driven runners stream progress on stdout only; tee their
        # stdout straight into the polled progress file so the legacy
        # progress-line bridge picks it up live.
        return String("target/serenity_trainer_progress.log")
    return String(IDEOGRAM4_LIVE_LOG)


def _runner_train_config_path(target: String) -> String:
    return String("target/serenity_") + target.copy() + String("_train_config.json")


def _ideogram4_sample_prompt_path() -> String:
    return String("target/serenity_ideogram4_sample_prompt.json")


def _write_ideogram4_sample_prompt(cfg: TrainerUIConfig) raises -> String:
    if len(cfg.samples) == 0:
        raise Error("Ideogram4 inline sampling needs a JSON sample prompt")
    var path = _ideogram4_sample_prompt_path()
    var f = open(path.copy(), "w")
    f.write(cfg.samples[0].prompt.copy())
    f.close()
    return path^


def _live_runner_command(cfg: TrainerUIConfig, rt: TrainerUIRuntime) -> String:
    var steps = Int(cfg.max_train_steps)
    if steps < 1:
        steps = Int(trainer_ui_total_steps(cfg))
    var rank = Int(cfg.lora_rank)
    if rank < 1:
        rank = 1
    var alpha = cfg.lora_alpha
    if alpha <= 0.0:
        alpha = Float32(rank)
    var save_every = Int(cfg.save_every)
    if save_every < 0:
        save_every = 0
    var target = cfg.backend_target.copy()
    var ideogram4_sample_prompt_arg = String("-")
    if target == String("ideogram4") and Int(cfg.sample_after) > 0 and len(cfg.samples) > 0:
        ideogram4_sample_prompt_arg = _ideogram4_sample_prompt_path()
    var args: String
    if _is_krea2_target(target):
        # train_krea2 keeps the cache path positional because the cache is the
        # primary dataset artifact. The TrainConfig JSON still carries checkpoint,
        # VAE, recipe, sampler cadence, workspace, and lever keys.
        args = (
            _shell_quote(cfg.cache_dir.copy())
            + String(" ")
            + String(steps)
            + String(" ")
            + _shell_quote(_runner_train_config_path(target.copy()))
        )
    elif _is_config_runner_target(target):
        # serenitymojo train_<m>_real runners: `<train_config.json> <steps>`.
        # The recipe (lr/rank/alpha/steps/save/cache/ckpt) is in the config
        # JSON written at launch by trainer_ui_launch_live_runner.
        args = (
            _shell_quote(_runner_train_config_path(target.copy()))
            + String(" ")
            + String(steps)
        )
    elif _is_hidream_target(target):
        # train_hidream_o1_real positional argv (header precedence rules):
        # argv wins for steps/lr/rank/out_dir; ema_decay "-" so the config
        # owns EMA (full SimpleTuner schedule when enabled); trailing
        # [config.json] delivers the T1 levers + T2.B quantized_resident.
        # stage_dir := cfg.cache_dir (stage-A images + raw captions).
        args = (
            _shell_quote(cfg.cache_dir.copy())
            + String(" ")
            + String(steps)
            + String(" ")
            + String(cfg.learning_rate)
            + String(" ")
            + String(rank)
            + String(" ")
            + _shell_quote(String(SERENITY_TRAINER_OUTPUT_DIR) + String("/hidream_o1_lora"))
            + String(" - ")
            + _shell_quote(_runner_train_config_path(target.copy()))
        )
    else:
        args = (
            _shell_quote(rt.progress_file_path.copy())
            + String(" ")
            + _shell_quote(_ideogram4_transformer_path(cfg))
            + String(" ")
            + _shell_quote(cfg.cache_dir.copy())
            + String(" ")
            + _shell_quote(String(SERENITY_TRAINER_OUTPUT_DIR))
            + String(" ")
            + String(steps)
            + String(" ")
            + String(rank)
            + String(" ")
            + String(alpha)
            + String(" ")
            + String(cfg.learning_rate)
            + String(" ")
            + String(save_every)
            # T1 lever delivery (Ideogram4LiveTrainer argv contract):
            # argv 10 = caption_dropout_prob (0.0 default = never drop);
            # argv 11 = levers config JSON path, or "-" when every lever is
            # default-off so default runs stay byte-identical (C13).
            # argv 12-15 = inline sampler cadence/steps/CFG/seed. Sampling is
            # not a T1 lever; it is delivered positionally so resident sampling
            # works even when the lever JSON is skipped.
            # argv 16 = optional resume LoRA path (UI skip sentinel here).
            # argv 17 = JSON prompt file for the real inline sampler.
            # argv 18 = requested square sample resolution.
            + String(" ")
            + String(cfg.caption_dropout)
            + String(" ")
            + _shell_quote(
                trainer_ui_ideogram4_levers_path_or_skip(
                    cfg, _runner_train_config_path(target.copy())
                )
            )
            + String(" ")
            + String(Int(cfg.sample_after))
            + String(" ")
            + String(Int(cfg.sample_steps))
            + String(" ")
            + String(cfg.sample_cfg)
            + String(" ")
            + String(Int(cfg.seed))
            + String(" - ")
            + _shell_quote(ideogram4_sample_prompt_arg)
            + String(" ")
            + cfg.resolution.copy()
        )
    var mkdirs = String("")
    if _is_hidream_target(target.copy()):
        # train_hidream_o1_real saves into out_dir without creating it.
        mkdirs = String(
            "mkdir -p /home/alex/mojodiffusion/output/hidream_o1_lora && "
        )
    if _is_krea2_target(target.copy()):
        # train_krea2 saves LoRAs and inline sample PNGs under workspace_dir.
        mkdirs = (
            String("mkdir -p ")
            + _shell_quote(cfg.workspace_dir.copy())
            + String(" ")
            + _shell_quote(cfg.workspace_dir.copy() + String("/samples"))
            + String(" && ")
        )
    if _is_config_runner_target(target.copy()):
        # The config-driven runners save to fixed per-model output dirs and do
        # not create them (Chroma raised "failed to open for write" on a
        # missing dir after an otherwise-good step). Self-heal at launch.
        if target == String("klein"):
            # train_klein_real saves LoRAs/samples into its comptime
            # LORA_DIR/SAMPLE_DIR (alina_train); it mkdirs SAMPLE_DIR itself
            # but self-heal here too. Klein-only so the other config runners'
            # launch commands stay byte-identical (C13).
            mkdirs = String(
                "mkdir -p /home/alex/mojodiffusion/output/alina_train"
                " /home/alex/mojodiffusion/output && "
            )
        else:
            mkdirs = String(
                "mkdir -p /home/alex/mojodiffusion/output/chroma_boxjana"
                " /home/alex/mojodiffusion/output/alina_sdxl"
                " /home/alex/mojodiffusion/output/alina_zimage"
                " /home/alex/mojodiffusion/output/alina_l2p"
                " /home/alex/mojodiffusion/serenitymojo/output"
                " /home/alex/mojodiffusion/output && "
            )
    return (
        String("cd /home/alex/serenity-trainer && ")
        + mkdirs
        + String("rm -f ")
        + _shell_quote(_pid_path(target.copy()))
        + String(" ")
        + _shell_quote(_term_pid_path(target.copy()))
        + String(" && bash ")
        + String(TERMINAL_LAUNCHER)
        + String(" ")
        + _shell_quote(_backend_label(target.copy()))
        + String(" ")
        + _shell_quote(_runner_path(target.copy()))
        + String(" ")
        + _shell_quote(_pid_path(target.copy()))
        + String(" ")
        + _shell_quote(_log_path(target.copy()))
        + String(" ")
        + args
    )


def _any_live_runner_active() -> Bool:
    var rc = _sys_system(String("pgrep -f '^(/home/alex/serenity-trainer/)?target/[s]erenity_(ideogram4|klein|chroma|ernie|anima|sdxl|zimage|l2p|hidream|krea2)_live_trainer( |$)' > /dev/null 2>&1"))
    return rc == 0


def _owned_live_runner_active(target: String) -> Bool:
    # kill -0 alone false-positives on a RECYCLED pid from a stale pidfile
    # (wrapper SIGKILL/logout paths can leave the pidfile behind), which
    # blocks every future launch with "Trainer already running" and no
    # terminal. Require the pid's cmdline to actually be this runner.
    var rc = _sys_system(
        String("pid=$(cat ")
        + _shell_quote(_pid_path(target.copy()))
        + String(" 2>/dev/null); [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null")
        + String(" && grep -aq ")
        + _shell_quote(_runner_path(target.copy()))
        + String(" \"/proc/$pid/cmdline\" 2>/dev/null")
    )
    return rc == 0


def _term_pid_path(target: String) -> String:
    # Terminal-wrapper pid written by scripts/serenity_terminal_launcher.sh's
    # run script ("$pidfile.term"). TERMing the WRAPPER is what closes the
    # gnome-terminal window: its trap kills the runner and exits. The runner
    # pid alone cannot close the window — the runner shares the wrapper's
    # process group, so the old group-kill on the runner pid always failed
    # and the window lingered at a "Press Enter" prompt (measured 2026-06-11).
    return _pid_path(target.copy()) + String(".term")


def _stop_live_runner(target: String) -> Bool:
    var pid_q = _shell_quote(_pid_path(target.copy()))
    var term_q = _shell_quote(_term_pid_path(target.copy()))
    var rc = _sys_system(
        String("term=$(cat ") + term_q + String(" 2>/dev/null); ")
        + String("pid=$(cat ") + pid_q + String(" 2>/dev/null); ")
        + String("ok=1; ")
        # Preferred: TERM the terminal wrapper — its trap stops the runner
        # AND closes the terminal window.
        + String("if [ -n \"$term\" ] && kill -TERM \"$term\" 2>/dev/null; then ok=0; fi; ")
        # Fallback (wrapper already gone / legacy launch): TERM the runner.
        + String("if [ -n \"$pid\" ] && kill -TERM \"$pid\" 2>/dev/null; then ok=0; fi; ")
        # Remove the pidfiles only when a kill actually landed. Unconditional
        # removal blinded the ownership guard while the process could still be
        # alive (UI audit 2026-07-03, bug 5); stale files on the failure path
        # are handled by the recycled-pid cmdline check + pre-launch rm.
        + String("if [ $ok -eq 0 ]; then rm -f ") + pid_q + String(" ") + term_q + String("; fi; ")
        + String("exit $ok")
    )
    return rc == 0


def _record_live_runner_exit(mut rt: TrainerUIRuntime):
    _append_runtime_log(
        rt,
        rt.backend_label.copy()
        + String(" live trainer process exited; stdout log: ")
        + _log_path(rt.backend_target.copy()),
    )
    try:
        var text = _read_file_text(_log_path(rt.backend_target.copy()))
        _ = _apply_progress_text_from(rt, text.copy(), 0)
    except:
        pass
    _ = _sys_system(
        String("rm -f ")
        + _shell_quote(_pid_path(rt.backend_target.copy()))
        + String(" ")
        + _shell_quote(_term_pid_path(rt.backend_target.copy()))
    )


def _write_runner_train_config(cfg: TrainerUIConfig) raises -> String:
    var path = _runner_train_config_path(cfg.backend_target.copy())
    var f = open(path.copy(), "w")
    f.write(trainer_ui_runner_train_config_json(cfg))
    f.close()
    return path^


def trainer_ui_launch_live_runner(cfg: TrainerUIConfig, mut rt: TrainerUIRuntime) -> Bool:
    if (
        cfg.backend_target != String("ideogram4")
        and not _is_config_runner_target(cfg.backend_target.copy())
        and not _is_hidream_target(cfg.backend_target.copy())
        and not _is_krea2_target(cfg.backend_target.copy())
    ):
        _append_runtime_log(
            rt,
            String("no live trainer runner wired for backend: ") + cfg.backend_target.copy(),
        )
        rt.status_text = String("No trainer wired for ") + cfg.backend_target.copy()
        return False
    rt.backend_target = cfg.backend_target.copy()
    rt.backend_label = _backend_label(cfg.backend_target.copy())
    if not rt.live_launch_enabled:
        _append_runtime_log(rt, String("live trainer launch disabled for this runtime"))
        return False
    if _owned_live_runner_active(rt.backend_target.copy()):
        _append_runtime_log(rt, String("live trainer launch skipped: existing ") + rt.backend_label.copy() + String(" runner is active"))
        rt.status_text = String("Trainer already running")
        return False
    if _any_live_runner_active():
        _append_runtime_log(rt, String("live trainer launch skipped: another Serenity runner is active"))
        rt.status_text = String("Another trainer is running")
        return False
    var needs_runner_config = (
        _is_config_runner_target(rt.backend_target.copy())
        or _is_hidream_target(rt.backend_target.copy())
        or _is_krea2_target(rt.backend_target.copy())
        # ideogram4: write the levers JSON ONLY when a lever is set — the
        # no-levers launch passes "-" (argv 11 skip sentinel, C13).
        or (
            rt.backend_target == String("ideogram4")
            and trainer_ui_ideogram4_levers_set(cfg)
        )
    )
    if needs_runner_config:
        try:
            var cfg_path = _write_runner_train_config(cfg)
            _append_runtime_log(rt, String("wrote runner train config: ") + cfg_path.copy())
        except e:
            _append_runtime_log(rt, String("runner train config write failed: ") + String(e))
            rt.status_text = String("Runner config write failed")
            return False
    if rt.backend_target == String("ideogram4") and Int(cfg.sample_after) > 0:
        try:
            var prompt_path = _write_ideogram4_sample_prompt(cfg)
            _append_runtime_log(rt, String("wrote Ideogram4 JSON sample prompt: ") + prompt_path.copy())
        except e:
            _append_runtime_log(rt, String("Ideogram4 sample prompt write failed: ") + String(e))
            rt.status_text = String("Sample prompt write failed")
            return False
    var rc = _sys_system(_live_runner_command(cfg, rt))
    if rc == 0:
        _append_runtime_log(rt, String("launched ") + rt.backend_label.copy() + String(" live trainer: ") + _log_path(rt.backend_target.copy()))
        rt.status_text = String("Launching ") + rt.backend_label.copy() + String(" trainer")
        return True
    _append_runtime_log(rt, rt.backend_label.copy() + String(" live trainer launch failed rc=") + String(rc))
    rt.status_text = String("Live trainer launch failed")
    return False


def trainer_ui_progress_fraction(rt: TrainerUIRuntime) -> Float32:
    if rt.live.total_steps <= 0:
        return 0.0
    if rt.live.total_epochs > 0:
        var total = rt.live.total_steps * rt.live.total_epochs
        if total > 0:
            return Float32(rt.live.global_step) / Float32(total)
    return Float32(rt.live.step) / Float32(rt.live.total_steps)


def trainer_ui_eta_label(rt: TrainerUIRuntime) -> String:
    if rt.has_running and rt.live.global_step <= 30:
        return String("Estimating ...")
    if rt.live.eta_secs < 0:
        return String("")
    var days = rt.live.eta_secs // 86400
    var rem = rt.live.eta_secs - days * 86400
    var hours = rem // 3600
    rem = rem - hours * 3600
    var minutes = rem // 60
    var seconds = rem - minutes * 60
    if days > 0:
        return String(days) + String("d ") + String(hours) + String("h")
    if hours > 0:
        return String(hours) + String("h ") + String(minutes) + String("m")
    if minutes > 0:
        return String(minutes) + String("m ") + String(seconds) + String("s")
    return String(seconds) + String("s")


def _update_eta_like_serenity(mut rt: TrainerUIRuntime):
    if rt.live.total_steps <= 0 or rt.live.total_epochs <= 0:
        rt.live.eta_secs = 0
        return
    # Epochs are 1-based in trainer output ("epoch 1/1"); clamp 0-based staging
    # lines up to 1 so the basis matches the global_step math below (the old
    # 0-based formula showed ETA 0 for the ENTIRE last epoch of a 1-based
    # trainer — UI audit 2026-07-03, bug 3).
    var epoch_1b = rt.live.epoch
    if epoch_1b < 1:
        epoch_1b = 1
    var steps_done = (epoch_1b - 1) * rt.live.total_steps + rt.live.step
    var remaining_steps = (
        (rt.live.total_epochs - epoch_1b) * rt.live.total_steps
        + (rt.live.total_steps - rt.live.step)
    )
    if steps_done <= 0:
        rt.live.eta_secs = -1
        return
    if remaining_steps < 0:
        remaining_steps = 0
    var now = perf_counter()
    var spent_total = now - rt.start_time
    if spent_total < 0.0:
        spent_total = 0.0
    rt.live.eta_secs = Int32((spent_total / Float64(steps_done)) * Float64(remaining_steps))


def trainer_ui_on_update_status(mut rt: TrainerUIRuntime, status: String):
    """Serenity-shaped train status surface."""
    rt.status_text = status.copy()
    rt.last_validation_summary = status.copy()


def trainer_ui_on_update_train_progress(
    mut rt: TrainerUIRuntime,
    epoch: Int32,
    epoch_step: Int32,
    max_step: Int32,
    max_epoch: Int32,
):
    """Serenity-shaped train progress surface.

    Serenity calls this after `train_progress.next_step(batch_size)` with
    `(train_progress, current_epoch_length, config.epochs)`. The UI progress is
    epoch-local step progress plus epoch progress, not a terminal tail.
    """
    var now = perf_counter()
    if rt.start_time <= 0.0:
        rt.start_time = now
    if rt.last_progress_time > 0.0:
        var delta_step = epoch_step - rt.last_progress_step
        if delta_step <= 0 and epoch_step == 0:
            delta_step = 1
        var delta_t = now - rt.last_progress_time
        if delta_step > 0 and delta_t > 0.0:
            rt.live.speed_it_s = Float32(delta_t / Float64(delta_step))
    rt.last_progress_time = now
    rt.last_progress_step = epoch_step
    rt.live.epoch = epoch
    rt.live.step = epoch_step
    rt.live.total_steps = max_step
    rt.live.total_epochs = max_epoch
    var completed_epochs = epoch - 1
    if completed_epochs < 0:
        completed_epochs = 0
    rt.live.global_step = completed_epochs * max_step + epoch_step
    rt.using_live_progress = True
    rt.using_callback_progress = True
    _update_eta_like_serenity(rt)


def trainer_ui_on_optimizer_step(
    mut rt: TrainerUIRuntime,
    loss: Float32,
    smooth_loss: Float32,
    grad_norm: Float32,
    learning_rate: Float32,
):
    """Loss/grad update surface mirroring GenericTrainer's update-step block."""
    rt.live.loss = loss
    rt.live.smooth_loss = smooth_loss
    rt.live.grad_norm = grad_norm
    rt.live.learning_rate = learning_rate


def _live_progress_complete(rt: TrainerUIRuntime) -> Bool:
    if rt.live.total_steps <= 0:
        return False
    if rt.live.total_epochs > 0:
        var total = rt.live.total_steps * rt.live.total_epochs
        if total > 0 and rt.live.global_step >= total:
            return True
    if rt.live.step < rt.live.total_steps:
        return False
    if rt.live.total_epochs <= 0:
        return True
    if rt.live.epoch >= rt.live.total_epochs:
        return True
    return False


def trainer_ui_refresh_system_metrics(mut rt: TrainerUIRuntime):
    var metrics = Backend.system_metrics()
    if metrics.gpu_available:
        rt.gpu_name = metrics.gpu_name.copy()
        rt.gpu_driver = metrics.gpu_driver.copy()
        rt.live.gpu_util = Float32(metrics.gpu_util_percent)
        rt.live.temp_c = metrics.gpu_temperature_c
        rt.live.vram_gb = Float32(metrics.gpu_memory_used_mb) / 1024.0
        rt.live.vram_total_gb = Float32(metrics.gpu_memory_total_mb) / 1024.0
    if metrics.cpu_name.byte_length() > 0:
        rt.cpu_name = metrics.cpu_name.copy()
    rt.live.cpu_util = Float32(metrics.cpu_util_percent)
    rt.live.ram_gb = Float32(metrics.ram_used_mb) / 1024.0
    rt.ram_total_gb = Float32(metrics.ram_total_mb) / 1024.0


def trainer_ui_submit_current(cfg: TrainerUIConfig, mut rt: TrainerUIRuntime) -> UInt64:
    rt.backend_target = cfg.backend_target.copy()
    rt.backend_label = _backend_label(cfg.backend_target.copy())
    if rt.has_running:
        _append_runtime_log(rt, String("submit ignored: run already active"))
        rt.last_command = String("submit ignored")
        return rt.run_id
    rt.last_validation_summary = trainer_ui_validate(cfg)
    if rt.last_validation_summary != String("Ready"):
        _append_runtime_log(rt, String("validation failed: ") + rt.last_validation_summary.copy())
        return UInt64(0)
    # Capability-table honesty (UI wave 2): launch proceeds, but every
    # non-default widget the selected runner does not consume is named LOUDLY
    # in the status rail + run log so the run record cannot silently lie.
    var ignored = trainer_ui_ignored_lever_summary(cfg)
    if ignored.byte_length() > 0:
        _append_runtime_log(rt, String("WARNING before launch: ") + ignored.copy())
        rt.last_validation_summary = String("WARNING ") + ignored.copy()
    var config_path: String
    try:
        config_path = _save_config_snapshot(cfg)
    except e:
        _append_runtime_log(rt, String("config save failed: ") + String(e))
        rt.last_command = String("submit failed")
        rt.last_validation_summary = String("Config save failed")
        return UInt64(0)
    rt.run_id = rt.run_id + 1
    rt.has_running = False
    rt.paused = False
    rt.frame_counter = 0
    rt.start_time = perf_counter()
    rt.last_progress_time = 0.0
    rt.last_progress_step = 0
    rt.progress_file_bytes = _file_byte_length(rt.progress_file_path.copy())
    rt.using_live_progress = False
    rt.using_callback_progress = False
    rt.last_command = String("submit ") + cfg.backend_target.copy()
    rt.status_text = String("Waiting for trainer callbacks")
    rt.live.step = 0
    rt.live.total_steps = trainer_ui_total_steps(cfg)
    rt.live.epoch = 0
    rt.live.total_epochs = Int32(cfg.epochs)
    rt.live.global_step = 0
    rt.live.loss = 0.0
    rt.live.smooth_loss = 0.0
    rt.live.grad_norm = 0.0
    rt.live.learning_rate = cfg.learning_rate
    rt.live.speed_it_s = 0.0
    rt.live.eta_secs = 0
    rt.live.gpu_util = 0.0
    rt.live.vram_gb = 0.0
    rt.live.vram_total_gb = 0.0
    rt.live.temp_c = 0
    rt.live.cpu_util = 0.0
    rt.live.ram_gb = 0.0
    trainer_ui_refresh_system_metrics(rt)
    _append_runtime_log(rt, String("submitted ") + cfg.backend_target.copy() + String(" trainer target: ") + cfg.base_model_name.copy())
    _append_runtime_log(rt, String("saved trainer config: ") + config_path.copy())
    _append_runtime_log(rt, String("watching Serenity callback bridge: ") + rt.progress_file_path.copy())
    if _append_command_event(rt, String("start"), config_path.copy()):
        _append_runtime_log(rt, String("command bridge start event: ") + rt.command_file_path.copy())
    var launched = trainer_ui_launch_live_runner(cfg, rt)
    if rt.live_launch_enabled and not launched:
        rt.last_command = String("launch failed")
        return UInt64(0)
    rt.has_running = True
    rt.runs.append(String("#") + String(rt.run_id) + String(" ") + cfg.run_name.copy() + String(" - ") + cfg.backend_target.copy() + String(" target"))
    return rt.run_id


def trainer_ui_pause(mut rt: TrainerUIRuntime) -> Bool:
    if not rt.has_running or rt.paused:
        return False
    rt.paused = True
    rt.last_command = String("pause")
    rt.status_text = String("Paused")
    rt.live.gpu_util = 0.0
    _append_runtime_log(rt, String("paused #") + String(rt.run_id))
    _ = _append_command_event(rt, String("pause"))
    return True


def trainer_ui_resume(mut rt: TrainerUIRuntime) -> Bool:
    if not rt.has_running or not rt.paused:
        return False
    rt.paused = False
    rt.last_command = String("resume")
    rt.status_text = String("Waiting for trainer callbacks")
    _append_runtime_log(rt, String("resumed #") + String(rt.run_id))
    _ = _append_command_event(rt, String("resume"))
    return True


def trainer_ui_cancel(mut rt: TrainerUIRuntime):
    if rt.has_running:
        _append_runtime_log(rt, String("cancelled #") + String(rt.run_id))
    if rt.live_launch_enabled and _owned_live_runner_active(rt.backend_target.copy()):
        if _stop_live_runner(rt.backend_target.copy()):
            _append_runtime_log(rt, String("stopped ") + rt.backend_label.copy() + String(" live trainer process"))
        else:
            _append_runtime_log(rt, rt.backend_label.copy() + String(" live trainer stop signal failed"))
    rt.has_running = False
    rt.paused = False
    rt.frame_counter = 0
    rt.last_command = String("cancel")
    rt.status_text = String("Stopped")
    rt.using_live_progress = False
    rt.using_callback_progress = False
    rt.live.gpu_util = 0.0
    rt.live.cpu_util = 0.0
    _ = _append_command_event(rt, String("stop"))


def trainer_ui_sample_now(mut rt: TrainerUIRuntime) -> Bool:
    if not rt.has_running:
        _append_runtime_log(rt, String("sample requested while idle"))
        return False
    _append_runtime_log(rt, String("sample command requested at step ") + String(rt.live.step))
    rt.last_command = String("sample")
    _ = _append_command_event(rt, String("sample"))
    return True


def _save_config_snapshot(cfg: TrainerUIConfig) raises -> String:
    var path = String("target/serenity_trainer_ui_config.json")
    var f = open(path, "w")
    f.write(trainer_ui_config_json_snapshot(cfg))
    f.write("\n")
    f.close()
    return path


def trainer_ui_save_checkpoint_now(cfg: TrainerUIConfig, mut rt: TrainerUIRuntime) -> Bool:
    var config_path: String
    try:
        config_path = _save_config_snapshot(cfg)
        _append_runtime_log(rt, String("config saved: ") + config_path.copy())
    except e:
        _append_runtime_log(rt, String("config save failed: ") + String(e))
        rt.last_command = String("save failed")
        return False
    if not rt.has_running:
        rt.last_command = String("save config")
        return False
    _append_runtime_log(rt, String("checkpoint command requested at step ") + String(rt.live.step))
    rt.last_command = String("save checkpoint")
    _ = _append_command_event(rt, String("save"), config_path.copy())
    return True


# NOTE: these helpers scan RAW BYTES. Trainer stdout can contain multi-byte
# UTF-8 (e.g. "—" in banner lines); String codepoint indexing at an arbitrary
# byte offset asserts on a mid-codepoint boundary, so never use
# `String(line[byte=i])` to scan — byte compares are boundary-safe.
def _find_token(line: String, token: String, start: Int = 0) -> Int:
    var lb = line.as_bytes()
    var tb = token.as_bytes()
    var line_len = len(lb)
    var token_len = len(tb)
    if token_len <= 0 or line_len < token_len:
        return -1
    var i = start
    if i < 0:
        i = 0
    var last = line_len - token_len
    while i <= last:
        var match_here = True
        for j in range(token_len):
            if lb[i + j] != tb[j]:
                match_here = False
                break
        if match_here:
            return i
        i = i + 1
    return -1


def _find_char(line: String, needle: String, start: Int) -> Int:
    var lb = line.as_bytes()
    var nb = needle.as_bytes()
    if len(nb) != 1:
        return -1
    var i = start
    if i < 0:
        i = 0
    while i < len(lb):
        if lb[i] == nb[0]:
            return i
        i = i + 1
    return -1


def _token_end(line: String, start: Int) -> Int:
    var lb = line.as_bytes()
    var i = start
    if i < 0:
        i = 0
    while i < len(lb):
        if lb[i] == 0x20 or lb[i] == 0x7C:  # ' ' or '|'
            return i
        i = i + 1
    return i


def _read_int_between(line: String, start: Int, end: Int) raises -> Int32:
    if start < 0 or end <= start:
        raise Error("empty int token")
    return Int32(atol(String(line[byte=start:end])))


def _read_float_between(line: String, start: Int, end: Int) raises -> Float32:
    if start < 0 or end <= start:
        raise Error("empty float token")
    return Float32(atof(String(line[byte=start:end])))


def _parse_seconds(text: String) raises -> Int32:
    var first = _find_char(text, String(":"), 0)
    if first < 0:
        return Int32(atol(text))
    var second = _find_char(text, String(":"), first + 1)
    if second < 0:
        var mins = atol(String(text[byte=0:first]))
        var secs = atol(String(text[byte=first + 1:]))
        return Int32(mins * 60 + secs)
    var hours = atol(String(text[byte=0:first]))
    var mins = atol(String(text[byte=first + 1:second]))
    var secs = atol(String(text[byte=second + 1:]))
    return Int32(hours * 3600 + mins * 60 + secs)


def _read_after_token_float(line: String, token: String) raises -> Float32:
    var pos = _find_token(line.copy(), token.copy())
    if pos < 0:
        raise Error(String("missing token ") + token.copy())
    var start = pos + token.byte_length()
    var end = _token_end(line.copy(), start)
    return _read_float_between(line.copy(), start, end)


def _read_after_token_int(line: String, token: String) raises -> Int32:
    var pos = _find_token(line.copy(), token.copy())
    if pos < 0:
        raise Error(String("missing token ") + token.copy())
    var start = pos + token.byte_length()
    var end = _token_end(line.copy(), start)
    return _read_int_between(line.copy(), start, end)


def trainer_ui_apply_serenity_callback_line(mut rt: TrainerUIRuntime, line: String) -> Bool:
    """Apply a Serenity UI bridge progress line.

    Expected shape:
      [Serenity-callback] progress epoch 1/10 | step 44/120 |
      global_step 164 | loss 0.5909 | smooth_loss 0.6123 |
      grad_norm 0.1527 | lr 0.000100 | status Training

    This is not a terminal view. It is the file-backed equivalent of
    TrainCallbacks.on_update_train_progress plus the update-step loss values
    Serenity trainer streams to the status rail.
    """
    if _find_token(line.copy(), String("[Serenity-callback]")) < 0:
        return False
    try:
        var epoch_pos = _find_token(line.copy(), String("epoch "))
        var step_pos = _find_token(line.copy(), String("step "))
        if epoch_pos < 0 or step_pos < 0:
            return False

        var epoch_start = epoch_pos + 6
        var epoch_slash = _find_char(line.copy(), String("/"), epoch_start)
        var epoch_end = _token_end(line.copy(), epoch_slash + 1)
        var epoch = _read_int_between(line.copy(), epoch_start, epoch_slash)
        var total_epoch = _read_int_between(line.copy(), epoch_slash + 1, epoch_end)

        var step_start = step_pos + 5
        var step_slash = _find_char(line.copy(), String("/"), step_start)
        var step_end = _token_end(line.copy(), step_slash + 1)
        var step = _read_int_between(line.copy(), step_start, step_slash)
        var total_step = _read_int_between(line.copy(), step_slash + 1, step_end)

        trainer_ui_on_update_train_progress(rt, epoch, step, total_step, total_epoch)
        rt.live.global_step = _read_after_token_int(line.copy(), String("global_step "))

        try:
            var loss = _read_after_token_float(line.copy(), String("loss "))
            var smooth = _read_after_token_float(line.copy(), String("smooth_loss "))
            var grad = _read_after_token_float(line.copy(), String("grad_norm "))
            var lr = _read_after_token_float(line.copy(), String("lr "))
            trainer_ui_on_optimizer_step(rt, loss, smooth, grad, lr)
        except:
            pass

        var status_pos = _find_token(line.copy(), String("status "))
        if status_pos >= 0:
            trainer_ui_on_update_status(
                rt,
                String(line[byte=status_pos + 7:]),
            )
        else:
            trainer_ui_on_update_status(rt, String("Training ..."))
        rt.has_running = not _live_progress_complete(rt)
        rt.paused = False
        rt.last_command = String("callback")
        rt.using_live_progress = True
        rt.using_callback_progress = True
        _append_runtime_log(rt, line.copy())
        return True
    except:
        return False


def trainer_ui_apply_progress_line(mut rt: TrainerUIRuntime, line: String) -> Bool:
    """Parse a Serenity/Klein progress line and update live stats.

    Expected shape:
      [Klein-lora] step 1613/2000 | epoch 14/17 | loss 0.5909 |
      grad_norm 0.1527 | 2.1s/step | elapsed 0:55:37 | ETA 0:13:20
    """
    try:
        # Parse EVERYTHING into locals first; commit only after the full line
        # validates. The old code wrote rt.live.step before the rest parsed, so
        # inline-SAMPLER lines ("[krea2-sample-inline] step 5 / 30 ...") that
        # fail later left a stale step behind and made the progress rail jump
        # to the sampler's counter during every cadence render (UI audit
        # 2026-07-03, bug 1).
        var step_pos = _find_token(line.copy(), String("step "))
        if step_pos < 0:
            return False
        var step_start = step_pos + 5
        var step_slash = _find_char(line.copy(), String("/"), step_start)
        var step_end = _token_end(line.copy(), step_slash + 1)
        var new_step = _read_int_between(line.copy(), step_start, step_slash)
        var new_total_steps = _read_int_between(line.copy(), step_slash + 1, step_end)

        var new_epoch = rt.live.epoch
        var new_total_epochs = rt.live.total_epochs
        var epoch_pos = _find_token(line.copy(), String("epoch "), step_end)
        if epoch_pos >= 0:
            var epoch_start = epoch_pos + 6
            var epoch_slash = _find_char(line.copy(), String("/"), epoch_start)
            var epoch_end = _token_end(line.copy(), epoch_slash + 1)
            new_epoch = _read_int_between(line.copy(), epoch_start, epoch_slash)
            new_total_epochs = _read_int_between(line.copy(), epoch_slash + 1, epoch_end)

        var new_loss = _read_after_token_float(line.copy(), String("loss "))
        var new_grad = _read_after_token_float(line.copy(), String("grad_norm "))

        # Line fully validated - commit.
        rt.live.step = new_step
        rt.live.total_steps = new_total_steps
        rt.live.epoch = new_epoch
        rt.live.total_epochs = new_total_epochs
        rt.live.loss = new_loss
        rt.live.grad_norm = new_grad
        # Prefer the trainer's own smooth_loss when the line carries one
        # (ideogram4 does); the UI-side EMA is only a fallback for trainers
        # that print raw loss only (UI audit 2026-07-03, bug 4).
        try:
            rt.live.smooth_loss = _read_after_token_float(line.copy(), String("smooth_loss "))
        except:
            if rt.live.smooth_loss <= 0.0:
                rt.live.smooth_loss = rt.live.loss
            else:
                rt.live.smooth_loss = rt.live.smooth_loss * 0.99 + rt.live.loss * 0.01

        var speed_marker = _find_token(line.copy(), String("s/step"))
        if speed_marker >= 0:
            var lb = line.as_bytes()
            var speed_start = speed_marker - 1
            while speed_start > 0:
                var ch = lb[speed_start - 1]
                if ch == 0x20 or ch == 0x7C:  # ' ' or '|'
                    break
                speed_start = speed_start - 1
            rt.live.speed_it_s = _read_float_between(line.copy(), speed_start, speed_marker)

        var eta_pos = _find_token(line.copy(), String("ETA "))
        if eta_pos >= 0:
            var eta_start = eta_pos + 4
            var eta_end = _token_end(line.copy(), eta_start)
            rt.live.eta_secs = _parse_seconds(String(line[byte=eta_start:eta_end]))

        var completed_epochs = rt.live.epoch - 1
        if completed_epochs < 0:
            completed_epochs = 0
        rt.live.global_step = completed_epochs * rt.live.total_steps + rt.live.step
        rt.has_running = not _live_progress_complete(rt)
        rt.paused = False
        rt.last_command = String("stream")
        rt.status_text = String("Training ...")
        rt.last_validation_summary = String("Ready")
        _append_runtime_log(rt, line.copy())
        return True
    except:
        return False


def _read_file_text(path: String) raises -> String:
    var f = open(path, "r")
    var text = f.read()
    f.close()
    return text^


def _file_byte_length(path: String) -> Int:
    try:
        return _read_file_text(path).byte_length()
    except:
        return 0


def _apply_progress_text_from(mut rt: TrainerUIRuntime, text: String, start: Int) -> Bool:
    var begin = start
    if begin < 0:
        begin = 0
    if begin > text.byte_length():
        begin = 0
    var applied = False
    while begin < text.byte_length():
        var end = _find_char(text.copy(), String("\n"), begin)
        if end < 0:
            end = text.byte_length()
        if end > begin:
            var line = String(text[byte=begin:end])
            if trainer_ui_apply_serenity_callback_line(rt, line):
                applied = True
            elif trainer_ui_apply_progress_line(rt, line):
                applied = True
        begin = end + 1
    return applied


def trainer_ui_poll_progress_file(mut rt: TrainerUIRuntime) -> Bool:
    """Tail the configured progress file and apply newly appended UI events.

    Preferred real path mirrors Serenity TrainCallbacks:
    write `[Serenity-callback] ...` event lines and the right rail updates
    from those callback values. Legacy Klein/tqdm-style lines are accepted only
    as a fallback bridge while the full in-process runner is being wired.
    """
    try:
        var text = _read_file_text(rt.progress_file_path.copy())
        var start = rt.progress_file_bytes
        if start > text.byte_length():
            start = 0
        if text.byte_length() <= start:
            return False
        var applied = _apply_progress_text_from(rt, text.copy(), start)
        rt.progress_file_bytes = text.byte_length()
        if applied:
            rt.using_live_progress = True
        return applied
    except:
        return False


def trainer_ui_tick_and_apply(mut rt: TrainerUIRuntime):
    rt.frame_counter = rt.frame_counter + 1
    if rt.frame_counter >= 60:
        rt.frame_counter = 0
        trainer_ui_refresh_system_metrics(rt)
    if not rt.has_running or rt.paused:
        return
    # Throttle bridge I/O off the 60 Hz render thread:
    # - the progress poll re-reads the WHOLE progress file (tee'd full trainer
    #   stdout, grows monotonically) — at 60 Hz that is O(file bytes) per
    #   frame; a ~100 ms cadence is plenty for the live rail.
    # - the liveness probe forks a shell (cat + kill -0 + grep) — 1 Hz.
    if rt.frame_counter % 6 == 0:
        if trainer_ui_poll_progress_file(rt):
            return
    if rt.live_launch_enabled and rt.start_time > 0.0 and rt.frame_counter == 30:
        var elapsed = perf_counter() - rt.start_time
        if elapsed > 3.0 and not _owned_live_runner_active(rt.backend_target.copy()):
            _record_live_runner_exit(rt)
            rt.has_running = False
            rt.status_text = String("Trainer process exited")
            rt.last_command = String("process exit")
            rt.using_live_progress = False
            rt.using_callback_progress = False
            return
    if not rt.using_live_progress:
        rt.status_text = String("Waiting for trainer callbacks")
