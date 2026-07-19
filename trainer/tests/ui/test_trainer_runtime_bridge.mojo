"""Focused tests for the Serenity trainer runtime bridge."""

from serenity_trainer.ui.TrainerRuntimeBridge import (
    TrainerUIRuntime,
    _live_runner_command,
    trainer_ui_apply_serenity_callback_line,
    trainer_ui_apply_progress_line,
    trainer_ui_on_optimizer_step,
    trainer_ui_on_update_status,
    trainer_ui_on_update_train_progress,
    trainer_ui_pause,
    trainer_ui_poll_progress_file,
    trainer_ui_refresh_system_metrics,
    trainer_ui_resume,
    trainer_ui_sample_now,
    trainer_ui_save_checkpoint_now,
    trainer_ui_submit_current,
    trainer_ui_tick_and_apply,
    trainer_ui_cancel,
)
from serenity_trainer.ui.TrainerConfigModel import (
    TrainerUIConfig,
    trainer_ui_apply_model_preset,
    trainer_ui_runner_train_config_json,
)


def _expect(cond: Bool, msg: String) raises:
    if not cond:
        print("FAIL:", msg)
        raise Error(msg)


def _contains(text: String, token: String) -> Bool:
    var text_len = text.byte_length()
    var token_len = token.byte_length()
    if token_len <= 0:
        return True
    if text_len < token_len:
        return False
    var last = text_len - token_len
    for i in range(last + 1):
        if String(text[byte=i:i + token_len]) == token:
            return True
    return False


def _read_text(path: String) raises -> String:
    var f = open(path, "r")
    var text = f.read()
    f.close()
    return text^


def test_klein_progress_line() raises:
    var rt = TrainerUIRuntime()
    var ok = trainer_ui_apply_progress_line(
        rt,
        String("[Klein-lora] step 1613/2000 | epoch 14/17 | loss 0.5909 | grad_norm 0.1527 | 2.1s/step | elapsed 0:55:37 | ETA 0:13:20"),
    )
    _expect(ok, "progress parser should accept Klein-style output")
    _expect(rt.live.step == 1613, "step parsed")
    _expect(rt.live.total_steps == 2000, "total steps parsed")
    _expect(rt.live.epoch == 14, "epoch parsed")
    _expect(rt.live.total_epochs == 17, "total epochs parsed")
    _expect(rt.live.loss > 0.590 and rt.live.loss < 0.592, "loss parsed")
    _expect(rt.live.grad_norm > 0.152 and rt.live.grad_norm < 0.153, "grad parsed")
    _expect(rt.live.speed_it_s > 2.09 and rt.live.speed_it_s < 2.11, "speed parsed")
    _expect(rt.live.eta_secs == 800, "ETA parsed")
    _expect(rt.has_running, "run remains active before final step")
    _expect(rt.last_command == String("stream"), "command reflects stream update")


def test_ideogram_single_epoch_line_stays_running() raises:
    var rt = TrainerUIRuntime()
    var ok = trainer_ui_apply_progress_line(
        rt,
        String("[Ideogram4-lora] model IDEOGRAM_4 | type LoRA | step 8/3000 | epoch 1/1 | loss 0.6634249 | smooth_loss 0.9483377 | grad_norm 0.0000 | 27.163357s/step | elapsed 0:03:37 | ETA 22:34:32"),
    )
    _expect(ok, "Ideogram progress parser should accept model/type output")
    _expect(rt.live.step == 8, "Ideogram step parsed")
    _expect(rt.live.total_steps == 3000, "Ideogram total parsed")
    _expect(rt.live.epoch == 1, "Ideogram epoch parsed")
    _expect(rt.live.total_epochs == 1, "Ideogram total epoch parsed")
    _expect(rt.has_running, "single-epoch run stays active until final step")
    _expect(rt.status_text == String("Training ..."), "Ideogram status running")


def test_progress_file_tail() raises:
    var path = String("/tmp/serenity_trainer_progress_test.log")
    var f = open(path, "w")
    f.write(String("[Klein-lora] step 10/20 | epoch 1/2 | loss 0.7500 | grad_norm 0.0100 | 2.0s/step | elapsed 0:00:20 | ETA 0:00:20\n"))
    f.close()

    var rt = TrainerUIRuntime()
    rt.progress_file_path = path.copy()
    var ok = trainer_ui_poll_progress_file(rt)
    _expect(ok, "progress file should apply")
    _expect(rt.live.step == 10, "file step parsed")
    _expect(rt.live.total_steps == 20, "file total parsed")
    _expect(rt.using_live_progress, "live progress mode enabled")

    var again = trainer_ui_poll_progress_file(rt)
    _expect(not again, "second poll should not replay already consumed bytes")


def test_serenity_callback_progress_line() raises:
    var rt = TrainerUIRuntime()
    var ok = trainer_ui_apply_serenity_callback_line(
        rt,
        String("[Serenity-callback] progress epoch 2/10 | step 44/120 | global_step 284 | loss 0.5909 | smooth_loss 0.6123 | grad_norm 0.1527 | lr 0.0001 | status Training ..."),
    )
    _expect(ok, "callback parser should accept Serenity-shaped progress")
    _expect(rt.live.epoch == 2, "callback epoch parsed")
    _expect(rt.live.total_epochs == 10, "callback total epochs parsed")
    _expect(rt.live.step == 44, "callback step parsed")
    _expect(rt.live.total_steps == 120, "callback total steps parsed")
    _expect(rt.live.global_step == 284, "callback global step parsed")
    _expect(rt.live.loss > 0.590 and rt.live.loss < 0.592, "callback loss parsed")
    _expect(rt.live.smooth_loss > 0.612 and rt.live.smooth_loss < 0.613, "callback smooth loss parsed")
    _expect(rt.live.grad_norm > 0.152 and rt.live.grad_norm < 0.153, "callback grad parsed")
    _expect(rt.live.learning_rate > 0.00009 and rt.live.learning_rate < 0.00011, "callback lr parsed")
    _expect(rt.status_text == String("Training ..."), "callback status parsed")
    _expect(rt.last_command == String("callback"), "callback command set")
    _expect(rt.using_callback_progress, "callback source enabled")


def test_direct_serenity_callback_surface() raises:
    var rt = TrainerUIRuntime()
    trainer_ui_on_update_status(rt, String("Training ..."))
    trainer_ui_on_update_train_progress(rt, 1, 12, 120, 10)
    trainer_ui_on_optimizer_step(rt, 0.42, 0.50, 0.02, 0.0001)
    _expect(rt.live.epoch == 1, "direct callback epoch")
    _expect(rt.live.step == 12, "direct callback step")
    _expect(rt.live.total_steps == 120, "direct callback max step")
    _expect(rt.live.total_epochs == 10, "direct callback max epoch")
    _expect(rt.live.loss > 0.41 and rt.live.loss < 0.43, "direct callback loss")
    _expect(rt.live.smooth_loss > 0.49 and rt.live.smooth_loss < 0.51, "direct callback smooth loss")
    _expect(rt.status_text == String("Training ..."), "direct callback status")
    _expect(rt.using_callback_progress, "direct callback source")


def test_start_waits_for_real_progress() raises:
    var cfg = TrainerUIConfig()
    var rt = TrainerUIRuntime()
    rt.live_launch_enabled = False
    rt.progress_file_path = String("/tmp/serenity_missing_progress_file.log")
    rt.command_file_path = String("/tmp/serenity_start_wait_commands.jsonl")
    _ = trainer_ui_submit_current(cfg, rt)
    trainer_ui_tick_and_apply(rt)
    _expect(rt.has_running, "submitted run waits for bridge")
    _expect(rt.live.step == 0, "no synthetic step advance")
    _expect(rt.live.loss == 0.0, "no synthetic loss")
    _expect(not rt.using_live_progress, "no live source until progress file emits")
    _expect(rt.status_text == String("Waiting for trainer callbacks"), "waiting status")


def test_command_bridge_events() raises:
    var command_path = String("/tmp/serenity_trainer_commands_test.jsonl")
    var f = open(command_path, "w")
    f.write(String(""))
    f.close()

    var cfg = TrainerUIConfig()
    var rt = TrainerUIRuntime()
    rt.live_launch_enabled = False
    rt.progress_file_path = String("/tmp/serenity_command_bridge_progress.log")
    rt.command_file_path = command_path.copy()
    var id = trainer_ui_submit_current(cfg, rt)
    _expect(id != UInt64(0), "submit succeeds")
    _expect(trainer_ui_pause(rt), "pause succeeds")
    _expect(trainer_ui_resume(rt), "resume succeeds")
    _expect(trainer_ui_sample_now(rt), "sample command succeeds")
    _expect(trainer_ui_save_checkpoint_now(cfg, rt), "save command succeeds")
    trainer_ui_cancel(rt)

    var text = _read_text(command_path)
    _expect(_contains(text, String("\"action\":\"start\"")), "start command written")
    _expect(_contains(text, String("\"action\":\"pause\"")), "pause command written")
    _expect(_contains(text, String("\"action\":\"resume\"")), "resume command written")
    _expect(_contains(text, String("\"action\":\"sample\"")), "sample command written")
    _expect(_contains(text, String("\"action\":\"save\"")), "save command written")
    _expect(_contains(text, String("\"action\":\"stop\"")), "stop command written")


def test_ideogram4_launch_argv_contract() raises:
    # Ideogram4LiveTrainer argv 10/11 (T1 lever delivery): default-off ->
    # argv 10 = 0.0 and argv 11 = '-' (skip sentinel, C13); levers set ->
    # argv 11 = the written levers config JSON path.
    var cfg = TrainerUIConfig()
    cfg.model_type_index = 0  # IDEOGRAM_4
    trainer_ui_apply_model_preset(cfg, True)
    var rt = TrainerUIRuntime()

    var cmd0 = _live_runner_command(cfg, rt)
    _expect(
        _contains(cmd0, String("target/serenity_ideogram4_live_trainer")),
        "ideogram4 runner path in command",
    )
    _expect(
        _contains(cmd0, String(" 0.0 '-'")),
        "default-off launch carries argv10=0.0 argv11='-' (got: " + cmd0 + ")",
    )
    _expect(
        _contains(cmd0, String(" '-' 500 20 7.0 42")),
        "default launch carries sample argv12-15 (got: " + cmd0 + ")",
    )

    cfg.loss_fn = String("huber")
    var cmd1 = _live_runner_command(cfg, rt)
    _expect(
        _contains(
            cmd1,
            String(" 0.0 'target/serenity_ideogram4_train_config.json'"),
        ),
        "levers-on launch delivers the levers JSON path (got: " + cmd1 + ")",
    )
    _expect(
        _contains(cmd1, String("'target/serenity_ideogram4_train_config.json' 500 20 7.0 42")),
        "levers-on launch preserves sample argv12-15 (got: " + cmd1 + ")",
    )


def test_klein_launch_argv_contract() raises:
    # Klein is config-driven (UI wave 2): the launch must be the shared
    # `<train_config.json> <steps>` config-runner shape (serenitymojo
    # train_klein_real argv), NOT the old 11-arg positional KleinLiveTrainer
    # shape (progress_file/ckpt/cache/.../vae) which carried no levers.
    var cfg = TrainerUIConfig()
    cfg.model_type_index = 1  # FLUX_2 -> klein preset
    trainer_ui_apply_model_preset(cfg, True)
    var rt = TrainerUIRuntime()
    var cmd = _live_runner_command(cfg, rt)
    _expect(
        _contains(cmd, String("target/serenity_klein_live_trainer")),
        "klein runner path in command",
    )
    _expect(
        _contains(cmd, String("'target/serenity_klein_train_config.json' 3000")),
        "klein launch is <config.json> <steps> (got: " + cmd + ")",
    )
    _expect(
        _contains(cmd, String("/home/alex/mojodiffusion/output/alina_train")),
        "klein LORA_DIR/SAMPLE_DIR self-heal mkdir (got: " + cmd + ")",
    )
    _expect(
        not _contains(cmd, String("flux2-vae.safetensors")),
        "old positional vae argv is gone (vae now travels in the config JSON)",
    )
    _expect(
        not _contains(cmd, String("/tmp/serenity_klein_live_trainer.log")),
        "klein stdout tees into the shared progress log, not the legacy log",
    )


def test_hidream_launch_argv_contract() raises:
    # train_hidream_o1_real positional argv: <stage_dir> <steps> <lr> <rank>
    # <out_dir> - <config.json> (config delivers levers + quantized_resident).
    var cfg = TrainerUIConfig()
    cfg.model_type_index = 11  # HIDREAM_O1
    trainer_ui_apply_model_preset(cfg, True)
    var rt = TrainerUIRuntime()
    var cmd = _live_runner_command(cfg, rt)
    _expect(
        _contains(cmd, String("target/serenity_hidream_live_trainer")),
        "hidream runner path in command",
    )
    _expect(
        _contains(cmd, String("'/home/alex/trainings/ideogram4_giger_stage'")),
        "hidream stage dir is argv 1 (got: " + cmd + ")",
    )
    _expect(
        _contains(
            cmd,
            String(
                "'/home/alex/mojodiffusion/output/hidream_o1_lora'"
                " - 'target/serenity_hidream_train_config.json'"
            ),
        ),
        "hidream out_dir, ema '-', config json tail (got: " + cmd + ")",
    )


def test_krea2_launch_argv_contract() raises:
    # Krea2 is config-backed but not a `<config.json> <steps>` runner. The
    # mojodiffusion train_krea2 argv contract is:
    #   <cache.safetensors> <steps> <train_config.json>
    var cfg = TrainerUIConfig()
    cfg.model_type_index = 12  # KREA_2
    trainer_ui_apply_model_preset(cfg, True)
    var rt = TrainerUIRuntime()
    var cmd = _live_runner_command(cfg, rt)
    _expect(
        _contains(cmd, String("target/serenity_krea2_live_trainer")),
        "krea2 runner path in command",
    )
    _expect(
        _contains(
            cmd,
            String(
                "'/home/alex/eri2_stage_512/cache.safetensors'"
                " 2000 'target/serenity_krea2_train_config.json'"
            ),
        ),
        "krea2 launch is <cache> <steps> <config> (got: " + cmd + ")",
    )
    _expect(
        _contains(cmd, String("/home/alex/trainings/krea2_eri2_lora/samples")),
        "krea2 workspace/sample mkdir is present (got: " + cmd + ")",
    )
    _expect(
        not _contains(cmd, String("'target/serenity_krea2_train_config.json' 2000")),
        "krea2 must not use the config-runner argv shape (got: " + cmd + ")",
    )

    var json = trainer_ui_runner_train_config_json(cfg)
    _expect(_contains(json, String("\"model_type\": \"krea2\"")), "krea2 config model_type")
    _expect(_contains(json, String("\"quantized_resident\": \"fp8_e4m3\"")), "krea2 fp8 resident")
    _expect(_contains(json, String("\"validation_prompts_file\": \"/home/alex/mojodiffusion/serenitymojo/configs/krea2_samples.json\"")), "krea2 sample prompt file")
    _expect(_contains(json, String("\"workspace_dir\": \"/home/alex/trainings/krea2_eri2_lora\"")), "krea2 workspace")
    _expect(_contains(json, String("\"lora_rank\": 64")), "krea2 rank")


def test_system_metrics_refresh() raises:
    var rt = TrainerUIRuntime()
    trainer_ui_refresh_system_metrics(rt)
    _expect(rt.live.gpu_util >= 0.0, "gpu util non-negative")
    _expect(rt.live.cpu_util >= 0.0, "cpu util non-negative")
    _expect(rt.live.ram_gb >= 0.0, "ram used non-negative")
    _expect(rt.ram_total_gb >= 0.0, "ram total non-negative")


def main() raises:
    test_klein_progress_line()
    test_ideogram_single_epoch_line_stays_running()
    test_progress_file_tail()
    test_serenity_callback_progress_line()
    test_direct_serenity_callback_surface()
    test_start_waits_for_real_progress()
    test_command_bridge_events()
    test_ideogram4_launch_argv_contract()
    test_klein_launch_argv_contract()
    test_hidream_launch_argv_contract()
    test_krea2_launch_argv_contract()
    test_system_metrics_refresh()
    print("PASS: trainer runtime bridge parser")
