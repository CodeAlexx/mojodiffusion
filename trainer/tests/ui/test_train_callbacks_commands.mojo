"""Focused tests for Serenity-shaped callback and command state."""

from serenity_trainer.util.TrainProgress import TrainProgress
from serenity_trainer.util.callbacks.TrainCallbacks import TrainCallbacks
from serenity_trainer.util.callbacks.TrainCallbacks import train_callback_progress_line
from serenity_trainer.util.commands.TrainCommands import TrainCommands
from serenity_trainer.ui.TrainerRuntimeBridge import (
    TrainerUIRuntime,
    trainer_ui_apply_serenity_callback_line,
)


def _expect(cond: Bool, msg: String) raises:
    if not cond:
        print("FAIL:", msg)
        raise Error(msg)


def test_callbacks() raises:
    var callbacks = TrainCallbacks()
    var progress = TrainProgress.zero()
    progress.next_step(2)
    callbacks.on_update_train_progress(progress, 120, 10)
    callbacks.on_update_status(String("Training ..."))
    callbacks.on_update_sample_default_progress(4, 28)
    callbacks.on_sample_default()
    _expect(callbacks.progress.global_step == 1, "callback progress global step")
    _expect(callbacks.max_step == 120, "callback max step")
    _expect(callbacks.max_epoch == 10, "callback max epoch")
    _expect(callbacks.status == String("Training ..."), "callback status")
    _expect(callbacks.sample_default_step == 4, "sample progress")
    _expect(callbacks.sample_default_count == 1, "sample count")
    var line = train_callback_progress_line(callbacks, 0.7, 0.75, 0.2, 0.0001)
    var rt = TrainerUIRuntime()
    _expect(trainer_ui_apply_serenity_callback_line(rt, line), "callback line consumed by UI runtime")
    _expect(rt.live.global_step == Int32(callbacks.progress.global_step), "callback line global step")


def test_commands() raises:
    var commands = TrainCommands()
    commands.sample_default()
    commands.sample_custom(String("manual sample"))
    commands.backup()
    commands.save()
    _expect(commands.get_and_reset_sample_default_command(), "sample default one-shot")
    _expect(not commands.get_and_reset_sample_default_command(), "sample default resets")
    var custom = commands.get_and_reset_sample_custom_commands()
    _expect(len(custom) == 1, "custom sample one-shot length")
    _expect(custom[0] == String("manual sample"), "custom sample payload")
    _expect(commands.get_and_reset_backup_command(), "backup one-shot")
    _expect(commands.get_and_reset_save_command(), "save one-shot")
    commands.stop()
    _expect(commands.get_stop_command(), "stop remains sticky")


def main() raises:
    test_callbacks()
    test_commands()
    print("PASS: train callbacks/commands")
