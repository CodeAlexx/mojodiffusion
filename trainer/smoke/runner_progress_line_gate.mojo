# runner_progress_line_gate.mojo — gate the runner->UI progress seam.
#
# Feeds the progress file produced by a REAL config-driven runner launch
# (target/serenity_trainer_progress.log) through the live bridge parser and
# asserts the UI live stats update. Run after a runner smoke (e.g. the anima
# 2-step run) has populated the progress file.

from serenity_trainer.ui.TrainerRuntimeBridge import (
    TrainerUIRuntime,
    trainer_ui_apply_progress_line,
)


def main() raises:
    var rt = TrainerUIRuntime()
    var f = open(String("target/serenity_trainer_progress.log"), "r")
    var text = f.read()
    f.close()

    var applied = 0
    var bs = text.as_bytes()
    var begin = 0
    while begin < len(bs):
        var end = begin
        while end < len(bs) and bs[end] != 0x0A:
            end = end + 1
        if end > begin:
            if trainer_ui_apply_progress_line(rt, String(text[byte=begin:end])):
                applied = applied + 1
        begin = end + 1

    print("applied progress lines:", applied)
    print(
        "live: step ", rt.live.step, "/", rt.live.total_steps,
        " epoch ", rt.live.epoch, "/", rt.live.total_epochs,
        " loss ", rt.live.loss, " grad_norm ", rt.live.grad_norm,
        " speed ", rt.live.speed_it_s, "s/step eta ", rt.live.eta_secs,
    )
    if applied < 2:
        raise Error("FAIL: expected >=2 parsed runner progress lines")
    if rt.live.step != rt.live.total_steps or rt.live.total_steps < 2:
        raise Error("FAIL: final step/total not reflected in live stats")
    if rt.live.loss <= 0.0 or rt.live.grad_norm <= 0.0:
        raise Error("FAIL: loss/grad_norm did not flow into live stats")
    print("GATE PASS — real runner progress lines drive the UI live stats")
