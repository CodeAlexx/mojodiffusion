# serenitymojo/models/ltx2/parity/ltx2_av_loss_combine_gate.mojo
#
# LTX-2 P6.2 AV loss-combine + DEAD-KEY end-to-end host gate.
#
# The MANDATE (lead): a config JSON carrying video_loss_weight / audio_loss_weight
# must MEASURABLY change the combined AV loss (video*vw + audio*aw) — a key that
# parses but doesn't move the number is still dead. This gate drives the REAL
# reader (read_model_config -> cfg.levers.{video,audio}_loss_weight) and the
# ADOPTED masked_loss reducer, and asserts:
#   * defaults-absent -> cfg.levers video=1.0 / audio=0.0 -> combined == video_loss
#     (C13: the video/v2v arms are byte-identical — audio weight 0 contributes 0);
#   * a config with audio_loss_weight>0 -> combined strictly larger (audio LIVE);
#   * distinct (vw,aw) -> distinct combined values (both keys reach the number).
#
# Run: rm -f serenitymojo.mojopkg; pixi run mojo build -O2 -I . -Xlinker -lm \
#   -Xlinker -lcuda serenitymojo/models/ltx2/parity/ltx2_av_loss_combine_gate.mojo \
#   -o /tmp/ltx2_av_loss_combine_gate && /tmp/ltx2_av_loss_combine_gate

from std.collections import List
from std.math import abs

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.ltx2.masked_loss import (
    masked_loss_unmasked, av_combine_loss, LTX2_LOSS_MSE,
)


# thin wrapper: the combine reads the MIGRATED levers weights the trainer reads.
def _combine(video_loss: Float32, audio_loss: Float32, levers: TrainConfig) -> Float32:
    return av_combine_loss(video_loss, audio_loss, levers.video_loss_weight, levers.audio_loss_weight)


def _write(path: String, body: String) raises:
    var f = open(path, "w")
    f.write(body)
    f.close()


def _close(a: Float32, b: Float32) -> Bool:
    return abs(a - b) <= Float32(1e-6)




def _expect(tag: String, cond: Bool, detail: String):
    print("  ", "PASS" if cond else "FAIL", tag, detail)


def main() raises:
    print("=== LTX-2 P6.2 AV loss-combine + dead-key end-to-end gate ===")

    # synthetic per-modality losses (fixed, non-degenerate).
    var vp = List[Float32](); var vt = List[Float32]()
    var ap = List[Float32](); var at = List[Float32]()
    for i in range(64):
        vp.append(Float32(i) * Float32(0.01)); vt.append(Float32(0.0))
        ap.append(Float32(i) * Float32(0.02)); at.append(Float32(0.0))
    var vloss = masked_loss_unmasked(vp, vt, LTX2_LOSS_MSE)
    var aloss = masked_loss_unmasked(ap, at, LTX2_LOSS_MSE)
    print("  video_loss=", vloss, " audio_loss=", aloss)

    var fails = 0

    # (1) defaults-absent config -> levers video=1.0 / audio=0.0 -> combined==video_loss (C13).
    var d = TrainConfig.default()
    var c_def = _combine(vloss, aloss, d)
    var ok_def = _close(c_def, vloss)
    _expect(String("defaults C13 (combined==video_loss, audio w=0)"), ok_def,
            String("combined=") + String(c_def) + " video=" + String(vloss))
    if not ok_def: fails += 1

    # (2) a config JSON with audio_loss_weight>0 -> combined strictly LARGER (audio LIVE).
    _write(String("/tmp/ltx2_p62_av.json"),
           String("{\"video_loss_weight\": 1.0, \"audio_loss_weight\": 0.5}"))
    var lv = read_model_config(String("/tmp/ltx2_p62_av.json"))
    var c_av = _combine(vloss, aloss, lv)
    var expect_av = vloss * Float32(1.0) + aloss * Float32(0.5)
    var ok_av = _close(c_av, expect_av) and c_av > c_def + Float32(1e-6)
    _expect(String("config audio_loss_weight=0.5 MOVES the number (not dead)"), ok_av,
            String("combined=") + String(c_av) + " expect=" + String(expect_av)
            + " vs default=" + String(c_def))
    if not ok_av: fails += 1

    # (3) distinct (vw,aw) -> distinct combined (BOTH keys reach the number).
    _write(String("/tmp/ltx2_p62_av2.json"),
           String("{\"video_loss_weight\": 2.0, \"audio_loss_weight\": 1.0}"))
    var lv2 = read_model_config(String("/tmp/ltx2_p62_av2.json"))
    var c_av2 = _combine(vloss, aloss, lv2)
    var expect_av2 = vloss * Float32(2.0) + aloss * Float32(1.0)
    var ok_av2 = _close(c_av2, expect_av2) and not _close(c_av2, c_av)
    _expect(String("config (vw=2,aw=1) distinct from (vw=1,aw=0.5)"), ok_av2,
            String("combined=") + String(c_av2) + " expect=" + String(expect_av2))
    if not ok_av2: fails += 1

    # (4) video_loss_weight ALSO reaches the number (config vw=0 -> combined==audio only).
    _write(String("/tmp/ltx2_p62_av3.json"),
           String("{\"video_loss_weight\": 0.0, \"audio_loss_weight\": 1.0}"))
    var lv3 = read_model_config(String("/tmp/ltx2_p62_av3.json"))
    var c_av3 = _combine(vloss, aloss, lv3)
    var ok_av3 = _close(c_av3, aloss)
    _expect(String("config video_loss_weight=0 -> combined==audio_loss (video key LIVE)"), ok_av3,
            String("combined=") + String(c_av3) + " audio=" + String(aloss))
    if not ok_av3: fails += 1

    print("")
    if fails == 0:
        print("LTX2 P6.2 AV loss-combine + dead-key gate PASS: config video/audio_loss_weight",
              "measurably move the combined loss; defaults C13")
    else:
        raise Error(String("LTX2 P6.2 AV loss-combine gate FAIL: ") + String(fails))
