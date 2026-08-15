# ltx2_config_torchref_huber_smoke.mojo — gate the torchref-huber -> smooth_l1 remap
# in LTX2TrainerConfig.from_args (skeptic S1). torchref's loss_type "huber" IS
# F.smooth_l1_loss(beta=huber_delta) (hv_train_network.py:499-506), so an ltx2
# --config with loss_fn "huber" must land as smooth_l1(beta=huber_delta) — NOT
# torch-huber (clamped-grad). Also asserts an all-off config stays mse (C13).
# The remap lives in from_args (config.mojo:579-584), so this drives that exact
# path (not read_model_config, which keeps the reader's torch-huber meaning).
#
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/ltx2/parity/ltx2_config_torchref_huber_smoke.mojo \
#     -o /tmp/ltx2_huber_gate && /tmp/ltx2_huber_gate

from serenitymojo.training.ltx2.config import LTX2TrainerConfig
from serenitymojo.training.train_config import LOSS_FN_MSE, LOSS_FN_SMOOTH_L1


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _close32(v: Float32, exp: Float32) -> Bool:
    var d = v - exp
    if d < Float32(0.0):
        d = -d
    return d < Float32(1.0e-5)


def _write(path: String, body: String) raises:
    var f = open(path, "w")
    f.write(body)
    f.close()


def _args(cfg_path: String) -> List[String]:
    # argv shape from_args expects: args[0]=prog (skipped), then flags.
    var a = List[String]()
    a.append(String("ltx2"))
    a.append(String("--config"))
    a.append(cfg_path)
    return a^


def main() raises:
    # 1) torchref-huber remap: loss_fn "huber" + huber_delta 0.7 -> smooth_l1(0.7).
    var hp = String("/tmp/ltx2_ref_huber_smoke.json")
    _write(hp, String('{ "model_type": "ltx2", "loss_fn": "huber", "huber_delta": 0.7 }'))
    var ch = LTX2TrainerConfig.from_args(_args(hp))
    _require(
        ch.levers.loss_fn == LOSS_FN_SMOOTH_L1,
        String("huber must remap to smooth_l1; got loss_fn=") + String(ch.levers.loss_fn),
    )
    _require(
        _close32(ch.levers.smooth_l1_beta, Float32(0.7)),
        String("smooth_l1_beta must == huber_delta 0.7; got ") + String(ch.levers.smooth_l1_beta),
    )
    print("  PASS torchref-huber-remap  loss_fn=", ch.levers.loss_fn,
          " smooth_l1_beta=", ch.levers.smooth_l1_beta)

    # 2) all-off config leaves loss_fn == mse (C13 default-off).
    var op = String("/tmp/ltx2_ref_alloff_smoke.json")
    _write(op, String('{ "model_type": "ltx2" }'))
    var co = LTX2TrainerConfig.from_args(_args(op))
    _require(
        co.levers.loss_fn == LOSS_FN_MSE,
        String("all-off must stay mse; got loss_fn=") + String(co.levers.loss_fn),
    )
    print("  PASS all-off-mse  loss_fn=", co.levers.loss_fn)

    print("ALL PASS")
