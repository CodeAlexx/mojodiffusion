# image_preprocess_probe.mojo — parity gate for the pure-Mojo i2v condition
# image preprocessor (models/lingbotvideo/lingbot_image_preprocess.mojo).
#
# Runs load_condition_image on the SAME fixed test PNG the Python oracle used,
# then compares the resulting [1,3,1,H,W] tensor to oracle_image_preprocess's
# expected tensor (cos + max_abs). Gate: cos >= 0.999 (resampler interpolation
# differences are OK; a big divergence => a layout/normalize bug).
#
# Run:
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#     serenitymojo/models/lingbotvideo/parity/image_preprocess_probe.mojo
#
# Prereq: generate the oracle first with
#   /home/alex/SerenityTrainer/venv/bin/python \
#     serenitymojo/models/lingbotvideo/parity/oracle_image_preprocess.py

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.lingbotvideo.lingbot_image_preprocess import (
    load_condition_image,
)

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime TEST_IMG = "/home/alex/mojodiffusion/output/media/lingbotvideo/mojo_e_generated.png"
comptime H = 576
comptime W = 320


def _sh(s: List[Int]) -> String:
    var out = String("[")
    for i in range(len(s)):
        if i > 0:
            out += ", "
        out += String(s[i])
    out += "]"
    return out


def main() raises:
    var ctx = DeviceContext()

    print("[img-prep] load_condition_image on", TEST_IMG)
    var mine = load_condition_image(String(TEST_IMG), H, W, ctx)
    print("  mine shape =", _sh(mine.shape()))
    var mine_host = mine.to_host(ctx)

    var oracle_path = String(PARITY_DIR) + "/oracle_image_preprocess.safetensors"
    print("[img-prep] loading oracle", oracle_path)
    var st = ShardedSafeTensors.open(oracle_path)
    var reference = Tensor.from_view_as_f32(st.tensor_view("image"), ctx)
    print("  ref  shape =", _sh(reference.shape()))
    var ref_host = reference.to_host(ctx)

    var harness = ParityHarness(0.999)
    var res = harness.compare_host(mine_host, ref_host)
    print("[img-prep] cos      =", res.cos)
    print("[img-prep] max_abs  =", res.max_abs)
    if res.passed:
        print("[img-prep] GATE PASS (cos >= 0.999)")
    else:
        print("[img-prep] GATE FAIL (cos < 0.999)")
