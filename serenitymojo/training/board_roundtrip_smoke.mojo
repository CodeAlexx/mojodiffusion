# training/board_roundtrip_smoke.mojo — SerenityBoard SQLite roundtrip (item 1a).
#
# Opens a SerenityBoardWriter on dir /tmp/wave1_board (open() appends /board.db,
# so the real DB file is /tmp/wave1_board/board.db), logs 3 train steps with
# KNOWN loss / grad_norm / lr / step_secs (sps derived), closes, and prints what
# it wrote. An external `sqlite3` CLI query (run by the driver, NOT chained here)
# reads back the `scalars` table and confirms the values match.
#
# Verifies the pure-Mojo board writer (serenityboard.mojo, the SQLite FFI path)
# actually persists scalars — no train run, no GPU. The writer mirrors what the
# real trainers call (log_train_step), so a green roundtrip means the board
# emission spine is sound.
#
# Build/run: the SQLite FFI does NOT link under JIT `mojo run` (sqlite3_open/exec
# unresolved); must AOT-build with the sqlite lib explicitly:
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#     -Xlinker -L"$PWD/.pixi/envs/default/lib" -Xlinker -lsqlite3 \
#     serenitymojo/training/board_roundtrip_smoke.mojo -o /tmp/board_rt_smoke
#   LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:$LD_LIBRARY_PATH" /tmp/board_rt_smoke
# Then (SEPARATE command — never `&&`-chain, per MOJO_CONVENTIONS §6); sqlite3 is
# NOT on PATH, the working CLI is the nsight binary:
#   /opt/nvidia/nsight-systems/2023.4.4/target-linux-x64/sqlite3 \
#     /tmp/wave1_board/board.db "SELECT tag,step,value FROM scalars ORDER BY tag,step;"

from serenitymojo.training.serenityboard import SerenityBoardWriter


def main() raises:
    var db_dir = String("/tmp/wave1_board")  # writer appends /board.db
    var board = SerenityBoardWriter.open(db_dir, String("wave1_sess"), 0)

    # 3 steps with known values (sps = 1/step_secs is derived by log_train_step).
    # step, loss, grad_norm, lr, step_secs, noise_speed
    board.log_train_step(1, Float32(0.75), Float64(1.5), Float32(0.0002), Float64(0.5), Float64(1000.0))
    board.log_train_step(2, Float32(0.50), Float64(1.2), Float32(0.0002), Float64(0.25), Float64(2000.0))
    board.log_train_step(3, Float32(0.40), Float64(0.9), Float32(0.0002), Float64(0.20), Float64(2500.0))

    board.close()

    print("WROTE /tmp/wave1_board/board.db")
    print("expect scalars:")
    print("  loss/train            step1=0.75 step2=0.5 step3=0.4")
    print("  grad_norm             step1=1.5  step2=1.2 step3=0.9")
    print("  lr/default            step1=0.0002 (all steps)")
    print("  perf/steps_per_sec    step1=2.0 step2=4.0 step3=5.0")
    print("board_roundtrip_smoke WRITE PASS")
