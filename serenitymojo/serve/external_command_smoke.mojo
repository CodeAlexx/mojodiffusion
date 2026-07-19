# No-GPU smoke for serenitymojo.serve.external_command.
#
# Run:
#   pixi run mojo run -I . -I /home/alex/MOJO-libs \
#     serenitymojo/serve/external_command_smoke.mojo

from serenitymojo.serve.external_command import ExternalCommand


def _wait_done(mut cmd: ExternalCommand, label: String) raises:
    var spins = 0
    while spins < 1000000:
        if cmd.poll():
            return
        spins += 1
    cmd.kill()
    raise Error(String("external_command_smoke timeout: ") + label)


def main() raises:
    var ok = ExternalCommand()
    ok.start(String("true"), String("true"))
    _wait_done(ok, String("true"))
    ok.require_success()

    var fail = ExternalCommand()
    fail.start(String("exit-seven"), String("exit 7"))
    _wait_done(fail, String("exit-seven"))
    if fail.last_status == 0:
        raise Error("external_command_smoke expected nonzero raw status")

    print("external_command_smoke PASS")
