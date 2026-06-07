# zimage_sample_supervisor.mojo — process-separated Z-Image sampler runner.
#
# The Z-Image train loop queues `serenity.zimage.sample_request.v1` manifests
# at sample cadence. This supervisor runs those requests after the trainer has
# exited or released memory, so Qwen3 + DiT + VAE sampling does not co-reside
# with the training stack.

from std.sys import argv

from serenitymojo.io.ffi import sys_system, sys_open, sys_close, O_RDONLY


comptime DEFAULT_REQUEST = "/home/alex/mojodiffusion/output/alina_zimage/sample_requests/step500_request.json"
comptime DEFAULT_SAMPLER_BIN = "/tmp/zimage_generate_prod"


def _path_exists(path: String) -> Bool:
    if path == String(""):
        return False
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _usage() -> String:
    return (
        String("usage: zimage_sample_supervisor [request_json] [sampler_bin] [mode] [sample_id]\n")
        + String("  mode: run | dryrun\n")
        + String("  default request: ") + String(DEFAULT_REQUEST) + String("\n")
        + String("  default sampler: ") + String(DEFAULT_SAMPLER_BIN)
    )


def _run_command(label: String, cmd: String) raises:
    print("[ZImage-supervisor]", label, "cmd:", cmd)
    var status = sys_system(cmd)
    if status != 0:
        raise Error(
            String("zimage_sample_supervisor: command failed status=")
            + String(status) + String(" label=") + label
        )


def main() raises:
    var a = argv()
    var request_path = String(DEFAULT_REQUEST)
    if len(a) >= 2:
        request_path = String(a[1])
    if request_path == String("--help") or request_path == String("-h"):
        print(_usage())
        return

    var sampler_bin = String(DEFAULT_SAMPLER_BIN)
    if len(a) >= 3:
        sampler_bin = String(a[2])
    var mode = String("run")
    if len(a) >= 4:
        mode = String(a[3])
    var sample_id = String("")
    if len(a) >= 5:
        sample_id = String(a[4])

    if not _path_exists(request_path):
        raise Error(String("zimage_sample_supervisor: missing request manifest: ") + request_path)
    if mode != String("dryrun") and not _path_exists(sampler_bin):
        raise Error(String("zimage_sample_supervisor: missing sampler binary: ") + sampler_bin)

    var cmd = (
        String("MODULAR_DEVICE_CONTEXT_SYNC_MODE=true ")
        + sampler_bin + String(" --request ") + request_path
    )
    if sample_id != String(""):
        cmd += String(" ") + sample_id

    print("=== Z-Image sample supervisor ===")
    print("  request:", request_path)
    print("  sampler:", sampler_bin)
    print("  mode:", mode)
    print("  sample_id:", sample_id)
    print("  sampler contract: serenity.zimage.sample_request.v1 -> serenity.zimage.sample_result.v1")
    print("  note: process separation only; not sampler parity or speed parity")

    if mode == String("dryrun"):
        print("[ZImage-supervisor] dryrun command:", cmd)
        print("DONE Z-Image supervisor dryrun")
        return
    if mode != String("run"):
        raise Error(String("zimage_sample_supervisor: unsupported mode: ") + mode)

    _run_command(String("sample request"), cmd)
    print("DONE Z-Image supervisor request:", request_path)
