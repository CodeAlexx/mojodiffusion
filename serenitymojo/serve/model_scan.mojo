# serenitymojo.serve.model_scan — model/LoRA disk scanner for GET /v1/models.
#
# Scans (HEADER READS ONLY — never loads weights):
#   * /home/alex/.serenity/models/checkpoints/*.safetensors  -> models
#   * /home/alex/.serenity/models/loras/*.safetensors        -> loras
#   * known model DIRECTORIES (zimage_base/, anima/)          -> models
#
# Arch tags come from cheap substring probes on the safetensors JSON header
# (the first 8 bytes give the header length; we read the header text and look
# for distinctive tensor-name markers — verified against the real files on
# this box, 2026-06-10):
#   noise_refiner.                       -> zimage      (Z-Image DiT)
#   double_stream_modulation_img         -> flux-2/klein (shared-mod tables)
#   distilled_guidance_layer             -> chroma
#   double_blocks.                       -> flux        (flux1-dev family)
#   audio_vae.                           -> ltx2        (audio-video bundle)
#   model.diffusion_model.joint_blocks   -> sd3         (MMDiT full ckpt)
#   model.diffusion_model.input_blocks   -> sdxl        (UNet full ckpt)
#   input_blocks.0.                      -> sdxl        (bare UNet export)
#   txt_norm.                            -> qwen-image
#   time_projection.                     -> wan
#   else                                 -> unknown
#
# Directory listing goes through `find` into a temp file (no readdir FFI in
# the stack); the header probe uses pread on the first min(header_len, 16 MiB)
# bytes only.

from std.ffi import external_call
from std.memory import alloc

from http.request import byte_substr

from serenitymojo.io.ffi import BytePtr, sys_open, sys_close, sys_pread, file_size, O_RDONLY

comptime CHECKPOINTS_DIR = "/home/alex/.serenity/models/checkpoints"
comptime LORAS_DIR = "/home/alex/.serenity/models/loras"
comptime SCAN_TMP = "/tmp/serenity_model_scan.txt"
comptime HEADER_PROBE_CAP = 16 * 1024 * 1024  # bound the header read


struct ScanEntry(Copyable, Movable):
    var name: String   # file stem (".safetensors" dropped) or directory name
    var path: String
    var arch: String   # "" for loras (not probed)
    var size: Int

    def __init__(out self, name: String, path: String, arch: String, size: Int):
        self.name = name
        self.path = path
        self.arch = arch
        self.size = size


def _shell(cmd: String) -> Int:
    var n = cmd.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = cmd.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    var status = Int(external_call["system", Int32](BytePtr(unsafe_from_address=Int(buf))))
    buf.free()
    return status


def _read_text_file(path: String) raises -> String:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("model_scan: cannot open ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        return String("")
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var k = sys_pread(fd, buf + done, n - done, done)
        if k <= 0:
            buf.free()
            _ = sys_close(fd)
            raise Error(String("model_scan: read failed on ") + path)
        done += k
    _ = sys_close(fd)
    var s = String(StringSlice(ptr=buf, length=n))
    buf.free()
    return s^


def _list_safetensors(dir: String) raises -> List[ScanEntry]:
    """`find <dir> -maxdepth 1 -name '*.safetensors' -printf '%f\\t%s\\n'` into
    a temp file, parsed into (name, path, size) entries (arch left "")."""
    var entries = List[ScanEntry]()
    var cmd = (  # -L: follow symlinks (HF-cache links live in checkpoints/)
        String("find -L '") + dir + "' -maxdepth 1 -type f -name '*.safetensors'"
        + " -printf '%f\\t%s\\n' 2>/dev/null | sort > " + SCAN_TMP
    )
    if _shell(cmd) != 0:
        return entries^
    var text = _read_text_file(String(SCAN_TMP))
    for line in text.split("\n"):
        var l = String(line)
        var tab = l.find("\t")
        if tab <= 0:
            continue
        var fname = byte_substr(l, 0, tab)
        var size = Int(byte_substr(l, tab + 1, l.byte_length()))
        var stem = fname.copy()
        if stem.endswith(".safetensors"):
            stem = byte_substr(stem, 0, stem.byte_length() - 12)
        entries.append(ScanEntry(stem^, dir + "/" + fname, String(""), size))
    return entries^


def _header_text(path: String) raises -> String:
    """First min(header_len, 16 MiB) bytes of the safetensors JSON header."""
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("model_scan: cannot open ") + path)
    var lenbuf = alloc[UInt8](8)
    var k = sys_pread(fd, lenbuf, 8, 0)
    if k != 8:
        lenbuf.free()
        _ = sys_close(fd)
        raise Error(String("model_scan: short header-length read on ") + path)
    var header_len = 0
    for i in range(8):
        header_len = header_len | (Int(lenbuf[i]) << (8 * i))
    lenbuf.free()
    if header_len <= 0:
        _ = sys_close(fd)
        return String("")
    var want = header_len if header_len < HEADER_PROBE_CAP else HEADER_PROBE_CAP
    var buf = alloc[UInt8](want)
    var done = 0
    while done < want:
        var r = sys_pread(fd, buf + done, want - done, 8 + done)
        if r <= 0:
            break
        done += r
    _ = sys_close(fd)
    var s = String(StringSlice(ptr=buf, length=done))
    buf.free()
    return s^


def detect_arch(header: String) -> String:
    """Substring probes, most-specific first (see file comment for evidence)."""
    if header.find('"noise_refiner.') >= 0:
        return String("zimage")
    if header.find('"double_stream_modulation_img') >= 0:
        return String("flux-2/klein")
    if header.find('"distilled_guidance_layer') >= 0:
        return String("chroma")
    if header.find('"double_blocks.') >= 0:
        return String("flux")
    if header.find('"audio_vae.') >= 0:
        return String("ltx2")
    if header.find('"model.diffusion_model.joint_blocks') >= 0:
        return String("sd3")
    if header.find('"model.diffusion_model.input_blocks') >= 0:
        return String("sdxl")
    if header.find('"input_blocks.0.') >= 0:
        return String("sdxl")
    if header.find('"txt_norm.') >= 0:
        return String("qwen-image")
    if header.find('"time_projection.') >= 0:
        return String("wan")
    return String("unknown")


def _dir_size(dir: String) -> Int:
    if _shell(String("du -sb '") + dir + "' 2>/dev/null | cut -f1 > " + SCAN_TMP) != 0:
        return 0
    try:
        var text = _read_text_file(String(SCAN_TMP))
        return Int(String(text.strip()))
    except:
        return 0


def _dir_exists(dir: String) -> Bool:
    return _shell(String("test -d '") + dir + "'") == 0


def scan_checkpoints() raises -> List[ScanEntry]:
    """checkpoints/*.safetensors with header-probed arch tags + known dirs."""
    var out = List[ScanEntry]()
    var files = _list_safetensors(String(CHECKPOINTS_DIR))
    for i in range(len(files)):
        var arch = String("unknown")
        try:
            arch = detect_arch(_header_text(files[i].path))
        except:
            pass  # unreadable/corrupt header -> unknown
        out.append(ScanEntry(files[i].name.copy(), files[i].path.copy(), arch^, files[i].size))
    # known model directories (diffusers-style trees; arch by identity)
    var known_names: List[String] = ["zimage_base", "anima"]
    var known_archs: List[String] = ["zimage", "anima"]
    for i in range(len(known_names)):
        var dir = String("/home/alex/.serenity/models/") + known_names[i]
        if _dir_exists(dir):
            out.append(ScanEntry(
                known_names[i].copy(), dir.copy(), known_archs[i].copy(), _dir_size(dir)
            ))
    # known diffusers-tree checkpoints UNDER checkpoints/ (multi-shard
    # transformer/vae subdirs — not flat .safetensors files, so the file scan
    # above misses them). These are the resident-backend targets the daemon can
    # switch to (model_scan.name == the backend.resident_model() string).
    var ckpt_names: List[String] = ["qwen-image-2512"]
    var ckpt_archs: List[String] = ["qwen-image"]
    for i in range(len(ckpt_names)):
        var dir = String(CHECKPOINTS_DIR) + "/" + ckpt_names[i]
        if _dir_exists(dir):
            out.append(ScanEntry(
                ckpt_names[i].copy(), dir.copy(), ckpt_archs[i].copy(), _dir_size(dir)
            ))
    return out^


def scan_loras() raises -> List[ScanEntry]:
    """loras/*.safetensors — name/path/size only (no header probe needed)."""
    return _list_safetensors(String(LORAS_DIR))
