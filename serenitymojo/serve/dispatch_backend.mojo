# serenitymojo.serve.dispatch_backend — multi-model residency + on-demand switch.
#
# THE PROBLEM run_daemon[B: GenBackend] is monomorphic over ONE backend type,
# but Phase-4 needs the daemon to hold DIFFERENT backend structs (Z-Image,
# Qwen-Image) one-at-a-time and SWITCH between them on demand. DispatchBackend
# is the single GenBackend the daemon is parametrized over; internally it holds
# AT MOST ONE concrete backend resident (List[ZImageBackend] | List[Qwen...] —
# 0/1 each) and routes start()/step()/cancel() to whichever is active.
#
# SWITCH PROTOCOL (G-PERF2). The daemon calls start(params) only when idle
# (no job in flight — single worker, serial). start() reads params.model,
# maps it to a backend KIND, and if that differs from the resident kind:
#   1. FREE the current backend (drop its List entry → all its DeviceBuffers
#      are released; ctx.synchronize() forces the frees to complete).
#   2. CONSTRUCT the new backend (its weights load lazily on the first step()).
# Because the switch only happens at a job boundary (start, never mid-job), no
# job ever sees two models resident. The first job after a switch reloads the
# new model's weights (the resident win returns from the SECOND job on the new
# model). resident_model() / model_name() / backend_name() always reflect the
# active backend, so /v1/health + /v1/models track the live residency.
#
# MODEL → KIND mapping (by the /v1/models scan `name`):
#   "" | "zimage_base" | <anything starting "zimage">  -> Z-Image (default)
#   "qwen-image-2512" | <anything containing "qwen">    -> Qwen-Image
#   "ideogram-4-fp8" | <anything containing "ideogram"> -> Ideogram-4
#   <anything containing "klein">                       -> Klein backend contract
#                                                        adapter; fails loud until
#                                                        cap-cache execution bridge
#   <flux2-dev/flux-2-dev>                              -> explicit unsupported
#                                                        (not a Klein runner)
# An unknown model name is REJECTED at start() (fail-loud — no silent fallback
# to a model the user didn't ask for).

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current, cu_mem_get_info
from serenitymojo.serve.backend import GenBackend, JobParams, StepResult
from serenitymojo.serve.zimage_backend import ZImageBackend
from serenitymojo.serve.qwenimage_backend import QwenImageBackend
from serenitymojo.serve.ideogram4_backend import Ideogram4Backend
from serenitymojo.serve.klein_backend import KleinBackend

comptime KIND_NONE = 0
comptime KIND_ZIMAGE = 1
comptime KIND_QWEN = 2
comptime KIND_IDEOGRAM4 = 3
comptime KIND_KLEIN = 4


def _kind_for_model(model: String) raises -> Int:
    """Map a /v1/generate `model` string to a backend kind. Empty -> Z-Image
    (the default first backend). Unknown -> raise (fail-loud)."""
    if model == String("") or model == String("zimage_base"):
        return KIND_ZIMAGE
    if model == String("qwen-image-2512"):
        return KIND_QWEN
    if model == String("ideogram-4-fp8"):
        return KIND_IDEOGRAM4
    # tolerant matching for the scanner's many zimage/qwen checkpoint names
    # (z_image_base_bf16, z_image_turbo_bf16, qwen_image_fp8_e4m3fn, …) and
    # hand-typed names / arch tags. NOTE: the actual resident weights served are
    # always the canonical zimage_base / qwen-image-2512 dirs (the per-variant
    # checkpoints aren't wired as distinct backends yet) — a variant name maps
    # to its family's resident backend so the UI's disk-scan model list works.
    var lo = model.lower()
    if lo.find("ideogram") >= 0:
        return KIND_IDEOGRAM4
    if lo.find("flux2-dev") >= 0 or lo.find("flux-2-dev") >= 0 or lo.find("flux2_dev") >= 0:
        raise Error(
            String("flux2-dev model '") + model
            + "' cannot run through the Klein daemon backend; Flux2-dev uses a "
            + "different transformer/text contract and has no GenBackend product path yet"
        )
    if lo.find("klein") >= 0:
        return KIND_KLEIN
    if lo.find("qwen") >= 0:
        return KIND_QWEN
    if lo.find("zimage") >= 0 or lo.find("z_image") >= 0 or lo.find("z-image") >= 0:
        return KIND_ZIMAGE
    raise Error(
        String("unknown model '") + model + "' — switchable resident models are"
        + " Z-Image (name contains 'zimage'/'z_image'; default), Qwen-Image"
        + " (name contains 'qwen'), and Ideogram-4 (name contains 'ideogram');"
        + " served weights are zimage_base, qwen-image-2512, ideogram-4-fp8,"
        + " and Klein admission-check routing; Flux2-dev remains explicitly unsupported"
    )


def _kind_name(kind: Int) -> String:
    if kind == KIND_ZIMAGE:
        return String("zimage")
    if kind == KIND_QWEN:
        return String("qwenimage")
    if kind == KIND_IDEOGRAM4:
        return String("ideogram4")
    if kind == KIND_KLEIN:
        return String("klein")
    return String("none")


struct DispatchBackend(GenBackend, Movable):
    var ctx: DeviceContext
    var kind: Int                            # KIND_* of resident backend (NONE = idle)
    # 0/1 each. Backends are Movable-not-Copyable; ArcPointer is Copyable, so
    # these can live in Lists.
    var z: List[ArcPointer[ZImageBackend]]
    var q: List[ArcPointer[QwenImageBackend]]
    var i4: List[ArcPointer[Ideogram4Backend]]
    var k: List[ArcPointer[KleinBackend]]

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.kind = KIND_NONE
        self.z = List[ArcPointer[ZImageBackend]]()
        self.q = List[ArcPointer[QwenImageBackend]]()
        self.i4 = List[ArcPointer[Ideogram4Backend]]()
        self.k = List[ArcPointer[KleinBackend]]()

    def backend_name(self) -> String:
        if self.kind == KIND_ZIMAGE:
            return self.z[0][].backend_name()
        if self.kind == KIND_QWEN:
            return self.q[0][].backend_name()
        if self.kind == KIND_IDEOGRAM4:
            return self.i4[0][].backend_name()
        if self.kind == KIND_KLEIN:
            return self.k[0][].backend_name()
        return String("dispatch")  # idle, no backend constructed yet

    def model_name(self) -> String:
        if self.kind == KIND_ZIMAGE:
            return self.z[0][].model_name()
        if self.kind == KIND_QWEN:
            return self.q[0][].model_name()
        if self.kind == KIND_IDEOGRAM4:
            return self.i4[0][].model_name()
        if self.kind == KIND_KLEIN:
            return self.k[0][].model_name()
        return String("-")

    def resident_model(self) -> String:
        if self.kind == KIND_ZIMAGE:
            return self.z[0][].resident_model()
        if self.kind == KIND_QWEN:
            return self.q[0][].resident_model()
        if self.kind == KIND_IDEOGRAM4:
            return self.i4[0][].resident_model()
        if self.kind == KIND_KLEIN:
            return self.k[0][].resident_model()
        return String("")

    # ── free the resident backend (drop its DeviceBuffers, force the frees) ──
    def _free_current(mut self) raises:
        if self.kind == KIND_NONE:
            return
        print("[dispatch] freeing resident backend:", _kind_name(self.kind),
              "(", self.resident_model(), ")")
        var before = cu_mem_get_info()
        # Dropping the single-element List runs the backend's destructor, which
        # frees every DeviceBuffer it owns (resident weights + any per-job
        # tensors) back to the Mojo-runtime caching pool. synchronize() ensures the
        # device-side frees complete before we trim.
        self.z = List[ArcPointer[ZImageBackend]]()
        self.q = List[ArcPointer[QwenImageBackend]]()
        self.i4 = List[ArcPointer[Ideogram4Backend]]()
        self.k = List[ArcPointer[KleinBackend]]()
        self.kind = KIND_NONE
        self.ctx.synchronize()
        # F3 (authorized internal change, MEASURED no-op in THIS Mojo-runtime build):
        # the Mojo GPU runtime DeviceContext is a SINGLETON whose caching allocator frees
        # buffers back to its OWN pool and never returns them to the OS while
        # any DeviceContext value lives (refcount never hits 0). We call
        # cuMemPoolTrimTo on the CUDA *default* stream-ordered pool here, but it
        # RECLAIMS 0 MiB because the runtime does NOT allocate from that default pool —
        # so the bytes stay pinned and switching zimage(~21GB peak) -> qwen
        # OOMs in qwen's 1024² forward. The call is left in (harmless, correct
        # if a future Mojo-runtime build routes through the default pool). The real fix
        # is process isolation per resident model — see SERENITYUI_TODO.md
        # "Phase 4 VRAM work" F3 for the full measured finding + what's needed.
        try:
            cu_mempool_trim_current(0)
        except e:
            print("[dispatch] WARNING: pool trim failed (continuing):", e)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        print("[dispatch] VRAM after free+trim: used",
              before.used_bytes() // (1024 * 1024), "->",
              after.used_bytes() // (1024 * 1024), "MiB (driver free view)")

    def _ensure_kind(mut self, kind: Int) raises:
        """Make `kind` the resident backend, switching (free old + build new)
        if it isn't already. Called only at a job boundary (start)."""
        if self.kind == kind:
            return
        if self.kind != KIND_NONE:
            self._free_current()
        if kind == KIND_ZIMAGE:
            print("[dispatch] constructing Z-Image backend")
            self.z = List[ArcPointer[ZImageBackend]]()
            self.z.append(ArcPointer(ZImageBackend()))
        elif kind == KIND_QWEN:
            print("[dispatch] constructing Qwen-Image backend")
            self.q = List[ArcPointer[QwenImageBackend]]()
            self.q.append(ArcPointer(QwenImageBackend()))
        elif kind == KIND_IDEOGRAM4:
            print("[dispatch] constructing Ideogram-4 backend")
            self.i4 = List[ArcPointer[Ideogram4Backend]]()
            self.i4.append(ArcPointer(Ideogram4Backend()))
        elif kind == KIND_KLEIN:
            print("[dispatch] constructing Flux2/Klein backend contract adapter")
            self.k = List[ArcPointer[KleinBackend]]()
            self.k.append(ArcPointer(KleinBackend()))
        else:
            raise Error("dispatch: invalid backend kind")
        self.kind = kind

    # ── job admission (the switch point) ──────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        var want = _kind_for_model(params.model)
        self._ensure_kind(want)
        if self.kind == KIND_ZIMAGE:
            self.z[0][].start(params)
        elif self.kind == KIND_QWEN:
            self.q[0][].start(params)
        elif self.kind == KIND_IDEOGRAM4:
            self.i4[0][].start(params)
        else:
            self.k[0][].start(params)

    def step(mut self) raises -> StepResult:
        if self.kind == KIND_ZIMAGE:
            return self.z[0][].step()
        if self.kind == KIND_QWEN:
            return self.q[0][].step()
        if self.kind == KIND_IDEOGRAM4:
            return self.i4[0][].step()
        if self.kind == KIND_KLEIN:
            return self.k[0][].step()
        var r = StepResult()
        r.failed = True
        r.error = String("dispatch: no active backend")
        return r^

    def cancel(mut self):
        if self.kind == KIND_ZIMAGE:
            self.z[0][].cancel()
        elif self.kind == KIND_QWEN:
            self.q[0][].cancel()
        elif self.kind == KIND_IDEOGRAM4:
            self.i4[0][].cancel()
        elif self.kind == KIND_KLEIN:
            self.k[0][].cancel()

    # ── F3 between-jobs pool trim (no switch) ──────────────────────────────────
    def between_jobs_trim(mut self) raises:
        """Reclaim the just-finished job's transient peak (per-job text encoder
        ~7.5-16 GB, decode activations) back to the OS via cuMemPoolTrimTo. The
        RESIDENT weights of the still-loaded backend have live suballocations and
        are NOT reclaimed — only the freed-to-pool transient chunks are. Called
        by the daemon at every job boundary so idle VRAM tracks the resident
        footprint, not the high-water mark (F3)."""
        var before = cu_mem_get_info()
        self.ctx.synchronize()
        cu_mempool_trim_current(0)
        self.ctx.synchronize()
        var after = cu_mem_get_info()
        var delta_mib = (before.used_bytes() - after.used_bytes()) // (1024 * 1024)
        print("[dispatch] between-jobs trim: used",
              before.used_bytes() // (1024 * 1024), "->",
              after.used_bytes() // (1024 * 1024), "MiB (reclaimed",
              delta_mib, "MiB)")
