# serenitymojo.serve.process_isolated_backend — Phase-5 parent dispatcher.
#
# A GenBackend that runs each resident model in a CHILD PROCESS (serve/worker.mojo)
# and talks to it over an AF_UNIX socket. A model switch = kill the child (the OS
# reclaims ALL its VRAM on exit — the one reclaim that actually works on this
# runtime, MEASURED in F3: 21416 -> 788 MiB) + fork+execv a fresh child. Same-model
# consecutive jobs REUSE the resident child (no reload). The parent itself never
# initializes CUDA, so its fork() is clean. See PHASE5_PROCESS_ISOLATION_DESIGN.md.

from json.parser import loads

from serenitymojo.serve.backend import GenBackend, JobParams, StepResult
from serenitymojo.serve.dispatch_backend import _kind_for_model, _kind_name
from serenitymojo.serve.proc_ipc import (
    make_socketpair, proc_kill_wait, set_nonblock, write_msg, build_argv, cstr,
    LineReader, sys_execv, sys__exit, SIGKILL, SELF_EXE,
)
from serenitymojo.serve.ipc_codec import encode_start, encode_cancel, decode_ev
from net.syscalls import sys_fork, sys_close, errno_str


struct ProcessIsolatedBackend(GenBackend, Movable):
    var child_pid: Int32        # -1 = no child
    var child_kind: String      # "" | "stub" | "zimage" | "qwenimage" | "ideogram4" | "klein"
    var parent_fd: Int32        # -1 = none; our end of the socketpair (O_NONBLOCK)
    var reader: LineReader
    var active: Bool            # a job is in flight in the child
    var resident: String        # model name the child currently holds ("" = none)
    var cur_step: Int
    var cur_total: Int

    def __init__(out self):
        self.child_pid = -1
        self.child_kind = String("")
        self.parent_fd = -1
        self.reader = LineReader(-1)
        self.active = False
        self.resident = String("")
        self.cur_step = 0
        self.cur_total = 0

    def backend_name(self) -> String:
        return String("isolated")

    def model_name(self) -> String:
        return self.resident if self.resident != "" else String("-")

    def resident_model(self) -> String:
        return self.resident

    # ── child lifecycle ──────────────────────────────────────────────────────
    def _kill_child(mut self):
        # SIGKILL, not SIGTERM: the worker was fork()'d from this daemon AFTER
        # run_daemon installed its signalfd, which BLOCKS SIGTERM/SIGINT in the
        # process mask; fork inherits the mask and execv preserves it, so a
        # SIGTERM to the worker stays pending forever and waitpid() would hang
        # (observed). SIGKILL can't be blocked — and a hard exit is exactly what
        # process isolation wants: the OS reclaims the child's VRAM regardless of
        # graceful cleanup (the F3-measured reclaim path).
        if self.child_pid > 0:
            proc_kill_wait(self.child_pid, SIGKILL)  # blocking reap -> VRAM freed
        if self.parent_fd >= 0:
            _ = sys_close(self.parent_fd)
        self.child_pid = -1
        self.child_kind = String("")
        self.parent_fd = -1
        self.resident = String("")

    def _spawn_child(mut self, kind: String) raises:
        var pair = make_socketpair()
        var parent_end = pair.parent_end
        var child_end = pair.child_end
        # build argv + path BEFORE fork (no allocation between fork and execv)
        var args = List[String]()
        args.append(String("serenity_daemon"))
        args.append(String("worker"))
        args.append(kind)
        args.append(String(Int(child_end)))
        var argv = build_argv(args)
        var path = cstr(SELF_EXE)
        var pid = sys_fork()
        if pid == 0:
            # CHILD: async-signal-safe calls only, then execv into a fresh image
            _ = sys_close(parent_end)
            _ = sys_execv(path, argv)
            sys__exit(127)               # execv failed
        if pid < 0:
            _ = sys_close(parent_end)
            _ = sys_close(child_end)
            raise Error("fork failed: " + errno_str())
        # PARENT
        _ = sys_close(child_end)
        set_nonblock(parent_end)
        self.child_pid = pid
        self.child_kind = kind
        self.parent_fd = parent_end
        self.reader = LineReader(parent_end)
        print("[isolated] spawned", kind, "worker pid", pid)

    def _isolated_kind(self, model: String) raises -> String:
        """Map a model to the worker kind. Reuses the dispatch rule for real
        models; adds CPU-only stub kinds ("stub"/"stub2" -> distinct stub
        workers) so the spawn / IPC / kill+respawn-switch machinery is verifiable
        end-to-end WITHOUT a GPU (the two distinct kinds force a real switch)."""
        if model == "stub":
            return String("stub")
        if model == "stub2":
            return String("stub2")
        return _kind_name(_kind_for_model(model))

    # ── GenBackend ───────────────────────────────────────────────────────────
    def start(mut self, params: JobParams) raises:
        var want = self._isolated_kind(params.model)  # raises on unknown
        if self.child_kind != want:
            if self.child_pid > 0:
                print("[isolated] switch", self.child_kind, "->", want,
                      ": killing child (OS reclaims its VRAM)")
            self._kill_child()
            self._spawn_child(want)
        write_msg(self.parent_fd, encode_start(params))
        self.active = True
        self.resident = params.model
        self.cur_step = 0
        self.cur_total = params.steps

    def step(mut self) raises -> StepResult:
        var still_open = True
        var line = self.reader.next_line(still_open)
        if not still_open:
            # child process died (crash / OOM-kill) — surface as a failed job;
            # next start() respawns lazily (daemon-survives at process granularity).
            self.active = False
            self._kill_child()
            var f = StepResult()
            f.failed = True
            f.error = String("worker process exited unexpectedly")
            return f^
        if line == "":
            var idle = StepResult()       # no message this tick — bounded, responsive
            idle.step = self.cur_step
            idle.total = self.cur_total
            return idle^
        var obj = loads(line)
        if obj.contains("ev") and obj["ev"].as_string() == "ready":
            var rdy = StepResult()        # child announced readiness — no-op tick
            rdy.step = self.cur_step
            rdy.total = self.cur_total
            return rdy^
        var r = decode_ev(obj)
        if r.total > 0:
            self.cur_total = r.total
        if r.step > 0:
            self.cur_step = r.step
        else:
            r.step = self.cur_step
        if r.total == 0:
            r.total = self.cur_total
        if r.is_terminal():
            self.active = False           # job done; child stays resident for reuse
        return r^

    def cancel(mut self):
        if self.active and self.parent_fd >= 0:
            try:
                write_msg(self.parent_fd, encode_cancel())
            except:
                pass  # if the pipe is gone the job is already ending

    def between_jobs_trim(mut self) raises:
        # No-op: the child holds the VRAM and reclaim happens on _kill_child at a
        # model SWITCH (real OS reclaim), not via an in-process pool trim (which
        # MEASURED 0 MiB on this runtime). Same-model jobs intentionally keep the
        # resident child to avoid reloading weights.
        pass
