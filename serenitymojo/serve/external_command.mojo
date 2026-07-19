# serenitymojo.serve.external_command -- nonblocking fork/exec/waitpid runner.
#
# Used by daemon backends for staged sidecar binaries. Commands are launched
# through /bin/sh -c so existing env-prefix command strings keep their behavior,
# while the daemon can keep polling IPC and cancellation state.

from std.memory import alloc
from std.builtin.type_aliases import MutExternalOrigin
from std.memory import UnsafePointer

from net.syscalls import BytePtr, sys_fork, errno_str
from serenitymojo.serve.proc_ipc import (
    build_argv, cstr, proc_kill_wait, sys_execv, sys_waitpid, sys__exit,
    SIGKILL, WNOHANG,
)


comptime SHELL_PATH = "/bin/sh"


struct ExternalCommand(Movable):
    var pid: Int32
    var label: String
    var command: String
    var last_status: Int32

    def __init__(out self):
        self.pid = -1
        self.label = String("")
        self.command = String("")
        self.last_status = 0

    def __del__(deinit self):
        if self.pid > 0:
            proc_kill_wait(self.pid, SIGKILL)

    def running(self) -> Bool:
        return self.pid > 0

    def start(mut self, label: String, command: String) raises:
        if self.running():
            raise Error(
                String("external command already running label=") + self.label
            )
        print("[external-command]", label, "cmd:", command)
        var args = List[String]()
        args.append(String("sh"))
        args.append(String("-c"))
        args.append(command.copy())
        var argv = build_argv(args)
        var path = cstr(String(SHELL_PATH))
        var pid = sys_fork()
        if pid == 0:
            _ = sys_execv(path, argv)
            sys__exit(127)
        if pid < 0:
            raise Error("fork failed: " + errno_str())
        self.pid = pid
        self.label = label.copy()
        self.command = command.copy()
        self.last_status = 0

    def poll(mut self) raises -> Bool:
        if self.pid <= 0:
            return True
        var st = alloc[Int32](1)
        st[0] = 0
        var rc = sys_waitpid(
            self.pid,
            rebind[UnsafePointer[Int32, MutExternalOrigin]](st),
            WNOHANG,
        )
        var raw = st[0]
        st.free()
        if rc == 0:
            return False
        if rc < 0:
            var old_pid = self.pid
            self.pid = -1
            raise Error(
                String("waitpid failed for external command pid=")
                + String(old_pid)
                + String(" label=")
                + self.label
                + String(": ")
                + errno_str()
            )
        self.pid = -1
        self.last_status = raw
        return True

    def require_success(self) raises:
        if self.last_status != 0:
            raise Error(
                String("external command failed status=")
                + String(self.last_status)
                + String(" label=")
                + self.label
                + String(" command=")
                + self.command
            )

    def kill(mut self):
        if self.pid > 0:
            proc_kill_wait(self.pid, SIGKILL)
        self.pid = -1
