#!/usr/bin/env bash
set -euo pipefail

root="/home/alex/serenity-trainer"
terminal_script="$root/target/serenity_terminal_run.sh"

if [ "$#" -lt 4 ]; then
    echo "usage: $0 backend_label runner pidfile logfile [runner args...]" >&2
    exit 2
fi

backend_label="$1"
runner="$2"
pidfile="$3"
logfile="$4"
shift 4

# Terminal-wrapper pid lives NEXT TO the runner pidfile. The UI stop path
# TERMs the wrapper (its trap tears down the runner and exits, which closes
# the gnome-terminal window); the runner pidfile alone cannot close the
# terminal — the runner shares the wrapper's process group, so a group-kill
# on the runner pid fails and the wrapper used to linger at a read prompt
# forever (the "trainer closed but terminal stays open" bug).
termpidfile="$pidfile.term"

mkdir -p "$root/target" "$(dirname "$logfile")"
rm -f "$pidfile" "$termpidfile" "$logfile"

cat > "$terminal_script" <<'EOS'
#!/usr/bin/env bash
set -u

root="$1"
pidfile="$2"
logfile="$3"
backend_label="$4"
runner="$5"
shift 5

termpidfile="$pidfile.term"

cd "$root"
rm -f "$pidfile"
echo "$$" > "$termpidfile"

echo "Serenity ${backend_label} trainer"
echo "Working dir: $root"
echo "Runner: $runner"
echo "Log: $logfile"
echo

"$runner" "$@" > >(tee "$logfile") 2>&1 &
child="$!"
echo "$child" > "$pidfile"

stop_child() {
    # UI Stop / window close / logout: kill the runner, clean up, and exit
    # immediately so the terminal window closes (no "Press Enter" hold).
    if [ -n "${child:-}" ]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    rm -f "$pidfile" "$termpidfile"
    exit 143
}

trap stop_child TERM INT HUP

wait "$child"
status="$?"
rm -f "$pidfile"

echo
echo "Serenity ${backend_label} trainer exited with status $status"
if [ "$status" -ne 0 ]; then
    # Keep failures visible for 30s, then close anyway (full output is in
    # $logfile either way). Successful/stopped runs close immediately.
    read -r -t 30 -p "Trainer failed (status $status); closing in 30s, Enter to close now..." || true
fi
rm -f "$termpidfile"
exit "$status"
EOS
chmod +x "$terminal_script"

env -i \
    HOME=/home/alex \
    USER=alex \
    LOGNAME=alex \
    SHELL=/bin/bash \
    LANG="${LANG:-en_US.UTF-8}" \
    DISPLAY="${DISPLAY:-:1}" \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/1000}" \
    DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/1000/bus}" \
    XDG_CURRENT_DESKTOP="${XDG_CURRENT_DESKTOP:-ubuntu:GNOME}" \
    DESKTOP_SESSION="${DESKTOP_SESSION:-ubuntu}" \
    GDMSESSION="${GDMSESSION:-ubuntu}" \
    PATH="/home/alex/.local/bin:/usr/local/cuda-12.6/bin:/usr/local/cuda/bin:/usr/local/cuda-12.4/bin:/usr/local/cuda-11.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin" \
    LD_LIBRARY_PATH="/home/alex/.local/lib:/usr/local/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:/usr/local/cuda-11.8/lib64" \
    /usr/bin/gnome-terminal \
        --title="Serenity ${backend_label} Trainer" \
        -- bash "$terminal_script" \
            "$root" \
            "$pidfile" \
            "$logfile" \
            "$backend_label" \
            "$runner" \
            "$@"

# gnome-terminal hands the spawn to gnome-terminal-server over D-Bus and
# returns immediately — rc 0 does NOT mean the wrapper started. Wait for the
# runner pidfile so a silent spawn failure surfaces as a nonzero rc in the UI
# ("training started but no terminal ever opened" bug). 20s budget covers a
# loaded desktop; idle spawn is < 1s.
for _ in $(seq 1 200); do
    # pidfile = runner registered; logfile = the wrapper's tee opened it the
    # instant the runner spawned (covers an instant-exit runner that already
    # removed its pidfile again).
    if [ -s "$pidfile" ] || [ -e "$logfile" ]; then
        exit 0
    fi
    sleep 0.1
done
echo "serenity_terminal_launcher: terminal did not start the runner within 20s" >&2
exit 1
