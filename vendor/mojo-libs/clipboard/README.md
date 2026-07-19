# clipboard

Desktop clipboard helpers for Mojo apps.

The module is Linux-first and dependency-light: it links only libc from Mojo and
uses the desktop provider already present at runtime:

- Wayland: `wl-copy` and `wl-paste` from `wl-clipboard`
- X11: `xclip` or `xsel`
- Terminal fallback: OSC52 write sequence, explicit only

Payload text is sent over provider stdin/stdout. It is not interpolated into a
shell command, so paths, prompts, and generated text do not become shell input.

## Modules

| Module | What it is |
|---|---|
| `clipboard.mojo` | Public text clipboard API: `write_text`, `read_text`, `clear`, `detect_backend`, `backend_available`, `availability_report`, `osc52_sequence`, and `write_text_osc52`. Uses libc `popen`/`fread`/`fwrite` to stream payload bytes to trusted provider commands. |
| `tests/clipboard_test.mojo` | Compile-safe checks for backend names, selection names, provider reporting, Base64 vectors, and OSC52 sequences. With `CLIPBOARD_TEST_REAL=1`, runs a real Wayland/X11 round-trip and restores prior non-empty clipboard text. |

## API

```mojo
from clipboard.clipboard import (
    read_text, write_text, clear, detect_backend, availability_report,
    SELECTION_CLIPBOARD, SELECTION_PRIMARY,
)

def main() raises:
    print(availability_report())
    write_text(String("/tmp/generated/image.png"))
    var pasted = read_text()
    print("clipboard:", pasted)

    write_text(String("primary selection"), SELECTION_PRIMARY)
```

Backends are selected automatically by `detect_backend()`:

1. Wayland when `WAYLAND_DISPLAY` is set and `wl-copy`/`wl-paste` exist.
2. X11 `xclip` when `DISPLAY` is set and `xclip` exists.
3. X11 `xsel` when `DISPLAY` is set and `xsel` exists.

Explicit backends are available through `BACKEND_WAYLAND`, `BACKEND_XCLIP`,
`BACKEND_XSEL`, and `BACKEND_OSC52`.

## Backend Behavior

| Backend | Read | Write | Selection support | Runtime requirement |
|---|---:|---:|---|---|
| Wayland | yes | yes | clipboard + primary | `WAYLAND_DISPLAY`, `wl-copy`, `wl-paste` |
| X11 `xclip` | yes | yes | clipboard + primary | `DISPLAY`, `xclip` |
| X11 `xsel` | yes | yes | clipboard + primary | `DISPLAY`, `xsel` |
| OSC52 | no | yes | clipboard + primary target codes | terminal that accepts OSC52 |

`BACKEND_AUTO` chooses the first full read/write provider in that order. It does
not auto-select OSC52 because OSC52 is write-only and visibly emits terminal
escape sequences.

## OSC52

OSC52 is write-only and terminal-dependent, so it is never selected
automatically:

```mojo
from clipboard.clipboard import write_text_osc52

def main() raises:
    write_text_osc52(String("copied through the terminal"))
```

Use it only for terminal apps where emitting an escape sequence to stdout or a
terminal writer is expected behavior. `write_text_osc52()` prints a convenience
sequence to stdout; use `osc52_sequence()` when your app needs exact byte control.

## Limits

- Text API only. UTF-8 is preserved byte-for-byte through the provider pipe.
- Clipboard persistence is owned by the provider. On some X11 setups, the
  provider process may need a running X server until ownership is transferred.
- `read_text(max_bytes=...)` defaults to 64 MiB to protect apps from accidental
  unbounded reads.
- No macOS/Windows backend yet. Add native providers behind the same public API
  rather than changing app code.

## Tests

Compile-safe tests:

```bash
pixi run mojo run -I . clipboard/tests/clipboard_test.mojo
```

Real clipboard round-trip is opt-in because it mutates the user's clipboard:

```bash
CLIPBOARD_TEST_REAL=1 pixi run mojo run -I . clipboard/tests/clipboard_test.mojo
```

Observed validation on the development machine:

- `pixi run --manifest-path /home/alex/mojodiffusion/pixi.toml mojo run -I . -I /home/alex/MOJO-libs /home/alex/MOJO-libs/clipboard/tests/clipboard_test.mojo` → `14 passed, 0 failed`.
- `CLIPBOARD_TEST_REAL=1 pixi run --manifest-path /home/alex/mojodiffusion/pixi.toml mojo run -I . -I /home/alex/MOJO-libs /home/alex/MOJO-libs/clipboard/tests/clipboard_test.mojo` → `xsel`, `16 passed, 0 failed`.
