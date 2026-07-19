# clipboard - desktop clipboard helpers for Mojo apps.
#
# Public surface:
#   from clipboard.clipboard import (
#       BACKEND_AUTO, BACKEND_WAYLAND, BACKEND_XCLIP, BACKEND_XSEL,
#       BACKEND_OSC52, SELECTION_CLIPBOARD, SELECTION_PRIMARY,
#       detect_backend, backend_available, backend_name, availability_report,
#       read_text, write_text, clear, osc52_sequence, write_text_osc52,
#   )
#
# See clipboard/README.md for backend requirements and limits.
