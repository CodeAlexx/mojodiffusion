"""Expose host-only metric dependencies after the Musubi virtualenv.

The training environment intentionally owns torch and CUDA packages.  The host
site has pyarrow, which Musubi's optional metrics writer needs, so append that
site instead of prepending it and accidentally shadowing the venv's torch.
"""

import sys


HOST_SITE = "/home/alex/.local/lib/python3.12/site-packages"
if HOST_SITE not in sys.path:
    sys.path.append(HOST_SITE)
