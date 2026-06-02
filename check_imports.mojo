# Phase-1 build proof: confirms the SDK forward ops inventoried via `mojo doc`
# are importable inside the project. Compile = pass. Run with:
#   pixi run mojo run check_imports.mojo
from linalg.matmul import matmul
from nn.normalization import rms_norm, layer_norm, group_norm
from nn.rope import apply_rope
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor


def main() raises:
    print("serenitymojo: SDK forward-op imports OK (matmul, rms_norm,")
    print("  layer_norm, group_norm, apply_rope, DeviceContext, LayoutTensor)")
