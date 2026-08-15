# serenitymojo/models/minimax_h3/h3_train_modgrid.mojo
#
# HOST-side AdaLN sigma grid for H3 training: the exact modcache tables at
# N uniform sigma nodes, built ONCE with the gated modcache pass and saved
# to a sidecar safetensors; training re-opens the sidecar mmap and fetches
# ONLY the current node's rows per step ([3, 6D] per block + [1, 2D] final,
# ~10MB/step H2D). This frees the ~9.7GB the device-resident grid held —
# the fp8-resident base needs that VRAM.
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_memcpy, BytePtr, sys_open, sys_close, O_RDONLY
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.models.dit.minimax_h3_dit import MiniMaxH3DiTConfig
from serenitymojo.models.dit.minimax_h3_frontend import minimax_h3_timestep_embedding
from serenitymojo.models.dit.minimax_h3_modcache import (
    minimax_h3_build_modulation_cache,
)

comptime TArc = ArcPointer[Tensor]


def _blk_name(i: Int) -> String:
    return String("blk_") + String(i)


struct H3TrainModGrid(Movable):
    var st: SafeTensors
    var nodes: Int
    var blocks: Int
    var block_width: Int   # 6*hidden
    var final_width: Int   # 2*hidden

    def __init__(
        out self, var st: SafeTensors,
        nodes: Int, blocks: Int, block_width: Int, final_width: Int,
    ):
        self.st = st^
        self.nodes = nodes
        self.blocks = blocks
        self.block_width = block_width
        self.final_width = final_width

    @staticmethod
    def build_or_load(
        path: String,
        shards: ShardedSafeTensors,
        frontend_w: Dict[String, ArcPointer[Tensor]],
        config: MiniMaxH3DiTConfig,
        nodes: Int,
        ctx: DeviceContext,
    ) raises -> H3TrainModGrid:
        var need_build = True
        try:
            var probe = SafeTensors.open(path)
            if probe.has_tensor(_blk_name(config.num_layers - 1)) and probe.has_tensor(String("final")):
                var sh = probe.tensor_info(_blk_name(0)).shape.copy()
                if sh[0] == nodes * 3:
                    need_build = False
        except:
            need_build = True
        if need_build:
            print("[modgrid] building", nodes, "-node adaln grid (one streamed pass)...")
            var ts = List[Float32]()
            for k in range(nodes):
                var sigma = Float32(k) / Float32(nodes - 1)
                ts.append(Float32(1.0) - sigma)
            var tsh: List[Int] = [nodes]
            var ts_t = Tensor.from_host(ts, tsh^, STDtype.F32, ctx)
            var temb = minimax_h3_timestep_embedding(ts_t, frontend_w, config, ctx)
            var cache = minimax_h3_build_modulation_cache(shards, temb, config, ctx)
            ctx.synchronize()
            var names = List[String]()
            var tensors = List[TArc]()
            for i in range(config.num_layers):
                names.append(_blk_name(i))
                tensors.append(cache.block_mod[i].copy())
            names.append(String("final"))
            tensors.append(cache.final_mod.copy())
            save_safetensors(names, tensors, path, ctx)
            _ = cache^
            print("[modgrid] saved:", path)
        var st = SafeTensors.open(path)
        var hidden = config.hidden_size
        return H3TrainModGrid(
            st^, nodes, config.num_layers, 6 * hidden, 2 * hidden,
        )

    def block_rows(
        self, layer: Int, node: Int, ctx: DeviceContext
    ) raises -> Tensor:
        """[3, 6D] BF16 device rows for sigma node `node` (3 modality tags)."""
        if layer < 0 or layer >= self.blocks or node < 0 or node >= self.nodes:
            raise Error("modgrid: out of range")
        var span = self.st.tensor_bytes(_blk_name(layer))
        var row_bytes = self.block_width * 2
        var off = node * 3 * row_bytes
        var sub = span[off : off + 3 * row_bytes]
        var shape: List[Int] = [3, self.block_width]
        return Tensor.from_view(from_parts(STDtype.BF16, shape^, sub), ctx)

    def final_row(self, node: Int, ctx: DeviceContext) raises -> Tensor:
        """[1, 2D] BF16 device row for sigma node `node`."""
        if node < 0 or node >= self.nodes:
            raise Error("modgrid: node out of range")
        var span = self.st.tensor_bytes(String("final"))
        var row_bytes = self.final_width * 2
        var off = node * row_bytes
        var sub = span[off : off + row_bytes]
        var shape: List[Int] = [1, self.final_width]
        return Tensor.from_view(from_parts(STDtype.BF16, shape^, sub), ctx)
