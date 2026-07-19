from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd import Tape, backward
from serenity_trainer.module.LoRAModule import make_lora_adapter, lora_linear_forward, LoraAdapter

def main() raises:
    var ctx = DeviceContext()
    comptime M = 4
    comptime IN = 8
    comptime OUT = 6
    var xv = List[Float32]()
    for i in range(M*IN): xv.append(Float32((i*7)%13-6)*0.05)
    var wv = List[Float32]()
    for i in range(OUT*IN): wv.append(Float32((i*5)%11-5)*0.05)
    var xs = List[Int](); xs.append(M); xs.append(IN)
    var ws = List[Int](); ws.append(OUT); ws.append(IN)
    var x = Tensor.from_host(xv, xs.copy(), STDtype.BF16, ctx)
    var w = Tensor.from_host(wv, ws.copy(), STDtype.BF16, ctx)
    var adp = make_lora_adapter(IN, OUT, 4, Float32(8.0), UInt64(1), ctx)
    var tape = Tape()
    tape.track(adp.a); tape.track(adp.b)
    var out = lora_linear_forward(tape, x, w, adp, ctx)
    print("lora out dtype =", out.dtype().name(), " a.id =", adp.a.id, " b.id =", adp.b.id)
    print("LORA COMPILE-CHECK OK")
