# Parity vs Serenity lr_ref.json + masked_loss_ref.json
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenity_trainer.util.lr_scheduler_util import LrSchedule, LR_COSINE
from serenity_trainer.util.loss.masked_loss import masked_losses

def main() raises:
    var ctx = DeviceContext()
    # cosine, scheduler_steps=1000, no warmup, min_factor=0 → match reference trainer lr_ref
    var sch = LrSchedule(LR_COSINE, 0, 1000, 1.0, 0.0)
    print("lr cosine (MOJO): @0=", sch.factor(0), " @250=", sch.factor(250),
          " @500=", sch.factor(500), " @750=", sch.factor(750), " @999=", sch.factor(999))
    print("  reference trainer ref         : @0=1.0 @250=0.853553 @500=0.5 @750=0.146447 @999=2e-06")

    # masked_loss: losses[1,1,2,2]=[0.1,0.2,0.3,0.4], mask=[1,0,0.5,1], uw=0.1
    var lv = List[Float32](); lv.append(0.1); lv.append(0.2); lv.append(0.3); lv.append(0.4)
    var mv = List[Float32](); mv.append(1.0); mv.append(0.0); mv.append(0.5); mv.append(1.0)
    var sh = List[Int](); sh.append(1); sh.append(1); sh.append(2); sh.append(2)
    var losses = Tensor.from_host(lv, sh.copy(), STDtype.F32, ctx)
    var mask = Tensor.from_host(mv, sh.copy(), STDtype.F32, ctx)
    var out = masked_losses(losses, mask, Float32(0.1), False, ctx)
    var oh = out.to_host(ctx)
    print("masked_loss (MOJO)=", oh[0], oh[1], oh[2], oh[3], "  reference trainer ref=[0.1,0.02,0.15,0.4]")
