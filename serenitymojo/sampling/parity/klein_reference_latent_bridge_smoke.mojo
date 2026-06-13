# sampling/parity/klein_reference_latent_bridge_smoke.mojo
#
# No-heavy gate for the Klein ReferenceLatent bridge. It proves the daemon's
# flattened SerenityFlow edit metadata can be converted into the target/ref
# token and image-id layout the future edit sampler must consume.
#
# Run:
#   pixi run mojo run -I . serenitymojo/sampling/parity/klein_reference_latent_bridge_smoke.mojo

from std.gpu.host import DeviceContext

from serenitymojo.serve.backend import JobParams
from serenitymojo.sampling.klein_reference_latent_bridge import (
    KleinReferenceLatentPlan,
    plan_klein_reference_latent_bridge,
    build_klein_reference_combined_img_ids,
    build_klein_reference_combined_tokens,
    synthetic_klein_reference_latent,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(msg)


def _check_eq(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(name + String(" got=") + String(got) + String(" expected=") + String(expected))


def _check_f32(name: String, got: Float32, expected: Float32) raises:
    var diff = got - expected
    if diff < 0.0:
        diff = -diff
    if diff > 1.0e-4:
        raise Error(name + String(" got=") + String(got) + String(" expected=") + String(expected))


def _edit_params(width: Int, height: Int, model: String) -> JobParams:
    var p = JobParams()
    p.model = model
    p.width = width
    p.height = height
    p.steps = 35
    p.seed = 42
    p.cfg = 3.5
    p.sampler = String("euler")
    p.scheduler = String("flux2")
    p.creativity = 1.0
    p.init_image = String("input.png")
    p.reference_image = String("input.png")
    p.reference_latent_method = String("index")
    p.reference_latent_count = 2
    return p^


def _check_plan(plan: KleinReferenceLatentPlan, width: Int, latent: Int) raises:
    _check_eq(String("plan width"), plan.width, width)
    _check_eq(String("plan height"), plan.height, width)
    _check_eq(String("plan latent_h"), plan.latent_h, latent)
    _check_eq(String("plan latent_w"), plan.latent_w, latent)
    _check_eq(String("target tokens"), plan.target_tokens, latent * latent)
    _check_eq(String("reference tokens"), plan.reference_tokens, latent * latent)
    _check_eq(String("combined image tokens"), plan.combined_image_tokens, 2 * latent * latent)
    _check_eq(String("text tokens"), plan.text_tokens, 512)
    _check_eq(String("edit sequence tokens"), plan.edit_sequence_tokens, 512 + 2 * latent * latent)
    _check_eq(String("latent channels"), plan.latent_channels, 128)
    _check_eq(String("reference links"), plan.reference_latent_links, 2)
    _check_eq(String("target start"), plan.target_token_start, 0)
    _check_eq(String("reference start"), plan.reference_token_start, latent * latent)
    _check_f32(String("reference t offset"), plan.reference_t_offset, 10.0)


def _check_bridge_tensors(
    plan: KleinReferenceLatentPlan, ctx: DeviceContext
) raises:
    var noise = synthetic_klein_reference_latent(plan, 0.0, ctx)
    var reference = synthetic_klein_reference_latent(plan, 2000000.0, ctx)

    var ids = build_klein_reference_combined_img_ids(plan, ctx)
    var ids_shape = ids.shape()
    _check(len(ids_shape) == 2, String("combined ids rank"))
    _check_eq(String("combined ids rows"), ids_shape[0], plan.combined_image_tokens)
    _check_eq(String("combined ids cols"), ids_shape[1], 4)
    var ids_h = ids.to_host(ctx)
    _check_f32(String("first target id t"), ids_h[0], 0.0)
    _check_f32(String("first target id row"), ids_h[1], 0.0)
    _check_f32(String("first target id col"), ids_h[2], 0.0)
    _check_f32(String("first reference id t"), ids_h[plan.target_tokens * 4], plan.reference_t_offset)
    _check_f32(String("first reference id row"), ids_h[plan.target_tokens * 4 + 1], 0.0)
    _check_f32(String("first reference id col"), ids_h[plan.target_tokens * 4 + 2], 0.0)

    var combined = build_klein_reference_combined_tokens(noise, reference, plan, ctx)
    var cshape = combined.shape()
    _check(len(cshape) == 3, String("combined tokens rank"))
    _check_eq(String("combined token batch"), cshape[0], 1)
    _check_eq(String("combined token sequence"), cshape[1], plan.combined_image_tokens)
    _check_eq(String("combined token channels"), cshape[2], plan.latent_channels)
    var combined_h = combined.to_host(ctx)
    _check_f32(String("target token c0"), combined_h[0], 0.0)
    _check_f32(String("target token c1"), combined_h[1], 10000.0)
    _check_f32(String("reference token c0"), combined_h[plan.target_tokens * plan.latent_channels], 2000000.0)
    _check_f32(String("reference token c1"), combined_h[plan.target_tokens * plan.latent_channels + 1], 2010000.0)


def main() raises:
    print("=== Klein ReferenceLatent bridge no-heavy gate ===")
    var ctx = DeviceContext()

    var p512 = _edit_params(512, 512, String("flux2-klein-9b.safetensors"))
    var plan512 = plan_klein_reference_latent_bridge(p512)
    _check_plan(plan512, 512, 32)
    _check_bridge_tensors(plan512, ctx)
    print("  512 edit bridge: plan + token/id tensors OK")

    var p1024 = _edit_params(1024, 1024, String("flux2-klein-4b.safetensors"))
    var plan1024 = plan_klein_reference_latent_bridge(p1024)
    _check_plan(plan1024, 1024, 64)
    _check_bridge_tensors(plan1024, ctx)
    print("  1024 edit bridge: plan + token/id tensors OK")

    print("PASS: Klein ReferenceLatent bridge no-heavy gate")
