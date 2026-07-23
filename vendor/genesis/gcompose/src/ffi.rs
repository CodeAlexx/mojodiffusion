//! Safe-ish Rust wrappers over the C engine shims (vendored from MojoMedia/ffi).
//!
//! Phase 0 surface: open a media file, decode one frame letterboxed into an RGBA8
//! buffer, close. The C side (fpx_decode.c) owns all the FFmpeg complexity.

use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_longlong, c_void};

extern "C" {
    fn fpx_open(path: *const c_char) -> *mut c_void;
    fn fpx_decode_frame_letterbox(
        h: *mut c_void,
        frame_index: c_int,
        out: *mut u8,
        ow: c_int,
        oh: c_int,
    ) -> c_int;
    fn fpx_close(h: *mut c_void);
    // Total video frame count (stream nb_frames, else estimated from duration*fps). 0 if unknown.
    fn fpx_nframes(h: *mut c_void) -> c_int;

    // OpenCL compute shim (fpx_gpu.c). Fixed working resolution GVW x GVH.
    fn fpx_gpu_init() -> c_int;
    fn fpx_gpu_upload_u8(slot: c_int, rgba8: *const u8) -> c_int;
    fn fpx_gpu_track1(tt: c_int, t: f32, param: f32);
    // P31: `blend` (0=Normal..7=Difference) selects the per-channel blend of the over RGB with the
    // base before the alpha-over. blend==0 (Normal) is byte-identical to the pre-P31 plain composite.
    fn fpx_gpu_pip(op: f32, blend: c_int, px: f32, py: f32, pw: f32, ph: f32);
    fn fpx_gpu_grade(bright: f32, contrast: f32, sat: f32);
    // Per-clip grade (Triad-B P1): grades the PiP-composite buffer (INB) IN PLACE before the program
    // grade, so a later fpx_gpu_grade stacks on top (documented "per-clip first, then program" order).
    // A neutral grade (0/1/1) is a no-op. Run between fpx_gpu_pip and fpx_gpu_grade.
    fn fpx_gpu_grade_clip(bright: f32, contrast: f32, sat: f32);
    // P2 TRANSFORM (Shotcut-parity): rotate (degrees) + uniform scale the BASE frame about its
    // center, bilinear sample. Runs RIGHT AFTER fpx_gpu_track1 (before pip). Identity at
    // rot_deg=0,scale=1 (skipped, zero cost). Uses a device scratch copy (cannot read+write in place).
    fn fpx_gpu_transform(rot_deg: f32, scale: f32);
    // P2 LGG (3-way color wheels): per-channel out=clamp01(pow(clamp01(in*gain+lift),1/gamma)) IN
    // PLACE on the grade-result buffer (OUTB), AFTER fpx_gpu_grade, BEFORE fpx_gpu_look. Identity at
    // lift 0 / gamma 1 / gain 1 (skipped). White balance is folded into the gains by the UI.
    fn fpx_gpu_lgg(
        lr: f32, lg: f32, lb: f32,
        gar: f32, gag: f32, gab: f32,
        gnr: f32, gng: f32, gnb: f32,
    );
    // P2 BLUR: separable gaussian (2 passes via device scratch), IN PLACE on OUTB. radius=ceil(2*sigma)
    // capped at 32; sigma<=0 => no-op. Runs AFTER fpx_gpu_lgg, BEFORE fpx_gpu_look.
    fn fpx_gpu_blur(sigma: f32);
    // P5 master tone curve (5-point piecewise-linear), in place on OUTB AFTER blur, BEFORE look.
    // Identity (0,.25,.5,.75,1) is skipped engine-side, so an un-curved clip is a no-op.
    fn fpx_gpu_curve(y0: f32, y1: f32, y2: f32, y3: f32, y4: f32);
    // P6 STYLIZE/UTILITY filters — all run on the composited OUTB AFTER the curve, BEFORE the look,
    // in the pinned order simplefx -> vignette -> sharpen -> flip. Each is a no-op at its default
    // (kind==0 / amt<=0 / mode==0), skipped engine-side so an unfiltered clip is byte-identical.
    //   fpx_gpu_simplefx(kind): in place on OUTB. 1 invert, 2 sepia, 3 grayscale, 4 posterize.
    fn fpx_gpu_simplefx(kind: c_int);
    //   fpx_gpu_vignette(amt): in place on OUTB. Radial edge darken by `amt` (smoothstep falloff).
    fn fpx_gpu_vignette(amt: f32);
    //   fpx_gpu_sharpen(amt): unsharp. The C wrapper copies OUTB->g_tmp, then reads g_tmp neighbours
    //   into OUTB (center*(1+4a) - a*(left+right+up+down), clamped).
    fn fpx_gpu_sharpen(amt: f32);
    //   fpx_gpu_flip(mode): mirror. 1 H, 2 V, 3 both. The C wrapper copies OUTB->g_tmp, then samples
    //   g_tmp at the flipped coord into OUTB.
    fn fpx_gpu_flip(mode: c_int);
    // P7 COLOR filters — both run on the composited OUTB AFTER the P6 filters (flip), BEFORE the look,
    // in the pinned order HSL -> LEVELS. Each is a no-op at its identity default (skipped engine-side)
    // so an unfiltered clip is byte-identical.
    //   fpx_gpu_hsl(hue_deg, sat, light): in place on OUTB. RGB->HSL, hue += hue_deg (wrap 360),
    //   saturation *= sat, lightness += light, HSL->RGB, clamp01. Identity hue_deg=0,sat=1,light=0.
    fn fpx_gpu_hsl(hue_deg: f32, sat: f32, light: f32);
    //   fpx_gpu_levels(in_black, in_white, gamma): in place on OUTB, per channel
    //   out=clamp01(pow(clamp01((c-in_black)/max(in_white-in_black,1e-3)),1/max(gamma,1e-3))).
    //   Identity in_black=0,in_white=1,gamma=1.
    fn fpx_gpu_levels(in_black: f32, in_white: f32, gamma: f32);
    // P8 STYLIZE-2 filters — both run on the composited OUTB AFTER the P7 color filters (levels),
    // BEFORE the look, in the pinned order MOSAIC -> GRADIENT-MAP. Each is a no-op at its default
    // (skipped engine-side) so an unfiltered clip is byte-identical.
    //   fpx_gpu_mosaic(block): pixelate. The C wrapper copies OUTB->g_tmp, then samples g_tmp's block
    //   top-left (bx=(x/block)*block, by=(y/block)*block) into OUTB. block<=1 = skip (no-op default).
    //   `block` is the block size in pixels; the C/kernel side use `int`, so this param is c_int and
    //   the worker prints the model's `mosaic: u32` as a plain decimal that parses back as i32.
    fn fpx_gpu_mosaic(block: c_int);
    //   fpx_gpu_gmap(amt, lo_*, hi_*): gradient map (luma -> colour ramp), in place on OUTB.
    //   luma=dot(rgb,[.299,.587,.114]); mapped=mix(lo,hi,luma); rgb=mix(rgb,mapped,amt). amt<=0 = skip
    //   (no-op default). (lo_r,lo_g,lo_b)=shadow colour, (hi_r,hi_g,hi_b)=highlight colour.
    fn fpx_gpu_gmap(amt: f32, lo_r: f32, lo_g: f32, lo_b: f32, hi_r: f32, hi_g: f32, hi_b: f32);
    // P9 FX filters — all run on the composited OUTB AFTER the P8 stylize-2 filters (gradient map),
    // BEFORE the look, in the pinned order denoise -> glow -> rgb-shift. Each is a no-op at its
    // default (denoise<=0 / glow amt<=0 / rgbshift px<=0), skipped engine-side so an unfiltered clip
    // is byte-identical.
    //   fpx_gpu_denoise(strength): edge-preserving 5x5 bilateral smooth blended by `strength` (0..1).
    //   The C wrapper copies OUTB->g_tmp, then reads g_tmp's 5x5 neighbourhood (spatial gaussian *
    //   colour-range gaussian) into OUTB, blending centre->bilateral by `strength`. strength<=0 = skip.
    fn fpx_gpu_denoise(strength: f32);
    //   fpx_gpu_glow(amt, thr): bloom. The C wrapper extracts the bright pass (luma>thr ? rgb : 0) into
    //   a 2nd scratch (g_tmp2), blurs it at a fixed sigma (reusing the separable gaussian), then adds
    //   amt*blurred back onto OUTB (clamped). amt<=0 = skip (no-op default); `thr` is the luma threshold.
    fn fpx_gpu_glow(amt: f32, thr: f32);
    //   fpx_gpu_rgbshift(px): chromatic aberration. The C wrapper copies OUTB->g_tmp, then samples r at
    //   (x+round(px),y), b at (x-round(px),y), g/a at (x,y) (x clamped to [0,VW-1]) into OUTB. px<=0 =
    //   skip (no-op default); `px` is the channel offset in pixels.
    fn fpx_gpu_rgbshift(px: f32);
    // P10 STYLIZE-4 filters — all run on the composited OUTB AFTER the P9 fx filters (rgb-shift),
    // BEFORE the look, in the pinned order halftone -> emboss -> edge. Each is a no-op at its default
    // (halftone cell<=1 / emboss amt<=0 / edge mix<=0), skipped engine-side so an unfiltered clip is
    // byte-identical. ALL THREE are spatial: the C wrapper copies OUTB->g_tmp, then the kernel reads
    // g_tmp ('s') and writes OUTB ('d').
    //   fpx_gpu_halftone(cell): luma-driven dot screen. cell = dot cell size in px (the model's
    //   `halftone: u32` is printed on the wire as a plain decimal that round-trips to i32, so this
    //   param is c_int). cell<=1 = skip. The C wrapper samples each cell's centre luma from g_tmp and
    //   paints a black dot (radius=(1-luma)*0.5*cell) on a white field into OUTB.
    fn fpx_gpu_halftone(cell: c_int);
    //   fpx_gpu_emboss(amt): directional relief. amt = relief strength (0..1). amt<=0 = skip. Per
    //   channel out=clamp01(0.5+amt*(centre - NW)) reading the g_tmp copy (NW = (x-1,y-1), clamped).
    fn fpx_gpu_emboss(amt: f32);
    //   fpx_gpu_edge(mix): Sobel edge/sketch mixed back. mix = edge mix (0..1). mix<=0 = skip. Sobel
    //   gradient magnitude on g_tmp luma; out.rgb = mix*vec3(mag) + (1-mix)*orig (3x3 neighbours clamped).
    fn fpx_gpu_edge(mix: f32);
    // P13 OLD-FILM/DISTORT filters — all run on the composited OUTB AFTER the P10 stylize-4 filters
    // (edge), BEFORE the look, in the pinned order grain -> scratches -> diffusion. Each is a no-op at
    // its default (grain<=0 / scratches<=0 / diffusion<=0), skipped engine-side so an unfiltered clip
    // is byte-identical. ALL THREE are spatial: the C wrapper copies OUTB->g_tmp, then the kernel
    // reads g_tmp ('s') and writes OUTB ('d'). The pseudo-randomness is a DETERMINISTIC integer hash
    // of the pixel coords (same input frame => same output), so regression gates stay stable.
    //   fpx_gpu_grain(amt): film noise. amt = noise strength 0..1. amt<=0 = skip. A per-pixel hashed
    //   luma noise n=(hash(x,y)*2-1)*amt is added to all 3 channels (achromatic grain), then clamped.
    fn fpx_gpu_grain(amt: f32);
    //   fpx_gpu_scratches(amt): old-film vertical lines. amt = scratch density/amount 0..1. amt<=0 =
    //   skip. A column is a scratch when hash(x,0) < amt*0.06; on a scratch column a column-wide signed
    //   offset (hash(x,7)-0.5)*0.9 is added to rgb (a bright/dark vertical line).
    fn fpx_gpu_scratches(amt: f32);
    //   fpx_gpu_diffusion(radius): frosted-glass jitter. radius = jitter radius in px 0..16. radius<=0
    //   = skip. Each pixel samples a deterministic hashed neighbour within +/- round(radius) (x/y
    //   offsets from two independent hash streams), clamped to [0,VW-1]x[0,VH-1].
    fn fpx_gpu_diffusion(radius: f32);
    // P16 DISTORT filters — all run on the composited OUTB AFTER the P13 old-film filters (diffusion),
    // BEFORE the look, in the pinned order wave -> swirl -> threshold. Each is a no-op at its default
    // (wave amp<=0 / swirl strength<=0 / threshold level<=0), skipped engine-side so an unfiltered clip
    // is byte-identical. WAVE and SWIRL are spatial: the C wrapper copies OUTB->g_tmp, then the kernel
    // reads g_tmp ('s') and writes OUTB ('d'). THRESHOLD is per-pixel IN PLACE on OUTB (no scratch).
    //   fpx_gpu_wave(amp): horizontal sinusoidal row displacement. amp = displacement amplitude in px.
    //   amp<=0 = skip. Each row y is shifted by sin(y*0.05)*amp; the source column is bilinear-sampled
    //   from g_tmp (x clamped to [0,VW-1]), so a straight vertical edge becomes a per-row wavy boundary.
    fn fpx_gpu_wave(amp: f32);
    //   fpx_gpu_swirl(strength): rotational distortion around the image centre. strength = rotation in
    //   radians at the centre. strength<=0 = skip. The rotation angle falls off linearly from the
    //   centre to the rim (ang=strength*(1-clamp(r/maxr,0,1))); the rotated source coord (clamped) is
    //   nearest-sampled from g_tmp, so an edge near the centre curves and the rim is untouched.
    fn fpx_gpu_swirl(strength: f32);
    //   fpx_gpu_threshold(level): luma binarize, in place on OUTB. level = luma cutoff in [0,1].
    //   level<=0 = skip. luma=dot(rgb,[.299,.587,.114]); out.rgb = luma>=level ? 1 : 0 (alpha
    //   passthrough), so a varied image collapses to pure black/white.
    fn fpx_gpu_threshold(level: f32);
    // P17 GEOMETRIC filters — all run on the composited OUTB AFTER the P16 distort filters (threshold),
    // BEFORE the look, in the pinned order lens -> crop -> glitch. Each is a no-op at its default (lens
    // k==0 / crop margin<=0 / glitch maxpx<=0), skipped engine-side so an unfiltered clip is
    // byte-identical. LENS and GLITCH are spatial: the C wrapper copies OUTB->g_tmp, then the kernel
    // reads g_tmp ('s') and writes OUTB ('d'). CROP is per-pixel IN PLACE on OUTB (no scratch).
    //   fpx_gpu_lens(k): radial barrel/pincushion distortion. k = radial coefficient (k>0 barrel,
    //   k<0 pincushion). k==0 = skip (BOTH signs active, only exact 0 is the no-op). For each output
    //   pixel the source is scaled radially about the centre (f=1+k*r2) and nearest-sampled (clamped)
    //   from g_tmp, so the frame bulges out (barrel) or pinches in (pincushion).
    fn fpx_gpu_lens(k: f32);
    //   fpx_gpu_crop(margin): margin-to-black, in place on OUTB. margin = border fraction in 0..0.49.
    //   margin<=0 = skip. Any pixel outside the centred keep-rect [margin..1-margin] in both axes has
    //   its RGB zeroed (alpha untouched), so the outer border goes black and the centre is unchanged.
    fn fpx_gpu_crop(margin: f32);
    //   fpx_gpu_glitch(maxpx): per-band horizontal channel shift. maxpx = max horizontal shift in px.
    //   maxpx<=0 = skip. The frame is split into 24px-high bands; each band gets a DETERMINISTIC signed
    //   integer shift (band hash, no time/RNG), then out.r samples g_tmp at x+sh, out.b at x-sh, g/a at
    //   x — so a sharp edge breaks into per-band displacements with R/B colour separation.
    fn fpx_gpu_glitch(maxpx: f32);
    // P23 360 REFRAME (equirectangular -> rectilinear): runs on the composited OUTB AFTER the P17
    // geometric filters (glitch), BEFORE the look. No-op at its default (enable==0, or fov out of
    // (0,180)) — engine returns immediately and OUTB is untouched, so an un-reframed clip is
    // byte-identical to pre-P23. SPATIAL: the C wrapper copies OUTB->g_tmp, then the kernel reads g_tmp
    // ('s', treated as a full 360x180 equirect panorama) and writes the reprojected rectilinear view
    // into OUTB ('d').
    //   fpx_gpu_eq2rect(enable, yaw_deg, pitch_deg, fov_deg): pinhole "360 viewer" reproject. enable
    //   (1/0) gates the kernel. yaw_deg/pitch_deg = view yaw/pitch in degrees (identity 0/0); fov_deg =
    //   horizontal field of view in degrees (default 90). yaw>0 pans the view RIGHT (samples u>0.5 of
    //   the equirect), yaw<0 LEFT (u<0.5). enable==0 = skip (no-op default).
    fn fpx_gpu_eq2rect(enable: c_int, yaw_deg: f32, pitch_deg: f32, fov_deg: f32);
    // P34 SHAPE MASK (Shotcut-parity mask_shape): zero (to black) the pixels OUTSIDE a centred
    // rectangle (shape==1) or ellipse (shape==2), with a feathered edge and optional invert. Runs on
    // the composited OUTB AFTER the P23 360 reframe, BEFORE the look — the SAME slot the P17 geometry
    // filters use. No-op at its default (shape==0) → engine returns immediately → OUTB untouched →
    // byte-identical to pre-P34. IN-PLACE on OUTB (each pixel scales only itself, like crop — no scratch).
    //   fpx_gpu_mask(shape, cx, cy, rw, rh, feather, inv): shape (0=none 1=rect 2=ellipse) gates the
    //   kernel. cx/cy = mask centre (normalized 0..1, identity 0.5/0.5); rw/rh = half-extents
    //   (normalized, default 0.5/0.5); feather = soft-edge band width (normalized, default 0); inv
    //   (1/0) flips inside<->outside. shape==0 = skip (no-op default).
    fn fpx_gpu_mask(shape: c_int, cx: f32, cy: f32, rw: f32, rh: f32, feather: f32, inv: c_int);
    // P38 DISTORTION BATCH (Shotcut-parity distort family): three per-clip OUTB filters run AFTER the
    // P34 shape mask, BEFORE the look — the SAME slot the P17/P23/P34 OUTB filters use. Each is a no-op
    // at its default → engine returns immediately → OUTB untouched → byte-identical to pre-P38.
    //   fpx_gpu_mirror(on): on==1 → the RIGHT half becomes a mirror of the LEFT half; on==0 = skip.
    //   fpx_gpu_kaleido(seg): seg>=2 → N-fold radial mirror; seg<2 (0/1) = skip (no-op default).
    //   fpx_gpu_dither(amt): amt>0 → ordered 4x4 Bayer-dithered posterize (~8 levels); amt<=0 = skip.
    fn fpx_gpu_mirror(on: c_int);
    fn fpx_gpu_kaleido(seg: c_int);
    fn fpx_gpu_dither(amt: f32);
    // P39 SELECTIVE COLOR (Shotcut-parity selective-color / hue-vs-hue): adjust ONE hue band on the
    // composited OUTB AFTER the P38 distort filters (dither), BEFORE the look — the SAME slot the
    // P17/P23/P34/P38 OUTB filters use. No-op at its default (sel_band==0) → engine returns immediately
    // → OUTB untouched → byte-identical to pre-P39. IN-PLACE on OUTB (each pixel touches only itself).
    //   fpx_gpu_selcolor(sel_band, sel_hshift, sel_sat): sel_band (0=off 1=Red 2=Yellow 3=Green 4=Cyan
    //   5=Blue 6=Magenta) gates the kernel and selects the hue band (centres at (band-1)/6, ~30° half-
    //   band, feathered to the edge). sel_hshift (-1..1) rotates the band's hue; sel_sat (default 1.0)
    //   scales its saturation. Greyscale pixels and pixels outside the band are left untouched.
    fn fpx_gpu_selcolor(sel_band: c_int, sel_hshift: f32, sel_sat: f32);
    // P41 SOLARIZE + COLOUR TEMPERATURE: two per-clip IN-PLACE OUTB pixel ops applied AFTER the P39
    // selective color, BEFORE the look. Each is a no-op at its default (sol_thr<=0 / temp==0) → engine
    // returns immediately → OUTB untouched → byte-identical to pre-P41.
    //   fpx_gpu_solarize(thr): classic darkroom solarize, per channel v>thr -> 1-v. thr 0 = off; (0,1].
    //   fpx_gpu_temp(t): warm (t>0) raises R & lowers B; cool (t<0) the reverse; green unchanged. -1..1.
    fn fpx_gpu_solarize(thr: f32);
    fn fpx_gpu_temp(t: f32);
    // P45 VIDEO FADE-TO-BLACK: multiply OUTB rgb by the per-frame fade factor f (from the clip's
    // fade_in/fade_out, computed in resolve_frame). f >= 1.0 = no fade (engine skips) → byte-identical.
    fn fpx_gpu_fade(f: f32);
    // P4 CHROMA KEY (green-screen): zero/soften the OVER buffer's ALPHA where the pixel's CHROMA
    // (luma-removed RGB) is within `sim`(+`smooth` edge band) of the key colour (kr,kg,kb) — RGB is
    // never touched. Runs on the OVER buffer AFTER its upload/transform and BEFORE fpx_gpu_pip, so the
    // pip composite (`over.a*op`) shows the base through the keyed pixels. Call ONLY when the clip's
    // chroma is enabled; a disabled clip skips it (OVER alpha untouched → byte-identical to P3).
    // P37: `spill` (>0) adds green-spill suppression in the SAME kernel, AFTER the alpha key. It pulls
    // a kept pixel's green channel down toward max(r,b) for a green-dominant key, removing the key tint
    // bled onto the subject's edges. spill==0 is a no-op (byte-identical to pre-P37). It rides the wire
    // as the LAST f32 field (after the P34 mask fields).
    fn fpx_gpu_chroma(kr: f32, kg: f32, kb: f32, sim: f32, smooth: f32, spill: f32);
    // look kind: 0=none (final=OUTB), 1=VHS, 2=LUT3D (both → final=LOOKB). amt = mix 0..1; lut_n =
    // the LUT grid N (cube root of the uploaded 3D LUT, only read when kind==2). Returns 1 when the
    // final composed frame lives in the LOOK buffer (kind 1/2), 0 when it stays in OUTB (kind 0).
    fn fpx_gpu_look(kind: c_int, amt: f32, lut_n: c_int) -> c_int;
    // Upload a 3D LUT (N*N*N*3 interleaved RGB floats in [0,1]) to the device for the LUT3D look.
    // `nfloats` must equal N*N*N*3 and be <= the shim's MAXLUTF capacity (33^3*3). Returns 0 on
    // success, negative on a bad arg / CL write failure.
    fn fpx_gpu_upload_lut(lut: *const f32, nfloats: c_int) -> c_int;
    fn fpx_gpu_download_u8(final_is_look: c_int, out: *mut u8);
    fn fpx_gpu_download_f32(final_is_look: c_int, out: *mut f32);
    fn fpx_gpu_finish();

    // Scope kernels (fpx_gpu.c) — run on the LAST composed GPU buffer (g_buf[OUTB] when
    // final_is_look==0, g_buf[LOOKB] when 1). These read the persistent on-device frame buffer; no
    // re-compose happens here, so the caller MUST have composed the wanted frame first (a PREVIEW)
    // and must NOT have run any other compose in between.
    //   fpx_gpu_histogram   -> 768 ints: R 0..255, G 256..511, B 512..767 (NOT a rendered image).
    //   fpx_gpu_waveform    -> 256*256*4 RGBA8 luma-waveform image.
    //   fpx_gpu_vectorscope -> 256*256*4 RGBA8 U/V vectorscope image.
    fn fpx_gpu_histogram(final_is_look: c_int, out_hist: *mut c_int);
    fn fpx_gpu_waveform(final_is_look: c_int, out: *mut u8);
    fn fpx_gpu_vectorscope(final_is_look: c_int, out: *mut u8);
    //   fpx_gpu_parade -> 256*256*4 RGBA8 RGB-parade image (3 side-by-side per-channel column
    //   waveforms, R|G|B). Triad-B P1 scope kind 3.
    fn fpx_gpu_parade(final_is_look: c_int, out: *mut u8);

    // Encode/mux shim (fpx_encode.c). RGBA f32 [0,1] frames -> mp4. Call order mirrors
    // MojoMedia main_editor.mojo: open -> config_video[ -> config_audio] -> start ->
    // (video_frame_f32 per frame in pts order)[ -> audio_samples_f32 ] -> finish -> close.
    fn fpx_enc_open(url: *const c_char) -> *mut c_void;
    fn fpx_enc_config_video(
        h: *mut c_void,
        codec_name: *const c_char,
        in_w: c_int,
        in_h: c_int,
        width: c_int,
        height: c_int,
        fps_num: c_int,
        fps_den: c_int,
        bit_rate: c_longlong,
        gop: c_int,
        preset: *const c_char,
    ) -> c_int;
    fn fpx_enc_config_audio(
        h: *mut c_void,
        codec_name: *const c_char,
        channels: c_int,
        sample_rate: c_int,
        bit_rate: c_longlong,
    ) -> c_int;
    // Constant-quality (CRF) rate control (Triad-B P1 export controls). Call AFTER config_video and
    // BEFORE start when the export uses rate_mode=1 (constant quality). Re-opens the video codec with
    // the crf private option set (x264/x265) or a global_quality/qscale fallback (mpeg4). Returns 0
    // on success, negative on error (best-effort: a failure leaves the bitrate config in place).
    fn fpx_enc_set_quality(
        h: *mut c_void,
        crf: c_int,
        gop: c_int,
        preset: *const c_char,
    ) -> c_int;
    fn fpx_enc_start(h: *mut c_void) -> c_int;
    fn fpx_enc_video_frame_f32(
        h: *mut c_void,
        rgba_f32: *const f32,
        in_w: c_int,
        in_h: c_int,
        ts_sec: c_double,
    ) -> c_int;
    fn fpx_enc_audio_samples_f32(h: *mut c_void, input: *const f32, nb: c_int) -> c_int;
    fn fpx_enc_finish(h: *mut c_void) -> c_int;
    fn fpx_enc_close(h: *mut c_void);

    // Audio FILTER shim (fpx_audio.c). Applies an arbitrary libavfilter chain (volume/pan/aeq/
    // anequalizer/acompressor/agate/loudnorm/...) to interleaved-float audio. P3 Triad-B: the
    // per-clip AudioFx chain is applied to a decoded clip range BEFORE the gain+offset mix.
    //   fpx_au_apply(sr, ch, chain, in, nb, out, out_cap):
    //     `in` is `nb` interleaved-float samples-per-channel (ch channels); `chain` is a libavfilter
    //     chain string with NO spaces (commas between filters, '=' / ':' inside). Writes up to
    //     out_cap floats into `out`; returns OUTPUT samples-per-channel (>= 0), or negative on error.
    //     A filter that changes the sample count (loudnorm/acompressor latency) is fine — the caller
    //     uses the returned count. `chain == NULL` or empty applies a pass-through (anull).
    fn fpx_au_apply(
        sr: c_int,
        ch: c_int,
        chain: *const c_char,
        input: *const f32,
        nb: c_int,
        out: *mut f32,
        out_cap: c_int,
    ) -> c_int;

    // Audio / asset shims (fpx_aread.c).
    //   fpx_audio_envelope: whole-track peak envelope into out[nbuckets] (0..1).
    //   fpx_decode_audio_range: decode [start,start+dur) -> interleaved f32 (out_ch).
    //   fpx_load_cube: parse a .cube 3D LUT into out (N*N*N*3 interleaved RGB floats); returns the
    //     grid size N on success (so out holds N^3*3 floats), or a negative error (-1 null arg,
    //     -2 open fail, -3 1D LUT unsupported, -4 out too small, -5 no LUT_3D_SIZE, -6 incomplete).
    fn fpx_audio_envelope(path: *const c_char, nbuckets: c_int, out: *mut f32) -> c_int;
    fn fpx_load_cube(path: *const c_char, out: *mut f32, max_floats: c_int) -> c_int;
    fn fpx_decode_audio_range(
        path: *const c_char,
        start_sec: c_double,
        dur_sec: c_double,
        out_sr: c_int,
        out_ch: c_int,
        out: *mut f32,
        cap: c_int,
    ) -> c_int;
}

/// The OpenCL shim's fixed working resolution (matches GVW/GVH in fpx_gpu.c).
pub const GVW: usize = 1280;
pub const GVH: usize = 856;

/// Scope-image dimensions (fpx_gpu.c renders waveform/vectorscope as a fixed SVW×SVH RGBA8 image,
/// and the histogram is rendered into the same size here). Matches the C shim's hard-coded 256×256
/// scope grid / image buffers, and the pinned worker.rs SW/SH the UI reads back.
pub const SVW: usize = 256;
pub const SVH: usize = 256;
/// Number of histogram bins fpx_gpu_histogram fills: 256 each for R, G, B = 768 ints.
pub const HIST_BINS: usize = 768;

/// Max LUT grid size the OpenCL shim accepts (matches `MAXLUTF = 33*33*33*3` in fpx_gpu.c). A
/// `.cube` whose `LUT_3D_SIZE` exceeds 33 will overflow `fpx_load_cube`'s `max_floats` guard and
/// return -4 (caller degrades to no look). `MAX_LUT_N` is the grid edge; `MAX_LUT_FLOATS` the
/// interleaved-RGB float count we stage.
pub const MAX_LUT_N: usize = 33;
pub const MAX_LUT_FLOATS: usize = MAX_LUT_N * MAX_LUT_N * MAX_LUT_N * 3;

/// Handle to the OpenCL compute pipeline. `init()` compiles the kernels once.
pub struct Gpu {
    _priv: (),
}

impl Gpu {
    /// Initialize OpenCL + compile kernels. Returns None if no usable device / build fails.
    pub fn init() -> Option<Gpu> {
        eprintln!("[gpu] init...");
        let rc = unsafe { fpx_gpu_init() };
        eprintln!("[gpu] init rc={rc}");
        if rc == 0 {
            Some(Gpu { _priv: () })
        } else {
            eprintln!("fpx_gpu_init failed: {rc}");
            None
        }
    }

    /// Initialize OpenCL with a few in-process retries for a SOFT init failure (`fpx_gpu_init`
    /// returning rc != 0). The hard flake is a process-death segfault inside the driver, which no
    /// in-process retry can fix (only the client respawn can) — but a clean non-zero rc (transient
    /// driver/device-busy) is worth retrying a couple of times before giving up. Returns None only
    /// if every attempt fails; the caller (serve) then exits non-zero so the client respawns.
    ///
    /// LIMITATION (finding #2): this retry only meaningfully recovers a genuinely TRANSIENT failure
    /// (device-busy on `clGetDeviceIDs`, a momentarily unavailable device). The C `fpx_gpu_init`
    /// returns early-0 only once `g_ready` is set, and its error paths (rc -1..-13) leave the
    /// partially-created OpenCL objects (`g_ctx`/`g_q`/`g_prog`/`g_buf`…) allocated WITHOUT freeing
    /// them — so each retry re-creates and leaks the prior partial handles. For a DETERMINISTIC
    /// failure (e.g. a kernel build error, rc -6) every attempt fails identically and only leaks
    /// `attempts-1` extra contexts before we give up and exit (the OS then reclaims on process
    /// exit). Keeping `attempts` small bounds that leak. Real handle-reuse recovery would require
    /// the C side to `goto cleanup`/release partial handles on each failure path; that lives in
    /// `csrc/fpx_gpu.c` (not this crate's wrapper) and is out of scope for this slice.
    pub fn init_retry(attempts: usize) -> Option<Gpu> {
        let attempts = attempts.max(1);
        for a in 0..attempts {
            if let Some(g) = Gpu::init() {
                return Some(g);
            }
            eprintln!("[gpu] init attempt {} of {attempts} failed (rc != 0)", a + 1);
        }
        None
    }

    /// Tiny end-to-end self-check after init: upload a KNOWN non-black frame, run an identity
    /// compose (no overlay, no grade change), download the result, and confirm the GPU actually
    /// ROUND-TRIPPED the pixels — not merely that the buffer is the right length. This catches a
    /// compositor that "init'd" (g_ready==1) but whose kernels can't launch / whose
    /// `clEnqueueReadBuffer` silently fails (the C download swallows every CL error and returns
    /// void), so `--serve` fails fast + clean BEFORE printing "serve ready" rather than serving a
    /// broken worker whose first real PREVIEW/ENC would produce garbage.
    ///
    /// Finding #1: the old check only asserted `out.len() == GVW*GVH*4`. But `compose()` allocates
    /// `vec![0u8; GVW*GVH*4]` and the C void downloads never resize it, so the length is ALWAYS
    /// exact and the check was a tautology that could never fail. The real signal is the pixel
    /// VALUES: we pre-fill the output with a sentinel that the upload value can't produce, upload a
    /// mid-gray (0x7F) frame, run an identity grade, and require the download to have (a) overwritten
    /// the sentinel and (b) landed near mid-gray. The `k_unpack`→`k_pack` round-trip of 0x7F is
    /// `round(127/255*255) == 127`, so an identity pipeline must yield ≈0x7F; an all-zero (dead
    /// read) or all-sentinel (read never ran) buffer fails.
    ///
    /// The check is cheap (one frame, op=0 so the overlay path is skipped) and side-effect-free
    /// w.r.t. the encoder/decoder caches: it only touches the GPU slots, which every real compose
    /// overwrites anyway.
    pub fn self_check(&self) -> bool {
        const FILL: u8 = 0x7F; // mid-gray upload; identity grade should preserve it (≈127 out)
        const SENTINEL: u8 = 0xAB; // pre-fill the download buffer with a value upload can't produce

        // Upload a uniform mid-gray frame into slot 0; op=0 disables PiP so slot 1 isn't required.
        let gray = vec![FILL; GVW * GVH * 4];
        self.upload(0, &gray);

        // Identity grade (bright=0, contrast=1, sat=1) and look=none: out should ≈ the uploaded gray.
        // We pre-seed `out` with SENTINEL so a download that never actually ran (CL read failed and
        // was swallowed) leaves the sentinel and is detectable.
        let mut out = vec![SENTINEL; GVW * GVH * 4];
        unsafe {
            fpx_gpu_track1(-1, 0.0, 4.0); // no transition: copy base (slot 0)
            fpx_gpu_pip(0.0, 0, 0.0, 0.0, 1.0, 1.0); // op=0: no overlay (blend=0 Normal)
            fpx_gpu_grade(0.0, 1.0, 1.0); // identity grade
            let fin = fpx_gpu_look(0, 0.0, 0); // look kind 0 = none
            fpx_gpu_download_u8(fin, out.as_mut_ptr());
            fpx_gpu_finish();
        }

        if out.len() != GVW * GVH * 4 {
            eprintln!(
                "[gpu] self-check FAILED: compose returned {} bytes (expected {})",
                out.len(),
                GVW * GVH * 4
            );
            return false;
        }

        // The download must have OVERWRITTEN our sentinel (proves the read ran) and produced a
        // non-degenerate, near-mid-gray result (proves the kernels ran). Sample on a stride so a
        // huge frame doesn't make the check expensive; the frame is uniform so a stride is faithful.
        const STRIDE: usize = 997; // coprime-ish stride to spread the samples across the buffer
        let mut samples = 0usize;
        let mut in_band = 0usize; // count of sampled bytes within an identity-of-0x7F band
        let mut sentinel_seen = 0usize;
        let mut nonzero_seen = false;
        let mut i = 0usize;
        while i < out.len() {
            let b = out[i];
            samples += 1;
            if b == SENTINEL {
                sentinel_seen += 1;
            }
            if b != 0 {
                nonzero_seen = true;
            }
            // Identity round-trip of 0x7F (127) should land at 127; allow generous slack for any
            // rounding in unpack/pack. Alpha bytes also round-trip 0x7F here (uniform fill).
            if (0x60..=0x9F).contains(&b) {
                in_band += 1;
            }
            i += STRIDE;
        }

        if samples == 0 {
            eprintln!("[gpu] self-check FAILED: no samples (empty buffer)");
            return false;
        }
        // Sentinel survivors mean the download never wrote those bytes (read failed silently).
        if sentinel_seen > 0 {
            eprintln!(
                "[gpu] self-check FAILED: {sentinel_seen}/{samples} sampled bytes still hold the \
                 pre-download sentinel (0x{SENTINEL:02X}) — GPU download did not run"
            );
            return false;
        }
        if !nonzero_seen {
            eprintln!("[gpu] self-check FAILED: download is all-zero (dead read / kernels not run)");
            return false;
        }
        // The vast majority of an identity-graded uniform-gray frame must land in the gray band.
        // (A few stragglers tolerated for any driver-specific rounding, but a broken pipeline that
        // returns black/white/garbage will miss the band wholesale.)
        if in_band * 2 < samples {
            eprintln!(
                "[gpu] self-check FAILED: only {in_band}/{samples} sampled bytes near mid-gray after \
                 an identity compose of a 0x{FILL:02X} frame — kernels are not composing correctly"
            );
            return false;
        }
        true
    }

    /// Upload an RGBA8 GVW×GVH frame to a slot (0=base/V1, 1=over/V2, 2=transition partner).
    pub fn upload(&self, slot: i32, rgba: &[u8]) {
        debug_assert_eq!(rgba.len(), GVW * GVH * 4);
        unsafe { fpx_gpu_upload_u8(slot as c_int, rgba.as_ptr()) };
    }

    /// Run the FIRST stage of the on-device pipeline directly (the transition / track-1 blend).
    ///
    /// `tt` = transition kind: -1 = none (copy slot-0 base into the track-1 buffer, today's
    /// no-transition behavior), 0..7 = the fpx_gpu transition kernels (0=crossfade, 1=wipe_lr,
    /// 2=wipe_rl, 3=wipe_up, 4=wipe_down, 5=slide_lr, 6=zoom, 7=dissolve). `t` is the transition
    /// progress in [0,1]; `param` is the per-transition parameter (4.0 default, dissolve Power).
    /// When `tt` in 0..7 the caller MUST have `upload(2, rgba)`'d the INCOMING (slot-2 / partner)
    /// frame first — the kernel blends slot-0 base toward slot-2 trans by `t`. Mirrors MojoMedia's
    /// `fpx_gpu_track1(tt_id, rtt, tt_p)` (main_editor.mojo ~699 preview / ~1300 render).
    ///
    /// This is exposed so the serve loop can drive a non-(-1) transition; the bundled `compose`/
    /// `compose_f32` keep their hardcoded no-transition `track1(-1,..)` for callers that never
    /// transition. `compose_trans`/`compose_trans_f32` below thread a real `tt` through instead.
    // Retained low-level wrapper: every live compose path uses compose_trans* (which call the C
    // fpx_gpu_track1 internally), so this standalone wrapper is currently unused.
    #[allow(dead_code)]
    pub fn track1(&self, tt: i32, t: f32, param: f32) {
        unsafe { fpx_gpu_track1(tt as c_int, t, param) };
    }

    /// Upload a parsed 3D LUT to the device for the LUT3D look. `lut` must be `N*N*N*3` interleaved
    /// RGB floats (exactly what `load_cube` returns). Returns true on success; a bad length / CL
    /// failure returns false (the caller then degrades the look to none). Must be called before a
    /// `compose(.., look_kind=2, .., lut_n=N)` so the LUT3D kernel reads the intended grid.
    pub fn upload_lut(&self, lut: &Lut) -> bool {
        let n = lut.n;
        let want = n * n * n * 3;
        if n == 0 || lut.data.len() < want || want > MAX_LUT_FLOATS {
            return false;
        }
        // Pass exactly N^3*3 floats (load_cube may have a longer staging buffer; only the first
        // want floats are the LUT). The C side validates 0 < nfloats <= MAXLUTF.
        let rc = unsafe { fpx_gpu_upload_lut(lut.data.as_ptr(), want as c_int) };
        rc == 0
    }

    /// Run the on-device pipeline (no transition → PiP composite of slot1 over slot0 → grade →
    /// look) and download the result as an RGBA8 GVW×GVH buffer.
    ///
    /// `look_kind` selects the look (0=none, 1=VHS, 2=LUT3D), `look_amt` is the mix, and `lut_n` is
    /// the uploaded LUT's grid size N (only read by the kernel when `look_kind==2`; pass 0 otherwise).
    /// For a LUT3D look the caller MUST `upload_lut` the matching grid first. Returns the composed
    /// RGBA8 buffer AND `final_is_look` (true when the frame ended up in the LOOK buffer, i.e. kind
    /// 1/2) so the serve loop can point a subsequent SCOPE at the post-look buffer.
    pub fn compose(
        &self,
        op: f32,
        px: f32,
        py: f32,
        pw: f32,
        ph: f32,
        bright: f32,
        contrast: f32,
        sat: f32,
        look_kind: i32,
        look_amt: f32,
        lut_n: i32,
    ) -> (Vec<u8>, bool) {
        let mut out = vec![0u8; GVW * GVH * 4];
        let fin = unsafe {
            fpx_gpu_track1(-1, 0.0, 4.0); // no transition: copy base (slot 0)
            fpx_gpu_pip(op, 0, px, py, pw, ph); // composite slot 1 over, into the PiP rect (blend=0 Normal)
            fpx_gpu_grade(bright, contrast, sat);
            let fin = fpx_gpu_look(look_kind as c_int, look_amt, lut_n as c_int);
            fpx_gpu_download_u8(fin, out.as_mut_ptr());
            fpx_gpu_finish();
            fin
        };
        (out, fin != 0)
    }

    /// Like `compose`, but runs a TRANSITION at the start of the pipeline (Wave 8). `tt` is the
    /// transition kind (-1 = none, copy base — identical to `compose`; 0..7 = a transition kernel),
    /// `trans_prog` is the progress in [0,1], `trans_param` the per-transition parameter (default
    /// 4.0). For `tt` in 0..7 the caller MUST have `upload(2, rgba)`'d the INCOMING frame first.
    /// Pipeline order matches MojoMedia: track1(tt, prog, param) → pip → grade → look. Returns the
    /// composed RGBA8 buffer + `final_is_look` (see `compose`).
    #[allow(clippy::too_many_arguments)]
    pub fn compose_trans(
        &self,
        tt: i32,
        trans_prog: f32,
        trans_param: f32,
        op: f32,
        // P31 BLEND MODE of the OVER (V2) clip: 0=Normal 1=Multiply 2=Screen 3=Overlay 4=Add
        // 5=Darken 6=Lighten 7=Difference. Rides the wire IMMEDIATELY AFTER `op` (over-opacity).
        // blend==0 (Normal) => fpx_blend returns the over colour => byte-identical to pre-P31.
        blend: i32,
        px: f32,
        py: f32,
        pw: f32,
        ph: f32,
        cbright: f32,
        ccontrast: f32,
        csat: f32,
        bright: f32,
        contrast: f32,
        sat: f32,
        look_kind: i32,
        look_amt: f32,
        lut_n: i32,
        // P2 per-clip effects (pinned wire order: lift3, gamma3, gain3, rot, scale, blur).
        lift_r: f32,
        lift_g: f32,
        lift_b: f32,
        gamma_r: f32,
        gamma_g: f32,
        gamma_b: f32,
        gain_r: f32,
        gain_g: f32,
        gain_b: f32,
        rot: f32,
        scale: f32,
        blur: f32,
        // P4 per-clip CHROMA KEY on the OVER (V2) buffer. `ck_on` (1/0) gates the kernel: when 0 the
        // OVER alpha is left untouched and the composite is byte-identical to P3. (kr,kg,kb)=key colour
        // in [0,1], `ck_sim` the chroma-distance threshold, `ck_smooth` the soft edge band.
        ck_on: i32,
        ck_r: f32,
        ck_g: f32,
        ck_b: f32,
        ck_sim: f32,
        ck_smooth: f32,
        // P37: green-spill suppression strength (0..1). spill==0 = no-op (byte-identical to pre-P37);
        // >0 pulls a kept pixel's green toward max(r,b) for a green-dominant key. Rides the wire as the
        // LAST f32 field (appended after the P34 mask fields), so the pre-P37 ck_* / mask indices are
        // unchanged. Threaded into fpx_gpu_chroma as its final arg.
        ck_spill: f32,
        // P5 master tone curve: 5 outputs at fixed inputs 0/.25/.5/.75/1 (identity = [0,.25,.5,.75,1]).
        curve: [f32; 5],
        // P6 stylize/utility (pinned wire order, after curve): vig sharp flip fx. All no-op at their
        // defaults (vig 0, sharp 0, flip 0, fx 0) → engine skips → byte-identical. Applied on OUTB
        // AFTER the curve, BEFORE the look, in order simplefx(fx) -> vignette(vig) -> sharpen(sharp)
        // -> flip(flip).
        vig: f32,
        sharp: f32,
        flip: i32,
        fx: i32,
        // P7 per-clip COLOR filters (pinned wire order, after the P6 fx): hue sat light inb inw gam.
        // All no-op at their defaults (hue 0, sat 1, light 0, inb 0, inw 1, gam 1) → engine skips →
        // byte-identical. Applied on OUTB AFTER the P6 flip, BEFORE the look, in order
        // hsl(hue,sat,light) -> levels(inb,inw,gam).
        hue: f32,
        sat_hsl: f32,
        light: f32,
        inb: f32,
        inw: f32,
        gam: f32,
        // P8 per-clip STYLIZE-2 filters (pinned wire order, after the P7 gam): mosaic gmap_amt glo3
        // ghi3. All no-op at their defaults (mosaic 0, gmap_amt 0) → engine skips → byte-identical.
        // Applied on OUTB AFTER the P7 levels, BEFORE the look, in order mosaic(mosaic) ->
        // gmap(gmap_amt, glo, ghi). `mosaic` is the block size in px (i32; 0/1 = off).
        mosaic: i32,
        gmap_amt: f32,
        glo_r: f32,
        glo_g: f32,
        glo_b: f32,
        ghi_r: f32,
        ghi_g: f32,
        ghi_b: f32,
        // P9 per-clip FX filters (pinned wire order, after the P8 ghi_b): denoise glow_amt glow_thr
        // rgbshift. All no-op at their defaults (denoise 0, glow_amt 0, rgbshift 0) → engine skips →
        // byte-identical. Applied on OUTB AFTER the P8 gradient map, BEFORE the look, in order
        // denoise(denoise) -> glow(glow_amt, glow_thr) -> rgbshift(rgbshift).
        denoise: f32,
        glow_amt: f32,
        glow_thr: f32,
        rgbshift: f32,
        // P10 per-clip STYLIZE-4 filters (pinned wire order, after the P9 rgbshift): halftone emboss
        // edge. All no-op at their defaults (halftone 0, emboss 0, edge 0) → engine skips →
        // byte-identical. Applied on OUTB AFTER the P9 rgb-shift, BEFORE the look, in order
        // halftone(halftone) -> emboss(emboss) -> edge(edge). `halftone` is the dot cell size in px
        // (i32; 0/1 = off — the model's u32 prints as a plain decimal that round-trips to i32).
        halftone: i32,
        emboss: f32,
        edge: f32,
        // P13 per-clip OLD-FILM/DISTORT filters (pinned wire order, after the P10 edge): grain
        // scratches diffusion. All no-op at their defaults (grain 0, scratches 0, diffusion 0) →
        // engine skips → byte-identical. Applied on OUTB AFTER the P10 edge, BEFORE the look, in order
        // grain(grain) -> scratches(scratches) -> diffusion(diffusion). grain=film-noise strength
        // 0..1; scratches=scratch density/amount 0..1; diffusion=jitter radius in px (0..16).
        grain: f32,
        scratches: f32,
        diffusion: f32,
        // P16 per-clip DISTORT filters (pinned wire order, after the P13 diffusion): wave swirl
        // threshold. All no-op at their defaults (wave 0, swirl 0, threshold 0) → engine skips →
        // byte-identical. Applied on OUTB AFTER the P13 diffusion, BEFORE the look, in order
        // wave(wave) -> swirl(swirl) -> threshold(threshold). wave=sinusoidal amplitude in px;
        // swirl=rotation strength in radians at the centre; threshold=luma binarize level 0..1.
        wave: f32,
        swirl: f32,
        threshold: f32,
        // P17 per-clip GEOMETRIC filters (pinned wire order, after the P16 threshold): lens crop
        // glitch. All no-op at their defaults (lens 0, crop 0, glitch 0) → engine skips →
        // byte-identical. Applied on OUTB AFTER the P16 threshold, BEFORE the look, in order
        // lens(lens) -> crop(crop) -> glitch(glitch). lens=radial coefficient (+barrel/-pincushion,
        // 0=off); crop=border-to-black fraction 0..0.49; glitch=max per-band horizontal shift in px.
        lens: f32,
        crop: f32,
        glitch: f32,
        // P23 per-clip 360 REFRAME (pinned wire order, after the P17 glitch): eq360 eq_yaw eq_pitch
        // eq_fov. No-op at its default (eq360==0) → engine returns immediately → byte-identical to
        // pre-P23. Applied on OUTB AFTER the P17 glitch, BEFORE the look, via
        // eq2rect(eq360, eq_yaw, eq_pitch, eq_fov). eq360 (1/0) = treat the clip as a 360x180 equirect
        // panorama and reproject to a flat rectilinear view; eq_yaw/eq_pitch = view yaw/pitch in degrees
        // (identity 0/0); eq_fov = horizontal field of view in degrees (default 90).
        eq360: i32,
        eq_yaw: f32,
        eq_pitch: f32,
        eq_fov: f32,
        // P34 per-clip SHAPE MASK (pinned wire order, after the P23 eq_fov): mask_shape mask_cx mask_cy
        // mask_rw mask_rh mask_feather mask_invert. No-op at its default (mask_shape==0) → engine returns
        // immediately → byte-identical to pre-P34. Applied on OUTB AFTER the P23 reframe, BEFORE the look,
        // via mask(mask_shape, mask_cx, mask_cy, mask_rw, mask_rh, mask_feather, mask_invert). mask_shape
        // (0=none 1=rect 2=ellipse) zeroes the pixels OUTSIDE a centred rect/ellipse with a feathered
        // edge; mask_cx/mask_cy = centre (0..1, identity 0.5/0.5); mask_rw/mask_rh = half-extents (0..1,
        // default 0.5/0.5); mask_feather = soft-edge band (default 0); mask_invert (1/0) flips it.
        mask_shape: i32,
        mask_cx: f32,
        mask_cy: f32,
        mask_rw: f32,
        mask_rh: f32,
        mask_feather: f32,
        mask_invert: i32,
        // P38 distortion batch (pinned wire order, after the P34 mask fields): mirror_x kaleido dither.
        // No-op at defaults (mirror_x 0 / kaleido <2 / dither 0) → engine skips → byte-identical to
        // pre-P38. Applied on OUTB AFTER the P34 mask, BEFORE the look, via mirror(mirror_x) ->
        // kaleido(kaleido) -> dither(dither). mirror_x (0/1) mirrors the left half onto the right;
        // kaleido (>=2) is an N-fold radial mirror; dither (0..1) is a Bayer-dithered posterize.
        mirror_x: i32,
        kaleido: i32,
        dither: f32,
        // P39 selective color (pinned wire order, after the P38 dither field): sel_band sel_hshift
        // sel_sat. No-op at its default (sel_band==0) → engine returns immediately → byte-identical to
        // pre-P39. Applied on OUTB AFTER the P38 dither, BEFORE the look, via selcolor(sel_band,
        // sel_hshift, sel_sat). sel_band (0=off 1=Red 2=Yellow 3=Green 4=Cyan 5=Blue 6=Magenta) selects
        // ONE hue band; sel_hshift (-1..1) rotates its hue; sel_sat (default 1.0) scales its saturation.
        sel_band: i32,
        sel_hshift: f32,
        sel_sat: f32,
        // P41 solarize + colour temperature (pinned wire order `sol_thr temp`, after the P39 sel_sat
        // field). Each is a no-op at its default (sol_thr<=0 / temp==0) → engine skips → byte-identical
        // to pre-P41. Applied on OUTB AFTER the P39 selective color, BEFORE the look, via solarize(thr)
        // then temp(t). sol_thr is the solarize threshold (0=off, (0,1]); temp is the warm/cool shift
        // (0=neutral, -1..1; t>0 warms by raising R / lowering B; green untouched).
        sol_thr: f32,
        temp: f32,
        // P45 VIDEO FADE: per-frame brightness factor in [0,1] (1.0 = no fade). Applied LAST on OUTB
        // (after the P41 solarize/temp), so the whole composed frame fades to black at clip head/tail.
        fade: f32,
    ) -> (Vec<u8>, bool) {
        let mut out = vec![0u8; GVW * GVH * 4];
        let fin = unsafe {
            fpx_gpu_track1(tt as c_int, trans_prog, trans_param); // transition (or -1 copy base)
            fpx_gpu_transform(rot, scale); // P2: rotate+scale the BASE frame (TRACK1), before pip
            // P4: chroma-key the OVER buffer (alpha only) BEFORE pip, ONLY when enabled — identity off.
            if ck_on != 0 {
                // P37: ck_spill is the final arg — 0 leaves green untouched (byte-identical to pre-P37).
                fpx_gpu_chroma(ck_r, ck_g, ck_b, ck_sim, ck_smooth, ck_spill);
            }
            fpx_gpu_pip(op, blend as c_int, px, py, pw, ph); // composite slot 1 over, into the PiP rect (P31 blend mode)
            fpx_gpu_grade_clip(cbright, ccontrast, csat); // PER-CLIP grade (in place on INB), P1
            fpx_gpu_grade(bright, contrast, sat); // PROGRAM grade, stacked on top
            // P2: 3-way color wheels (LGG) then gaussian blur, in place on OUTB, before look.
            fpx_gpu_lgg(lift_r, lift_g, lift_b, gamma_r, gamma_g, gamma_b, gain_r, gain_g, gain_b);
            fpx_gpu_blur(blur);
            fpx_gpu_curve(curve[0], curve[1], curve[2], curve[3], curve[4]); // P5 master tone curve
            // P6 stylize/utility, on OUTB after curve, before look: simplefx -> vignette -> sharpen -> flip.
            fpx_gpu_simplefx(fx as c_int);
            fpx_gpu_vignette(vig);
            fpx_gpu_sharpen(sharp);
            fpx_gpu_flip(flip as c_int);
            // P7 color filters, on OUTB after the P6 flip, before the look: hsl -> levels.
            fpx_gpu_hsl(hue, sat_hsl, light);
            fpx_gpu_levels(inb, inw, gam);
            // P8 stylize-2, on OUTB after the P7 levels, before the look: mosaic -> gradient map.
            fpx_gpu_mosaic(mosaic as c_int);
            fpx_gpu_gmap(gmap_amt, glo_r, glo_g, glo_b, ghi_r, ghi_g, ghi_b);
            // P9 fx, on OUTB after the P8 gradient map, before the look: denoise -> glow -> rgb-shift.
            fpx_gpu_denoise(denoise);
            fpx_gpu_glow(glow_amt, glow_thr);
            fpx_gpu_rgbshift(rgbshift);
            // P10 stylize-4, on OUTB after the P9 rgb-shift, before the look: halftone -> emboss -> edge.
            fpx_gpu_halftone(halftone as c_int);
            fpx_gpu_emboss(emboss);
            fpx_gpu_edge(edge);
            // P13 old-film/distort, on OUTB after the P10 edge, before the look: grain -> scratches -> diffusion.
            fpx_gpu_grain(grain);
            fpx_gpu_scratches(scratches);
            fpx_gpu_diffusion(diffusion);
            // P16 distort, on OUTB after the P13 diffusion, before the look: wave -> swirl -> threshold.
            fpx_gpu_wave(wave);
            fpx_gpu_swirl(swirl);
            fpx_gpu_threshold(threshold);
            // P17 geometric, on OUTB after the P16 threshold, before the look: lens -> crop -> glitch.
            fpx_gpu_lens(lens);
            fpx_gpu_crop(crop);
            fpx_gpu_glitch(glitch);
            // P23 360 reframe, on OUTB after the P17 glitch, before the look. eq360==0 = no-op (engine
            // returns immediately → byte-identical to pre-P23).
            fpx_gpu_eq2rect(eq360 as c_int, eq_yaw, eq_pitch, eq_fov);
            // P34 shape mask, on OUTB after the P23 reframe, before the look. mask_shape==0 = no-op
            // (engine returns immediately → byte-identical to pre-P34).
            fpx_gpu_mask(mask_shape as c_int, mask_cx, mask_cy, mask_rw, mask_rh, mask_feather, mask_invert as c_int);
            // P38 distortion batch, on OUTB after the P34 mask, before the look: mirror -> kaleido ->
            // dither. Each is a no-op at its default (mirror_x 0 / kaleido <2 / dither 0) → engine skips
            // → byte-identical to pre-P38.
            fpx_gpu_mirror(mirror_x as c_int);
            fpx_gpu_kaleido(kaleido as c_int);
            fpx_gpu_dither(dither);
            // P39 selective color, on OUTB after the P38 dither, before the look. sel_band==0 = no-op
            // (engine returns immediately → byte-identical to pre-P39).
            fpx_gpu_selcolor(sel_band as c_int, sel_hshift, sel_sat);
            // P41 solarize + colour temperature, on OUTB after the P39 selective color, before the look.
            // Each is a no-op at its default (sol_thr<=0 / temp==0) → engine skips → byte-identical to
            // pre-P41.
            fpx_gpu_solarize(sol_thr);
            fpx_gpu_temp(temp);
            // P45 video fade-to-black — the LAST OUTB op, so the whole composed frame fades. f>=1 = skip.
            fpx_gpu_fade(fade);
            let fin = fpx_gpu_look(look_kind as c_int, look_amt, lut_n as c_int);
            fpx_gpu_download_u8(fin, out.as_mut_ptr());
            fpx_gpu_finish();
            fin
        };
        (out, fin != 0)
    }

    /// f32 sibling of `compose_trans` (Wave 8): same transition-first pipeline, but downloads RGBA
    /// **f32** in [0,1] for `Encoder::video_frame`. Mirrors MojoMedia's render loop, which runs
    /// `track1(r_tt_id, rtt, r_tt_p)` → pip → grade → look → `download_f32` (main_editor.mojo
    /// ~1300-1308). Same args + `final_is_look` return as `compose_trans`.
    #[allow(clippy::too_many_arguments)]
    pub fn compose_trans_f32(
        &self,
        tt: i32,
        trans_prog: f32,
        trans_param: f32,
        op: f32,
        // P31 BLEND MODE of the OVER (V2) clip: 0=Normal 1=Multiply 2=Screen 3=Overlay 4=Add
        // 5=Darken 6=Lighten 7=Difference. Rides the wire IMMEDIATELY AFTER `op` (over-opacity).
        // blend==0 (Normal) => fpx_blend returns the over colour => byte-identical to pre-P31.
        blend: i32,
        px: f32,
        py: f32,
        pw: f32,
        ph: f32,
        cbright: f32,
        ccontrast: f32,
        csat: f32,
        bright: f32,
        contrast: f32,
        sat: f32,
        look_kind: i32,
        look_amt: f32,
        lut_n: i32,
        // P2 per-clip effects (pinned wire order: lift3, gamma3, gain3, rot, scale, blur).
        lift_r: f32,
        lift_g: f32,
        lift_b: f32,
        gamma_r: f32,
        gamma_g: f32,
        gamma_b: f32,
        gain_r: f32,
        gain_g: f32,
        gain_b: f32,
        rot: f32,
        scale: f32,
        blur: f32,
        // P4 per-clip CHROMA KEY on the OVER (V2) buffer (see `compose_trans`). `ck_on` (1/0) gates it;
        // 0 => OVER alpha untouched => byte-identical to P3.
        ck_on: i32,
        ck_r: f32,
        ck_g: f32,
        ck_b: f32,
        ck_sim: f32,
        ck_smooth: f32,
        // P37: green-spill suppression strength (0..1). spill==0 = no-op (byte-identical to pre-P37);
        // >0 pulls a kept pixel's green toward max(r,b) for a green-dominant key. Rides the wire as the
        // LAST f32 field (appended after the P34 mask fields), so the pre-P37 ck_* / mask indices are
        // unchanged. Threaded into fpx_gpu_chroma as its final arg.
        ck_spill: f32,
        // P5 master tone curve: 5 outputs at fixed inputs 0/.25/.5/.75/1 (identity = [0,.25,.5,.75,1]).
        curve: [f32; 5],
        // P6 stylize/utility (pinned wire order, after curve): vig sharp flip fx. All no-op at their
        // defaults (vig 0, sharp 0, flip 0, fx 0) → engine skips → byte-identical. Applied on OUTB
        // AFTER the curve, BEFORE the look, in order simplefx(fx) -> vignette(vig) -> sharpen(sharp)
        // -> flip(flip).
        vig: f32,
        sharp: f32,
        flip: i32,
        fx: i32,
        // P7 per-clip COLOR filters (pinned wire order, after the P6 fx): hue sat light inb inw gam.
        // All no-op at their defaults (hue 0, sat 1, light 0, inb 0, inw 1, gam 1) → engine skips →
        // byte-identical. Applied on OUTB AFTER the P6 flip, BEFORE the look, in order
        // hsl(hue,sat,light) -> levels(inb,inw,gam).
        hue: f32,
        sat_hsl: f32,
        light: f32,
        inb: f32,
        inw: f32,
        gam: f32,
        // P8 per-clip STYLIZE-2 filters (pinned wire order, after the P7 gam): mosaic gmap_amt glo3
        // ghi3. All no-op at their defaults (mosaic 0, gmap_amt 0) → engine skips → byte-identical.
        // Applied on OUTB AFTER the P7 levels, BEFORE the look, in order mosaic(mosaic) ->
        // gmap(gmap_amt, glo, ghi). `mosaic` is the block size in px (i32; 0/1 = off).
        mosaic: i32,
        gmap_amt: f32,
        glo_r: f32,
        glo_g: f32,
        glo_b: f32,
        ghi_r: f32,
        ghi_g: f32,
        ghi_b: f32,
        // P9 per-clip FX filters (pinned wire order, after the P8 ghi_b): denoise glow_amt glow_thr
        // rgbshift. All no-op at their defaults (denoise 0, glow_amt 0, rgbshift 0) → engine skips →
        // byte-identical. Applied on OUTB AFTER the P8 gradient map, BEFORE the look, in order
        // denoise(denoise) -> glow(glow_amt, glow_thr) -> rgbshift(rgbshift).
        denoise: f32,
        glow_amt: f32,
        glow_thr: f32,
        rgbshift: f32,
        // P10 per-clip STYLIZE-4 filters (pinned wire order, after the P9 rgbshift): halftone emboss
        // edge. All no-op at their defaults (halftone 0, emboss 0, edge 0) → engine skips →
        // byte-identical. Applied on OUTB AFTER the P9 rgb-shift, BEFORE the look, in order
        // halftone(halftone) -> emboss(emboss) -> edge(edge). `halftone` is the dot cell size in px
        // (i32; 0/1 = off — the model's u32 prints as a plain decimal that round-trips to i32).
        halftone: i32,
        emboss: f32,
        edge: f32,
        // P13 per-clip OLD-FILM/DISTORT filters (pinned wire order, after the P10 edge): grain
        // scratches diffusion. All no-op at their defaults (grain 0, scratches 0, diffusion 0) →
        // engine skips → byte-identical. Applied on OUTB AFTER the P10 edge, BEFORE the look, in order
        // grain(grain) -> scratches(scratches) -> diffusion(diffusion). grain=film-noise strength
        // 0..1; scratches=scratch density/amount 0..1; diffusion=jitter radius in px (0..16).
        grain: f32,
        scratches: f32,
        diffusion: f32,
        // P16 per-clip DISTORT filters (pinned wire order, after the P13 diffusion): wave swirl
        // threshold. All no-op at their defaults (wave 0, swirl 0, threshold 0) → engine skips →
        // byte-identical. Applied on OUTB AFTER the P13 diffusion, BEFORE the look, in order
        // wave(wave) -> swirl(swirl) -> threshold(threshold). wave=sinusoidal amplitude in px;
        // swirl=rotation strength in radians at the centre; threshold=luma binarize level 0..1.
        wave: f32,
        swirl: f32,
        threshold: f32,
        // P17 per-clip GEOMETRIC filters (pinned wire order, after the P16 threshold): lens crop
        // glitch. All no-op at their defaults (lens 0, crop 0, glitch 0) → engine skips →
        // byte-identical. Applied on OUTB AFTER the P16 threshold, BEFORE the look, in order
        // lens(lens) -> crop(crop) -> glitch(glitch). lens=radial coefficient (+barrel/-pincushion,
        // 0=off); crop=border-to-black fraction 0..0.49; glitch=max per-band horizontal shift in px.
        lens: f32,
        crop: f32,
        glitch: f32,
        // P23 per-clip 360 REFRAME (pinned wire order, after the P17 glitch): eq360 eq_yaw eq_pitch
        // eq_fov. No-op at its default (eq360==0) → engine returns immediately → byte-identical to
        // pre-P23. Applied on OUTB AFTER the P17 glitch, BEFORE the look, via
        // eq2rect(eq360, eq_yaw, eq_pitch, eq_fov). eq360 (1/0) = treat the clip as a 360x180 equirect
        // panorama and reproject to a flat rectilinear view; eq_yaw/eq_pitch = view yaw/pitch in degrees
        // (identity 0/0); eq_fov = horizontal field of view in degrees (default 90).
        eq360: i32,
        eq_yaw: f32,
        eq_pitch: f32,
        eq_fov: f32,
        // P34 per-clip SHAPE MASK (pinned wire order, after the P23 eq_fov): mask_shape mask_cx mask_cy
        // mask_rw mask_rh mask_feather mask_invert. No-op at its default (mask_shape==0) → engine returns
        // immediately → byte-identical to pre-P34. Applied on OUTB AFTER the P23 reframe, BEFORE the look,
        // via mask(mask_shape, mask_cx, mask_cy, mask_rw, mask_rh, mask_feather, mask_invert). mask_shape
        // (0=none 1=rect 2=ellipse) zeroes the pixels OUTSIDE a centred rect/ellipse with a feathered
        // edge; mask_cx/mask_cy = centre (0..1, identity 0.5/0.5); mask_rw/mask_rh = half-extents (0..1,
        // default 0.5/0.5); mask_feather = soft-edge band (default 0); mask_invert (1/0) flips it.
        mask_shape: i32,
        mask_cx: f32,
        mask_cy: f32,
        mask_rw: f32,
        mask_rh: f32,
        mask_feather: f32,
        mask_invert: i32,
        // P38 distortion batch (pinned wire order, after the P34 mask fields): mirror_x kaleido dither.
        // No-op at defaults (mirror_x 0 / kaleido <2 / dither 0) → engine skips → byte-identical to
        // pre-P38. Applied on OUTB AFTER the P34 mask, BEFORE the look, via mirror(mirror_x) ->
        // kaleido(kaleido) -> dither(dither). mirror_x (0/1) mirrors the left half onto the right;
        // kaleido (>=2) is an N-fold radial mirror; dither (0..1) is a Bayer-dithered posterize.
        mirror_x: i32,
        kaleido: i32,
        dither: f32,
        // P39 selective color (pinned wire order, after the P38 dither field): sel_band sel_hshift
        // sel_sat. No-op at its default (sel_band==0) → engine returns immediately → byte-identical to
        // pre-P39. Applied on OUTB AFTER the P38 dither, BEFORE the look, via selcolor(sel_band,
        // sel_hshift, sel_sat). sel_band (0=off 1=Red 2=Yellow 3=Green 4=Cyan 5=Blue 6=Magenta) selects
        // ONE hue band; sel_hshift (-1..1) rotates its hue; sel_sat (default 1.0) scales its saturation.
        sel_band: i32,
        sel_hshift: f32,
        sel_sat: f32,
        // P41 solarize + colour temperature (pinned wire order `sol_thr temp`, after the P39 sel_sat
        // field). Each is a no-op at its default (sol_thr<=0 / temp==0) → engine skips → byte-identical
        // to pre-P41. Applied on OUTB AFTER the P39 selective color, BEFORE the look, via solarize(thr)
        // then temp(t). sol_thr is the solarize threshold (0=off, (0,1]); temp is the warm/cool shift
        // (0=neutral, -1..1; t>0 warms by raising R / lowering B; green untouched).
        sol_thr: f32,
        temp: f32,
        // P45 VIDEO FADE: per-frame brightness factor in [0,1] (1.0 = no fade). Applied LAST on OUTB.
        fade: f32,
    ) -> (Vec<f32>, bool) {
        let mut out = vec![0f32; GVW * GVH * 4];
        let fin = unsafe {
            fpx_gpu_track1(tt as c_int, trans_prog, trans_param); // transition (or -1 copy base)
            fpx_gpu_transform(rot, scale); // P2: rotate+scale the BASE frame (TRACK1), before pip
            // P4: chroma-key the OVER buffer (alpha only) BEFORE pip, ONLY when enabled — identity off.
            if ck_on != 0 {
                // P37: ck_spill is the final arg — 0 leaves green untouched (byte-identical to pre-P37).
                fpx_gpu_chroma(ck_r, ck_g, ck_b, ck_sim, ck_smooth, ck_spill);
            }
            fpx_gpu_pip(op, blend as c_int, px, py, pw, ph); // composite slot 1 over, into the PiP rect (P31 blend mode)
            fpx_gpu_grade_clip(cbright, ccontrast, csat); // PER-CLIP grade (in place on INB), P1
            fpx_gpu_grade(bright, contrast, sat); // PROGRAM grade, stacked on top
            // P2: 3-way color wheels (LGG) then gaussian blur, in place on OUTB, before look.
            fpx_gpu_lgg(lift_r, lift_g, lift_b, gamma_r, gamma_g, gamma_b, gain_r, gain_g, gain_b);
            fpx_gpu_blur(blur);
            fpx_gpu_curve(curve[0], curve[1], curve[2], curve[3], curve[4]); // P5 master tone curve
            // P6 stylize/utility, on OUTB after curve, before look: simplefx -> vignette -> sharpen -> flip.
            fpx_gpu_simplefx(fx as c_int);
            fpx_gpu_vignette(vig);
            fpx_gpu_sharpen(sharp);
            fpx_gpu_flip(flip as c_int);
            // P7 color filters, on OUTB after the P6 flip, before the look: hsl -> levels.
            fpx_gpu_hsl(hue, sat_hsl, light);
            fpx_gpu_levels(inb, inw, gam);
            // P8 stylize-2, on OUTB after the P7 levels, before the look: mosaic -> gradient map.
            fpx_gpu_mosaic(mosaic as c_int);
            fpx_gpu_gmap(gmap_amt, glo_r, glo_g, glo_b, ghi_r, ghi_g, ghi_b);
            // P9 fx, on OUTB after the P8 gradient map, before the look: denoise -> glow -> rgb-shift.
            fpx_gpu_denoise(denoise);
            fpx_gpu_glow(glow_amt, glow_thr);
            fpx_gpu_rgbshift(rgbshift);
            // P10 stylize-4, on OUTB after the P9 rgb-shift, before the look: halftone -> emboss -> edge.
            fpx_gpu_halftone(halftone as c_int);
            fpx_gpu_emboss(emboss);
            fpx_gpu_edge(edge);
            // P13 old-film/distort, on OUTB after the P10 edge, before the look: grain -> scratches -> diffusion.
            fpx_gpu_grain(grain);
            fpx_gpu_scratches(scratches);
            fpx_gpu_diffusion(diffusion);
            // P16 distort, on OUTB after the P13 diffusion, before the look: wave -> swirl -> threshold.
            fpx_gpu_wave(wave);
            fpx_gpu_swirl(swirl);
            fpx_gpu_threshold(threshold);
            // P17 geometric, on OUTB after the P16 threshold, before the look: lens -> crop -> glitch.
            fpx_gpu_lens(lens);
            fpx_gpu_crop(crop);
            fpx_gpu_glitch(glitch);
            // P23 360 reframe, on OUTB after the P17 glitch, before the look. eq360==0 = no-op (engine
            // returns immediately → byte-identical to pre-P23).
            fpx_gpu_eq2rect(eq360 as c_int, eq_yaw, eq_pitch, eq_fov);
            // P34 shape mask, on OUTB after the P23 reframe, before the look. mask_shape==0 = no-op
            // (engine returns immediately → byte-identical to pre-P34).
            fpx_gpu_mask(mask_shape as c_int, mask_cx, mask_cy, mask_rw, mask_rh, mask_feather, mask_invert as c_int);
            // P38 distortion batch, on OUTB after the P34 mask, before the look: mirror -> kaleido ->
            // dither. Each is a no-op at its default (mirror_x 0 / kaleido <2 / dither 0) → engine skips
            // → byte-identical to pre-P38.
            fpx_gpu_mirror(mirror_x as c_int);
            fpx_gpu_kaleido(kaleido as c_int);
            fpx_gpu_dither(dither);
            // P39 selective color, on OUTB after the P38 dither, before the look. sel_band==0 = no-op
            // (engine returns immediately → byte-identical to pre-P39).
            fpx_gpu_selcolor(sel_band as c_int, sel_hshift, sel_sat);
            // P41 solarize + colour temperature, on OUTB after the P39 selective color, before the look.
            // Each is a no-op at its default (sol_thr<=0 / temp==0) → engine skips → byte-identical to
            // pre-P41.
            fpx_gpu_solarize(sol_thr);
            fpx_gpu_temp(temp);
            // P45 video fade-to-black — the LAST OUTB op, so the whole composed frame fades. f>=1 = skip.
            fpx_gpu_fade(fade);
            let fin = fpx_gpu_look(look_kind as c_int, look_amt, lut_n as c_int);
            fpx_gpu_download_f32(fin, out.as_mut_ptr());
            fpx_gpu_finish();
            fin
        };
        (out, fin != 0)
    }

    /// Same pipeline as `compose`, but downloads the result as RGBA **f32** in [0,1] — the
    /// exact buffer `Encoder::video_frame` (fpx_enc_video_frame_f32) expects. Mirrors
    /// MojoMedia's render loop, which feeds the encoder via `fpx_gpu_download_f32`. Same look
    /// arguments + `final_is_look` return as `compose`.
    // Retained no-transition f32 compose: the render path uses compose_trans_f32 (threads a real
    // transition kind); this hardcoded-no-transition variant is currently unused.
    #[allow(dead_code)]
    pub fn compose_f32(
        &self,
        op: f32,
        px: f32,
        py: f32,
        pw: f32,
        ph: f32,
        bright: f32,
        contrast: f32,
        sat: f32,
        look_kind: i32,
        look_amt: f32,
        lut_n: i32,
    ) -> (Vec<f32>, bool) {
        let mut out = vec![0f32; GVW * GVH * 4];
        let fin = unsafe {
            fpx_gpu_track1(-1, 0.0, 4.0); // no transition: copy base (slot 0)
            fpx_gpu_pip(op, 0, px, py, pw, ph); // composite slot 1 over, into the PiP rect (blend=0 Normal)
            fpx_gpu_grade(bright, contrast, sat);
            let fin = fpx_gpu_look(look_kind as c_int, look_amt, lut_n as c_int);
            fpx_gpu_download_f32(fin, out.as_mut_ptr());
            fpx_gpu_finish();
            fin
        };
        (out, fin != 0)
    }

    /// RGB histogram of the LAST composed buffer (`final_is_look`=0 reads g_buf[OUTB], =1 reads
    /// g_buf[LOOKB]). Returns `HIST_BINS` (768) int bins: indices 0..256 = R, 256..512 = G,
    /// 512..768 = B, each bin = pixel count at that 8-bit value. The C shim does NOT render these
    /// into an image (unlike waveform/vectorscope) — `main.rs` rasterizes the bins into a 256×256
    /// RGBA graph for the SCOPE command. `fpx_gpu_finish` is called so the blocking read is complete
    /// before we return the buffer.
    pub fn histogram(&self, final_is_look: bool) -> Vec<i32> {
        let mut bins = vec![0i32; HIST_BINS];
        unsafe {
            fpx_gpu_histogram(final_is_look as c_int, bins.as_mut_ptr());
            fpx_gpu_finish();
        }
        bins
    }

    /// GPU-rendered luma-waveform image of the LAST composed buffer -> RGBA8 SVW×SVH (256×256).
    /// Reads g_buf[OUTB] (final_is_look=false) or g_buf[LOOKB] (true). The C shim clears its grid,
    /// accumulates over the frame, and renders directly to a 256×256×4 byte image; we just receive
    /// it. The Genesis preview path always composes with look=none, so callers pass false.
    pub fn waveform(&self, final_is_look: bool) -> Vec<u8> {
        let mut out = vec![0u8; SVW * SVH * 4];
        unsafe {
            fpx_gpu_waveform(final_is_look as c_int, out.as_mut_ptr());
            fpx_gpu_finish();
        }
        out
    }

    /// GPU-rendered vectorscope (U/V scatter) image of the LAST composed buffer -> RGBA8 SVW×SVH
    /// (256×256). Reads g_buf[OUTB] (final_is_look=false) or g_buf[LOOKB] (true). Same direct-image
    /// path as `waveform`.
    pub fn vectorscope(&self, final_is_look: bool) -> Vec<u8> {
        let mut out = vec![0u8; SVW * SVH * 4];
        unsafe {
            fpx_gpu_vectorscope(final_is_look as c_int, out.as_mut_ptr());
            fpx_gpu_finish();
        }
        out
    }

    /// GPU-rendered RGB PARADE (Triad-B P1, scope kind 3) of the LAST composed buffer -> RGBA8
    /// SVW×SVH (256×256): three side-by-side per-channel column waveforms (R|G|B), value on the
    /// y-axis. Reads g_buf[OUTB] (final_is_look=false) or g_buf[LOOKB] (true). Same direct-image path
    /// as `waveform`/`vectorscope`, but its own 3-panel kernel + dedicated 3×256×256 grid.
    pub fn parade(&self, final_is_look: bool) -> Vec<u8> {
        let mut out = vec![0u8; SVW * SVH * 4];
        unsafe {
            fpx_gpu_parade(final_is_look as c_int, out.as_mut_ptr());
            fpx_gpu_finish();
        }
        out
    }
}

/// A live encoder/muxer over the fpx_encode.c shim. Configured for video (and optionally
/// audio), then fed RGBA f32 frames in pts order, then finished. Closes on drop.
///
/// Call order (enforced by the type's method sequence, mirroring MojoMedia render):
///   `Encoder::open` -> `config_video` -> [`config_audio`] -> `start`
///   -> `video_frame(..)*` [ -> `audio_samples(..)` ] -> `finish` -> (drop closes).
pub struct Encoder {
    h: *mut c_void,
    in_w: usize,
    in_h: usize,
}

impl Encoder {
    /// Allocate the output container for `url` (e.g. "/tmp/out.mp4"). None on failure.
    pub fn open(url: &str) -> Option<Encoder> {
        let c = CString::new(url).ok()?;
        let h = unsafe { fpx_enc_open(c.as_ptr()) };
        if h.is_null() {
            None
        } else {
            Some(Encoder { h, in_w: 0, in_h: 0 })
        }
    }

    /// Configure the video stream. `in_w/in_h` = source RGBA dims fed per frame; `width/height`
    /// = encoded dims (usually equal). Returns true on success (stream index >= 0).
    ///
    /// P25 export depth: `gop` is the keyframe interval in frames (<=0 leaves the encoder's default
    /// gop_size untouched) and `preset` is the x264/x265 encoder preset (empty string => set no
    /// preset). Both are applied on the video codec context BEFORE `avcodec_open2`. Defaults
    /// (gop=0, preset="") reproduce the pre-P25 encode byte-for-byte.
    #[allow(clippy::too_many_arguments)]
    pub fn config_video(
        &mut self,
        codec: &str,
        in_w: usize,
        in_h: usize,
        width: usize,
        height: usize,
        fps_num: i32,
        fps_den: i32,
        bit_rate: i64,
        gop: i32,
        preset: &str,
    ) -> bool {
        let c = match CString::new(codec) {
            Ok(c) => c,
            Err(_) => return false,
        };
        // The CString must outlive the FFI call — bind it (don't pass a temporary's as_ptr()). An
        // empty preset yields a valid pointer to a lone NUL byte; the C side checks preset[0], so ""
        // is treated as "no preset".
        let p = match CString::new(preset) {
            Ok(p) => p,
            Err(_) => return false,
        };
        let rc = unsafe {
            fpx_enc_config_video(
                self.h,
                c.as_ptr(),
                in_w as c_int,
                in_h as c_int,
                width as c_int,
                height as c_int,
                fps_num as c_int,
                fps_den as c_int,
                bit_rate as c_longlong,
                gop as c_int,
                p.as_ptr(),
            )
        };
        if rc >= 0 {
            self.in_w = in_w;
            self.in_h = in_h;
            true
        } else {
            false
        }
    }

    /// Configure the audio stream (interleaved float input). Returns true on success.
    pub fn config_audio(&mut self, codec: &str, channels: i32, sample_rate: i32, bit_rate: i64) -> bool {
        let c = match CString::new(codec) {
            Ok(c) => c,
            Err(_) => return false,
        };
        let rc = unsafe {
            fpx_enc_config_audio(
                self.h,
                c.as_ptr(),
                channels as c_int,
                sample_rate as c_int,
                bit_rate as c_longlong,
            )
        };
        rc >= 0
    }

    /// Set constant-quality (CRF) rate control. Call AFTER `config_video` and BEFORE `start` for a
    /// rate_mode=1 export. `crf` is the quality value (lower = better). Returns true on success;
    /// best-effort — a false return leaves the average-bitrate config in place (still a valid encode).
    ///
    /// P25 export depth: the CRF path RE-OPENS the video codec context, so the `gop`/`preset` that
    /// `config_video` set on the old context would be LOST — they must be re-applied here. `gop`<=0
    /// carries forward the gop_size config_video stamped; an empty `preset` sets none. With defaults
    /// (gop=0, preset="") the re-opened CRF context matches the pre-P25 CRF encode.
    pub fn set_quality(&mut self, crf: i32, gop: i32, preset: &str) -> bool {
        // Bind the CString so it outlives the FFI call; "" => valid lone-NUL pointer (preset[0]==0).
        let p = match CString::new(preset) {
            Ok(p) => p,
            Err(_) => return false,
        };
        unsafe { fpx_enc_set_quality(self.h, crf as c_int, gop as c_int, p.as_ptr()) >= 0 }
    }

    /// Write the container header. Must be called after config and before any frame. true=ok.
    pub fn start(&mut self) -> bool {
        unsafe { fpx_enc_start(self.h) >= 0 }
    }

    /// Encode one RGBA f32 [0,1] frame (`in_w*in_h*4` floats) at timestamp `ts_sec`. true=ok.
    pub fn video_frame(&mut self, rgba_f32: &[f32], ts_sec: f64) -> bool {
        debug_assert_eq!(rgba_f32.len(), self.in_w * self.in_h * 4);
        let rc = unsafe {
            fpx_enc_video_frame_f32(
                self.h,
                rgba_f32.as_ptr(),
                self.in_w as c_int,
                self.in_h as c_int,
                ts_sec as c_double,
            )
        };
        rc >= 0
    }

    /// Feed `nb` interleaved-float samples-per-channel (`samples.len() == nb*channels`). true=ok.
    ///
    /// Wired by the AUDIO serve command (program audio): the worker decodes each clip's audio
    /// range with `decode_audio_range` (2ch @ 48k) and feeds the interleaved floats here, passing
    /// `nb = floats / channels` (mirrors MojoMedia's `fpx_enc_audio_samples_f32(e, audmix,
    /// prog_floats // 2)`).
    pub fn audio_samples(&mut self, samples: &[f32], nb: usize) -> bool {
        if nb == 0 {
            return true; // nothing to feed is not a failure (empty clip range).
        }
        let rc = unsafe { fpx_enc_audio_samples_f32(self.h, samples.as_ptr(), nb as c_int) };
        rc >= 0
    }

    /// Flush encoders + write the trailer. Call exactly once before drop. true=ok.
    pub fn finish(&mut self) -> bool {
        unsafe { fpx_enc_finish(self.h) >= 0 }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        unsafe { fpx_enc_close(self.h) };
    }
}

/// A parsed 3D LUT: the grid edge `n` (so the data is `n*n*n*3` interleaved RGB floats in [0,1])
/// plus the float payload. Produced by `load_cube`, consumed by `Gpu::upload_lut`. Cached per
/// `.cube` path by the serve loop so repeated frames with the same look don't reparse the file.
#[derive(Clone)]
pub struct Lut {
    pub n: usize,
    pub data: Vec<f32>,
}

/// Parse a `.cube` 3D LUT file via the C `fpx_load_cube` shim. Returns the loaded `Lut` (grid N +
/// `N^3*3` floats) on success, or None on any failure — a missing/malformed/too-large/1D LUT.
/// The caller treats None as "no look" (degrade gracefully, never fail the frame). The staging
/// buffer is sized to the shim's `MAX_LUT_FLOATS` capacity; on success it is truncated to the
/// exact `N^3*3` floats the LUT used.
pub fn load_cube(path: &str) -> Option<Lut> {
    let c = CString::new(path).ok()?;
    let mut data = vec![0f32; MAX_LUT_FLOATS];
    let n = unsafe { fpx_load_cube(c.as_ptr(), data.as_mut_ptr(), MAX_LUT_FLOATS as c_int) };
    if n <= 0 {
        // negative = parse/open error; 0 should never happen (C returns N>0 or negative).
        return None;
    }
    let n = n as usize;
    let want = n.checked_mul(n)?.checked_mul(n)?.checked_mul(3)?;
    // Defensive: the C side guarantees count == N^3*3 <= max_floats before returning N, but clamp
    // anyway so a future C change can't hand us a length we can't slice.
    if want == 0 || want > data.len() {
        return None;
    }
    data.truncate(want);
    Some(Lut { n, data })
}

/// Whole-track peak-amplitude envelope: `buckets` peaks in [0,1] across the file's audio.
/// Returns None if the file has no audio / can't be read.
pub fn audio_envelope(path: &str, buckets: usize) -> Option<Vec<f32>> {
    if buckets == 0 {
        return None;
    }
    let c = CString::new(path).ok()?;
    let mut out = vec![0f32; buckets];
    let rc = unsafe { fpx_audio_envelope(c.as_ptr(), buckets as c_int, out.as_mut_ptr()) };
    // C returns nbuckets on success, 0 if the file has no audio stream, negative on error.
    if rc as usize == buckets {
        Some(out)
    } else {
        None
    }
}

/// Decode `[start_sec, start_sec+dur_sec)` of `path`'s audio -> interleaved f32 (`out_ch`),
/// resampled to `out_sr`. Returns the decoded samples (length = floats written), or an empty
/// Vec if the file has no audio. None only on a hard error.
///
/// Wired by the AUDIO serve command (program audio): the render path decodes each clip's source
/// range with this and feeds the floats to `Encoder::audio_samples`. The C side
/// (`fpx_decode_audio_range`) returns the number of FLOATS written (frames * out_ch), 0 when the
/// file has no audio stream, negative on a hard error — mirrored here.
pub fn decode_audio_range(
    path: &str,
    start_sec: f64,
    dur_sec: f64,
    out_sr: i32,
    out_ch: i32,
    cap: usize,
) -> Option<Vec<f32>> {
    if cap == 0 {
        return Some(Vec::new());
    }
    // Guard the usize -> c_int narrowing (finding #4): the C contract takes `cap` as a c_int, so a
    // `cap` above c_int::MAX would wrap to a negative/small value and either be rejected by C's
    // `cap <= 0` guard or silently truncate the decoded audio. Callers already clamp (see
    // audio_feed's CAP_MAX), but clamp here too so this wrapper is sound for ANY caller. We shrink
    // the requested cap to the largest value the c_int can carry rather than over-allocating.
    let cap = cap.min(c_int::MAX as usize);
    let c = CString::new(path).ok()?;
    let mut buf = vec![0f32; cap];
    let rc = unsafe {
        fpx_decode_audio_range(
            c.as_ptr(),
            start_sec as c_double,
            dur_sec as c_double,
            out_sr as c_int,
            out_ch as c_int,
            buf.as_mut_ptr(),
            cap as c_int,
        )
    };
    if rc < 0 {
        return None; // hard error (open/decode/resample failure)
    }
    // rc == 0 means "no audio stream" -> an empty Vec (caller skips the clip, doesn't abort).
    buf.truncate(rc as usize);
    Some(buf)
}

/// Apply a libavfilter `chain` (NO spaces; commas between filters, `=`/`:` inside) to `input`
/// (interleaved-float, `ch` channels, `nb` samples-per-channel) via the C `fpx_au_apply` shim.
/// Returns the FILTERED interleaved-float samples (length = out_frames * ch), or None on a hard
/// filter-graph error (bad chain / alloc fail) so the caller can fall back to the UNFILTERED input.
///
/// `chain` should be a real filter expression; an empty string is a pass-through (the C side maps
/// it to `anull`), but the P3 caller only ever calls this when the chain is non-trivial. The output
/// can have a DIFFERENT sample count than the input (loudnorm/acompressor add latency or trim), so
/// the returned Vec is truncated to exactly the floats the filter produced.
///
/// CAPACITY: a generous headroom over the input length is allocated (`nb*ch` + 1 s + slack) so a
/// filter that lengthens the stream isn't truncated for the common per-clip range. A filter that
/// produces MORE than that headroom has its tail clamped by the C side (`(total+n)*ch <= out_cap`),
/// which for these effects is inaudible (sub-frame mixing tail); the returned count still matches
/// the bytes actually written.
pub fn au_apply(chain: &str, samples: &[f32], sr: i32, ch: i32) -> Option<Vec<f32>> {
    if ch <= 0 || samples.is_empty() {
        return Some(samples.to_vec()); // nothing to filter; pass through.
    }
    let ch_us = ch as usize;
    let nb = samples.len() / ch_us; // samples-per-channel
    if nb == 0 {
        return Some(samples.to_vec());
    }
    let c = CString::new(chain).ok()?;
    // Output headroom: input frames + 1 s of slack (covers filter latency) per channel, clamped to
    // a c_int. Bounds the temp buffer and keeps the `as c_int` narrowing lossless and positive.
    let extra_frames = sr.max(0) as usize; // ~1 s of slack at this sample rate
    let cap_frames = nb.saturating_add(extra_frames).saturating_add(4096);
    let cap = cap_frames.saturating_mul(ch_us).min(c_int::MAX as usize);
    let mut out = vec![0f32; cap];
    let rc = unsafe {
        fpx_au_apply(
            sr as c_int,
            ch as c_int,
            c.as_ptr(),
            samples.as_ptr(),
            nb as c_int,
            out.as_mut_ptr(),
            cap as c_int,
        )
    };
    if rc < 0 {
        return None; // hard graph error: caller falls back to the unfiltered range.
    }
    // rc = OUTPUT samples-per-channel; truncate to the floats actually produced (clamp to cap so a
    // filter reporting more than it wrote — shouldn't happen — never over-reads the buffer).
    let out_floats = (rc as usize).saturating_mul(ch_us).min(out.len());
    out.truncate(out_floats);
    Some(out)
}

/// An open media decoder handle. Closes on drop.
///
/// Holds a raw `*mut c_void` (the C decoder handle). `Decoder` is NOT `Hash`/`Eq` itself, but
/// it is fine to store in a `HashMap<String, Decoder>` keyed by the media path — that is how the
/// persistent serve loop caches one open handle per file and reuses it for repeated frames.
pub struct Decoder {
    h: *mut c_void,
}

impl Decoder {
    /// Open `path`. Returns None if the file can't be opened.
    pub fn open(path: &str) -> Option<Decoder> {
        let c = CString::new(path).ok()?;
        let h = unsafe { fpx_open(c.as_ptr()) };
        if h.is_null() {
            None
        } else {
            Some(Decoder { h })
        }
    }

    /// Decode `frame_index` letterboxed into a fresh `w*h*4` RGBA8 buffer. Returns the pixel buffer,
    /// or None on decode failure.
    ///
    /// Frame-accurate (the C returns the frame whose pts >= target). The C now ALSO skips the
    /// seek-to-keyframe when `frame_index` advances forward within a bounded window of the decoder's
    /// current position (the sequential playback/render case), decoding forward via a drain-first
    /// loop instead — so this stays a thin wrapper while the per-frame re-seek cost is gone. Random
    /// access / backward scrubbing still seeks. See `fpx_decode_frame_letterbox` in fpx_decode.c.
    pub fn decode_rgba(&mut self, frame_index: i32, w: usize, h: usize) -> Option<Vec<u8>> {
        let mut buf = vec![0u8; w * h * 4];
        let rc = unsafe {
            fpx_decode_frame_letterbox(self.h, frame_index, buf.as_mut_ptr(), w as c_int, h as c_int)
        };
        if rc >= 0 {
            Some(buf)
        } else {
            None
        }
    }

    /// Total video frame count of the open media (0 if unknown). Used to clamp a clip's length so it
    /// never references source frames past the media end.
    pub fn nframes(&self) -> i32 {
        unsafe { fpx_nframes(self.h) }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe { fpx_close(self.h) };
    }
}
