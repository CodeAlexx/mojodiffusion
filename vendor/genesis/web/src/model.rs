//! Project data model — the Rust win over MojoMedia's flat parallel lists: real structs.
//!
//! SHARED CONTRACT. Owned by the timeline/model team; consumed by worker + app + pool.
//! Frame units are timeline frames (30 fps assumed for now).

#[derive(Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Clip {
    pub media: usize,  // index into Project.media
    pub src_in: i64,   // source in-point (frames)
    pub len: i64,      // length on the timeline (frames)
    pub t0: i64,       // timeline start (frames)
    pub track: u8,     // 0 = V1, 1 = V2, 2 = A1
    pub look: i32,     // per-clip LOOK index (0 = none)
    pub look_amt: f32, // look mix 0..1
    pub fade_in: i64,
    pub fade_out: i64,
    pub px: f32, // PiP rect (fractions of frame)
    pub py: f32,
    pub pw: f32,
    pub ph: f32,
    // Per-clip LUT path for look == 2 (LUT3D); empty = none. PINNED this wave: produced here
    // (Team B), consumed by Team A (engine look — loaded via fpx_load_cube + uploaded with
    // fpx_gpu_upload_lut, cached per path) and Team C (Look picker UI). `#[serde(default)]` so
    // pre-LUT .json projects still deserialize (the field defaults to "" = no LUT). Clip.look
    // semantics: 0 = None, 1 = VHS, 2 = LUT3D (uses this `lut`).
    #[serde(default)]
    pub lut: String,

    // ----- Triad-B P1 per-clip AUDIO + COLOR (all #[serde(default ..)] so pre-P1 .json loads) -----
    // Per-clip audio gain as a LINEAR multiplier (1.0 = unity). Surfaced in the properties panel as
    // a dB slider (Shotcut "Gain / Volume" range −70..+24 dB → linear), but stored linear so the
    // worker can pass it straight onto the AUDIO wire line. The clip's fade_in/fade_out (already
    // used for VIDEO opacity) ALSO ramp this gain 0→1 / 1→0 at the clip edges (applied in gcompose
    // at mix time). Defaults to 1.0 via `default_gain` so a clip with no stored gain is unchanged.
    #[serde(default = "default_gain")]
    pub gain: f32,
    // Per-clip color grade, ADDITIVE on top of the program grade (grade_at). Same kernels/semantics
    // as the program grade: `bright` is added (−1..1), `contrast`/`sat` are multipliers (0..2, 1.0 =
    // identity). resolve_frame combines per-clip with program (see worker::resolve_frame for the
    // documented order). Defaults reproduce a neutral grade (0/1/1) so an un-graded clip is a no-op.
    #[serde(default)]
    pub bright: f32,
    #[serde(default = "default_one")]
    pub contrast: f32,
    #[serde(default = "default_one")]
    pub sat: f32,

    // ----- Triad-B P2 per-clip COLOR-WHEELS (LIFT/GAMMA/GAIN) + TRANSFORM + BLUR -----
    // PINNED P2 wire extension: these 9 + 3 scalar values are FOLDED+APPENDED to the ENC/PREVIEW
    // lines (after csat) so the engine's new fpx_gpu_lgg / fpx_gpu_transform / fpx_gpu_blur kernels
    // apply them per-clip. All `#[serde(default ..)]` with IDENTITY defaults so pre-P2 .json projects
    // load unchanged AND reproduce the current render (the engine no-ops at identity). Mirrors
    // Shotcut's movit.lift_gamma_gain (Color Grading), white balance, rotate, and blur_gaussian.
    //
    // 3-WAY COLOR WHEELS. Engine semantics (PINNED Team A): per channel
    //   out = pow(clamp(in*gain + lift, 0, 1), 1/gamma).
    // `lift` is an additive shadow offset (Shotcut lift_r = wheel.redF*2-1, range −1..1, def 0).
    // `gamma` is a midtone power (Shotcut gamma factor V0 = 2 → range 0..2, def 1).
    // `gain_rgb` is a highlight multiplier (Shotcut gain factor V0 = 4 → range 0..4, def 1).
    // NOTE: named `gain_rgb` to stay distinct from the P1 AUDIO `gain` (linear audio multiplier).
    #[serde(default = "default_lift")]
    pub lift: [f32; 3], // R,G,B additive lift, identity [0,0,0]
    #[serde(default = "default_gamma")]
    pub gamma: [f32; 3], // R,G,B gamma power, identity [1,1,1]
    #[serde(default = "default_gain_rgb")]
    pub gain_rgb: [f32; 3], // R,G,B highlight gain, identity [1,1,1]

    // WHITE BALANCE (NOT a wire field — folded into gain_rgb by the worker). `wb_temp` is a warm/
    // cool bias in [−1,1] (0 = neutral; >0 warmer → boosts gain_r, cuts gain_b; mirrors Shotcut's
    // color_temperature mapped about 6500 K). `wb_tint` is a green/magenta bias in [−1,1] (0 =
    // neutral; >0 greener → boosts gain_g, cuts gain_r/gain_b). The worker's resolve_frame folds
    // both into the 9 lift/gamma/gain values it sends, so the ENGINE only ever sees lift/gamma/gain.
    #[serde(default)]
    pub wb_temp: f32, // −1..1, def 0
    #[serde(default)]
    pub wb_tint: f32, // −1..1, def 0

    // TRANSFORM of the base frame (Shotcut rotate). `rot` is rotation in DEGREES about the frame
    // center (range −180..180 in the UI; def 0). `scale` is a uniform zoom about the center (UI
    // 0.1..4; def 1). Engine `fpx_gpu_transform(rot_deg, scale)` bilinear-samples; identity at 0/1.
    #[serde(default)]
    pub rot: f32, // degrees, def 0
    #[serde(default = "default_one")]
    pub scale: f32, // zoom, def 1

    // GAUSSIAN BLUR sigma (Shotcut blur_gaussian av.sigma). UI 0..~20; def 0 = no blur. Engine
    // `fpx_gpu_blur(sigma)` runs a separable gaussian; sigma <= 0 is a no-op.
    #[serde(default)]
    pub blur: f32, // sigma, def 0

    // ----- P3 per-clip AUDIO FX (consumed by the audio triad: worker builds a libavfilter chain
    // from these and passes it to gcompose's fpx_au_apply). All-neutral default = no audio change,
    // so pre-P3 projects load + render identically. Structured (not a raw filter string) so the UI
    // can present sliders/toggles; the worker maps them to volume/pan/equalizer/acompressor/agate/
    // loudnorm. Mirrors Shotcut's audio_gain/audio_pan/audio_eq3band/compressor/noisegate/normalize.
    #[serde(default)]
    pub audio_fx: AudioFx,

    // ----- P4 per-clip CHROMA KEY (consumed by the chroma triad: worker sends the key params on the
    // wire; gcompose's k_chroma zeroes the OVER clip's alpha where the pixel matches the key color so
    // pip composites only the non-keyed pixels over V1). Disabled by default = no change, so pre-P4
    // projects render identically. Mirrors Shotcut's bluescreen0r (Chroma Key: Simple).
    #[serde(default)]
    pub chroma: ChromaKey,

    // ----- P5 TEXT / TITLE overlay (consumed by the text triad: the worker rasterizes `title.text`
    // with ab_glyph into a full-frame transparent RGBA and composites it over the clip's frame).
    // Empty text = no title (default) so pre-P5 projects are unchanged. Mirrors Shotcut's
    // dynamictext (Text: Simple) filter.
    #[serde(default)]
    pub title: Title,

    // ----- P5 CURVE: a 5-point master tone curve (Shotcut Curves). The 5 outputs are at fixed
    // inputs 0, 0.25, 0.5, 0.75, 1.0; the engine piecewise-linear interpolates and applies it to all
    // 3 channels after blur, before look. Default = identity (y=x) so an un-curved clip is a no-op.
    #[serde(default = "default_curve")]
    pub curve: [f32; 5],

    // ----- P6 STYLIZE / UTILITY filters (consumed by the filter triad; all per-pixel/spatial on the
    // composited OUTB after the curve, before look). All defaults are no-ops so pre-P6 projects are
    // unchanged. Mirror Shotcut's vignette / sharpen / flip / invert+sepia+grayscale+posterize.
    #[serde(default)]
    pub vignette: f32, // 0 = off; 0..1 darkens the frame edges radially
    #[serde(default)]
    pub sharpen: f32, // 0 = off; unsharp amount (~0..2)
    #[serde(default)]
    pub flip: u8, // 0 none, 1 horizontal, 2 vertical, 3 both
    #[serde(default)]
    pub fx: i32, // simple per-pixel FX: 0 none, 1 invert, 2 sepia, 3 grayscale, 4 posterize

    // ----- P7 COLOR filters (consumed by the color triad; per-pixel on OUTB after the P6 filters,
    // before look). Identity defaults = no-ops. Mirror Shotcut's hue/lightness/saturation + levels.
    #[serde(default = "default_hsl")]
    pub hsl: [f32; 3], // [hue_shift_degrees (0), saturation_mult (1), lightness_add (0)]
    #[serde(default = "default_levels")]
    pub levels: [f32; 3], // [in_black (0), in_white (1), gamma (1)]

    // ----- P8 STYLIZE filters (consumed by the P8 triad; on OUTB after the P7 color filters, before
    // look). Identity defaults = no-ops. Mirror Shotcut's mosaic (pixelate) + gradient-map.
    #[serde(default)]
    pub mosaic: u32, // block size in px; 0 or 1 = off (no pixelation)
    #[serde(default)]
    pub gmap_amt: f32, // gradient-map mix 0..1; 0 = off
    #[serde(default = "default_zero3")]
    pub gmap_lo: [f32; 3], // shadow colour (luma 0), default black
    #[serde(default = "default_one3")]
    pub gmap_hi: [f32; 3], // highlight colour (luma 1), default white

    // ----- P9 STYLIZE-3 / FX filters (consumed by the P9 wave; on OUTB after the P8 stylize filters,
    // before look). Identity defaults = no-ops (engine skips each at its off value). Mirror Shotcut's
    // Reduce-Noise (denoise), Glow, and RGB-Shift (chromatic aberration).
    #[serde(default)]
    pub denoise: f32, // edge-preserving denoise strength 0..1; 0 = off
    #[serde(default)]
    pub glow_amt: f32, // glow/bloom mix 0..1; 0 = off
    #[serde(default = "default_glow_thr")]
    pub glow_thr: f32, // glow luma threshold (only pixels brighter than this bloom); default 0.7
    #[serde(default)]
    pub rgbshift: f32, // RGB-shift / chromatic-aberration offset in px (R +shift, B -shift); 0 = off

    // ----- P10 STYLIZE-4 filters (consumed by the P10 wave; on OUTB after the P9 FX filters, before
    // look). Identity defaults = no-ops (engine skips each at its off value). Mirror Shotcut's
    // Halftone, Emboss, and Sketch/Edge-detect.
    #[serde(default)]
    pub halftone: u32, // halftone cell size in px; 0 or 1 = off
    #[serde(default)]
    pub emboss: f32, // emboss relief strength 0..1; 0 = off
    #[serde(default)]
    pub edge: f32, // edge-detect (sketch) mix 0..1; 0 = off

    // ----- P13 OLD-FILM / DISTORT filters (consumed by the P13 wave; on OUTB after the P10 stylize-4
    // filters, before look). Identity defaults = no-ops (engine skips each at its off value). Mirror
    // Shotcut's Old Film: Grain, Old Film: Scratches, and a Diffusion (frosted-glass) distort.
    #[serde(default)]
    pub grain: f32, // film-grain noise strength 0..1; 0 = off
    #[serde(default)]
    pub scratches: f32, // old-film vertical scratch density/amount 0..1; 0 = off
    #[serde(default)]
    pub diffusion: f32, // diffusion / frosted-glass radius in px (0..16); 0 = off

    // ----- P16 DISTORT filters (consumed by the P16 wave; on OUTB after the P13 old-film filters,
    // before look). Identity defaults = no-ops (engine skips each at its off value). Mirror Shotcut's
    // Wave, Swirl, and Threshold distort/stylize filters.
    #[serde(default)]
    pub wave: f32, // sinusoidal wave displacement amplitude in px; 0 = off
    #[serde(default)]
    pub swirl: f32, // swirl rotation strength in radians (at centre); 0 = off
    #[serde(default)]
    pub threshold: f32, // luma threshold/binarize level 0..1; 0 = off

    // ----- P17 GEOMETRIC/DISTORT filters (consumed by the P17 wave; on OUTB after the P16 distort
    // filters, before look). Identity defaults = no-ops. Mirror Shotcut's Lens Correction, Crop, and
    // a Glitch (per-band channel shift).
    #[serde(default)]
    pub lens: f32, // lens distortion: + barrel / - pincushion (radial); 0 = off
    #[serde(default)]
    pub crop: f32, // crop margin fraction 0..0.49 (outside -> black); 0 = off
    #[serde(default)]
    pub glitch: f32, // glitch per-band horizontal channel shift, max px; 0 = off

    // ----- P23 360 REFRAME (consumed by the P23 wave; on OUTB after the P17 geometry filters, before
    // look). When `eq360` is true the clip is treated as a 360 equirectangular source and reprojected
    // to a flat rectilinear view at (eq_yaw, eq_pitch) with field-of-view eq_fov — the standard
    // equirect->rectilinear "360 viewer" (mirrors Shotcut/bigsh0t's 360 reframe). eq360=false = no-op.
    #[serde(default)]
    pub eq360: bool, // enable 360 equirectangular -> rectilinear reprojection
    #[serde(default)]
    pub eq_yaw: f32, // view yaw (degrees), 0 = forward
    #[serde(default)]
    pub eq_pitch: f32, // view pitch (degrees), 0 = level
    #[serde(default = "default_eq_fov")]
    pub eq_fov: f32, // view field-of-view (degrees), default 90

    // ----- P24 CLIP SPEED / TIME-REMAP + REVERSE (consumed by the P24 wave). Model A: the clip keeps
    // its timeline footprint (t0,len); `speed` scales how fast the SOURCE is consumed (2.0 = 2x faster
    // / reads every other source frame; 0.5 = slow-mo), and `reverse` plays the consumed source range
    // backward. Identity speed=1.0 + reverse=false reads src_in+(t-t0) exactly (byte-identical).
    #[serde(default = "default_speed")]
    pub speed: f32, // source consumption rate; 1.0 = normal, 2.0 = 2x faster, 0.5 = slow-mo
    #[serde(default)]
    pub reverse: bool, // play the consumed source range backward

    // ----- P49 NESTED SEQUENCE (compound clip). -1 (default) = a normal media clip (decodes
    // `media`). >=0 = a COMPOUND clip whose source is `Project.subseqs[seq]` (a sub-timeline): at
    // render the worker composes that sub-sequence's frame at the clip's inner time and feeds it as a
    // RAW: layer. serde-default -1 so pre-P49 .json loads as all-normal clips (byte-identical).
    #[serde(default = "default_seq")]
    pub seq: i32,

    // ----- P31 BLEND MODE (consumed by the P31 wave). When this clip is the V2 OVERLAY, its RGB is
    // combined with the V1 base by this blend mode before the alpha-over composite. 0 = Normal
    // (plain alpha-over, byte-identical to pre-P31). 1=Multiply 2=Screen 3=Overlay 4=Add 5=Darken
    // 6=Lighten 7=Difference. Only meaningful for the overlay clip; a base/single clip ignores it.
    #[serde(default)]
    pub blend_mode: u8,

    // ----- P34 SHAPE MASK (consumed by the P34 wave; on OUTB after the P17/P23 geometry filters,
    // before the look). mask_shape 0=none (no-op, byte-identical) / 1=rectangle / 2=ellipse, centred
    // at (mask_cx,mask_cy) with half-extents (mask_rw,mask_rh) in normalized [0,1] frame coords;
    // mask_feather softens the edge; mask_invert flips inside/outside. Pixels OUTSIDE the kept region
    // are zeroed to black (like the P17 crop, but shaped + feathered).
    #[serde(default)]
    pub mask_shape: u8,
    #[serde(default = "default_half")]
    pub mask_cx: f32,
    #[serde(default = "default_half")]
    pub mask_cy: f32,
    #[serde(default = "default_half")]
    pub mask_rw: f32,
    #[serde(default = "default_half")]
    pub mask_rh: f32,
    #[serde(default)]
    pub mask_feather: f32,
    #[serde(default)]
    pub mask_invert: bool,

    // ----- P38 DISTORT BATCH (consumed by the P38 wave; on OUTB after the P34 mask, before the look).
    // Each is a no-op at its default (byte-identical). mirror_x: mirror the LEFT half onto the right
    // (1=on/0=off). kaleido: N-fold radial kaleidoscope segments (0 or 1 = off; >=2 = segment count).
    // dither: ordered 4x4 Bayer dither strength 0..1 (0 = off; reduces colour banding).
    #[serde(default)]
    pub mirror_x: u8,
    #[serde(default)]
    pub kaleido: i32,
    #[serde(default)]
    pub dither: f32,

    // ----- P39 SELECTIVE COLOR (consumed by the P39 wave; on OUTB after the P38 distort, before the
    // look). Adjust only one HUE BAND. sel_band 0=off (no-op) / 1=Reds 2=Yellows 3=Greens 4=Cyans
    // 5=Blues 6=Magentas; sel_hshift = hue rotation applied to that band (-1..1 = -180..180 deg);
    // sel_sat = saturation MULTIPLIER for that band (1.0 = unchanged, 0 = desaturate to grey).
    #[serde(default)]
    pub sel_band: u8,
    #[serde(default)]
    pub sel_hshift: f32,
    #[serde(default = "default_one")]
    pub sel_sat: f32,

    // ----- P41 SOLARIZE + COLOR TEMPERATURE (consumed by the P41 wave; on OUTB after the P39 selective
    // color, before the look). Each is a no-op at its default (byte-identical). sol_thr: solarize
    // threshold — per channel, v>sol_thr → 1-v (classic darkroom solarize); 0.0 = OFF (no-op), active
    // range (0,1]. temp: colour temperature — warm (temp>0) raises R / lowers B, cool (temp<0) the
    // reverse; 0.0 = neutral/off (no-op), range -1..1.
    #[serde(default)]
    pub sol_thr: f32,
    #[serde(default)]
    pub temp: f32,

    // ----- P35 CLIP GROUPING (consumed purely in the timeline/model edit path; NOT a render/wire
    // field). `group` is a non-zero group id shared by every clip in the same group; 0 = ungrouped
    // (the default). When a grouped clip's BODY is dragged, all members of its group move together by
    // the same frame delta (see timeline body-MOVE). `#[serde(default)]` so pre-P35 .json projects
    // load with group 0 (no grouping) → byte-identical editing behaviour. Grouping never touches the
    // render: a clip's group id is invisible to the worker/engine.
    #[serde(default)]
    pub group: u32,
}

/// serde default for the P34 mask centre/half-extent fields: 0.5 (frame centre / full extent).
fn default_half() -> f32 {
    0.5
}

/// serde default for `Clip.eq_fov`: a 90° rectilinear field of view.
fn default_eq_fov() -> f32 {
    90.0
}

/// serde default for `Clip.speed`: normal (1.0) playback rate. Pre-P24 projects load at 1.0 → identity.
fn default_speed() -> f32 {
    1.0
}

/// serde default for `Clip.seq` (P49): -1 = a normal media clip (not a compound/nested-sequence clip).
fn default_seq() -> i32 {
    -1
}

/// serde default [0,0,0] (gradient-map shadow colour = black).
fn default_zero3() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}

/// serde default for `Clip.glow_thr`: only pixels brighter than 0.7 luma contribute to the glow.
fn default_glow_thr() -> f32 {
    0.7
}

/// serde default [1,1,1] (gradient-map highlight colour = white).
fn default_one3() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

/// serde default for `Clip.hsl`: identity (no hue shift, unit saturation, no lightness change).
fn default_hsl() -> [f32; 3] {
    [0.0, 1.0, 0.0]
}

/// serde default for `Clip.levels`: identity (in 0..1, gamma 1).
fn default_levels() -> [f32; 3] {
    [0.0, 1.0, 1.0]
}

/// serde default for `Clip.curve`: the identity tone curve (outputs == inputs at the 5 control points).
fn default_curve() -> [f32; 5] {
    [0.0, 0.25, 0.5, 0.75, 1.0]
}

/// Per-clip text/title overlay (P5). `text` empty (default) is a no-op. `size_frac` is the font
/// height as a fraction of the frame height; `x`/`y` are the normalized top-left anchor in [0,1];
/// `rgb` is the text colour in [0,1]. Mirrors Shotcut's Text: Simple (dynamictext).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Title {
    pub text: String,
    pub size_frac: f32, // font height / frame height, e.g. 0.1
    pub x: f32,         // normalized left anchor [0,1]
    pub y: f32,         // normalized top anchor [0,1]
    pub rgb: [f32; 3],  // text colour [0,1]
}

impl Default for Title {
    fn default() -> Self {
        Title { text: String::new(), size_frac: 0.1, x: 0.05, y: 0.05, rgb: [1.0, 1.0, 1.0] }
    }
}

impl Title {
    /// True when there is no text to render (the worker then composites the clip normally).
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// A "lower third" preset (Shotcut's common title placement): the given text anchored toward
    /// the lower-left of the frame at a modest size, white. A convenience for the title-editor UI's
    /// preset button — it only sets the layout/colour fields (worker reads them unchanged), so a
    /// project that never builds one is byte-identical. `size_frac`/`x`/`y` are normalized as on the
    /// struct (font height / frame height; top-left anchor in [0,1]).
    pub fn lower_third(text: &str) -> Title {
        Title { text: text.to_string(), size_frac: 0.07, x: 0.06, y: 0.78, rgb: [1.0, 1.0, 1.0] }
    }
}

/// Per-clip chroma-key (green-screen) settings (P4). `enabled=false` (default) is a no-op: the worker
/// sends a disabled sentinel and the engine skips keying, so the composite is identical to P3. Applies
/// to a clip when it is the V2 OVERLAY (keyed pixels become transparent so V1 shows through).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct ChromaKey {
    pub enabled: bool,
    pub key: [f32; 3],   // key colour RGB in [0,1], default green [0,1,0]
    pub similarity: f32, // 0..1 colour-distance threshold to key out (larger = more keyed), def 0.4
    pub smoothness: f32, // 0..1 edge softness band beyond `similarity`, def 0.1
    // P37 SPILL SUPPRESSION: 0..1, 0 = off (identity). Reduces the green/key-colour tint that bleeds
    // onto the kept subject's edges. Applied in k_chroma after the alpha key (green-dominant key only).
    #[serde(default)]
    pub spill: f32,
}

impl Default for ChromaKey {
    fn default() -> Self {
        ChromaKey { enabled: false, key: [0.0, 1.0, 0.0], similarity: 0.4, smoothness: 0.1, spill: 0.0 }
    }
}

/// Per-clip audio-filter settings (P3). Neutral default (all 0 / false) is a no-op: the worker emits
/// no audio filter chain, so the mix is byte-identical to P2. Ranges mirror Shotcut's audio filters.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioFx {
    pub eq_low_db: f32,   // low-shelf gain, dB (Shotcut EQ: 3-band), 0 = flat
    pub eq_mid_db: f32,   // mid peak gain, dB, 0 = flat
    pub eq_high_db: f32,  // high-shelf gain, dB, 0 = flat
    pub pan: f32,         // -1 = full left, 0 = center, +1 = full right
    pub compress: bool,   // acompressor (sensible defaults)
    pub gate: bool,       // agate
    pub normalize: bool,  // loudnorm (single-pass)
    // ----- P11 per-clip audio effects (Shotcut Reverb / Delay / Pitch). All `#[serde(default ..)]`
    // so pre-P11 .json (an audio_fx object lacking these keys) loads to the neutral, off state. Each
    // is a no-op at its default, so is_neutral() stays true and the chain stays "-" (P10 identity).
    #[serde(default)]
    pub reverb: f32,      // reverb amount 0..1 (0 = off) → multi-tap aecho
    #[serde(default)]
    pub delay_ms: f32,    // echo delay in ms (0 = off) → aecho
    #[serde(default = "default_delay_decay")]
    pub delay_decay: f32, // echo feedback 0..0.95 (only meaningful when delay_ms>0)
    #[serde(default)]
    pub pitch: f32,       // pitch shift in SEMITONES (0 = off) → rubberband (tempo-preserving)
    // ----- P12 per-clip audio filters (Shotcut Low Pass / High Pass / Tremolo). All `#[serde(default)]`
    // (each defaults to 0.0), so pre-P12 .json (an audio_fx object lacking these keys) loads to the
    // neutral, off state. Each is a no-op at its default, so is_neutral() stays true and the chain
    // stays "-" (identity preserved).
    #[serde(default)]
    pub lowpass_hz: f32,  // low-pass cutoff in Hz (0 = off) → lowpass=f=<hz>
    #[serde(default)]
    pub highpass_hz: f32, // high-pass cutoff in Hz (0 = off) → highpass=f=<hz>
    #[serde(default)]
    pub tremolo: f32,     // tremolo depth 0..0.95 (0 = off) → tremolo=f=5:d=<depth>
    // ----- P15 per-clip audio filters (Shotcut Bass & Treble / Notch / Chorus). All `#[serde(default)]`
    // (each defaults to 0.0), so pre-P15 .json (an audio_fx object lacking these keys) loads to the
    // neutral, off state. Each is a no-op at its default, so is_neutral() stays true and the chain
    // stays "-" (identity preserved).
    #[serde(default)]
    pub bass_db: f32,    // low-shelf gain in dB (0 = flat / off) → bass=g=<db>
    #[serde(default)]
    pub treble_db: f32,  // high-shelf gain in dB (0 = flat / off) → treble=g=<db>
    #[serde(default)]
    pub notch_hz: f32,   // band-reject centre frequency in Hz (0 = off) → bandreject=f=<hz>
    #[serde(default)]
    pub chorus: f32,     // chorus depth 0..1 (0 = off) → chorus=0.5:0.9:50:0.4:0.25:<2*depth ms>
    // ----- P22 per-clip audio filters (Shotcut Flanger / Phaser / Limiter). All `#[serde(default)]`
    // (each defaults to 0.0), so pre-P22 .json (an audio_fx object lacking these keys) loads to the
    // neutral, off state. Each is a no-op at its default, so is_neutral() stays true and the chain
    // stays "-" (identity preserved).
    #[serde(default)]
    pub flanger: f32,    // flanger depth 0..1 (0 = off) → flanger=depth=<0..8 ms>:speed=0.5
    #[serde(default)]
    pub phaser: f32,     // phaser intensity 0..1 (0 = off) → aphaser=speed=<0.1..2.1 Hz>
    #[serde(default)]
    pub limiter: f32,    // limiter peak ceiling 0..1 (0 = off) → alimiter=limit=<0.05..1.0 linear>
    // ----- P32 GRAPHIC EQ (Shotcut audio_eq15band-style). 10 ISO bands at 31/62/125/250/500/1k/2k/
    // 4k/8k/16k Hz, each a peaking gain in dB (0 = flat). All-zero (the default) => no filter added,
    // is_neutral() stays true, chain "-" (identity preserved). Each active band => one `equalizer`.
    #[serde(default)]
    pub geq: [f32; 10],
}

impl Default for AudioFx {
    fn default() -> Self {
        AudioFx {
            eq_low_db: 0.0,
            eq_mid_db: 0.0,
            eq_high_db: 0.0,
            pan: 0.0,
            compress: false,
            gate: false,
            normalize: false,
            reverb: 0.0,
            delay_ms: 0.0,
            delay_decay: default_delay_decay(),
            pitch: 0.0,
            lowpass_hz: 0.0,
            highpass_hz: 0.0,
            tremolo: 0.0,
            bass_db: 0.0,
            treble_db: 0.0,
            notch_hz: 0.0,
            chorus: 0.0,
            flanger: 0.0,
            phaser: 0.0,
            limiter: 0.0,
            geq: [0.0; 10],
        }
    }
}

/// serde default for `AudioFx.delay_decay`: 0.5 (echo feedback midpoint). Kept as a fn so a pre-P11
/// project that has an `audio_fx` object without `delay_decay` deserializes to 0.5 rather than 0.0 —
/// 0.5 is the neutral resting value the UI shows, and decay alone never makes the FX non-neutral.
fn default_delay_decay() -> f32 {
    0.5
}

impl AudioFx {
    /// True when every control is at its neutral value — the worker can then skip the audio filter
    /// chain entirely (no fpx_au_apply call), keeping the no-FX mix path byte-identical to P2.
    pub fn is_neutral(&self) -> bool {
        self.eq_low_db == 0.0
            && self.eq_mid_db == 0.0
            && self.eq_high_db == 0.0
            && self.pan == 0.0
            && !self.compress
            && !self.gate
            && !self.normalize
            // P11: only the "active when > 0" effects gate neutrality. `delay_decay` is a parameter
            // of the delay (not an effect by itself), so a clip with the default decay 0.5 but no
            // delay_ms stays neutral and still emits "-".
            && self.reverb == 0.0
            && self.delay_ms == 0.0
            && self.pitch == 0.0
            // P12: low-pass / high-pass / tremolo each gate neutrality only when active (> 0). All
            // default 0.0, so a pre-P12 clip (and any clip with these untouched) stays neutral → "-".
            && self.lowpass_hz == 0.0
            && self.highpass_hz == 0.0
            && self.tremolo == 0.0
            // P15: bass / treble shelves, notch (band-reject) and chorus each gate neutrality only
            // when active. bass_db / treble_db are "off" at 0 (flat shelf); notch_hz / chorus are
            // "off" at 0. All default 0.0, so a pre-P15 clip (and any clip untouched) stays neutral → "-".
            && self.bass_db == 0.0
            && self.treble_db == 0.0
            && self.notch_hz == 0.0
            && self.chorus == 0.0
            // P22: flanger / phaser / limiter each gate neutrality only when active (> 0). All
            // default 0.0, so a pre-P22 clip (and any clip untouched) stays neutral → "-".
            && self.flanger == 0.0
            && self.phaser == 0.0
            && self.limiter == 0.0
            && self.geq.iter().all(|&g| g == 0.0)
    }
}

/// serde default for `Clip.gain` (and any unity linear multiplier): 1.0.
fn default_gain() -> f32 {
    1.0
}

/// serde default for `Clip.contrast` / `Clip.sat` / `Clip.scale`: 1.0 (identity multiplier).
fn default_one() -> f32 {
    1.0
}

/// serde default for `Clip.lift`: [0,0,0] (no additive lift — identity).
fn default_lift() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}

/// serde default for `Clip.gamma`: [1,1,1] (unity gamma power — identity).
fn default_gamma() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

/// serde default for `Clip.gain_rgb`: [1,1,1] (unity highlight gain — identity).
fn default_gain_rgb() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

impl Clip {
    pub fn video(media: usize, t0: i64, len: i64, track: u8, name_hint: &str) -> Clip {
        let _ = name_hint;
        Clip {
            media, src_in: 0, len, t0, track, look: 0, look_amt: 1.0,
            fade_in: 0, fade_out: 0, px: 0.0, py: 0.0, pw: 1.0, ph: 1.0,
            lut: String::new(), gain: 1.0, bright: 0.0, contrast: 1.0, sat: 1.0,
            // P2 color-wheels / transform / blur — IDENTITY (no-op) so demo/render are unchanged.
            lift: [0.0, 0.0, 0.0],
            gamma: [1.0, 1.0, 1.0],
            gain_rgb: [1.0, 1.0, 1.0],
            wb_temp: 0.0,
            wb_tint: 0.0,
            rot: 0.0,
            scale: 1.0,
            blur: 0.0,
            audio_fx: AudioFx::default(),
            chroma: ChromaKey::default(),
            title: Title::default(),
            curve: default_curve(),
            vignette: 0.0,
            sharpen: 0.0,
            flip: 0,
            fx: 0,
            hsl: default_hsl(),
            levels: default_levels(),
            mosaic: 0,
            gmap_amt: 0.0,
            gmap_lo: default_zero3(),
            gmap_hi: default_one3(),
            denoise: 0.0,
            glow_amt: 0.0,
            glow_thr: default_glow_thr(),
            rgbshift: 0.0,
            halftone: 0,
            emboss: 0.0,
            edge: 0.0,
            grain: 0.0,
            scratches: 0.0,
            diffusion: 0.0,
            wave: 0.0,
            swirl: 0.0,
            threshold: 0.0,
            lens: 0.0,
            crop: 0.0,
            glitch: 0.0,
            eq360: false,
            eq_yaw: 0.0,
            eq_pitch: 0.0,
            eq_fov: default_eq_fov(),
            speed: default_speed(),
            reverse: false,
            seq: -1,
            blend_mode: 0,
            mask_shape: 0,
            mask_cx: default_half(),
            mask_cy: default_half(),
            mask_rw: default_half(),
            mask_rh: default_half(),
            mask_feather: 0.0,
            mask_invert: false,
            mirror_x: 0,
            kaleido: 0,
            dither: 0.0,
            sel_band: 0,
            sel_hshift: 0.0,
            sel_sat: default_one(),
            sol_thr: 0.0,
            temp: 0.0,
            group: 0,
        }
    }
    pub fn end(&self) -> i64 {
        self.t0 + self.len
    }

    /// P45 VIDEO FADE: the per-frame brightness factor in [0,1] for timeline frame `t` from this
    /// clip's `fade_in`/`fade_out` (frames). 1.0 = full (no fade). The fade-IN ramps 0->1 over the
    /// first `fade_in` frames; the fade-OUT ramps 1->0 over the last `fade_out` frames; the factor is
    /// the MIN of the two (so overlapping head+tail fades both apply). A clip with no fades returns
    /// 1.0 for every `t` (byte-identical render). The worker multiplies the composed frame by this.
    pub fn fade_factor(&self, t: i64) -> f32 {
        let mut f = 1.0f32;
        if self.fade_in > 0 {
            let local = t - self.t0; // 0 at the clip's first frame
            if local < self.fade_in {
                f = f.min(((local + 1) as f32 / self.fade_in as f32).clamp(0.0, 1.0));
            }
        }
        if self.fade_out > 0 {
            let remaining = self.end() - t; // 1 at the clip's last frame
            if remaining <= self.fade_out {
                f = f.min((remaining as f32 / self.fade_out as f32).clamp(0.0, 1.0));
            }
        }
        f.clamp(0.0, 1.0)
    }

    /// True when this clip carries a non-empty TITLE overlay (P5) — the worker then rasterizes
    /// `title.text` into a full-frame transparent RGBA and composites it over the clip's frame. An
    /// empty title (the default) returns false, so a pre-P5 / untitled clip is unchanged. Mirrors
    /// the `Title::is_empty` no-op contract: `is_title()` ≡ `!self.title.is_empty()`.
    // Retained predicate (unit-tested via `title_is_empty_and_clip_is_title`): the worker inlines
    // `!title.is_empty()` in resolve_frame, so this convenience accessor is currently unwired.
    #[allow(dead_code)]
    pub fn is_title(&self) -> bool {
        !self.title.is_empty()
    }
}

/// A per-boundary TRANSITION between two same-track clips (Wave 8). Unlike MojoMedia's
/// per-boundary `trans_type[boundary]` parallel list keyed by clip index, this is a real struct
/// keyed by (track, center) so it survives split/delete/reorder without re-indexing: the
/// transition is anchored at a timeline `center` frame on a given `track`, animated over the
/// half-open window `[center - dur/2, center + dur/2)`. `kind` maps to the fpx_gpu track1
/// transition ids (0=crossfade, 1=wipe_lr, 2=wipe_rl, 3=wipe_up, 4=wipe_down, 5=slide_lr,
/// 6=zoom, 7=dissolve). PINNED this wave: produced + edited here (Team B), consumed by Team A
/// (worker::resolve_frame → ENC/PREVIEW trans fields) and Team C (timeline transition UI).
#[derive(Clone, Copy)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Transition {
    pub track: u8,    // 0 = V1, 1 = V2, 2 = A1 (Clip.track index space)
    pub center: i64,  // timeline frame the transition is centered on (typically a clip boundary)
    pub dur: i64,     // window length in frames (clamped >= 2); window = [center - dur/2, center + dur/2)
    pub kind: i32,    // fpx_gpu track1 transition id 0..10 (0=crossfade .. 7=dissolve, 8=iris, 9=clock, 10=barn-door)
}

impl Transition {
    /// Start frame of the animated window (inclusive).
    pub fn start(&self) -> i64 {
        self.center - self.dur / 2
    }
    /// End frame of the animated window (exclusive).
    pub fn end(&self) -> i64 {
        self.center - self.dur / 2 + self.dur
    }
    /// True when timeline frame `t` is inside this transition's half-open window.
    pub fn contains(&self, t: i64) -> bool {
        t >= self.start() && t < self.end()
    }
    /// Animation progress 0..1 for frame `t`, mirroring MojoMedia's `rtt` ramp (clamped to the
    /// window so the worker never feeds track1 a `t` outside [0,1]). At window start prog=0
    /// (full outgoing clip), at window end prog→1 (full incoming clip).
    pub fn progress(&self, t: i64) -> f32 {
        let span = self.dur.max(1);
        let p = (t - self.start()) as f64 / span as f64;
        p.clamp(0.0, 1.0) as f32
    }
}

/// Keyframe interpolation TYPE (P14), mirroring Shotcut/MLT's practical keyframe modes. The interp
/// is PER-KEYFRAME and controls the curve of the segment whose LOWER keyframe carries it:
///   - `Discrete`: HOLD the lower keyframe's value until the next key (step). MLT "discrete".
///   - `Linear`:   straight-line blend between the two keys (the pre-P14 behavior). MLT "linear".
///   - `Smooth`:   smoothstep ease-in/out (`s = b*b*(3-2b)`). This is an HONEST approximation of
///                 Shotcut's "Smooth" mode — it is a smoothstep ease, NOT a bit-exact MLT
///                 Catmull-Rom spline. Endpoints match Linear (eased toward the mid).
/// `Default = Linear`, and the `#[serde(default)]` on the `interp` fields means a pre-P14 .json
/// keyframe (a bare `{t,v}` with no `interp` key) deserializes as `Linear` — so an old project's
/// render is byte-identical (Linear is the previous, only mode).
#[derive(Clone, Copy, PartialEq, Debug)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum KfInterp {
    Discrete,
    Linear,
    Smooth,
    // P19: MLT-exact Catmull-Rom smooth variants (mlt_animation.c interpolate_value /
    // catmull_rom_interpolate). Each maps to a (alpha, tension) pair fed to the same spline:
    //   SmoothNatural = centripetal, peak-flattening (alpha 0.5, tension -1.0)  [MLT smooth_natural]
    //   SmoothLoose   = uniform Catmull-Rom, overshoots    (alpha 0.0, tension 1.0)  [MLT smooth_loose / "~"]
    //   SmoothTight   = zero tangents (= smoothstep ease)  (alpha 0.5, tension 0.0)  [MLT smooth_tight]
    SmoothNatural,
    SmoothLoose,
    SmoothTight,
    // P20: MLT easing keyframe types (mlt_animation.c interpolate_value, Robert-Penner easings).
    // Each is a closed-form factor on the linear blend (no neighbours needed) — see `ease_factor`.
    SineIn, SineOut, SineInOut,
    QuadIn, QuadOut, QuadInOut,
    CubicIn, CubicOut, CubicInOut,
    QuartIn, QuartOut, QuartInOut,
    QuintIn, QuintOut, QuintInOut,
    ExpoIn, ExpoOut, ExpoInOut,
    CircIn, CircOut, CircInOut,
    BackIn, BackOut, BackInOut,
    ElasticIn, ElasticOut, ElasticInOut,
    BounceIn, BounceOut, BounceInOut,
}

impl KfInterp {
    /// The MLT (alpha, tension) pair for the Catmull-Rom variants; `None` for the non-spline kinds
    /// (Discrete/Linear/Smooth + the easings, which use the 2-point `interp_segment`).
    fn catmull_params(self) -> Option<(f64, f64)> {
        match self {
            KfInterp::SmoothNatural => Some((0.5, -1.0)),
            KfInterp::SmoothLoose => Some((0.0, 1.0)),
            KfInterp::SmoothTight => Some((0.5, 0.0)),
            _ => None,
        }
    }

    /// Human label for the keyframe-interp picker (single source of truth for the UI combo).
    pub fn label(self) -> &'static str {
        use KfInterp::*;
        match self {
            Discrete => "Discrete (hold)", Linear => "Linear", Smooth => "Smooth (eased)",
            SmoothNatural => "Smooth Natural", SmoothLoose => "Smooth Loose", SmoothTight => "Smooth Tight",
            SineIn => "Sine In", SineOut => "Sine Out", SineInOut => "Sine In-Out",
            QuadIn => "Quad In", QuadOut => "Quad Out", QuadInOut => "Quad In-Out",
            CubicIn => "Cubic In", CubicOut => "Cubic Out", CubicInOut => "Cubic In-Out",
            QuartIn => "Quart In", QuartOut => "Quart Out", QuartInOut => "Quart In-Out",
            QuintIn => "Quint In", QuintOut => "Quint Out", QuintInOut => "Quint In-Out",
            ExpoIn => "Expo In", ExpoOut => "Expo Out", ExpoInOut => "Expo In-Out",
            CircIn => "Circ In", CircOut => "Circ Out", CircInOut => "Circ In-Out",
            BackIn => "Back In", BackOut => "Back Out", BackInOut => "Back In-Out",
            ElasticIn => "Elastic In", ElasticOut => "Elastic Out", ElasticInOut => "Elastic In-Out",
            BounceIn => "Bounce In", BounceOut => "Bounce Out", BounceInOut => "Bounce In-Out",
        }
    }

    /// All keyframe-interp kinds, in picker order (the UI iterates this).
    pub const ALL: [KfInterp; 36] = {
        use KfInterp::*;
        [
            Discrete, Linear, Smooth, SmoothNatural, SmoothLoose, SmoothTight,
            SineIn, SineOut, SineInOut, QuadIn, QuadOut, QuadInOut,
            CubicIn, CubicOut, CubicInOut, QuartIn, QuartOut, QuartInOut,
            QuintIn, QuintOut, QuintInOut, ExpoIn, ExpoOut, ExpoInOut,
            CircIn, CircOut, CircInOut, BackIn, BackOut, BackInOut,
            ElasticIn, ElasticOut, ElasticInOut, BounceIn, BounceOut, BounceInOut,
        ]
    };
}

/// Easing direction (Robert-Penner): the three phases each easing family comes in.
#[derive(Clone, Copy)]
enum EaseDir {
    In,
    Out,
    InOut,
}

/// MLT easing FACTOR for an easing `KfInterp` at fractional progress `t` ∈ [0,1] — `None` for the
/// non-easing kinds. Each family is a verbatim port of the matching function in MLT's
/// mlt_animation.c (sinusoidal/power/exponential/circular/back/elastic/bounce). The caller applies
/// it as `y1 + (y2-y1)*factor`, exactly like MLT.
fn ease_factor(kind: KfInterp, t: f64) -> Option<f64> {
    use EaseDir::*;
    use KfInterp::*;
    let f = match kind {
        SineIn => ease_sine(t, In), SineOut => ease_sine(t, Out), SineInOut => ease_sine(t, InOut),
        QuadIn => ease_pow(t, 2.0, In), QuadOut => ease_pow(t, 2.0, Out), QuadInOut => ease_pow(t, 2.0, InOut),
        CubicIn => ease_pow(t, 3.0, In), CubicOut => ease_pow(t, 3.0, Out), CubicInOut => ease_pow(t, 3.0, InOut),
        QuartIn => ease_pow(t, 4.0, In), QuartOut => ease_pow(t, 4.0, Out), QuartInOut => ease_pow(t, 4.0, InOut),
        QuintIn => ease_pow(t, 5.0, In), QuintOut => ease_pow(t, 5.0, Out), QuintInOut => ease_pow(t, 5.0, InOut),
        ExpoIn => ease_expo(t, In), ExpoOut => ease_expo(t, Out), ExpoInOut => ease_expo(t, InOut),
        CircIn => ease_circ(t, In), CircOut => ease_circ(t, Out), CircInOut => ease_circ(t, InOut),
        BackIn => ease_back(t, In), BackOut => ease_back(t, Out), BackInOut => ease_back(t, InOut),
        ElasticIn => ease_elastic(t, In), ElasticOut => ease_elastic(t, Out), ElasticInOut => ease_elastic(t, InOut),
        BounceIn => ease_bounce(t, In), BounceOut => ease_bounce(t, Out), BounceInOut => ease_bounce(t, InOut),
        _ => return None,
    };
    Some(f)
}

// --- Robert-Penner easing factors, verbatim from MLT mlt_animation.c ---
fn ease_sine(t: f64, e: EaseDir) -> f64 {
    use std::f64::consts::{PI, FRAC_PI_2};
    match e {
        EaseDir::In => (t - 1.0).mul_add(FRAC_PI_2, 0.0).sin() + 1.0,
        EaseDir::Out => (t * FRAC_PI_2).sin(),
        EaseDir::InOut => 0.5 * (1.0 - (t * PI).cos()),
    }
}
fn ease_pow(t: f64, order: f64, e: EaseDir) -> f64 {
    match e {
        EaseDir::In => t.powf(order),
        EaseDir::Out => 1.0 - (1.0 - t).powf(order),
        EaseDir::InOut => {
            if t < 0.5 {
                2f64.powf(order) * t.powf(order) / 2.0
            } else {
                1.0 - (-2.0 * t + 2.0).powf(order) / 2.0
            }
        }
    }
}
fn ease_expo(t: f64, e: EaseDir) -> f64 {
    if t == 0.0 {
        return 0.0;
    }
    if t == 1.0 {
        return 1.0;
    }
    match e {
        EaseDir::In => 2f64.powf(10.0 * t - 10.0),
        EaseDir::Out => 1.0 - 2f64.powf(-10.0 * t),
        EaseDir::InOut => {
            if t < 0.5 {
                2f64.powf(20.0 * t - 10.0) / 2.0
            } else {
                (2.0 - 2f64.powf(-20.0 * t + 10.0)) / 2.0
            }
        }
    }
}
fn ease_circ(t: f64, e: EaseDir) -> f64 {
    match e {
        EaseDir::In => 1.0 - (1.0 - t.powi(2)).sqrt(),
        EaseDir::Out => (1.0 - (t - 1.0).powi(2)).sqrt(),
        EaseDir::InOut => {
            if t < 0.5 {
                0.5 * (1.0 - (1.0 - 4.0 * (t * t)).sqrt())
            } else {
                0.5 * ((-((2.0 * t) - 3.0) * ((2.0 * t) - 1.0)).sqrt() + 1.0)
            }
        }
    }
}
fn ease_back(t: f64, e: EaseDir) -> f64 {
    use std::f64::consts::PI;
    match e {
        EaseDir::In => t * t * t - t * (t * PI).sin(),
        EaseDir::Out => {
            let f = 1.0 - t;
            1.0 - (f * f * f - f * (f * PI).sin())
        }
        EaseDir::InOut => {
            if t < 0.5 {
                let f = 2.0 * t;
                0.5 * (f * f * f - f * (f * PI).sin())
            } else {
                let f = 1.0 - (2.0 * t - 1.0);
                0.5 * (1.0 - (f * f * f - f * (f * PI).sin())) + 0.5
            }
        }
    }
}
fn ease_elastic(t: f64, e: EaseDir) -> f64 {
    use std::f64::consts::FRAC_PI_2;
    let c = 13.0 * FRAC_PI_2;
    match e {
        EaseDir::In => (c * t).sin() * 2f64.powf(10.0 * (t - 1.0)),
        EaseDir::Out => (-c * (t + 1.0)).sin() * 2f64.powf(-10.0 * t) + 1.0,
        EaseDir::InOut => {
            if t < 0.5 {
                0.5 * (c * (2.0 * t)).sin() * 2f64.powf(10.0 * ((2.0 * t) - 1.0))
            } else {
                0.5 * ((-c * ((2.0 * t - 1.0) + 1.0)).sin() * 2f64.powf(-10.0 * (2.0 * t - 1.0)) + 2.0)
            }
        }
    }
}
fn ease_bounce(t: f64, e: EaseDir) -> f64 {
    match e {
        EaseDir::In => 1.0 - ease_bounce(1.0 - t, EaseDir::Out),
        EaseDir::Out => {
            if t < 4.0 / 11.0 {
                (121.0 * t * t) / 16.0
            } else if t < 8.0 / 11.0 {
                (363.0 / 40.0 * t * t) - (99.0 / 10.0 * t) + 17.0 / 5.0
            } else if t < 9.0 / 10.0 {
                (4356.0 / 361.0 * t * t) - (35442.0 / 1805.0 * t) + 16061.0 / 1805.0
            } else {
                (54.0 / 5.0 * t * t) - (513.0 / 25.0 * t) + 268.0 / 25.0
            }
        }
        EaseDir::InOut => {
            if t < 0.5 {
                0.5 * ease_bounce(t * 2.0, EaseDir::In)
            } else {
                0.5 * ease_bounce(2.0 * t - 1.0, EaseDir::Out) + 0.5
            }
        }
    }
}

impl Default for KfInterp {
    fn default() -> Self {
        KfInterp::Linear
    }
}

/// One keyframe on a scalar track: `v` is the value at timeline (or clip-local) frame `t`.
/// Mirrors MojoMedia's parallel `KfTrack { frames, values }` but as a real struct (the Rust
/// win): a `Vec<Kf>` kept sorted ascending by `t` replaces the two parallel lists.
/// `interp` (P14) is this keyframe's interpolation type; it controls the SEGMENT that STARTS at
/// this key (i.e. the curve from this key up to the next). `#[serde(default)]` → pre-P14 `{t,v}`
/// keyframes load as `Linear`.
#[derive(Clone, Copy)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Kf {
    pub t: i64,
    pub v: f32,
    #[serde(default)]
    pub interp: KfInterp,
}

/// One per-clip PiP keyframe, stored flat (mirrors MojoMedia `PipKf`): which clip, which
/// param, the CLIP-LOCAL frame, and the value. Flat storage (one Vec for the whole project)
/// is chosen over a Vec-per-clip so the set survives split/delete without re-indexing nested
/// vectors; see `remap_clip_keys` for the index-stability policy.
///
/// PINNED PAR REGISTRY (this flat store animates BOTH the PiP rect AND a curated set of per-clip
/// VIDEO filter params — P30). `eval_pip` is generic over `par`, so adding a param is purely a
/// registry + worker-substitution change; the store, serde, and interpolation are unchanged:
///   0 = px      1 = py       2 = pw      3 = ph       (PiP rect, EXISTING — applied to the overlay)
///   4 = bright  5 = contrast 6 = sat                  (per-clip GRADE  → Clip.bright/.contrast/.sat)
///   7 = blur    8 = rot      9 = scale                (per-clip TRANSFORM/BLUR → Clip.blur/.rot/.scale)
/// Each param falls back to the matching static `Clip` field when the clip has no keys for it, so a
/// pre-P30 project (or any clip without keys) is byte-identical: `clip_param_at` returns the static
/// value the worker already sent. `clip_param_at` (par 4..9) is the public entry; `add_clip_param_key`
/// keys one param from its live field; `clip_param_key_count` reports per-param key counts.
#[derive(Clone, Copy)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PipKey {
    pub clip: usize, // clip index these keys animate
    pub par: u8,     // 0=px 1=py 2=pw 3=ph | 4=bright 5=contrast 6=sat 7=blur 8=rot 9=scale
    pub t_local: i64, // clip-local frame (t - clip.t0)
    pub v: f32,
    // P14 interpolation type; controls the SEGMENT starting at this key (the lower key of an
    // (clip,par) pair). `#[serde(default)]` → pre-P14 flat PiP keyframes load as `Linear`.
    #[serde(default)]
    pub interp: KfInterp,
}

/// Keyframe eval shared by grade + PiP: value of a sorted-ascending `Vec<Kf>` at `t`, or
/// `fallback` when the track is empty. Clamps to the first/last value outside the range. The
/// interpolation of the segment `[i, i+1]` is selected by the LOWER keyframe's `interp` (P14):
/// `Discrete` HOLDS `track[i].v`, `Linear` is the pre-P14 straight blend, `Smooth` applies a
/// smoothstep ease. See `interp_segment` for the shared blend math (also used by `eval_pip`).
fn eval_track(track: &[Kf], t: i64, fallback: f32) -> f32 {
    let n = track.len();
    if n == 0 {
        return fallback;
    }
    if t <= track[0].t {
        return track[0].v;
    }
    if t >= track[n - 1].t {
        return track[n - 1].v;
    }
    // find i such that track[i].t <= t < track[i+1].t
    let mut i = 0;
    while i < n - 1 && track[i + 1].t <= t {
        i += 1;
    }
    let kind = track[i].interp;
    // P19: the MLT Catmull-Rom variants need the two NEIGHBOURING keys (the one before `i` and the
    // one after `i+1`) for their tangents. At the ends, duplicate the boundary key — MLT's
    // catmull_rom_interpolate then shoves the duplicate ±10000 frames away to make a horizontal end
    // tangent. `progress` is the fractional position in [i, i+1] (MLT's (frame-p1)/(p2-p1)).
    if let Some((alpha, tension)) = kind.catmull_params() {
        let p1 = track[i];
        let p2 = track[i + 1];
        let p0 = if i > 0 { track[i - 1] } else { p1 };
        let p3 = if i + 2 < n { track[i + 2] } else { p2 };
        let prog = (t - p1.t) as f64 / (p2.t - p1.t) as f64;
        return catmull_rom(
            p0.t as f64, p0.v as f64, p1.t as f64, p1.v as f64,
            p2.t as f64, p2.v as f64, p3.t as f64, p3.v as f64,
            prog, alpha, tension,
        ) as f32;
    }
    interp_segment(kind, track[i].t, track[i].v, track[i + 1].t, track[i + 1].v, t)
}

/// Euclidean distance between two control points (MLT `distance`, mlt_animation.c).
fn kf_distance(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
}

/// MLT-exact Catmull-Rom spline (mlt_animation.c `catmull_rom_interpolate`), translated line-for-line.
/// 4 control points by FRAME (x) + value (y): `(x0,y0)` before, `(x1,y1)` segment start, `(x2,y2)`
/// segment end, `(x3,y3)` after; `t` ∈ [0,1] is the fractional progress between p1 and p2; `alpha`
/// selects the parameterisation (0 uniform / 0.5 centripetal / 1 chordal); `tension` scales the
/// tangents (|tension|; sign + the monotonic-between-neighbours test gate whether a tangent is
/// computed at all, so a peak gets a flat tangent = no overshoot). Returns the interpolated value.
#[allow(clippy::too_many_arguments)]
fn catmull_rom(
    mut x0: f64, y0: f64, x1: f64, y1: f64, x2: f64, y2: f64, mut x3: f64, y3: f64,
    t: f64, alpha: f64, tension: f64,
) -> f64 {
    // Duplicated boundary point → push it far away so the end segment gets a horizontal tangent.
    if x0 == x1 {
        x0 -= 10000.0;
    }
    if x3 == x2 {
        x3 += 10000.0;
    }
    let mut m1 = 0.0;
    let mut m2 = 0.0;
    let t12 = kf_distance(x1, y1, x2, y2).powf(alpha);
    if tension > 0.0 || (y1 < y0 && y1 > y2) || (y1 > y0 && y1 < y2) {
        let t01 = kf_distance(x0, y0, x1, y1).powf(alpha);
        m1 = tension.abs() * (y2 - y1 + t12 * ((y1 - y0) / t01 - (y2 - y0) / (t01 + t12)));
    }
    if tension > 0.0 || (y2 < y1 && y2 > y3) || (y2 > y1 && y2 < y3) {
        let t23 = kf_distance(x2, y2, x3, y3).powf(alpha);
        m2 = tension.abs() * (y2 - y1 + t12 * ((y3 - y2) / t23 - (y3 - y1) / (t12 + t23)));
    }
    let a = 2.0 * (y1 - y2) + m1 + m2;
    let b = -3.0 * (y1 - y2) - m1 - m1 - m2;
    let c = m1;
    let d = y1;
    a * t * t * t + b * t * t + c * t + d
}

/// Evaluate a single keyframe SEGMENT at frame `t` using the lower key's interpolation `kind`.
/// `(fa, va)` is the lower keyframe, `(fb, vb)` the upper; `t` is assumed in `[fa, fb)` (the
/// endpoint-clamp cases are handled by the callers). With `blend = (t-fa)/(fb-fa) ∈ [0,1)`:
///   - `Discrete`: return `va` (HOLD until the next key — step interpolation).
///   - `Linear`:   `va + blend*(vb-va)` (the pre-P14 behavior, unchanged).
///   - `Smooth`:   `s = blend*blend*(3 - 2*blend)` (smoothstep ease-in/out), return `va + s*(vb-va)`.
/// Shared by `eval_track` (grade tracks) and `eval_pip` (flat PiP store) so both honor the same
/// per-segment curve. A degenerate `fb == fa` would only arise from coincident keys; the callers
/// never feed that case (eval_track advances past equal-frame keys; eval_pip's lo<t<hi guarantees
/// fb>fa), but Discrete still returns `va` safely regardless.
fn interp_segment(kind: KfInterp, fa: i64, va: f32, fb: i64, vb: f32, t: i64) -> f32 {
    match kind {
        KfInterp::Discrete => va,
        KfInterp::Linear => {
            let blend = (t - fa) as f64 / (fb - fa) as f64;
            (blend * (vb - va) as f64) as f32 + va
        }
        KfInterp::Smooth => {
            let blend = (t - fa) as f64 / (fb - fa) as f64;
            let s = blend * blend * (3.0 - 2.0 * blend); // smoothstep ease-in/out
            (s * (vb - va) as f64) as f32 + va
        }
        // Everything else: the easing kinds (P20) apply a closed-form factor on the linear blend;
        // the Catmull-Rom variants are NEIGHBOUR-aware and handled in `eval_track` (they never reach
        // this 2-point helper, but a linear fallback keeps this safe if mis-routed).
        kind => {
            let blend = (t - fa) as f64 / (fb - fa) as f64;
            let factor = ease_factor(kind, blend).unwrap_or(blend); // easing factor, else linear
            (factor * (vb - va) as f64) as f32 + va
        }
    }
}

/// Sorted insert-or-replace into a `Vec<Kf>` keyed on `t` (mirrors MojoMedia kf_set): if a
/// key already exists at `t` its value AND its `interp` are overwritten, otherwise the key is
/// inserted so the track stays ascending in `t`. P14: `interp` is the CURRENT create mode
/// (`Project.kf_interp`) threaded through by `add_grade_key`, so re-keying a frame while the mode
/// is Smooth makes that key Smooth.
fn set_track(track: &mut Vec<Kf>, t: i64, v: f32, interp: KfInterp) {
    match track.binary_search_by(|k| k.t.cmp(&t)) {
        Ok(idx) => {
            // replace at existing frame — value AND interp follow the current create mode.
            track[idx].v = v;
            track[idx].interp = interp;
        }
        Err(idx) => track.insert(idx, Kf { t, v, interp }), // sorted insert
    }
}

/// Export / render settings (Triad-B P1). Carried on the `Project` so `worker::render_program`
/// (which keeps its `(&Project, &str)` signature — no app.rs change) can read them and pass them
/// through the `OPEN` wire line to the encoder. The OpenCL working canvas stays GVW×GVH (1280×856);
/// these only drive the ENCODER: `out_w`/`out_h` are the swscaled output dims, `fps_num`/`fps_den`
/// the output framerate, `rate_mode` selects bitrate (0) vs CRF (1), `rate_value` is the bitrate in
/// bits/s (rate_mode 0) OR the CRF quality value (rate_mode 1), and `vcodec` is the encoder name.
///
/// DEFAULTS REPRODUCE TODAY'S BEHAVIOR (1280×856 @ 30/1, mpeg4, 4 Mbit/s bitrate) so existing render
/// gates pass unchanged. All fields are `#[serde(default = ..)]` so pre-P1 .json projects load with
/// the defaults. `mlt`-style values are intentionally avoided — these map straight to fpx_encode.c.
#[derive(Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ExportSettings {
    #[serde(default = "default_out_w")]
    pub out_w: u32,
    #[serde(default = "default_out_h")]
    pub out_h: u32,
    #[serde(default = "default_fps_num")]
    pub fps_num: u32,
    #[serde(default = "default_fps_den")]
    pub fps_den: u32,
    /// 0 = average bitrate, 1 = constant quality (CRF/qscale).
    #[serde(default)]
    pub rate_mode: u8,
    /// bits/s when `rate_mode == 0`; the CRF/quality value when `rate_mode == 1`.
    #[serde(default = "default_bitrate")]
    pub rate_value: i64,
    /// CRF/quality value (kept separate from `rate_value` so toggling rate_mode in the UI doesn't
    /// clobber the other mode's last value). Used as `rate_value` source when rate_mode switches to 1.
    #[serde(default = "default_crf")]
    pub crf: i64,
    #[serde(default = "default_vcodec")]
    pub vcodec: String,
    /// GOP / keyframe interval in frames (P25 export depth). 0 = leave the encoder default (identity
    /// with pre-P25). >0 forces a keyframe every `gop` frames (1 = all-intra).
    #[serde(default)]
    pub gop: i32,
    /// x264/x265 encoder preset (P25). "" = don't set (identity). e.g. "ultrafast".."veryslow".
    /// Ignored by codecs without a "preset" private option (mpeg4).
    #[serde(default)]
    pub preset: String,
    /// Audio bitrate in bits/s (P25). 0 = keep the engine default (identity). e.g. 128000, 192000.
    #[serde(default)]
    pub abitrate: i64,
    /// Audio codec name (P29 export depth). "" => the engine default "aac" (identity with pre-P29).
    /// e.g. "aac", "libmp3lame", "ac3", "pcm_s16le". Must be compatible with the chosen container.
    #[serde(default)]
    pub acodec: String,
}

fn default_out_w() -> u32 {
    1280
}
fn default_out_h() -> u32 {
    856
}
fn default_fps_num() -> u32 {
    30
}
fn default_fps_den() -> u32 {
    1
}
fn default_bitrate() -> i64 {
    4_000_000
}
fn default_crf() -> i64 {
    23
}
fn default_vcodec() -> String {
    "mpeg4".to_string()
}

impl Default for ExportSettings {
    fn default() -> Self {
        ExportSettings {
            out_w: default_out_w(),
            out_h: default_out_h(),
            fps_num: default_fps_num(),
            fps_den: default_fps_den(),
            rate_mode: 0,
            rate_value: default_bitrate(),
            crf: default_crf(),
            vcodec: default_vcodec(),
            gop: 0,
            preset: String::new(),
            abitrate: 0,
            acodec: String::new(),
        }
    }
}

#[derive(Clone, Default)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Project {
    pub media: Vec<String>, // media file paths; clips index into this
    pub names: Vec<String>, // display names per media
    // ----- P47 MEDIA BINS (organize the pool into named bins) — pre-added by integrator. `bin_names`
    // are the bins (default the single "Media" bin); `media_bin[i]` is the bin index of media i
    // (default 0). serde-default so pre-P47 .json loads as one flat "Media" bin. Parallel to `media`
    // like `names`. Bin assignment is pool-organization only — invisible to the render/timeline.
    #[serde(default = "default_bins")]
    pub bin_names: Vec<String>,
    #[serde(default)]
    pub media_bin: Vec<u32>,
    pub clips: Vec<Clip>,
    #[serde(default)]
    pub trans: Vec<i32>, // LEGACY transition id per boundary (-1 = none); unused — superseded by `transitions`
    // ----- per-boundary transitions (Wave 8; PINNED) -----------------------------------------
    // The real transition store: a list of (track, center, dur, kind) structs anchored at
    // timeline frames, replacing the legacy index-keyed `trans` Vec. `#[serde(default)]` so
    // pre-Wave-8 .json projects still deserialize (the field defaults to empty = no transitions).
    // Produced + edited here via add/remove_transition; queried by Team A (resolve) via
    // transition_at() and by Team C (timeline UI) via boundaries() + transition_at().
    #[serde(default)]
    pub transitions: Vec<Transition>,
    #[serde(default)]
    pub bright: f32,
    #[serde(default)]
    pub contrast: f32,
    #[serde(default)]
    pub sat: f32,
    #[serde(default)]
    pub markers: Vec<i64>, // timeline markers (frames); the scrub/playhead can snap to them
    // ----- P48 SUBTITLES (timeline-wide timed captions, rendered over the PROGRAM). serde-default
    // empty so pre-P48 .json loads byte-identical (no subtitles -> no render change). Each entry is
    // [start,end) timeline frames + text; the worker overlays the active one as the top RAW: layer.
    #[serde(default)]
    pub subtitles: Vec<Subtitle>,
    // ----- P49 NESTED SEQUENCES (compound clips). A list of self-contained sub-timelines; a Clip with
    // `seq >= 0` indexes into this and is composed (its frame at inner time) as a RAW: layer at render.
    // serde-default empty so pre-P49 .json loads byte-identical (no subseqs, every clip seq=-1).
    #[serde(default)]
    pub subseqs: Vec<SubSeq>,

    // ----- keyframe storage (Slice C; all #[serde(default)] so pre-keyframe .json loads) -----
    // Program-wide grade tracks, each a Vec<Kf> sorted ascending by t (timeline frames). An
    // EMPTY track means "use the static bright/contrast/sat field" — grade_at() falls back to
    // the static value so the existing non-animated grade keeps working unchanged. opacity_kf
    // is reserved for a future V2-opacity animation (worker does not yet read it; harmless to
    // store). Consumed by Team A worker::resolve_frame via grade_at(t).
    #[serde(default)]
    pub bright_kf: Vec<Kf>,
    #[serde(default)]
    pub contrast_kf: Vec<Kf>,
    #[serde(default)]
    pub sat_kf: Vec<Kf>,
    #[serde(default)]
    pub opacity_kf: Vec<Kf>,
    /// P27 MASTER AUDIO-GAIN automation track. Empty = flat gain (1.0). Keys are LINEAR (the engine
    /// linear-interps the gain envelope it receives on the GAINENV wire line). Applied per-sample to
    /// the program-audio mix at each sample's absolute timeline time.
    #[serde(default)]
    pub gain_kf: Vec<Kf>,

    // Per-clip PiP keyframes, flat (mirrors MojoMedia PipKf). Each entry binds (clip, param,
    // clip-local frame) -> value. An (clip, param) with NO entries falls back to that clip's
    // static px/py/pw/ph in pip_at(). Consumed by Team A worker::resolve_frame via pip_at().
    #[serde(default)]
    pub pip_kf: Vec<PipKey>,

    // ----- per-track state (PINNED this wave; index 0 = V1, 1 = V2, 2 = A1) -----
    // serde(default) so older .json projects (without these keys) still deserialize to
    // [false; 3]. These fields are exposed for the worker via is_hidden()/is_muted()/
    // is_locked() below. INTEGRATOR WIRING REQUIRED (Team A): worker.rs does NOT yet
    // consult them — resolve_frame() composites by Clip.track alone and build_audio_lines()
    // gates on a static track_is_audible(track). Until Team A calls project.is_hidden(track)
    // in resolve_frame (skip a hidden VIDEO track) and project.is_muted(track) in
    // build_audio_lines (drop a muted track's audio), these toggles change neither the
    // preview nor the export. track_lock is advisory this wave (edits to a locked track
    // should be blocked; timeline.rs already calls is_locked()).
    // P5 ARBITRARY TRACKS: an ordered list (bottom -> top) replacing the fixed V1/V2/A1 + [bool;3]
    // hide/mute/lock. Video tracks composite bottom-as-base + top-as-overlay; audio tracks all mix.
    // `Clip.track` indexes into this. serde default rebuilds the legacy 3 (V1 video, V2 video, A1
    // audio) so pre-P5 .json projects (no "tracks" field) load with today's layout.
    #[serde(default = "default_tracks")]
    pub tracks: Vec<Track>,

    // ----- Export / render settings (Triad-B P1) -----
    // serde(default) so pre-P1 .json projects deserialize with today's-behavior defaults
    // (1280×856 @ 30, mpeg4, 4 Mbit/s). Read by worker::render_program → OPEN wire line; edited via
    // the Export Settings block in panels::properties_ui. Decouples the OUTPUT resolution from the
    // fixed GVW×GVH OpenCL working canvas (the encoder swscales the composed frame to out_w×out_h).
    #[serde(default)]
    pub export: ExportSettings,

    // ----- P14 keyframe interpolation CREATE mode -----
    // The interpolation TYPE applied to NEW keyframes created via add_grade_key / add_pip_key (and
    // to a re-keyed frame). Per-keyframe interp lives on Kf/PipKey; this is the single "current
    // mode" the create path reads, so those add_* signatures stay unchanged (no-ripple design).
    // `#[serde(default)]` → pre-P14 .json (no "kf_interp" key) loads as `Linear`, and the derived
    // `Default for Project` also yields `Linear` (KfInterp::default), matching the pre-P14-only mode.
    #[serde(default)]
    pub kf_interp: KfInterp,

    // ----- P43 EXPORT IN/OUT REGION (editing batch — pre-added by integrator) -----
    // Optional sub-range of the timeline to render. -1 / -1 (the default) = export the WHOLE timeline
    // (byte-identical to pre-P43). When both are >=0 and export_out > export_in, render_program emits
    // only frames in [export_in, export_out). serde(default) so pre-P43 .json loads with no region.
    #[serde(default = "default_neg1")]
    pub export_in: i64,
    #[serde(default = "default_neg1")]
    pub export_out: i64,
}

/// serde default for the P43 export in/out marks: -1 (no region → whole timeline).
fn default_neg1() -> i64 {
    -1
}

/// serde default for the P47 media bins: a single "Media" bin (every media starts here).
fn default_bins() -> Vec<String> {
    vec!["Media".to_string()]
}

/// P48 SUBTITLES — one timed caption: timeline frames `[start, end)` + the (possibly multi-line) text.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Subtitle {
    pub start: i64,
    pub end: i64,
    pub text: String,
}

/// P49 NESTED SEQUENCE (compound clip) — a self-contained sub-timeline that can be placed as one clip
/// on a parent timeline. ONE level of nesting: a sub-sequence's own clips are plain media clips (seq
/// = -1). `clips`/`tracks` index into the PARENT project's shared `media` pool. `len` is the
/// compound clip's natural duration (frames).
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SubSeq {
    pub name: String,
    pub len: i64,
    pub clips: Vec<Clip>,
    #[serde(default = "default_tracks")]
    pub tracks: Vec<Track>,
}

/// P48 — parse SRT text into `Subtitle`s, converting `HH:MM:SS,mmm` timestamps to TIMELINE FRAMES at
/// `fps`. Tolerant: normalizes CRLF, splits on blank lines, finds each block's `start --> end` line,
/// joins the remaining lines as the text; skips blocks with no arrow / bad timestamps / empty text /
/// non-positive duration (never panics). Index lines are ignored.
pub fn parse_srt(s: &str, fps: f64) -> Vec<Subtitle> {
    fn ts_to_frames(t: &str, fps: f64) -> Option<i64> {
        let t = t.trim().replace(',', "."); // accept ',' or '.' ms separator
        let (hms, ms) = t.split_once('.').unwrap_or((t.as_str(), "0"));
        let parts: Vec<&str> = hms.split(':').collect();
        if parts.len() != 3 {
            return None;
        }
        let h: f64 = parts[0].trim().parse().ok()?;
        let m: f64 = parts[1].trim().parse().ok()?;
        let sec: f64 = parts[2].trim().parse().ok()?;
        let ms_frac: f64 = format!("0.{}", ms.trim()).parse().unwrap_or(0.0);
        Some(((h * 3600.0 + m * 60.0 + sec + ms_frac) * fps).round() as i64)
    }
    let norm = s.replace("\r\n", "\n").replace('\r', "\n");
    let mut out = Vec::new();
    for block in norm.split("\n\n") {
        let lines: Vec<&str> = block.lines().collect();
        let Some(ai) = lines.iter().position(|l| l.contains("-->")) else {
            continue;
        };
        let Some((a, b)) = lines[ai].split_once("-->") else {
            continue;
        };
        if let (Some(start), Some(end)) = (ts_to_frames(a, fps), ts_to_frames(b, fps)) {
            let text = lines[ai + 1..].join("\n").trim().to_string();
            if !text.is_empty() && end > start {
                out.push(Subtitle { start, end, text });
            }
        }
    }
    out
}

/// P46 AUDIO ALIGN — the integer lag (in samples) in `[-max_lag, max_lag]` that best aligns `b` onto
/// `a` by NORMALIZED cross-correlation. A POSITIVE return means `b` LAGS `a` (b's content appears that
/// many samples LATER), so to sync the two recordings you shift `b` EARLIER by the lag. Returns 0 if
/// either slice is empty or no lag has enough overlap. Property: if `b[i] == a[i-K]` (b is `a` delayed
/// by K samples), this returns `K`. Pure + deterministic (the gateable core of the align feature).
pub fn cross_correlation_offset(a: &[f32], b: &[f32], max_lag: usize) -> i64 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let max_lag = max_lag.min(a.len().max(b.len()).saturating_sub(1)) as i64;
    let mut best_lag = 0i64;
    let mut best_score = f64::NEG_INFINITY;
    for l in -max_lag..=max_lag {
        // score(L) = normalized sum a[i]*b[i+L] over the valid overlap. b[i+L]==a[i] for the delay.
        let i0 = if l < 0 { (-l) as usize } else { 0 };
        let i1 = a.len().min(((b.len() as i64) - l).max(0) as usize);
        if i1 <= i0 {
            continue;
        }
        let (mut dot, mut ea, mut eb, mut cnt) = (0f64, 0f64, 0f64, 0usize);
        for i in i0..i1 {
            let av = a[i] as f64;
            let bv = b[(i as i64 + l) as usize] as f64;
            dot += av * bv;
            ea += av * av;
            eb += bv * bv;
            cnt += 1;
        }
        if cnt < 8 {
            continue; // too little overlap to trust
        }
        let denom = (ea * eb).sqrt();
        let score = if denom > 1e-12 { dot / denom } else { 0.0 };
        if score > best_score {
            best_score = score;
            best_lag = l;
        }
    }
    best_lag
}

/// T1 — 3D-LUT LIBRARY. Scan `dir` for every `*.cube` file (case-insensitive extension) and return
/// `(display_name, full_path)` pairs SORTED by display name ascending. `display_name` is the file
/// stem (filename without its extension, e.g. `/luts/Teal_Orange.cube` -> `Teal_Orange`); the full
/// path is the joined `dir`/`filename` so a pick can be assigned straight onto `Clip.lut`.
///
/// PURE + dependency-free (std::fs only): a missing / unreadable directory, or one with no `.cube`
/// files, yields an empty `Vec` (never panics). Non-files and sub-directories are skipped. The UI
/// (panels.rs LUT-library combo) reads this to populate a per-clip LUT dropdown; selecting an entry
/// sets `clip.lut` + `clip.look = 2` so the engine renders the 3D LUT.
pub fn scan_lut_dir(dir: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    // read_dir on a missing/unreadable dir -> Err -> empty list (no panic).
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return out,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        // Only plain files (skip sub-dirs / symlinked dirs).
        if !path.is_file() {
            continue;
        }
        // Case-insensitive ".cube" extension match.
        let is_cube = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("cube"))
            .unwrap_or(false);
        if !is_cube {
            continue;
        }
        // display_name = file stem (no extension); skip anything without a valid UTF-8 stem.
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let full = path.to_string_lossy().into_owned();
        out.push((stem, full));
    }
    // SORTED by display name ascending (stable, case-sensitive byte order — matches MojoMedia's
    // luts list which sorted on the raw stem).
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Video vs audio track (P5 arbitrary tracks). Video tracks composite; audio tracks mix.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TrackKind {
    Video,
    Audio,
}

/// One timeline track (P5). `Project.tracks` is ordered bottom -> top; `Clip.track` indexes it.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Track {
    pub kind: TrackKind,
    pub name: String,
    #[serde(default)]
    pub hidden: bool,
    #[serde(default)]
    pub muted: bool,
    #[serde(default)]
    pub locked: bool,
    // ----- P42 AUDIO MIXER (per-track fader/pan/solo). No-ops at their defaults so pre-P42 projects
    // load + render byte-identical. `gain` is a linear track-level fader folded into each clip's
    // emitted per-clip gain (1.0 = unity). `pan` is a track L/R balance in [-1,1] folded into the
    // clip's audio fx_chain via stereotools (0 = centre). `solo` is the mixer solo flag: when ANY
    // audio track is soloed, only soloed (non-muted) tracks are audible.
    #[serde(default = "default_one")]
    pub gain: f32,
    #[serde(default)]
    pub pan: f32,
    #[serde(default)]
    pub solo: bool,
}

impl Track {
    pub fn new(kind: TrackKind, name: &str) -> Track {
        Track {
            kind,
            name: name.to_string(),
            hidden: false,
            muted: false,
            locked: false,
            gain: 1.0,
            pan: 0.0,
            solo: false,
        }
    }
}

/// The legacy default track set — V1 video, V2 video, A1 audio — matching the old fixed `Clip.track`
/// 0/1/2 so the demo and pre-P5 projects keep their layout. The serde default for `Project.tracks`.
pub fn default_tracks() -> Vec<Track> {
    vec![
        Track::new(TrackKind::Video, "V1"),
        Track::new(TrackKind::Video, "V2"),
        Track::new(TrackKind::Audio, "A1"),
    ]
}

impl Project {
    /// A demo project (3 clips) used until the media pool + import land.
    pub fn demo(media: String) -> Project {
        Project {
            media: vec![media],
            names: vec!["clip".into()],
            bin_names: vec!["Media".into()],
            media_bin: vec![0],
            subtitles: vec![],
            subseqs: vec![],
            clips: vec![
                Clip::video(0, 0, 120, 0, "intro"),
                Clip::video(0, 70, 90, 1, "overlay"),
                Clip::video(0, 0, 160, 2, "audio"),
            ],
            trans: vec![],
            transitions: vec![],
            bright: 0.0,
            contrast: 1.0,
            sat: 1.0,
            markers: vec![],
            tracks: default_tracks(),
            bright_kf: vec![],
            contrast_kf: vec![],
            sat_kf: vec![],
            opacity_kf: vec![],
            gain_kf: vec![],
            pip_kf: vec![],
            export: ExportSettings::default(),
            kf_interp: KfInterp::Linear,
            export_in: -1,
            export_out: -1,
        }
    }

    pub fn total_frames(&self) -> i64 {
        self.clips.iter().map(|c| c.end()).max().unwrap_or(1).max(1)
    }

    /// P43 — the effective export frame range `[start, end)` for a timeline of length `total`.
    /// Returns the marked in/out region when valid (both >=0 and out>in, clamped to [0,total]),
    /// otherwise the WHOLE timeline `(0, total)`. Default (-1/-1) → whole timeline.
    pub fn export_range(&self, total: i64) -> (i64, i64) {
        if self.export_in >= 0 && self.export_out > self.export_in {
            (self.export_in.max(0), self.export_out.min(total))
        } else {
            (0, total)
        }
    }

    /// P47 — the bin index of media `i` (0 if unassigned / out of range / no bins).
    pub fn bin_of(&self, media_idx: usize) -> u32 {
        self.media_bin.get(media_idx).copied().unwrap_or(0)
    }

    /// P47 — add a new media bin, returning its index. A blank name is ignored (returns 0).
    pub fn add_bin(&mut self, name: &str) -> usize {
        let name = name.trim();
        if name.is_empty() {
            return 0;
        }
        if self.bin_names.is_empty() {
            self.bin_names.push("Media".to_string()); // keep bin 0 stable
        }
        self.bin_names.push(name.to_string());
        self.bin_names.len() - 1
    }

    /// P47 — assign media `i` to bin index `bin` (clamped to a valid bin). Pads `media_bin` to cover
    /// `i`. No-op if `i` is out of range. Pool organization only — never touches the render.
    pub fn set_media_bin(&mut self, media_idx: usize, bin: u32) {
        if media_idx >= self.media.len() {
            return;
        }
        let nbins = self.bin_names.len().max(1) as u32;
        let bin = bin.min(nbins - 1);
        if self.media_bin.len() <= media_idx {
            self.media_bin.resize(media_idx + 1, 0);
        }
        self.media_bin[media_idx] = bin;
    }

    /// P47 — relink media `i` to a new file path (Shotcut "Replace / Relink"). Every clip referencing
    /// media `i` keeps its index, so it now decodes the new file. No-op if `i` is out of range or the
    /// path is empty. Returns true on success.
    pub fn relink_media(&mut self, media_idx: usize, new_path: &str) -> bool {
        if media_idx >= self.media.len() || new_path.is_empty() {
            return false;
        }
        self.media[media_idx] = new_path.to_string();
        true
    }

    /// P48 — the active subtitle at timeline frame `t` (first whose `[start, end)` contains `t`), or
    /// None. The worker overlays its text as the top program layer during render/preview.
    pub fn active_subtitle_at(&self, t: i64) -> Option<&Subtitle> {
        self.subtitles.iter().find(|s| t >= s.start && t < s.end)
    }

    /// P49 — a transient, parent-less `Project` that composes sub-sequence `idx` (its clips + tracks
    /// over the PARENT's shared `media` pool) so the worker can render one of its frames via the
    /// normal compose path (`resolve_frame`/the RAW: fold). None if `idx` is out of range. The view
    /// carries NO subseqs of its own (one level of nesting), so a compound clip inside a sub-sequence
    /// would render as a gap rather than recursing infinitely.
    pub fn subseq_view(&self, idx: usize) -> Option<Project> {
        let ss = self.subseqs.get(idx)?;
        Some(Project {
            media: self.media.clone(),
            names: self.names.clone(),
            clips: ss.clips.clone(),
            tracks: ss.tracks.clone(),
            // IDENTITY program grade — `Project::default()` derives contrast/sat = 0.0 (f32 default),
            // which would crush the whole sub-sequence to mid-grey; the sub-timeline must composite
            // with a neutral grade (bright 0, contrast 1, sat 1) like a fresh project.
            bright: 0.0,
            contrast: 1.0,
            sat: 1.0,
            ..Project::default()
        })
    }

    // ----- keyframe eval (PINNED; consumed by Team A worker::resolve_frame) -------------
    // Both methods return the STATIC field when the relevant track has no keys, so a project
    // with no keyframes behaves exactly as before (worker can call these unconditionally).

    /// (bright, contrast, sat) at timeline frame `t`. Each component linearly interpolates its
    /// keyframe track; an empty track falls back to the static `bright`/`contrast`/`sat` field.
    pub fn grade_at(&self, t: i64) -> (f32, f32, f32) {
        (
            eval_track(&self.bright_kf, t, self.bright),
            eval_track(&self.contrast_kf, t, self.contrast),
            eval_track(&self.sat_kf, t, self.sat),
        )
    }

    /// V2-overlay opacity multiplier at timeline frame `t` from the `opacity_kf` track (Triad-B P1
    /// wiring of the previously-stored-but-unread track, model.rs:177). Returns 1.0 (fully opaque)
    /// when the track is EMPTY, so a project with no opacity keyframes composites the overlay exactly
    /// as before; otherwise it linearly interpolates the keyframes (clamped to the first/last value
    /// outside the range), mirroring `grade_at`. Consumed by worker::resolve_frame, which multiplies
    /// the overlay's composite `op` by this so a keyframed fade of the V2 overlay actually changes
    /// its opacity in BOTH the preview and the render.
    pub fn opacity_at(&self, t: i64) -> f32 {
        eval_track(&self.opacity_kf, t, 1.0)
    }

    /// P27: master audio-gain multiplier at timeline frame `t` from the `gain_kf` automation track.
    /// Empty track → 1.0 (flat, identity). The track stores LINEAR keys; the engine receives the
    /// keyframes (sec:gain) on the GAINENV line and linear-interps per audio sample.
    pub fn master_gain_at(&self, t: i64) -> f32 {
        eval_track(&self.gain_kf, t, 1.0)
    }

    /// P27: drop a master-gain automation key at timeline frame `t`, taking the CURRENT envelope
    /// value at `t` (so inserting a key is non-destructive) with LINEAR interp.
    pub fn add_gain_key(&mut self, t: i64) {
        let v = self.master_gain_at(t);
        set_track(&mut self.gain_kf, t, v, KfInterp::Linear);
    }

    /// (px, py, pw, ph) for clip `clip_idx` at CLIP-LOCAL frame `t_local`. Each param linearly
    /// interpolates its per-clip PiP keyframes; a param with no keys for this clip falls back to
    /// the clip's static `px`/`py`/`pw`/`ph`. An out-of-range `clip_idx` returns the full-frame
    /// default (0,0,1,1) so the worker never panics on a stale index.
    pub fn pip_at(&self, clip_idx: usize, t_local: i64) -> (f32, f32, f32, f32) {
        let (sx, sy, sw, sh) = match self.clips.get(clip_idx) {
            Some(c) => (c.px, c.py, c.pw, c.ph),
            None => (0.0, 0.0, 1.0, 1.0),
        };
        (
            self.eval_pip(clip_idx, 0, t_local, sx),
            self.eval_pip(clip_idx, 1, t_local, sy),
            self.eval_pip(clip_idx, 2, t_local, sw),
            self.eval_pip(clip_idx, 3, t_local, sh),
        )
    }

    /// Interpolate one (clip, param) PiP track from the flat `pip_kf` store at clip-local frame
    /// `t`. Mirrors MojoMedia pip_eval: scan the flat list for matching (clip,par) entries,
    /// track the nearest key at/below `t` (lo) and above `t` (hi), then linearly blend; empty
    /// -> `fallback`, clamp to lo/hi at the ends. The flat list is unsorted, so this is an O(n)
    /// scan (n = total PiP keys, small) rather than a binary search.
    fn eval_pip(&self, clip: usize, par: u8, t: i64, fallback: f32) -> f32 {
        // P19: build the (clip,par) track as a sorted `Vec<Kf>` (t_local as the frame) and delegate
        // to the unified `eval_track`, so the flat PiP store gets the SAME interpolation as grade
        // tracks — including the neighbour-aware Catmull-Rom variants (eval_track needs the keys in
        // ascending order to find each segment's neighbours). The flat list is small, so the
        // collect+sort per query is cheap. Empty -> fallback; endpoint clamp handled by eval_track.
        let mut keys: Vec<Kf> = self
            .pip_kf
            .iter()
            .filter(|k| k.clip == clip && k.par == par)
            .map(|k| Kf {
                t: k.t_local,
                v: k.v,
                interp: k.interp,
            })
            .collect();
        keys.sort_by_key(|k| k.t);
        eval_track(&keys, t, fallback)
    }

    // ----- keyframe edit ops (Slice C; called by panels::properties_ui Key buttons) -----

    /// Snapshot the CURRENT static grade (bright/contrast/sat) into a keyframe at timeline
    /// frame `t` on all three grade tracks (sorted insert-or-replace). Mirrors MojoMedia's
    /// "K" buttons keying brightness/contrast/saturation at the playhead with the live values.
    pub fn add_grade_key(&mut self, t: i64) {
        let (b, c, s) = (self.bright, self.contrast, self.sat);
        // P14: the new keys take the project's CURRENT create mode. Same signature as before —
        // no caller (panels.rs) changes — the mode is read off the Project, not passed in.
        let interp = self.kf_interp;
        set_track(&mut self.bright_kf, t, b, interp);
        set_track(&mut self.contrast_kf, t, c, interp);
        set_track(&mut self.sat_kf, t, s, interp);
    }

    /// Snapshot clip `clip_idx`'s CURRENT static PiP rect (px/py/pw/ph) into PiP keyframes at
    /// CLIP-LOCAL frame `t_local` (one per param 0..3, insert-or-replace). Mirrors MojoMedia's
    /// "Key PiP" button keying all four params at the clip-local frame. No-op for a bad index.
    pub fn add_pip_key(&mut self, clip_idx: usize, t_local: i64) {
        let (px, py, pw, ph) = match self.clips.get(clip_idx) {
            Some(c) => (c.px, c.py, c.pw, c.ph),
            None => return,
        };
        // P14: all four new param keys take the project's CURRENT create mode. Signature
        // unchanged — the mode is read off the Project, so panels.rs's call site is untouched.
        let interp = self.kf_interp;
        self.set_pip(clip_idx, 0, t_local, px, interp);
        self.set_pip(clip_idx, 1, t_local, py, interp);
        self.set_pip(clip_idx, 2, t_local, pw, interp);
        self.set_pip(clip_idx, 3, t_local, ph, interp);
    }

    /// Insert-or-replace a single PiP keyframe in the flat store (mirrors MojoMedia pip_set):
    /// overwrite the value AND `interp` if an entry already exists for (clip, par, t_local), else
    /// append. P14: `interp` is the current create mode (`Project.kf_interp`) threaded through by
    /// `add_pip_key`, so re-keying a frame while the mode is Smooth makes that key Smooth.
    fn set_pip(&mut self, clip: usize, par: u8, t_local: i64, v: f32, interp: KfInterp) {
        if let Some(k) = self
            .pip_kf
            .iter_mut()
            .find(|k| k.clip == clip && k.par == par && k.t_local == t_local)
        {
            // replace — value AND interp follow the current create mode.
            k.v = v;
            k.interp = interp;
        } else {
            self.pip_kf.push(PipKey { clip, par, t_local, v, interp });
        }
    }

    /// Count of PiP keyframes bound to clip `clip_idx` (any param) — for the panel's key-count
    /// readout. O(n) over the small flat store.
    pub fn pip_key_count(&self, clip_idx: usize) -> usize {
        self.pip_kf.iter().filter(|k| k.clip == clip_idx).count()
    }

    // ----- P30 PER-CLIP VIDEO-PARAM KEYFRAMING (par 4..9; see PipKey registry) ----------------------
    // These three helpers expose the EXISTING flat `pip_kf` store + generic `eval_pip` for the new
    // per-clip video params (bright/contrast/sat/blur/rot/scale) so the worker can send the
    // INTERPOLATED value per frame instead of the static `Clip` field. No engine/wire change: the
    // worker already emits each clip's bright/contrast/sat/blur/rot/scale per frame — keyframing
    // only changes WHICH value is emitted (the interpolated key value vs the static field). A clip
    // with no keys for a param => `clip_param_at` returns `fallback` (the static field) => byte-
    // identical to pre-P30. `pip_kf` is already `#[serde(default)]`, so older projects deserialize
    // with an empty store and resolve to their static fields unchanged.

    /// Interpolated value of one per-clip video param at CLIP-LOCAL frame `t_local`, or `fallback`
    /// (the clip's static field) when the clip has no keyframes for `par`. Public entry point for the
    /// P30 video params (par 4=bright, 5=contrast, 6=sat, 7=blur, 8=rot, 9=scale); thin wrapper over
    /// the generic, private `eval_pip` (which the PiP rect path 0..3 uses too) so the new params get
    /// the SAME full 36-mode interpolation as the PiP/grade tracks. The worker passes the matching
    /// static `Clip` field as `fallback`, so a no-keys clip emits exactly the value it does today.
    pub fn clip_param_at(&self, clip: usize, par: u8, t_local: i64, fallback: f32) -> f32 {
        self.eval_pip(clip, par, t_local, fallback)
    }

    /// Snapshot the live value `v` of one per-clip video param into a keyframe at CLIP-LOCAL frame
    /// `t_local` (insert-or-replace). Mirrors `add_pip_key` but for a SINGLE param (the panel's per-
    /// slider "◆" buttons pass the slider's current field value); the new key takes the project's
    /// current create-mode interp (`kf_interp`), exactly like `add_pip_key`/`add_grade_key`.
    pub fn add_clip_param_key(&mut self, clip: usize, par: u8, t_local: i64, v: f32) {
        self.set_pip(clip, par, t_local, v, self.kf_interp);
    }

    /// Count of keyframes bound to (clip, par) — per-param, for the panel's per-slider key readout.
    /// (`pip_key_count` counts ALL params for a clip; this isolates a single param.) O(n) over the
    /// small flat store.
    pub fn clip_param_key_count(&self, clip: usize, par: u8) -> usize {
        self.pip_kf
            .iter()
            .filter(|k| k.clip == clip && k.par == par)
            .count()
    }

    // ----- keyframe DRAG/DELETE edit ops (Slice B; called by timeline diamond/tick drags) -----
    // `track` selects the grade keyframe track: 0 = bright_kf, 1 = contrast_kf, 2 = sat_kf,
    // 3 = opacity_kf. All ops bounds-check (bad track / idx -> no-op) so a stale index from a
    // mid-drag mutation can never panic. PiP ops address the flat `pip_kf` store by index.

    /// Mutable borrow of one grade track by its PINNED index, or `None` for an out-of-range
    /// track. Internal helper for the move/delete ops below.
    fn grade_track_mut(&mut self, track: u8) -> Option<&mut Vec<Kf>> {
        match track {
            0 => Some(&mut self.bright_kf),
            1 => Some(&mut self.contrast_kf),
            2 => Some(&mut self.sat_kf),
            3 => Some(&mut self.opacity_kf),
            4 => Some(&mut self.gain_kf),
            _ => None,
        }
    }

    /// Move grade keyframe `idx` of `track` to timeline frame `new_t` (clamped to `>= 0`), then
    /// re-sort that track ascending by `t` so eval/draw stay correct (mirrors MojoMedia kf_set's
    /// sorted ordering — here applied to a moved key rather than a fresh one). The key keeps its
    /// VALUE; only its frame changes. No-op for a bad track or `idx`. If the move lands the key
    /// exactly onto another key's frame, BOTH are kept (a stable sort preserves their relative
    /// order) — eval_track still returns a well-defined value, and a later add_grade_key at that
    /// frame would collapse them via set_track's replace.
    pub fn move_grade_key(&mut self, track: u8, idx: usize, new_t: i64) {
        let nt = new_t.max(0);
        if let Some(t) = self.grade_track_mut(track) {
            if idx < t.len() {
                t[idx].t = nt;
                // stable sort by frame so the moved key slots into ascending order. `sort_by_key`
                // is GUARANTEED stable by std (do NOT swap to `sort_unstable_by_key`): the doc
                // invariant "two keys landing on the same frame keep their relative order" relies
                // on it, so a moved key dropped exactly onto another's frame stays well-defined.
                t.sort_by_key(|k| k.t);
            }
        }
    }

    /// Delete grade keyframe `idx` of `track` (no-op for a bad track or `idx`). Removing a key
    /// keeps the track sorted (Vec::remove preserves order).
    pub fn delete_grade_key(&mut self, track: u8, idx: usize) {
        if let Some(t) = self.grade_track_mut(track) {
            if idx < t.len() {
                t.remove(idx);
            }
        }
    }

    /// Move PiP keyframe `idx` (index into the flat `pip_kf` store) to clip-local frame
    /// `new_t_local` (clamped to `>= 0`). The flat store is unsorted (eval_pip scans it), so no
    /// re-sort is needed; only the entry's `t_local` changes (its clip/par/value are preserved).
    /// No-op for an out-of-range `idx`.
    pub fn move_pip_key(&mut self, idx: usize, new_t_local: i64) {
        if let Some(k) = self.pip_kf.get_mut(idx) {
            k.t_local = new_t_local.max(0);
        }
    }

    /// Delete PiP keyframe `idx` from the flat store (no-op for an out-of-range `idx`).
    pub fn delete_pip_key(&mut self, idx: usize) {
        if idx < self.pip_kf.len() {
            self.pip_kf.remove(idx);
        }
    }

    // ----- per-track state helpers (P5: read Project.tracks) -----------------------------------
    // `track` is the Clip.track index into Project.tracks. Each helper bounds-checks the index
    // (an out-of-range track is treated as visible / audible / unlocked / video) so callers never
    // index out of bounds.

    /// True if the given track's VIDEO is hidden (skipped in base/over resolution).
    pub fn is_hidden(&self, track: u8) -> bool {
        self.tracks.get(track as usize).is_some_and(|t| t.hidden)
    }

    /// True if the given track's AUDIO is muted (contributes nothing to the render).
    pub fn is_muted(&self, track: u8) -> bool {
        self.tracks.get(track as usize).is_some_and(|t| t.muted)
    }

    /// True if edits to the given track are blocked.
    pub fn is_locked(&self, track: u8) -> bool {
        self.tracks.get(track as usize).is_some_and(|t| t.locked)
    }

    /// True if the given track is an AUDIO track (its clips contribute audio, not video).
    pub fn is_audio(&self, track: u8) -> bool {
        self.tracks.get(track as usize).is_some_and(|t| t.kind == TrackKind::Audio)
    }

    /// P42 mixer — the track's linear fader gain (1.0 = unity / out-of-range default). Folded into
    /// each clip's emitted per-clip gain by the worker's audio-emit loops.
    pub fn track_gain(&self, track: u8) -> f32 {
        self.tracks.get(track as usize).map_or(1.0, |t| t.gain)
    }

    /// P42 mixer — the track's L/R pan balance in [-1,1] (0 = centre / out-of-range default). Folded
    /// into the clip's audio fx_chain (stereotools balance_out) by the worker's audio-emit loops.
    pub fn track_pan(&self, track: u8) -> f32 {
        self.tracks.get(track as usize).map_or(0.0, |t| t.pan)
    }

    /// P42 mixer — true if the given track's solo flag is set.
    pub fn is_solo(&self, track: u8) -> bool {
        self.tracks.get(track as usize).is_some_and(|t| t.solo)
    }

    /// P42 mixer — true if ANY track is soloed. When true, only soloed (non-muted) tracks are
    /// audible (standard mixer solo behaviour); when false, solo has no effect.
    pub fn any_solo(&self) -> bool {
        self.tracks.iter().any(|t| t.solo)
    }

    /// Number of tracks (timeline lane count).
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Append a new track of `kind` (named V/A + the next ordinal). Returns its index.
    pub fn add_track(&mut self, kind: TrackKind) -> usize {
        let n = self.tracks.iter().filter(|t| t.kind == kind).count() + 1;
        let name = match kind {
            TrackKind::Video => format!("V{n}"),
            TrackKind::Audio => format!("A{n}"),
        };
        self.tracks.push(Track::new(kind, &name));
        self.tracks.len() - 1
    }

    /// Remove track `idx`: drop all clips on it and decrement the `track` index of every clip on a
    /// higher track (and rebase transitions/keyframes that key by track). No-op for the last track
    /// or an out-of-range index. Returns true if a track was removed.
    pub fn remove_track(&mut self, idx: usize) -> bool {
        if idx >= self.tracks.len() || self.tracks.len() <= 1 {
            return false;
        }
        let ti = idx as u8;
        // Remove clips on the track (descending so delete_clip's index/PiP remap stays valid).
        let doomed: Vec<usize> =
            self.clips.iter().enumerate().filter(|(_, c)| c.track == ti).map(|(i, _)| i).collect();
        for &i in doomed.iter().rev() {
            self.delete_clip(i);
        }
        // Shift higher clips + transitions down one track.
        for c in self.clips.iter_mut() {
            if c.track > ti {
                c.track -= 1;
            }
        }
        self.transitions.retain(|tr| tr.track != ti);
        for tr in self.transitions.iter_mut() {
            if tr.track > ti {
                tr.track -= 1;
            }
        }
        self.tracks.remove(idx);
        true
    }

    // ----- per-boundary transitions (Wave 8; PINNED) -----------------------------------------
    // Transitions are keyed by (track, center frame), NOT by clip index — so unlike the PiP
    // keyframe store they need NO remap on split/delete/reorder; a transition simply animates
    // whatever two clips happen to straddle its window. Callers pass CURRENT clip indices (from
    // boundaries()) into resolve and never cache them across a mutation.

    /// The transition on `track` whose window `[center - dur/2, center + dur/2)` contains `t`,
    /// or `None`. If several overlap `t` (windows can overlap after editing), the one whose
    /// `center` is NEAREST to `t` wins (and on a tie, the earliest in the list) so the worker
    /// gets a single deterministic transition per frame. Consumed by Team A (resolve_frame) to
    /// fill the ENC/PREVIEW `<trans_kind> <trans_prog> <trans_param> <trans_path> <trans_frame>`
    /// fields and by Team C (timeline) to highlight the active boundary.
    pub fn transition_at(&self, track: u8, t: i64) -> Option<&Transition> {
        self.transitions
            .iter()
            .filter(|tr| tr.track == track && tr.contains(t))
            .min_by_key(|tr| (tr.center - t).abs())
    }

    /// Add a transition on `track` centered at `center` over `dur` frames (clamped to `>= 2`)
    /// with the given `kind`. If a transition already exists on the SAME `track` at the SAME
    /// `center`, it is replaced in place (its `dur`/`kind` updated) rather than duplicated —
    /// mirroring MojoMedia's per-boundary "Cycle transition type" which mutates the existing
    /// boundary entry. Otherwise the new transition is pushed.
    pub fn add_transition(&mut self, track: u8, center: i64, dur: i64, kind: i32) {
        let d = dur.max(2);
        if let Some(tr) = self
            .transitions
            .iter_mut()
            .find(|tr| tr.track == track && tr.center == center)
        {
            tr.dur = d;
            tr.kind = kind;
        } else {
            self.transitions.push(Transition { track, center, dur: d, kind });
        }
    }

    /// Remove transition `idx` from the store (no-op for an out-of-range `idx`). `Vec::remove`
    /// preserves the relative order of the remaining transitions.
    pub fn remove_transition(&mut self, idx: usize) {
        if idx < self.transitions.len() {
            self.transitions.remove(idx);
        }
    }

    /// Adjacent/overlapping same-track clip boundaries as `(outgoing_clip_idx, incoming_clip_idx,
    /// boundary_frame)`. Scans the clips on `track` ordered by `t0` (without reordering the
    /// underlying `clips` Vec — indices in the result are into `self.clips`) and, for each
    /// consecutive pair (A then B) that touch or overlap within `BOUNDARY_GAP` frames, emits a
    /// boundary. The boundary frame is `A.end()` when the clips merely abut/gap, or the MIDPOINT
    /// of the overlap when they overlap (so a centered transition window straddles the seam).
    /// Consumed by Team C to place/seed transitions and by Team A to find the incoming partner
    /// clip to stage into slot 2.
    pub fn boundaries(&self, track: u8) -> Vec<(usize, usize, i64)> {
        // Collect (clip_index, t0, end) for this track, then sort by t0 (then end) for a stable
        // left-to-right scan. We sort a list of indices so the emitted indices stay valid into
        // self.clips.
        let mut order: Vec<usize> = (0..self.clips.len())
            .filter(|&i| self.clips[i].track == track)
            .collect();
        order.sort_by_key(|&i| (self.clips[i].t0, self.clips[i].end()));

        let mut out = Vec::new();
        for w in order.windows(2) {
            let a = w[0];
            let b = w[1];
            let a_end = self.clips[a].end();
            let b_t0 = self.clips[b].t0;
            let overlap = b_t0 < a_end; // B starts before A ends
            let gap = (b_t0 - a_end).abs();
            if overlap || gap <= BOUNDARY_GAP {
                // overlap -> seam at the midpoint of [b_t0, a_end); abut/gap -> at A.end().
                let boundary = if overlap {
                    (b_t0 + a_end) / 2
                } else {
                    a_end
                };
                out.push((a, b, boundary));
            }
        }
        out
    }

    // ----- on-clip fade + transition-length gateable clamp layer ------------------------------
    // The TIMELINE drag handles (timeline.rs) set Clip.fade_in/fade_out and Transition.dur through
    // THESE ops rather than mutating the fields inline, so the clamps live in one tested place. All
    // three are all-or-nothing and OOR-safe (return false + no mutation on a bad index), mirroring
    // the nudge_clip / replace_clip / detach_audio bool-edit-op style. The fade fields are already
    // drawn (draw_clip_fades) and applied to video opacity + audio gain at mix; `dur` is the
    // existing >=2-clamped transition window length (see add_transition).

    /// Set clip `clip_idx`'s FADE-IN length to `frames`, clamped to `[0, len]` (a fade can be at
    /// most the whole clip). Returns false (no mutation) when `clip_idx` is out of range.
    pub fn set_fade_in(&mut self, clip_idx: usize, frames: i64) -> bool {
        let Some(c) = self.clips.get_mut(clip_idx) else {
            return false;
        };
        c.fade_in = frames.clamp(0, c.len);
        true
    }

    /// Set clip `clip_idx`'s FADE-OUT length to `frames`, clamped to `[0, len]`. Returns false (no
    /// mutation) when `clip_idx` is out of range. Symmetric to `set_fade_in`.
    pub fn set_fade_out(&mut self, clip_idx: usize, frames: i64) -> bool {
        let Some(c) = self.clips.get_mut(clip_idx) else {
            return false;
        };
        c.fade_out = frames.clamp(0, c.len);
        true
    }

    /// Set transition `trans_idx`'s window length `dur` to `dur.max(2)` (the window stays >= 2,
    /// matching `add_transition`'s clamp). Returns false (no mutation) when `trans_idx` is out of
    /// range.
    pub fn set_transition_dur(&mut self, trans_idx: usize, dur: i64) -> bool {
        let Some(tr) = self.transitions.get_mut(trans_idx) else {
            return false;
        };
        tr.dur = dur.max(2);
        true
    }

    // ----- edit ops -----------------------------------------------------
    // Mirror MojoMedia editor/main_editor.mojo: positioned clips (explicit t0),
    // split keeps src_in/len/t0 math identical, trims clamp len>=1 and t0>=0.

    /// Append a clip to the timeline.
    pub fn add_clip(&mut self, clip: Clip) {
        self.clips.push(clip);
    }

    /// Remove clip `i` (no-op if out of range). Also keeps the flat PiP keyframe store
    /// clip-stable: drop every PiP key bound to the removed clip, then shift the `clip` index
    /// of keys for higher clips down by one so they keep pointing at the same clip after the
    /// `Vec::remove`. (Grade keys are program-wide and need no remap.)
    pub fn delete_clip(&mut self, i: usize) {
        if i < self.clips.len() {
            self.clips.remove(i);
            self.pip_kf.retain(|k| k.clip != i);
            for k in self.pip_kf.iter_mut() {
                if k.clip > i {
                    k.clip -= 1;
                }
            }
        }
    }

    /// Split clip `i` at timeline frame `t` into two clips. The second clip starts at
    /// `t` with `src_in += off`, `len -= off`, `t0 = t`, same track/look/etc. Mirrors
    /// the razor in main_editor.mojo (`off = cur_T - seg_t0`; split only when 0 < off < len).
    /// Returns the index of the new (right-hand) clip if a split occurred.
    pub fn split_clip(&mut self, i: usize, t: i64) -> Option<usize> {
        if i >= self.clips.len() {
            return None;
        }
        let off = t - self.clips[i].t0;
        if off <= 0 || off >= self.clips[i].len {
            return None; // playhead not strictly inside the clip body
        }
        let mut right = self.clips[i].clone();
        // left half: shorten to the cut point
        self.clips[i].len = off;
        // right half: advance source + start, shorten remaining length
        right.src_in += off;
        right.len -= off;
        right.t0 = self.clips[i].t0 + off;
        // a fresh fade-in on the right half is not inherited from the left's fade-in
        right.fade_in = 0;
        // the left half no longer carries a fade-out (that belongs to the right edge now)
        self.clips[i].fade_out = 0;
        let idx = i + 1;
        self.clips.insert(idx, right);
        // PiP keyframe clip-stability: inserting the right half at `idx` shifts every clip
        // above `i` up by one, so bump the `clip` index of keys for those clips to match. Keys
        // on the split clip itself (clip == i) stay with the LEFT half — their clip-local
        // frames still measure from the same t0, so they animate the (now shorter) left clip.
        // The right half starts with no PiP keys (it inherits the static rect, like a fresh
        // fade-in), mirroring MojoMedia's razor which does not duplicate per-clip keyframes.
        for k in self.pip_kf.iter_mut() {
            if k.clip > i {
                k.clip += 1;
            }
        }
        Some(idx)
    }

    /// RAZOR ALL TRACKS at timeline frame `t` (Shotcut "Split All Tracks", Shift+S): split EVERY
    /// clip on EVERY track that STRICTLY spans `t` (`t0 < t < end()`). Reuses `split_clip` verbatim
    /// for each cut, so the new right half keeps source continuity (`right.src_in = left.src_in +
    /// (t - left.t0)`), the fade-in/out hand-off matches the single-clip razor, and the flat PiP
    /// keyframe store stays clip-stable through each insert. A clip that only TOUCHES `t` at an edge
    /// (`t == t0` or `t == end()`) is left untouched — `split_clip`'s own `0 < off < len` guard
    /// rejects it. Returns the number of splits performed.
    ///
    /// Walk note: `split_clip(i, t)` inserts the right half at `i+1`, shifting every higher index up
    /// by one. That right half starts AT `t` (its `t0 == t`), so it can never itself span `t`; we
    /// therefore advance past it (`i += 2` after a split) and otherwise step by one. This visits
    /// each ORIGINAL clip exactly once with no double-splitting. Splitting is independent per track:
    /// `split_clip` only ever touches clip `i` and inserts beside it, never reordering across tracks.
    pub fn split_all_at(&mut self, t: i64) -> usize {
        let mut splits = 0usize;
        let mut i = 0usize;
        while i < self.clips.len() {
            let c = &self.clips[i];
            // Strict span test (mirrors split_clip's 0 < off < len with off = t - t0).
            if c.t0 < t && t < c.end() {
                if self.split_clip(i, t).is_some() {
                    splits += 1;
                    i += 2; // skip the freshly-inserted right half (it starts AT t, can't span t)
                    continue;
                }
            }
            i += 1;
        }
        splits
    }

    /// FRAME NUDGE: move clip `clip_idx` by `delta` frames along its own track (the keyboard ±1-frame
    /// nudge, Alt+Left / Alt+Right or `,` / `.`). Overlaps are allowed — like a free body-drag move —
    /// the only constraint is `t0 >= 0` (a left nudge past frame 0 clamps to 0; the length is never
    /// changed). Touches ONLY this clip (single-track free move; not a ripple). Returns false if
    /// `clip_idx` is out of range, true once the move (possibly clamped) is applied.
    pub fn nudge_clip(&mut self, clip_idx: usize, delta: i64) -> bool {
        let Some(c) = self.clips.get_mut(clip_idx) else {
            return false;
        };
        c.t0 = (c.t0 + delta).max(0);
        true
    }

    /// T4 FREEZE-FRAME (Shotcut "Freeze Frame" / hold): at timeline frame `t` STRICTLY inside clip
    /// `clip_idx` (`C.t0 < t < C.end()`), SPLIT C at `t` then INSERT a `dur`-frame freeze clip at `t`
    /// that HOLDS C's source frame at `t` as a silent still, rippling the right half + all later
    /// same-track clips RIGHT by `dur`. The freeze clip reuses EXISTING fields only:
    ///   - `src_in = C.src_in + (t - C.t0)` — the exact source frame visible at `t` (matches the
    ///     razor's `right.src_in = left.src_in + (t - left.t0)` continuity, so frame 0 of the freeze
    ///     is the same source frame the right half would have started on),
    ///   - `len = dur`,
    ///   - `speed = 0.0` — the P24 time-remap rate; speed 0 reads the SAME source frame for every
    ///     timeline frame of the clip (a held still). The worker's speed/atempo path guards against
    ///     a 0 divisor (see worker.rs clamp): video pins `src_in`, audio is silent via `gain`,
    ///   - `gain = 0.0` — the per-clip AUDIO gain; a held still is SILENT.
    /// Every other field is inherited from C (same media, grade, look, filters, track), so the still
    /// looks like a frozen C. Returns `false` (nothing mutated) when `clip_idx` is out of range, `dur
    /// <= 0`, or `t` is not STRICTLY inside C (`t <= C.t0` or `t >= C.end()`); `true` on success.
    ///
    /// Mechanism: we lean on `split_clip(clip_idx, t)`, which cuts C into a left half (keeps t0, len
    /// shortened to `t - C.t0`) and a right half inserted at `clip_idx+1` starting exactly at `t`
    /// with source continuity. We then shift every same-track clip whose `t0 >= t` (which now
    /// includes that right half) RIGHT by `dur` to open a `dur` hole at `t`, and push the freeze clip
    /// (built from C's pre-split fields) into that hole. Net timeline length grows by `dur`, and a
    /// new (3rd, on this track) clip with `speed==0 && gain==0 && len==dur` sits at `t0==t`.
    /// Edit-only of existing fields → pre-freeze `.json` projects load + render byte-identical
    /// (nothing runs unless the gesture fires).
    pub fn freeze_frame(&mut self, clip_idx: usize, t: i64, dur: i64) -> bool {
        // Guard: valid clip, positive duration, and `t` STRICTLY inside C's body.
        let Some(c) = self.clips.get(clip_idx) else {
            return false;
        };
        if dur <= 0 || t <= c.t0 || t >= c.end() {
            return false;
        }
        // Source frame visible at `t` (same continuity rule as the razor's right half).
        let freeze_src_in = c.src_in + (t - c.t0);
        let track = c.track;
        // Build the freeze still from C's fields (inherits media/grade/look/filters/track/...).
        let mut freeze = c.clone();
        freeze.t0 = t;
        freeze.len = dur;
        freeze.src_in = freeze_src_in;
        freeze.speed = 0.0; // HOLD the source frame (P24: speed 0 = no source advance)
        freeze.reverse = false; // a held still has no direction
        freeze.gain = 0.0; // SILENT still (per-clip audio gain 0)
        // A fresh independent still: no fades, not linked into C's group.
        freeze.fade_in = 0;
        freeze.fade_out = 0;
        freeze.group = 0;

        // Cut C at `t`: left half keeps [C.t0, t), right half (inserted at clip_idx+1) starts at `t`.
        // This must succeed because `t` is strictly inside C (the guard above mirrors split_clip's
        // own `0 < off < len` test exactly), so split_clip returns Some(_).
        let _ = self.split_clip(clip_idx, t);
        // Open a `dur` hole at `t`: shift every same-track clip at/after `t` (incl. C's right half)
        // RIGHT by `dur`. The freeze clip is not in `clips` yet, so nothing is excluded.
        self.shift_after(track, t, dur, usize::MAX);
        // Drop the freeze still into the opened hole (appended last; its index never collides with a
        // remapped PiP key — same contract as insert_clip / overwrite_clip).
        self.clips.push(freeze);
        true
    }

    /// Trim the start of clip `i` to a new timeline start `new_t0`, holding the right
    /// edge fixed. Advances src_in by the delta and shortens len, with len>=1 / t0>=0
    /// clamping. Mirrors a left-edge trim (the inverse of the right-edge trim drag).
    pub fn trim_start(&mut self, i: usize, new_t0: i64) {
        if i >= self.clips.len() {
            return;
        }
        let end = self.clips[i].end(); // right edge stays put
        let src_in = self.clips[i].src_in;
        let t0 = self.clips[i].t0;
        let mut nt0 = new_t0.max(0);
        if nt0 >= end {
            nt0 = end - 1; // keep len >= 1
        }
        // Frames trimmed off the head (negative = extend left). On an extend, the source
        // can only supply `src_in` extra frames before hitting frame 0; clamp the timeline
        // extension to that so len/src_in/t0 stay consistent (no out-of-range source request
        // in worker.rs). This mirrors MojoMedia holding the trim within available source.
        let mut delta = nt0 - t0;
        if delta < 0 && src_in + delta < 0 {
            delta = -src_in; // only extend by what the source has
            nt0 = t0 + delta;
        }
        self.clips[i].src_in = (src_in + delta).max(0);
        self.clips[i].t0 = nt0;
        self.clips[i].len = (end - nt0).max(1);
    }

    /// Trim the end of clip `i` to a new length `new_len` (right-edge drag). Clamps to
    /// MIN_CLIP frames (matching MojoMedia's `nl < 15 → 15` trim floor) so dragging the
    /// right edge left past the start — or snapping it to an earlier edge — cannot collapse
    /// the clip below a usable length.
    pub fn trim_end(&mut self, i: usize, new_len: i64) {
        if i >= self.clips.len() {
            return;
        }
        self.clips[i].len = new_len.max(MIN_CLIP);
    }

    // ----- RIPPLE edit ops (P3 editing slice; Shotcut "Ripple Delete" / ripple trim) ----------
    // Unlike the LIFT (`delete_clip`, leaves a gap) and the plain trims above (hold the far edge,
    // leave/open a gap), the RIPPLE ops CLOSE the gap on the SAME track: deleting/shortening a clip
    // shifts every later same-track clip left so the timeline has no hole; extending shifts them
    // right. Other tracks are never touched (Shotcut's default is ripple-current-track-only; the
    // "ripple all tracks" setting is out of scope this wave). PiP keyframes are clip-stable through
    // the underlying `delete_clip` (which already remaps the flat `pip_kf` store); the t0 shifts
    // here move clips on the timeline but keep their CLIP-LOCAL keyframes (t_local) intact, so PiP
    // animation rides along with the rippled clip exactly as it would with a body-drag move.

    /// Shift every clip on `track` whose `t0 >= from` by `delta` frames (t0 clamped to `>= 0`).
    /// Internal helper for the ripple ops: closes/opens the gap left by a ripple delete/trim. A
    /// `delta < 0` ripples earlier (gap-close); `delta > 0` ripples later (gap-open). `skip` is a
    /// clip index NOT to move (the clip whose edit triggered the ripple, when it must stay put);
    /// pass `usize::MAX` to move nothing-excluded.
    fn shift_after(&mut self, track: u8, from: i64, delta: i64, skip: usize) {
        if delta == 0 {
            return;
        }
        for (k, c) in self.clips.iter_mut().enumerate() {
            if k != skip && c.track == track && c.t0 >= from {
                c.t0 = (c.t0 + delta).max(0);
            }
        }
    }

    /// RIPPLE DELETE clip `i`: remove it AND shift every later same-track clip left by the deleted
    /// clip's length, closing the gap (Shotcut "Ripple Delete", X / Shift+Delete). Contrast with
    /// `delete_clip` (the LIFT), which removes the clip but leaves a hole. The downstream shift uses
    /// the deleted clip's `end()` as the cutoff and `-len` as the delta, computed BEFORE the
    /// `delete_clip` call (which renumbers higher clip indices via its PiP remap). No-op out of range.
    // Retained single-clip helper: the UI ripple-delete (X / Shift+Del) routes through
    // `ripple_delete_many` (handles multi-select + index/t0-order correctly), so this is unwired but
    // kept as the documented single-clip contract `ripple_delete_many` contrasts against.
    #[allow(dead_code)]
    pub fn ripple_delete(&mut self, i: usize) {
        let (track, end, len) = match self.clips.get(i) {
            Some(c) => (c.track, c.end(), c.len),
            None => return,
        };
        // Close the gap first (the surviving downstream clips keep their indices through this), then
        // remove the clip itself (which renumbers the higher indices + remaps PiP keys).
        self.shift_after(track, end, -len, i);
        self.delete_clip(i);
    }

    /// RIPPLE DELETE a SET of clips at once, closing the gap on each affected track correctly
    /// REGARDLESS of how the clips' Vec-index order relates to their t0 order (skeptic #1). The
    /// single-clip `ripple_delete` cannot be applied in a loop for a multi-clip selection: its
    /// per-call gap-close is t0-based (`shift_after`), so deleting one clip moves the *t0/end* of
    /// the other still-selected clips before they are removed, and the per-call `end()` cutoffs
    /// compound — over-shifting survivors when index order ≠ t0 order (e.g. two same-track clips
    /// where the lower index has the higher t0). Shotcut closes a multi-ripple by accumulating the
    /// removed length per track and shifting each survivor by the total removed length that sits
    /// at/before it. This does exactly that, atomically:
    ///   1. snapshot every selected clip's `(track, t0, len)` BEFORE any mutation;
    ///   2. remove all selected clips in ONE pass (descending index so Vec indices stay valid and
    ///      `delete_clip`'s PiP-key remap stays correct);
    ///   3. for each SURVIVING clip, subtract from its `t0` the summed `len` of every removed clip
    ///      on the SAME track whose original `t0 <= survivor.t0` (clamped to `>= 0`).
    /// Other tracks are untouched (ripple-current-track-only, like `ripple_delete`). PiP keyframes
    /// ride along with their clips via `delete_clip`'s remap; only survivor `t0`s move here, so
    /// clip-local keyframes are preserved exactly as in the single-clip path. Out-of-range / dup
    /// indices are ignored. No-op (no mutation) when `indices` selects nothing valid.
    pub fn ripple_delete_many(&mut self, indices: &[usize]) {
        // Gather valid, unique indices and snapshot the removed clips' (track, t0, len) up front.
        let mut idxs: Vec<usize> = Vec::new();
        for &i in indices {
            if i < self.clips.len() && !idxs.contains(&i) {
                idxs.push(i);
            }
        }
        if idxs.is_empty() {
            return;
        }
        // Removed-clip snapshots: their original positions, captured before any index renumbering.
        let removed: Vec<(u8, i64, i64)> =
            idxs.iter().map(|&i| (self.clips[i].track, self.clips[i].t0, self.clips[i].len)).collect();

        // Remove all selected clips in descending index order: higher indices first keeps the lower
        // indices (and `delete_clip`'s `clip > i` PiP remap) valid through the whole batch.
        idxs.sort_unstable();
        for &i in idxs.iter().rev() {
            self.delete_clip(i);
        }

        // Close the gaps: each survivor slides left by the total removed length on its track that
        // sat at/before its ORIGINAL t0. Using the pre-delete snapshot makes the result independent
        // of deletion order and of any index↔t0 mismatch.
        for c in self.clips.iter_mut() {
            let shift: i64 = removed
                .iter()
                .filter(|&&(rt, rt0, _)| rt == c.track && rt0 <= c.t0)
                .map(|&(_, _, rl)| rl)
                .sum();
            if shift != 0 {
                c.t0 = (c.t0 - shift).max(0);
            }
        }
    }

    /// RIPPLE TRIM the START of clip `i` to a new timeline start `new_t0`, then shift the downstream
    /// same-track clips by the resulting length delta so the gap stays closed (Shotcut ripple
    /// trim-in). Holds the clip's RIGHT edge via `trim_start` (which advances src_in + reshapes len),
    /// then moves every later same-track clip by `old_len - new_len` (a head trim that SHORTENS the
    /// clip ripples downstream LEFT; extending the head ripples them RIGHT). No-op out of range.
    pub fn ripple_trim_start(&mut self, i: usize, new_t0: i64) {
        let (track, old_t0, old_end) = match self.clips.get(i) {
            Some(c) => (c.track, c.t0, c.end()),
            None => return,
        };
        self.trim_start(i, new_t0);
        // Head-trim amount actually applied (trim_start clamps new_t0 into [0, end-1] / source limits).
        let d = self.clips[i].t0 - old_t0;
        // A head trim HOLDS the right edge, so it opens a gap at the FRONT (between old_t0 and the
        // new t0), NOT downstream. Ripple = close that gap: re-anchor the (now shorter) clip at its
        // original start and slide every later same-track clip left by the same amount, keeping the
        // sequence tight. Mirrors ripple_trim_end (clip stays anchored; followers ride the delta).
        self.clips[i].t0 = old_t0;
        self.shift_after(track, old_end, -d, i);
    }

    /// RIPPLE TRIM the END of clip `i` to a new length `new_len`, then shift the downstream same-track
    /// clips by the length delta so the gap stays closed (Shotcut ripple trim-out). `trim_end`
    /// applies the MIN_CLIP floor; we read the clip's ACTUAL new length back (so the ripple matches
    /// what was really applied) and shift every later same-track clip by `actual_new - old_len`
    /// (shortening ripples them LEFT, lengthening ripples them RIGHT). No-op out of range.
    pub fn ripple_trim_end(&mut self, i: usize, new_len: i64) {
        let (track, old_end, old_len) = match self.clips.get(i) {
            Some(c) => (c.track, c.end(), c.len),
            None => return,
        };
        self.trim_end(i, new_len);
        let actual_new = self.clips[i].len;
        let delta = actual_new - old_len;
        // Downstream = clips that started at/after this clip's ORIGINAL end (so the trimmed clip's
        // own t0 is untouched and only the followers slide). Use old_end as the cutoff.
        self.shift_after(track, old_end, delta, i);
    }

    /// SLIP clip `i` by `delta` source frames: re-time the SOURCE under a fixed timeline window
    /// (Shotcut slip / 3-point slip). `t0` and `len` are UNCHANGED — only `src_in` moves, so the
    /// clip occupies the exact same span on the timeline but shows an earlier/later part of its
    /// media. `src_in` is clamped to `>= 0` (a slip cannot pull source before frame 0). A positive
    /// `delta` slips the source forward (later media under the same window); negative, backward.
    /// No-op out of range. Mirrors `slipTrim` holding the timeline rect while sliding the cut.
    pub fn slip(&mut self, i: usize, delta: i64) {
        if let Some(c) = self.clips.get_mut(i) {
            c.src_in = (c.src_in + delta).max(0);
        }
    }

    /// REPLACE the media behind clip `i` with `new_media` (Shotcut "Replace" — swap the source of a
    /// clip while it keeps its EXACT timeline footprint and every edit on it). Only `Clip.media`
    /// changes: `t0`, `len`, `track`, `src_in`, all per-clip effects/grade/PiP/audio/etc. are
    /// preserved untouched, so the clip stays in place and only shows a different source. Returns
    /// `true` on success; `false` (a no-op, nothing mutated) when `i` is out of range OR `new_media`
    /// is not a valid index into `self.media`. Mirrors slip in being a pure in-place re-target.
    pub fn replace_clip(&mut self, idx: usize, new_media: usize) -> bool {
        if idx < self.clips.len() && new_media < self.media.len() {
            self.clips[idx].media = new_media;
            true
        } else {
            false
        }
    }

    /// DETACH AUDIO from clip `clip_idx` (Shotcut "Detach Audio"): split the clip's audio onto its own
    /// audio track using EXISTING fields only — no new model field. We CLONE the clip onto the first
    /// Audio-kind track (creating one if none exists), so the clone carries the SAME media/src_in/len/
    /// t0/audio_fx and the original's per-clip AUDIO `gain`; then we SILENCE the original clip's audio
    /// by setting its `gain = 0.0`. Because `gain` is the per-clip AUDIO multiplier (0.0 mutes ONLY
    /// that clip's contribution to the audio mix and never touches its video), the original clip's
    /// VIDEO is unchanged while its audio now plays from the detached audio-track clone at the original
    /// gain. Returns `false` (a no-op, nothing mutated) when `clip_idx` is out of range; `true` on
    /// success. Edit-only of existing fields → pre-detach `.json` projects load byte-identical.
    pub fn detach_audio(&mut self, clip_idx: usize) -> bool {
        if clip_idx >= self.clips.len() {
            return false;
        }
        // Find the first Audio-kind track; create one ("A{n}") if the project has none yet.
        let audio_track = match self.tracks.iter().position(|t| t.kind == TrackKind::Audio) {
            Some(i) => i,
            None => self.add_track(TrackKind::Audio),
        } as u8;
        // Clone the source clip: it inherits everything (media/src_in/len/t0/audio_fx + the per-clip
        // gain), then lands on the audio track so its audio is what the mix now hears for this clip.
        let mut detached = self.clips[clip_idx].clone();
        detached.track = audio_track;
        // The original clip is no longer in a group with its detached audio (avoid moving the silent
        // original when the audio clip is body-dragged, and vice-versa). The detached audio is a fresh
        // independent clip; clearing its group id keeps it ungrouped.
        detached.group = 0;
        self.clips.push(detached);
        // Silence the ORIGINAL clip's audio (video untouched): its sound now comes from the clone.
        self.clips[clip_idx].gain = 0.0;
        true
    }

    /// PASTE FILTERS from `src` onto clip `dst_idx` (Shotcut "Paste Filters"): overwrite dst's entire
    /// FILTER / GRADE / LOOK / audio-fx stack with src's, while PRESERVING dst's identity and position.
    /// `media`, `src_in`, `len`, `t0`, `track`, `group`, `fade_in`, and `fade_out` stay dst's own (so
    /// the clip does NOT move, change source, or change its grouping/fades); EVERYTHING else — every
    /// per-clip filter param (grade, color wheels, transform, blur, look/lut, all P6..P41 stylize/
    /// color/distort fields, mask, blend, speed/reverse, etc.) and `audio_fx` — becomes src's. No-op
    /// (nothing mutated) when `dst_idx` is out of range. Implemented by cloning `src` and restoring the
    /// preserved identity/position fields from the current dst, so any future per-clip filter field is
    /// copied automatically without touching this list.
    pub fn copy_filters_from(&mut self, dst_idx: usize, src: &Clip) {
        if dst_idx >= self.clips.len() {
            return;
        }
        // Snapshot the identity/position fields that must survive the paste.
        let media = self.clips[dst_idx].media;
        let src_in = self.clips[dst_idx].src_in;
        let len = self.clips[dst_idx].len;
        let t0 = self.clips[dst_idx].t0;
        let track = self.clips[dst_idx].track;
        let group = self.clips[dst_idx].group;
        let fade_in = self.clips[dst_idx].fade_in;
        let fade_out = self.clips[dst_idx].fade_out;
        // Start from a full clone of src (all filter/grade/look/audio_fx params) ...
        let mut pasted = src.clone();
        // ... then restore dst's identity/position so the clip stays exactly where + what it was.
        pasted.media = media;
        pasted.src_in = src_in;
        pasted.len = len;
        pasted.t0 = t0;
        pasted.track = track;
        pasted.group = group;
        pasted.fade_in = fade_in;
        pasted.fade_out = fade_out;
        self.clips[dst_idx] = pasted;
    }

    /// Convenience PASTE FILTERS by clip indices: copy clip `src_idx`'s filter stack onto clip
    /// `dst_idx` (see `copy_filters_from` for the exact preserved/overwritten field split). No-op when
    /// either index is out of range, or when `src_idx == dst_idx` (pasting a clip's filters onto
    /// itself is a no-op). Clones the source first to avoid aliasing the `self.clips` borrow.
    /// (The UI pastes via the clipboard `copy_filters_from(&Clip)` path; this index→index variant is
    /// exercised by the unit tests, hence `#[allow(dead_code)]`.)
    #[allow(dead_code)]
    pub fn paste_filters(&mut self, src_idx: usize, dst_idx: usize) {
        if src_idx >= self.clips.len() || dst_idx >= self.clips.len() || src_idx == dst_idx {
            return;
        }
        let src = self.clips[src_idx].clone();
        self.copy_filters_from(dst_idx, &src);
    }

    /// GROUP the clips at `indices` so they move together (Shotcut "Group" / linked clips). Assigns a
    /// FRESH group id to every VALID index — `1 + max(existing group over all clips)`, clamped to a
    /// minimum of 1 — so the new id never collides with an existing group and is always non-zero.
    /// Out-of-range indices are ignored. Returns the assigned id, or 0 when `indices` is empty / has
    /// no valid index (nothing is mutated in that case). The render is unaffected (group is edit-only).
    pub fn group_clips(&mut self, indices: &[usize]) -> u32 {
        // Are there any valid indices to group? (Bail with 0 — no mutation — if not.)
        let any_valid = indices.iter().any(|&i| i < self.clips.len());
        if !any_valid {
            return 0;
        }
        // Fresh id = 1 + the current max group across all clips (min 1), so it can't collide.
        let max_group = self.clips.iter().map(|c| c.group).max().unwrap_or(0);
        let id = max_group + 1; // max_group >= 0, so id >= 1 (non-zero) always.
        for &i in indices {
            if i < self.clips.len() {
                self.clips[i].group = id;
            }
        }
        id
    }

    /// UNGROUP: clear the group on every clip whose `group == group` (Shotcut "Ungroup"). A `group`
    /// of 0 is a no-op (0 is the ungrouped sentinel — clearing it would match every ungrouped clip).
    /// After this, the affected clips move independently again. Edit-only; render is unaffected.
    pub fn ungroup(&mut self, group: u32) {
        if group == 0 {
            return;
        }
        for c in self.clips.iter_mut() {
            if c.group == group {
                c.group = 0;
            }
        }
    }

    /// MOVE every clip in `group` by `dt` timeline frames together (the model side of a grouped
    /// body-drag). `group == 0` is a no-op (the ungrouped sentinel). `dt` is CLAMPED so the EARLIEST
    /// member's `t0` stays `>= 0` — i.e. a leftward move can shift the group at most by the smallest
    /// member `t0` (`dt = max(dt, -min_t0)`), so no member is ever pushed before frame 0. A rightward
    /// move is unbounded. Each member's `t0` shifts by the SAME (clamped) delta, so the group keeps
    /// its internal layout exactly. No-op when the group has no members. Edit-only (no render change).
    /// Public + test-covered group-move primitive; the interactive timeline drag uses per-member
    /// absolute re-derivation instead (independent >=0 clamping), so the bin itself does not call this.
    #[allow(dead_code)]
    pub fn move_group(&mut self, group: u32, dt: i64) {
        if group == 0 {
            return;
        }
        // Earliest member t0 (None => group has no members => nothing to move).
        let min_t0 = match self.clips.iter().filter(|c| c.group == group).map(|c| c.t0).min() {
            Some(m) => m,
            None => return,
        };
        // Clamp so the earliest member can't go below 0: dt >= -min_t0.
        let dt = dt.max(-min_t0);
        if dt == 0 {
            return;
        }
        for c in self.clips.iter_mut() {
            if c.group == group {
                c.t0 += dt; // already clamped; min_t0 + dt >= 0 holds for every member
            }
        }
    }

    /// ROLL the shared cut between two adjacent same-track clips by `delta` frames (Shotcut roll /
    /// 3-point roll edit): the LEFT clip's OUT point and the RIGHT clip's IN point move together so
    /// the boundary slides while the pair's combined timeline span is unchanged. `delta > 0` moves
    /// the cut RIGHT (left clip grows, right clip shrinks + starts later); `delta < 0` moves it LEFT.
    ///
    /// Both edges are clamped to keep each clip `>= MIN_CLIP` and the right clip's `src_in >= 0`, and
    /// the EFFECTIVE delta is the most either side can take, so the seam stays a single shared cut
    /// (no gap, no overlap). `left_i`/`right_i` must be distinct, same-track, and abut (right starts
    /// where left ends); otherwise it is a no-op. Returns the effective delta actually applied.
    pub fn roll_edit(&mut self, left_i: usize, right_i: usize, delta: i64) -> i64 {
        if left_i == right_i {
            return 0;
        }
        let (l_track, l_t0, l_len, l_end) = match self.clips.get(left_i) {
            Some(c) => (c.track, c.t0, c.len, c.end()),
            None => return 0,
        };
        let (r_track, r_t0, r_len, r_src) = match self.clips.get(right_i) {
            Some(c) => (c.track, c.t0, c.len, c.src_in),
            None => return 0,
        };
        // The two must share the cut (right starts exactly where left ends) and be on one track.
        if l_track != r_track || r_t0 != l_end {
            return 0;
        }
        // Clamp so neither clip drops below MIN_CLIP and the right source can't go negative.
        //   moving the cut right by d: left.len += d (max = anything), right.len -= d, right.src_in += d
        //   moving the cut left  by d (<0): left.len += d (>= MIN_CLIP), right grows.
        let mut d = delta;
        // left length floor: l_len + d >= MIN_CLIP  ->  d >= MIN_CLIP - l_len
        d = d.max(MIN_CLIP - l_len);
        // right length floor: r_len - d >= MIN_CLIP  ->  d <= r_len - MIN_CLIP
        d = d.min(r_len - MIN_CLIP);
        // right source floor: r_src + d >= 0  ->  d >= -r_src
        d = d.max(-r_src);
        let _ = l_t0; // left t0 is untouched by a roll (only its OUT point moves); bound for clarity.
        if d == 0 {
            return 0;
        }
        // Apply: left out-point moves by +d (grow/shrink its tail); right in-point moves by +d
        // (advance its source + start, shrink/grow its length), keeping the seam a single cut.
        self.clips[left_i].len = l_len + d;
        self.clips[right_i].t0 = r_t0 + d;
        self.clips[right_i].src_in = r_src + d;
        self.clips[right_i].len = r_len - d;
        d
    }

    /// Convenience ROLL by `boundary` frame: find the same-track clip pair whose shared cut sits at
    /// `boundary` on `track` (left.end() == boundary == right.t0) and roll it by `delta`. Returns the
    /// effective delta, or 0 if no abutting pair sits exactly at that boundary. Lets a caller roll by
    /// a timeline frame (e.g. a dragged boundary x) without resolving the two clip indices itself.
    // Retained roll-by-boundary convenience API (unit-tested): the timeline's roll hot-zone already
    // has the two clip indices, so it calls `roll_edit` directly — this frame-addressed variant is
    // unwired but kept (covered by `roll_moves_shared_cut_preserving_total`).
    #[allow(dead_code)]
    pub fn roll(&mut self, track: u8, boundary: i64, delta: i64) -> i64 {
        let mut left_i: Option<usize> = None;
        let mut right_i: Option<usize> = None;
        for (k, c) in self.clips.iter().enumerate() {
            if c.track != track {
                continue;
            }
            if c.end() == boundary {
                left_i = Some(k);
            }
            if c.t0 == boundary {
                right_i = Some(k);
            }
        }
        match (left_i, right_i) {
            (Some(l), Some(r)) if l != r => self.roll_edit(l, r, delta),
            _ => 0,
        }
    }

    /// SLIDE clip `clip_idx` in TIME by `delta` frames while its two abutting same-track NEIGHBOURS
    /// absorb the move (Shotcut slide / 4-point slide edit). The slid clip C keeps its OWN content
    /// (src_in/len unchanged) and just shifts on the timeline; the PREVIOUS clip P (the one whose
    /// `end() == C.t0`) extends/retracts its OUT to follow C's left edge, and the NEXT clip N (the
    /// one whose `t0 == C.end()`) trims/extends its HEAD to follow C's right edge. The track's TOTAL
    /// length is therefore invariant — only the two cuts around C move. This is distinct from MOVE
    /// (free, opens gaps), SLIP (content shifts, position fixed → `slip`) and ROLL (one shared cut
    /// moves → `roll_edit`).
    ///
    /// Both neighbours MUST exist and abut C on the same track; `delta > 0` slides C right (P grows,
    /// N shrinks from the head); `delta < 0` slides C left (P shrinks, N grows). CLAMP / ALL-OR-
    /// NOTHING: returns `false` and mutates NOTHING if P or N is missing/non-abutting, or if `delta`
    /// would drive `P.len <= 0`, `N.len <= 0`, or `N.src_in < 0`. On success returns `true` with:
    ///   C.t0 += delta            (content unchanged)
    ///   P.len += delta           (P's OUT follows C's left edge)
    ///   N.t0 += delta; N.len -= delta; N.src_in += delta   (N's HEAD follows C's right edge)
    pub fn slide(&mut self, clip_idx: usize, delta: i64) -> bool {
        // The clip being slid must exist; capture its abutment frames BEFORE any mutation.
        let (c_track, c_t0, c_end) = match self.clips.get(clip_idx) {
            Some(c) => (c.track, c.t0, c.end()),
            None => return false,
        };
        // delta == 0 is a no-op success (layout is already valid, nothing changes).
        if delta == 0 {
            return true;
        }
        // Find the abutting PREVIOUS clip P (end() == C.t0) and NEXT clip N (t0 == C.end()) on the
        // SAME track, distinct from C. Both must exist for a slide.
        let mut prev_i: Option<usize> = None;
        let mut next_i: Option<usize> = None;
        for (k, c) in self.clips.iter().enumerate() {
            if k == clip_idx || c.track != c_track {
                continue;
            }
            if c.end() == c_t0 {
                prev_i = Some(k);
            }
            if c.t0 == c_end {
                next_i = Some(k);
            }
        }
        let (p, n) = match (prev_i, next_i) {
            (Some(p), Some(n)) if p != n => (p, n),
            _ => return false, // a neighbour is missing / non-abutting (or P==N degenerate)
        };
        // Validate the resulting layout WITHOUT mutating: P keeps t0/src_in, only its len grows by
        // delta; N's head moves by delta (t0/src_in up, len down). Reject if any would be invalid.
        let new_p_len = self.clips[p].len + delta; // P's OUT follows C's left edge
        let new_n_t0 = self.clips[n].t0 + delta; // N's IN follows C's right edge
        let new_n_len = self.clips[n].len - delta;
        let new_n_src = self.clips[n].src_in + delta;
        if new_p_len <= 0 || new_n_len <= 0 || new_n_src < 0 {
            return false; // clamp: would collapse a neighbour or pull N's source before frame 0
        }
        // All checks passed — apply the slide (total track length is invariant: +delta on P, the
        // clip shifts, -delta on N).
        self.clips[clip_idx].t0 = c_t0 + delta; // C content (src_in/len) unchanged, only position
        self.clips[p].len = new_p_len;
        self.clips[n].t0 = new_n_t0;
        self.clips[n].len = new_n_len;
        self.clips[n].src_in = new_n_src;
        true
    }

    // ----- COPY / PASTE clipboard helpers (P3 editing slice; Shotcut Ctrl+C / Ctrl+V) -----------
    // The clipboard itself lives on the app (`Genesis.clipboard: Vec<Clip>`) so it survives across
    // edits and project loads independent of the model. These helpers do the OFFSET-PRESERVING math:
    // copy snapshots a selection (rebased so the earliest clip sits at t0 = 0); paste re-anchors that
    // snapshot at the playhead. Cloning keeps every per-clip field (look/grade/transform/audio_fx/
    // fades/PiP rect) — audio_fx is preserved verbatim (Team A never reads/writes it, just carries it).

    /// Snapshot the clips at `indices` into a fresh `Vec<Clip>`, REBASED so the earliest selected
    /// clip starts at `t0 = 0` (offsets between the copied clips, and their tracks, are preserved).
    /// Paste then re-anchors the whole group at the playhead. Out-of-range indices are skipped;
    /// duplicate indices are de-duped so a clip is never copied twice. Order follows ascending t0
    /// so the rebase origin is deterministic. Returns an empty Vec if nothing valid was selected.
    pub fn copy_clips(&self, indices: &[usize]) -> Vec<Clip> {
        // Gather valid, unique clip indices.
        let mut picked: Vec<usize> = Vec::new();
        for &i in indices {
            if i < self.clips.len() && !picked.contains(&i) {
                picked.push(i);
            }
        }
        if picked.is_empty() {
            return Vec::new();
        }
        // Rebase origin = the earliest t0 among the picked clips.
        let base = picked.iter().map(|&i| self.clips[i].t0).min().unwrap_or(0);
        // Sort the snapshot by t0 so paste lays them down left-to-right (cosmetic; offsets carry).
        picked.sort_by_key(|&i| self.clips[i].t0);
        picked
            .into_iter()
            .map(|i| {
                let mut c = self.clips[i].clone();
                c.t0 -= base; // rebase: earliest clip lands at 0, the rest keep their relative offset
                c
            })
            .collect()
    }

    /// PASTE a clipboard snapshot (from `copy_clips`) at timeline frame `at`, OFFSET-PRESERVING:
    /// each clipboard clip is cloned with `t0 += at` so the group lands with the same internal
    /// spacing/tracks, its earliest clip at `at` (Shotcut paste-at-playhead). Appends the new clips
    /// and returns the index of the FIRST pasted clip (for the caller to select), or `None` if the
    /// clipboard is empty. Drops onto LOCKED tracks are skipped (advisory lock enforcement, matching
    /// the drop path); if every clip is on a locked track nothing is added and `None` is returned.
    pub fn paste_clips(&mut self, clips: &[Clip], at: i64) -> Option<usize> {
        let first = self.clips.len();
        let mut added = 0usize;
        for c in clips {
            if self.is_locked(c.track) {
                continue; // refuse a paste onto a locked track (advisory; mirrors the drop path)
            }
            let mut nc = c.clone();
            nc.t0 = (c.t0 + at).max(0);
            self.clips.push(nc);
            added += 1;
        }
        if added == 0 {
            None
        } else {
            Some(first)
        }
    }

    // ----- 3-POINT EDITING ops (P4 editing slice; Shotcut Append / Overwrite / Insert) ----------
    // These three are the timeline-target half of a 3-point edit: a SOURCE clip (already cut to its
    // length, with its src_in/len/look/grade/audio_fx/chroma carried verbatim) is dropped onto a
    // TRACK at a TIME with one of three placement policies, mirroring Shotcut's MultitrackModel:
    //   * INSERT   (TimelineDock::insert / InsertCommand)    — RIPPLE: if a clip straddles `t0`,
    //     SPLIT it at `t0` (Shotcut's insertClip splits the clip under the insert point), then open
    //     a hole of `clip.len` at `t0` by shifting every same-track clip whose t0 >= t0 RIGHT by
    //     clip.len (incl. that right remnant), then drop the clip at t0. Downstream content is
    //     preserved, just pushed later. (Shotcut default-ripple V.)
    //   * OVERWRITE(TimelineDock::overwrite / OverwriteCommand) — REPLACE the range [t0, t0+len) on
    //     that track: trim/split/remove whatever the new clip covers, then drop the clip. NO ripple
    //     (the timeline length is unchanged; the range is simply replaced). (Shotcut B.)
    //   * APPEND   (TimelineDock::append / AppendCommand)    — drop the clip at the track's END
    //     (max end() of clips on that track, or 0 for an empty track). NO ripple. (Shotcut A.)
    // The caller (app.rs) clones the source clip, sets its `track`, pushes undo, then calls these;
    // each method sets the placed clip's `track`/`t0` itself from its arguments so the caller need
    // not pre-set t0. All three return the index of the newly-placed clip (so the caller can select
    // it). LOCKED tracks are refused (advisory, matching paste/drop): a locked target is a no-op and
    // returns `None`. PiP keyframes ride along correctly: ripple/overwrite only move/trim/remove
    // EXISTING clips via the t0 shift + the PiP-stable `delete_clip`/`split_clip`/`trim_*` already
    // in this file, and the freshly-placed clip starts with no PiP keys (it is appended LAST, so its
    // new index never collides with a remapped key). IDENTITY: none of these run unless the app
    // fires the gesture, so a project with no 3-point edit is byte-identical to before.

    /// INSERT (ripple) `clip` at timeline frame `t0` on `track` (the clip's own `track`/`t0` are set
    /// from these args). Opens a hole of `clip.len` frames at `t0`: if an existing same-track clip
    /// STRADDLES `t0` (its body spans the insert point), it is first SPLIT at `t0` so its right half
    /// starts exactly at `t0`; then every same-track clip whose `t0 >= t0` (which now includes that
    /// right half) is shifted RIGHT by `clip.len`, and the new clip is placed in the opened hole at
    /// `t0`. Returns the new clip's index, or `None` if `track` is locked. `t0` is clamped to `>= 0`.
    /// Mirrors Shotcut's InsertCommand → MultitrackModel::insertClip (multitrackmodel.cpp:1294 splits
    /// the clip under the insert point via `splitClip` when `position > clip_start(target)` before
    /// inserting; ripple-current-track-only — the "ripple all tracks" setting is out of scope).
    pub fn insert_clip(&mut self, track: u8, t0: i64, mut clip: Clip) -> Option<usize> {
        if self.is_locked(track) {
            return None;
        }
        let at = t0.max(0);
        let len = clip.len.max(1);
        // SHOTCUT PARITY: split the clip the insert point falls strictly inside, so its right
        // remnant starts at `at` and rides the ripple (rather than being overlapped by the new
        // clip). `split_clip` only acts when `at` is strictly inside a clip body (off>0 && off<len),
        // returns None otherwise, and remaps PiP keys for the inserted right half. There can be at
        // most one such straddling clip on a track (clips on a track don't overlap), so a single
        // scan + split suffices. We must do this BEFORE `shift_after`, which moves clips by `t0`:
        // the fresh right half has `t0 == at`, so the cutoff `at` catches it (`at >= at`).
        if let Some(i) = self
            .clips
            .iter()
            .position(|c| c.track == track && c.t0 < at && c.end() > at)
        {
            self.split_clip(i, at);
        }
        // Open the gap: shift every same-track clip at/after `at` right by `len`. `usize::MAX`
        // excludes nothing (the new clip is not in `clips` yet, so there is no index to skip).
        self.shift_after(track, at, len, usize::MAX);
        clip.track = track;
        clip.t0 = at;
        let idx = self.clips.len();
        self.clips.push(clip);
        Some(idx)
    }

    /// OVERWRITE `clip` onto `track` at timeline frame `t0`, REPLACING the range `[t0, t0+len)` on
    /// that track (no ripple). Every same-track clip the range touches is trimmed to its surviving
    /// portion(s) and/or removed via `lift_range`, then the new clip is placed. Returns the new
    /// clip's index, or `None` if `track` is locked. `t0` is clamped to `>= 0`. Mirrors Shotcut's
    /// OverwriteCommand (the timeline length is unchanged — only the covered range is replaced).
    pub fn overwrite_clip(&mut self, track: u8, t0: i64, mut clip: Clip) -> Option<usize> {
        if self.is_locked(track) {
            return None;
        }
        let at = t0.max(0);
        let len = clip.len.max(1);
        // Clear the covered range on this track (trim/split/remove existing clips under it).
        self.lift_range(track, at, at + len);
        clip.track = track;
        clip.t0 = at;
        let idx = self.clips.len();
        self.clips.push(clip);
        Some(idx)
    }

    /// APPEND `clip` to the END of `track` (no ripple): the clip is placed at the maximum `end()` of
    /// the clips already on that track, or at 0 for an empty track. Returns the new clip's index, or
    /// `None` if `track` is locked. Mirrors Shotcut's AppendCommand (Append, `A`).
    pub fn append_clip(&mut self, track: u8, mut clip: Clip) -> Option<usize> {
        if self.is_locked(track) {
            return None;
        }
        let end = self.track_end(track);
        clip.track = track;
        clip.t0 = end;
        let idx = self.clips.len();
        self.clips.push(clip);
        Some(idx)
    }

    /// The end frame of `track` = the maximum `end()` of every clip on it, or 0 if the track is
    /// empty. The landing point for `append_clip` (Shotcut appends after the last clip on the track).
    pub fn track_end(&self, track: u8) -> i64 {
        self.clips
            .iter()
            .filter(|c| c.track == track)
            .map(|c| c.end())
            .max()
            .unwrap_or(0)
    }

    /// Clear the timeline range `[from, to)` on `track`: every same-track clip is reshaped so that no
    /// part of it remains inside the range, by trimming its overlapping head/tail and/or splitting it
    /// (a clip that STRADDLES the whole range is cut into a left remnant before `from` and a right
    /// remnant after `to`). Used by `overwrite_clip` to make room for the overwriting clip (NO
    /// ripple — surviving remnants keep their timeline positions). A `to <= from` range is a no-op.
    ///
    /// Implementation reuses the PiP-stable primitives already in this file so keyframes ride along:
    ///   * fully-covered clip (from <= c.t0 && c.end() <= to)  -> `delete_clip` (remaps PiP keys).
    ///   * straddling clip (c.t0 < from && to < c.end())       -> `split_clip` at `from`, then the
    ///     right remnant is `trim_start`-ed to `to` (advancing its src_in), leaving a left remnant
    ///     ending at `from` and a right remnant starting at `to`.
    ///   * head-overlap (c.t0 < to && c.t0 >= from ... i.e. starts inside the range, ends after) ->
    ///     `trim_start` to `to`.
    ///   * tail-overlap (ends inside the range, starts before) -> `trim_end` so the clip ends at
    ///     `from`.
    /// The clip Vec is mutated (split inserts, delete removes), so we rescan from scratch until no
    /// same-track clip still intersects `[from, to)`. The loop terminates because each pass strictly
    /// reduces the total covered frames (every action removes or shrinks an intersecting clip), and
    /// a clip whose remaining length would fall below `MIN_CLIP` via a trim is removed instead so a
    /// trim can never get "stuck" at the floor while still intersecting.
    pub fn lift_range(&mut self, track: u8, from: i64, to: i64) {
        if to <= from {
            return;
        }
        // Bounded rescans (defensive cap: at most one action per existing clip, plus splits). Each
        // pass performs exactly ONE structural action then rescans, so the index it found stays valid.
        let max_passes = self.clips.len() * 4 + 8;
        for _ in 0..max_passes {
            // Find the first same-track clip that still intersects [from, to).
            let hit = self.clips.iter().position(|c| {
                c.track == track && c.t0 < to && c.end() > from
            });
            let i = match hit {
                Some(i) => i,
                None => return, // range is clear
            };
            let (c_t0, c_end) = (self.clips[i].t0, self.clips[i].end());

            if from <= c_t0 && c_end <= to {
                // fully covered -> remove it (delete_clip remaps PiP keys).
                self.delete_clip(i);
            } else if c_t0 < from && to < c_end {
                // STRADDLES the whole range. Each surviving remnant (left = [c_t0, from), right =
                // [to, c_end)) is kept only if it is at least MIN_CLIP long; otherwise that side is
                // dropped (a remnant clamped UP to MIN_CLIP by trim_* would spill back into the range
                // and the rescan would never clear it — an infinite loop). We split at `from` first,
                // which yields a left half [c_t0, from) and a right half [from, c_end); then advance
                // the right half's start to `to`. If a side is too short to survive, we remove it.
                let left_len = from - c_t0;
                let right_len = c_end - to;
                if left_len < MIN_CLIP {
                    // left remnant too small: drop the whole clip and let the (large enough) right
                    // side, if any, be re-created by re-processing — simplest correct path is to
                    // head-trim the original clip to `to` (keeps the right remnant, drops the left).
                    if right_len < MIN_CLIP {
                        self.delete_clip(i); // neither side survives -> remove entirely
                    } else {
                        self.trim_start(i, to); // keep only the right remnant
                    }
                } else if right_len < MIN_CLIP {
                    // right remnant too small: keep only the left remnant by tail-trimming to `from`.
                    self.trim_end(i, left_len);
                } else if let Some(right) = self.split_clip(i, from) {
                    // both remnants survive: split made the right half [from, c_end); advance it to `to`.
                    self.trim_start(right, to);
                } else {
                    // split refused (shouldn't happen here: from is strictly inside) — keep the left.
                    self.trim_end(i, left_len);
                }
            } else if c_t0 >= from {
                // HEAD-OVERLAP: starts inside the range, extends past `to`. The survivor is
                // [to, c_end); if that is shorter than MIN_CLIP, drop the clip (a trim_start clamped
                // up would leave it intersecting the range). Otherwise head-trim to `to`.
                if c_end - to < MIN_CLIP {
                    self.delete_clip(i);
                } else {
                    self.trim_start(i, to);
                }
            } else {
                // TAIL-OVERLAP: starts before `from`, ends inside the range. The survivor is
                // [c_t0, from); if that is shorter than MIN_CLIP, drop the clip (a trim_end clamped
                // up to MIN_CLIP would spill back into the range). Otherwise tail-trim to `from`.
                let left_len = from - c_t0;
                if left_len < MIN_CLIP {
                    self.delete_clip(i);
                } else {
                    self.trim_end(i, left_len);
                }
            }
        }
    }
}

/// Minimum clip length in frames. Mirrors MojoMedia's trim floor (`nl < 15 → 15`).
pub const MIN_CLIP: i64 = 15;

/// Max abutment/gap (frames) between two same-track clips for `boundaries()` to treat them as a
/// transition-eligible pair. Clips that touch (gap 0), slightly gap, or overlap within this many
/// frames produce a boundary; anything farther apart is two unrelated clips, not a seam.
pub const BOUNDARY_GAP: i64 = 30;

/// Snapshot-based undo/redo for the whole project. `Project` derives `Clone`, so each
/// snapshot is a full copy — simple + correct (mirrors MojoMedia's `Snap` stacks).
///
/// NOTE: not yet wired into `app.rs` (no caller pushes pre-edit state). `#[allow(dead_code)]`
/// keeps the build green under `-D warnings` until the integrator wires undo/redo keybindings
/// + a `push` before each edit. Remove the allow once consumed.
#[allow(dead_code)]
#[derive(Default)]
pub struct History {
    undo: Vec<Project>,
    redo: Vec<Project>,
}

impl History {
    pub fn new() -> History {
        History { undo: Vec::new(), redo: Vec::new() }
    }

    /// Push the *current* project state onto the undo stack before a mutation, clearing
    /// the redo stack (a new edit invalidates the redo future). Call this with the state
    /// as it is *before* applying an edit.
    pub fn push(&mut self, project: &Project) {
        self.undo.push(project.clone());
        self.redo.clear();
    }

    /// Undo: restore the most recent snapshot into `project`, stashing the current state
    /// onto the redo stack. Returns true if anything was undone.
    pub fn undo(&mut self, project: &mut Project) -> bool {
        if let Some(prev) = self.undo.pop() {
            self.redo.push(project.clone());
            *project = prev;
            true
        } else {
            false
        }
    }

    /// Redo: re-apply the most recently undone snapshot, stashing the current state back
    /// onto the undo stack. Returns true if anything was redone.
    pub fn redo(&mut self, project: &mut Project) -> bool {
        if let Some(next) = self.redo.pop() {
            self.undo.push(project.clone());
            *project = next;
            true
        } else {
            false
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Undo-stack DEPTH — a cheap, monotone-per-edit "has the project changed" signal for the
    /// auto-save loop (`app.rs`). Pure read: every `push` (pre-edit) grows it, undo/redo shift it,
    /// so any *change* in this value since the last auto-save means the project has been edited.
    /// Adds NO behavior — existing undo/redo/save/load are byte-unaffected.
    pub fn len(&self) -> usize {
        self.undo.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // T1 — 3D-LUT LIBRARY. Pure directory scan: a temp dir holding two `.cube` files returns both,
    // sorted by stem ("a","b"), with the joined full paths; a non-existent dir returns []. Uses only
    // std::fs + std::env::temp_dir (dependency-free) and a unique sub-dir so concurrent test runs
    // don't collide. Cleans up after itself.
    #[test]
    fn scan_lut_dir_lists_cubes_sorted_and_missing_is_empty() {
        // Unique temp dir (pid + nanos) so parallel `cargo test` invocations never share state.
        let uniq = format!(
            "genesis_lut_scan_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let dir = std::env::temp_dir().join(uniq);
        std::fs::create_dir_all(&dir).expect("create temp lut dir");

        // Write b.cube first to prove the result is SORTED (not insertion order). Also drop a
        // non-cube file in to prove it's filtered out.
        std::fs::write(dir.join("b.cube"), b"# b lut").expect("write b.cube");
        std::fs::write(dir.join("a.cube"), b"# a lut").expect("write a.cube");
        std::fs::write(dir.join("notes.txt"), b"ignore me").expect("write notes.txt");

        let dir_str = dir.to_string_lossy().into_owned();
        let found = scan_lut_dir(&dir_str);

        assert_eq!(found.len(), 2, "exactly two .cube files (the .txt is ignored): {found:?}");
        // Sorted ascending by stem -> "a" then "b".
        assert_eq!(found[0].0, "a", "first stem is 'a' (sorted)");
        assert_eq!(found[1].0, "b", "second stem is 'b' (sorted)");
        // Full path = dir joined with the filename, assignable straight onto Clip.lut.
        assert_eq!(found[0].1, dir.join("a.cube").to_string_lossy(), "a full path");
        assert_eq!(found[1].1, dir.join("b.cube").to_string_lossy(), "b full path");

        // A non-existent directory -> empty Vec (no panic).
        let missing = dir.join("does_not_exist_subdir");
        assert!(
            scan_lut_dir(&missing.to_string_lossy()).is_empty(),
            "missing dir scans to []"
        );

        // Best-effort cleanup (failure here must not fail the test).
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn grade_keyframe_interp() {
        let mut p = Project::demo("x".into());
        p.bright = 0.0;
        p.add_grade_key(0);
        p.bright = 1.0;
        p.add_grade_key(100);
        assert!((p.grade_at(0).0 - 0.0).abs() < 1e-4);
        assert!((p.grade_at(50).0 - 0.5).abs() < 1e-3, "b50={}", p.grade_at(50).0);
        assert!((p.grade_at(100).0 - 1.0).abs() < 1e-4);
        assert!((p.grade_at(150).0 - 1.0).abs() < 1e-4); // clamp past last key
    }

    #[test]
    fn pip_keyframe_interp() {
        let mut p = Project::demo("x".into());
        p.clips[0].px = 0.0;
        p.add_pip_key(0, 0);
        p.clips[0].px = 0.6;
        p.add_pip_key(0, 100);
        assert!((p.pip_at(0, 0).0 - 0.0).abs() < 1e-4);
        assert!((p.pip_at(0, 50).0 - 0.3).abs() < 1e-3, "px50={}", p.pip_at(0, 50).0);
        assert!((p.pip_at(0, 100).0 - 0.6).abs() < 1e-4);
    }

    // ----- P14 keyframe INTERPOLATION TYPES (eval_track + eval_pip) -------------------------------
    // All on a 2-key track 0@frame0 -> 10@frame10. The interp of the LOWER key (frame 0) selects
    // the segment curve. `kf_interp` is the create mode read by add_grade_key/add_pip_key.

    // Build a grade (bright) track 0@0 -> 10@10 with the given create interp; eval via grade_at.
    fn grade_0_10(interp: KfInterp) -> Project {
        let mut p = Project::demo("x".into());
        p.kf_interp = interp;
        p.bright = 0.0;
        p.add_grade_key(0); // lower key carries `interp`
        p.bright = 10.0;
        p.add_grade_key(10);
        p
    }

    // Build a PiP px track 0@0 -> 10@10 with the given create interp; eval via pip_at(.).0.
    fn pip_0_10(interp: KfInterp) -> Project {
        let mut p = Project::demo("x".into());
        p.kf_interp = interp;
        p.clips[0].px = 0.0;
        p.add_pip_key(0, 0); // lower key carries `interp`
        p.clips[0].px = 10.0;
        p.add_pip_key(0, 10);
        p
    }

    #[test]
    fn interp_linear_eval_track_and_pip() {
        // (a) LINEAR: eval@5 == 5.0 (midpoint), both tracks. This is the pre-P14 behavior and the
        //     KfInterp::default, so it is also what an old .json loads as.
        let g = grade_0_10(KfInterp::Linear);
        assert!((g.grade_at(5).0 - 5.0).abs() < 1e-3, "grade linear@5={}", g.grade_at(5).0);
        let pp = pip_0_10(KfInterp::Linear);
        assert!((pp.pip_at(0, 5).0 - 5.0).abs() < 1e-3, "pip linear@5={}", pp.pip_at(0, 5).0);
        // sanity: default create mode is Linear, so an un-set kf_interp behaves identically.
        assert_eq!(KfInterp::default(), KfInterp::Linear);
    }

    #[test]
    fn interp_discrete_eval_track_and_pip() {
        // (b) DISCRETE: HOLD the lower key's value across the segment. eval@5 == 0.0, eval@9 == 0.0,
        //     eval@10 == 10.0 (the upper key's frame is its own value — endpoint clamp / next key).
        let g = grade_0_10(KfInterp::Discrete);
        assert!((g.grade_at(5).0 - 0.0).abs() < 1e-3, "grade discrete@5={}", g.grade_at(5).0);
        assert!((g.grade_at(9).0 - 0.0).abs() < 1e-3, "grade discrete@9={}", g.grade_at(9).0);
        assert!((g.grade_at(10).0 - 10.0).abs() < 1e-3, "grade discrete@10={}", g.grade_at(10).0);
        let pp = pip_0_10(KfInterp::Discrete);
        assert!((pp.pip_at(0, 5).0 - 0.0).abs() < 1e-3, "pip discrete@5={}", pp.pip_at(0, 5).0);
        assert!((pp.pip_at(0, 9).0 - 0.0).abs() < 1e-3, "pip discrete@9={}", pp.pip_at(0, 9).0);
        assert!((pp.pip_at(0, 10).0 - 10.0).abs() < 1e-3, "pip discrete@10={}", pp.pip_at(0, 10).0);
    }

    #[test]
    fn interp_smooth_eval_track_and_pip() {
        // (c) SMOOTH: smoothstep s = b*b*(3-2b). Symmetric → eval@5 == 5.0; ease-IN below midpoint
        //     (eval@2 = smoothstep(0.2)*10 = 1.04 < 2.0); ease-OUT above midpoint
        //     (eval@8 = smoothstep(0.8)*10 = 8.96 > 8.0).
        let g = grade_0_10(KfInterp::Smooth);
        assert!((g.grade_at(5).0 - 5.0).abs() < 1e-3, "grade smooth@5={}", g.grade_at(5).0);
        assert!((g.grade_at(2).0 - 1.04).abs() < 1e-3, "grade smooth@2={}", g.grade_at(2).0);
        assert!(g.grade_at(2).0 < 2.0, "grade smooth ease-in@2={}", g.grade_at(2).0);
        assert!((g.grade_at(8).0 - 8.96).abs() < 1e-3, "grade smooth@8={}", g.grade_at(8).0);
        assert!(g.grade_at(8).0 > 8.0, "grade smooth ease-out@8={}", g.grade_at(8).0);
        let pp = pip_0_10(KfInterp::Smooth);
        assert!((pp.pip_at(0, 5).0 - 5.0).abs() < 1e-3, "pip smooth@5={}", pp.pip_at(0, 5).0);
        assert!((pp.pip_at(0, 2).0 - 1.04).abs() < 1e-3, "pip smooth@2={}", pp.pip_at(0, 2).0);
        assert!(pp.pip_at(0, 2).0 < 2.0, "pip smooth ease-in@2={}", pp.pip_at(0, 2).0);
        assert!((pp.pip_at(0, 8).0 - 8.96).abs() < 1e-3, "pip smooth@8={}", pp.pip_at(0, 8).0);
        assert!(pp.pip_at(0, 8).0 > 8.0, "pip smooth ease-out@8={}", pp.pip_at(0, 8).0);
    }

    #[test]
    fn interp_rekey_updates_interp() {
        // Re-keying the SAME frame while the create mode changed updates that key's interp (replace
        // path threads the new mode). Start Linear@0..10, then re-key frame 0 as Discrete -> hold.
        let mut g = grade_0_10(KfInterp::Linear);
        assert!((g.grade_at(5).0 - 5.0).abs() < 1e-3);
        g.kf_interp = KfInterp::Discrete;
        g.bright = 0.0;
        g.add_grade_key(0); // replace frame-0 key; its interp becomes Discrete
        assert!((g.grade_at(5).0 - 0.0).abs() < 1e-3, "rekey->discrete holds, @5={}", g.grade_at(5).0);

        let mut pp = pip_0_10(KfInterp::Linear);
        assert!((pp.pip_at(0, 5).0 - 5.0).abs() < 1e-3);
        pp.kf_interp = KfInterp::Discrete;
        pp.clips[0].px = 0.0;
        pp.add_pip_key(0, 0); // replace frame-0 param keys; their interp becomes Discrete
        assert!((pp.pip_at(0, 5).0 - 0.0).abs() < 1e-3, "pip rekey->discrete holds, @5={}", pp.pip_at(0, 5).0);
    }

    // P19: a 3-key grade track 0@f0 -> 10@f10 -> 30@f20 with the SEGMENT [10,20] (lower key = f10)
    // carrying the given Catmull variant. Frame 0 / frame 20 keys are Linear (their interp is
    // irrelevant to an eval inside [10,20]).
    fn grade_3key(seg_interp: KfInterp) -> Project {
        let mut p = Project::demo("x".into());
        p.kf_interp = KfInterp::Linear;
        p.bright = 0.0;
        p.add_grade_key(0);
        p.kf_interp = seg_interp;
        p.bright = 10.0;
        p.add_grade_key(10);
        p.kf_interp = KfInterp::Linear;
        p.bright = 30.0;
        p.add_grade_key(20);
        p
    }
    fn pip_3key(seg_interp: KfInterp) -> Project {
        let mut p = Project::demo("x".into());
        p.kf_interp = KfInterp::Linear;
        p.clips[0].px = 0.0;
        p.add_pip_key(0, 0);
        p.kf_interp = seg_interp;
        p.clips[0].px = 10.0;
        p.add_pip_key(0, 10);
        p.kf_interp = KfInterp::Linear;
        p.clips[0].px = 30.0;
        p.add_pip_key(0, 20);
        p
    }

    #[test]
    fn interp_catmull_variants_match_mlt() {
        // P19: values are the EXACT output of MLT's catmull_rom_interpolate (mlt_animation.c) computed
        // from the verbatim formula on the same track, segment [10,20] (p0=(0,0),p1=(10,10),
        // p2=(20,30),p3=dup(20,30)) at frame 15 (progress 0.5):
        //   smooth_loose   (alpha 0.0, tension  1.0) = 20.625000
        //   smooth_natural (alpha 0.5, tension -1.0) = 21.982970
        //   smooth_tight   (alpha 0.5, tension  0.0) = 20.000000  (zero tangents == smoothstep)
        for (interp, want) in [
            (KfInterp::SmoothLoose, 20.625_f32),
            (KfInterp::SmoothNatural, 21.982_97_f32),
            (KfInterp::SmoothTight, 20.0_f32),
        ] {
            let g = grade_3key(interp);
            assert!((g.grade_at(15).0 - want).abs() < 1e-2, "grade {:?}@15 = {} (want {})", interp, g.grade_at(15).0, want);
            let pp = pip_3key(interp);
            assert!((pp.pip_at(0, 15).0 - want).abs() < 1e-2, "pip {:?}@15 = {} (want {})", interp, pp.pip_at(0, 15).0, want);
        }
        // smooth_loose overshoot signature off the midpoint (MLT reference): @12=13.68, @18=27.12.
        let g = grade_3key(KfInterp::SmoothLoose);
        assert!((g.grade_at(12).0 - 13.68).abs() < 1e-2, "loose@12={}", g.grade_at(12).0);
        assert!((g.grade_at(18).0 - 27.12).abs() < 1e-2, "loose@18={}", g.grade_at(18).0);
    }

    // P20: a grade track 0@f0 -> 10@f100 with the given easing on the f0 (lower) key; grade_at(frame)
    // returns 10*factor(frame/100), so the expected values are the MLT easing factors *10.
    fn ease_track(interp: KfInterp) -> Project {
        let mut p = Project::demo("x".into());
        p.kf_interp = interp;
        p.bright = 0.0;
        p.add_grade_key(0);
        p.kf_interp = KfInterp::Linear;
        p.bright = 10.0;
        p.add_grade_key(100);
        p
    }

    #[test]
    fn interp_easings_match_mlt() {
        // Values are 10 * the easing factor at t = frame/100, computed from the VERBATIM MLT
        // mlt_animation.c easing functions (sinusoidal/power/exponential/circular/back/elastic/bounce).
        let cases = [
            (KfInterp::SineInOut, 50, 5.0_f32),
            (KfInterp::QuadIn, 25, 0.625),
            (KfInterp::CubicIn, 25, 0.15625),
            (KfInterp::CubicOut, 25, 5.78125),
            (KfInterp::QuartIn, 50, 0.625),
            (KfInterp::QuintIn, 50, 0.3125),
            (KfInterp::ExpoOut, 50, 9.6875),
            (KfInterp::CircIn, 50, 1.33975),
            (KfInterp::BackIn, 50, -3.75), // back anticipates BELOW the start value (overshoot)
            (KfInterp::ElasticOut, 50, 10.22097), // elastic overshoots ABOVE the target
            (KfInterp::BounceOut, 50, 7.1875),
            (KfInterp::BounceOut, 25, 4.72656),
        ];
        for (interp, frame, want) in cases {
            let g = ease_track(interp);
            let got = g.grade_at(frame).0;
            assert!((got - want).abs() < 1e-2, "{:?}@f{} = {} (want {})", interp, frame, got, want);
        }
        // every easing kind is reachable from ALL + has a label (UI invariant).
        assert_eq!(KfInterp::ALL.len(), 36);
        for k in KfInterp::ALL {
            assert!(!k.label().is_empty());
        }
    }

    #[test]
    fn transition_window_and_progress() {
        let mut p = Project::demo("x".into());
        // centered at 100, dur 20 -> window [90, 110), progress 0 at 90, ~1 at 109.
        p.add_transition(0, 100, 20, 0);
        assert_eq!(p.transitions.len(), 1);
        assert!(p.transition_at(0, 89).is_none(), "before window");
        assert!(p.transition_at(0, 90).is_some(), "window start inclusive");
        assert!(p.transition_at(0, 109).is_some(), "inside window");
        assert!(p.transition_at(0, 110).is_none(), "window end exclusive");
        assert!(p.transition_at(1, 100).is_none(), "other track has no transition");
        let tr = p.transition_at(0, 90).unwrap();
        assert!((tr.progress(90) - 0.0).abs() < 1e-4, "prog start=0");
        assert!((tr.progress(100) - 0.5).abs() < 1e-3, "prog mid=0.5 got {}", tr.progress(100));
        assert!((tr.progress(120) - 1.0).abs() < 1e-4, "prog clamped past end");
    }

    #[test]
    fn transition_add_replaces_same_center_and_clamps_dur() {
        let mut p = Project::demo("x".into());
        p.add_transition(0, 100, 20, 0);
        // same track+center -> replace in place (dur/kind updated), not a duplicate.
        p.add_transition(0, 100, 1, 7);
        assert_eq!(p.transitions.len(), 1, "same track+center replaced");
        assert_eq!(p.transitions[0].kind, 7);
        assert_eq!(p.transitions[0].dur, 2, "dur clamped to >= 2");
        // different center -> a new entry.
        p.add_transition(0, 200, 10, 1);
        assert_eq!(p.transitions.len(), 2);
        p.remove_transition(0);
        assert_eq!(p.transitions.len(), 1);
        assert_eq!(p.transitions[0].center, 200);
        p.remove_transition(99); // out of range -> no-op
        assert_eq!(p.transitions.len(), 1);
    }

    #[test]
    fn set_fade_in_clamps_to_len_and_floor_and_is_oor_safe() {
        let mut p = Project::demo("x".into());
        p.clips[0].len = 50;
        p.clips[0].fade_in = 0;
        // over the clip length -> clamped to len.
        assert!(p.set_fade_in(0, 1000));
        assert_eq!(p.clips[0].fade_in, 50, "fade_in clamped to len");
        // negative -> floored to 0.
        assert!(p.set_fade_in(0, -5));
        assert_eq!(p.clips[0].fade_in, 0, "fade_in floored to 0");
        // OOR index -> false + no mutation (set a known value first to prove it stays).
        p.clips[0].fade_in = 7;
        assert!(!p.set_fade_in(99, 3), "OOR idx returns false");
        assert_eq!(p.clips[0].fade_in, 7, "OOR set_fade_in did not mutate");
    }

    #[test]
    fn set_fade_out_clamps_to_len_and_floor_and_is_oor_safe() {
        let mut p = Project::demo("x".into());
        p.clips[0].len = 50;
        p.clips[0].fade_out = 0;
        assert!(p.set_fade_out(0, 1000));
        assert_eq!(p.clips[0].fade_out, 50, "fade_out clamped to len");
        assert!(p.set_fade_out(0, -5));
        assert_eq!(p.clips[0].fade_out, 0, "fade_out floored to 0");
        p.clips[0].fade_out = 9;
        assert!(!p.set_fade_out(99, 3), "OOR idx returns false");
        assert_eq!(p.clips[0].fade_out, 9, "OOR set_fade_out did not mutate");
    }

    #[test]
    fn set_transition_dur_floors_to_2_and_is_oor_safe() {
        let mut p = Project::demo("x".into());
        p.add_transition(0, 100, 20, 0);
        // dur 1 -> floored to 2 (window stays >= 2).
        assert!(p.set_transition_dur(0, 1));
        assert_eq!(p.transitions[0].dur, 2, "dur floored to 2");
        // a normal value passes through.
        assert!(p.set_transition_dur(0, 40));
        assert_eq!(p.transitions[0].dur, 40);
        // OOR index -> false + no mutation.
        assert!(!p.set_transition_dur(99, 8), "OOR idx returns false");
        assert_eq!(p.transitions[0].dur, 40, "OOR set_transition_dur did not mutate");
    }

    #[test]
    fn fade_factor_ramps_in_and_out_else_full() {
        // clip [t0=0, len=30); fade_in 15, fade_out 10.
        let mut c = Clip::video(0, 0, 30, 0, "C");
        c.fade_in = 15;
        c.fade_out = 10;
        // fade-IN ramp 0->1 over the first 15 frames: (local+1)/15.
        assert!((c.fade_factor(0) - 1.0 / 15.0).abs() < 1e-4, "first frame near black");
        assert!((c.fade_factor(7) - 8.0 / 15.0).abs() < 1e-4, "mid fade-in");
        assert!((c.fade_factor(14) - 1.0).abs() < 1e-4, "fade-in complete at frame 15");
        // middle: full brightness.
        assert!((c.fade_factor(18) - 1.0).abs() < 1e-4, "middle full");
        // fade-OUT ramp 1->0 over the last 10 frames: remaining/10 (remaining = end - t).
        assert!((c.fade_factor(20) - 1.0).abs() < 1e-4, "fade-out begins (remaining 10)");
        assert!((c.fade_factor(25) - 5.0 / 10.0).abs() < 1e-4, "mid fade-out");
        assert!((c.fade_factor(29) - 1.0 / 10.0).abs() < 1e-4, "last frame near black");
        // no fades -> always 1.0 (byte-identical render).
        let plain = Clip::video(0, 0, 30, 0, "P");
        assert_eq!(plain.fade_factor(0), 1.0);
        assert_eq!(plain.fade_factor(29), 1.0);
    }

    #[test]
    fn nested_subseq_view_and_inner_time() {
        // a sub-sequence with two media clips: red [0,30) then green [30,60).
        let mut p = Project::demo("/m/a.mp4".into());
        p.media = vec!["/m/red.mp4".into(), "/m/green.mp4".into()];
        let ss = SubSeq {
            name: "comp".into(),
            len: 60,
            clips: vec![Clip::video(0, 0, 30, 0, "r"), Clip::video(1, 30, 30, 0, "g")],
            tracks: default_tracks(),
        };
        p.subseqs = vec![ss];
        // a COMPOUND clip on the main timeline at t0=10 referencing subseq 0.
        let mut comp = Clip::video(0, 0, 60, 0, "compound");
        comp.seq = 0;
        comp.t0 = 10;
        comp.src_in = 0;
        p.clips = vec![comp];

        // subseq_view exposes the sub-timeline's clips over the SHARED media pool.
        let view = p.subseq_view(0).expect("subseq 0 exists");
        assert_eq!(view.clips.len(), 2);
        assert_eq!(view.media, p.media); // shares the parent media pool
        assert!(view.subseqs.is_empty()); // one level of nesting (no infinite recursion)
        assert!(p.subseq_view(9).is_none()); // OOR

        // inner time of the compound clip at outer frame t (speed 1 -> src_in + (t - t0); the worker
        // computes this via src_frame_at). outer t=10 -> inner 0 (subseq's red @0); outer t=45 ->
        // inner 35 (subseq's green, which spans inner [30,60)).
        let c = &p.clips[0];
        assert_eq!(c.src_in + (10 - c.t0), 0);
        assert_eq!(c.src_in + (45 - c.t0), 35);
        assert!(c.seq >= 0); // it's a compound clip
    }

    #[test]
    fn parse_srt_and_active_subtitle() {
        // two cues with a gap; multi-line text on the second. @30fps: 1s=30 frames.
        let srt = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n2\n00:00:03,000 --> 00:00:04,500\nWorld\nline2\n";
        let subs = crate::model::parse_srt(srt, 30.0);
        assert_eq!(subs.len(), 2);
        assert_eq!((subs[0].start, subs[0].end), (30, 60));
        assert_eq!(subs[0].text, "Hello");
        assert_eq!((subs[1].start, subs[1].end), (90, 135));
        assert_eq!(subs[1].text, "World\nline2");
        // a malformed block (no arrow) is skipped, not fatal.
        assert_eq!(crate::model::parse_srt("1\nnot a cue\n", 30.0).len(), 0);

        let mut p = Project::demo("x".into());
        p.subtitles = subs;
        assert_eq!(p.active_subtitle_at(45).map(|s| s.text.as_str()), Some("Hello")); // in [30,60)
        assert!(p.active_subtitle_at(75).is_none()); // the gap
        assert_eq!(p.active_subtitle_at(100).map(|s| s.text.as_str()), Some("World\nline2"));
        assert!(p.active_subtitle_at(0).is_none()); // before the first cue
    }

    #[test]
    fn cross_correlation_recovers_known_delay() {
        // A deterministic non-periodic-ish signal so the correlation peak is unambiguous.
        let n = 400usize;
        let a: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32;
                (x * 0.37).sin() + 0.6 * (x * 0.111).sin() + 0.3 * (x * 0.93).cos()
            })
            .collect();
        // b is `a` delayed by K=17 samples (b[i] == a[i-17], zero-padded front).
        let k = 17usize;
        let mut b = vec![0f32; n];
        for i in k..n {
            b[i] = a[i - k];
        }
        // positive return == b lags a by K.
        assert_eq!(crate::model::cross_correlation_offset(&a, &b, 64), k as i64);
        // symmetric: aligning a onto b recovers -K.
        assert_eq!(crate::model::cross_correlation_offset(&b, &a, 64), -(k as i64));
        // identical signals -> 0 lag (no false shift).
        assert_eq!(crate::model::cross_correlation_offset(&a, &a, 64), 0);
        // empty input -> 0 (safe).
        assert_eq!(crate::model::cross_correlation_offset(&[], &a, 64), 0);
    }

    #[test]
    fn transition_at_picks_nearest_center_on_overlap() {
        let mut p = Project::demo("x".into());
        // two overlapping windows: [90,110) center 100 and [95,115) center 105; at t=104 both
        // contain it, nearest center (105) wins.
        p.add_transition(0, 100, 20, 0);
        p.add_transition(0, 105, 20, 1);
        let tr = p.transition_at(0, 104).expect("some transition at 104");
        assert_eq!(tr.center, 105, "nearest center wins");
    }

    #[test]
    fn boundaries_abut_and_overlap() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        // V1 (track 0): A [0,100), B [100,200) abut exactly -> boundary at 100.
        p.clips.push(Clip::video(0, 0, 100, 0, "A"));   // idx 0
        p.clips.push(Clip::video(0, 100, 100, 0, "B")); // idx 1
        // V2 (track 1): C [0,100), D [80,180) overlap -> boundary at midpoint (80+100)/2 = 90.
        p.clips.push(Clip::video(0, 0, 100, 1, "C"));   // idx 2
        p.clips.push(Clip::video(0, 80, 100, 1, "D"));  // idx 3

        let bv1 = p.boundaries(0);
        assert_eq!(bv1.len(), 1);
        assert_eq!(bv1[0], (0, 1, 100));

        let bv2 = p.boundaries(1);
        assert_eq!(bv2.len(), 1);
        assert_eq!(bv2[0], (2, 3, 90));

        // a far-apart pair (gap > BOUNDARY_GAP) produces no boundary.
        let mut q = Project::demo("x".into());
        q.clips.clear();
        q.clips.push(Clip::video(0, 0, 100, 0, "A"));
        q.clips.push(Clip::video(0, 200, 100, 0, "B")); // gap 100 > 30
        assert!(q.boundaries(0).is_empty());
    }

    // ----- P3 EDITING ops -------------------------------------------------------------------

    #[test]
    fn ripple_delete_closes_gap_same_track_only() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        // V1: A [0,100), B [100,150), C [150,210). V2: D [120,220) must NOT move.
        p.clips.push(Clip::video(0, 0, 100, 0, "A"));   // idx 0
        p.clips.push(Clip::video(0, 100, 50, 0, "B"));  // idx 1
        p.clips.push(Clip::video(0, 150, 60, 0, "C"));  // idx 2
        p.clips.push(Clip::video(0, 120, 100, 1, "D")); // idx 3 (other track)
        p.ripple_delete(1); // delete B (len 50), C shifts left by 50
        // B gone -> 3 clips. C now at 100, ends 160. D unchanged at 120.
        assert_eq!(p.clips.len(), 3);
        // find C (media 0 track 0 len 60) and D (track 1)
        let c = p.clips.iter().find(|c| c.track == 0 && c.len == 60).unwrap();
        assert_eq!(c.t0, 100, "C rippled left by B.len");
        let d = p.clips.iter().find(|c| c.track == 1).unwrap();
        assert_eq!(d.t0, 120, "other-track clip unmoved");
    }

    #[test]
    fn ripple_delete_many_non_contiguous_survivor() {
        // Track 0: A [0,100), B [100,150), C [150,210). Ripple-delete A and C (non-contiguous):
        // B is after A (removed, 100 frames) -> shifts left 100; C is removed. B -> t0=0, len=50.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A")); // idx 0
        p.clips.push(Clip::video(0, 100, 50, 0, "B")); // idx 1
        p.clips.push(Clip::video(0, 150, 60, 0, "C")); // idx 2
        p.ripple_delete_many(&[0, 2]);
        assert_eq!(p.clips.len(), 1, "A and C removed, B survives");
        assert_eq!(p.clips[0].len, 50, "survivor is B");
        assert_eq!(p.clips[0].t0, 0, "B closed the 100-frame gap left by A");
    }

    #[test]
    fn ripple_delete_many_index_order_ne_t0_order() {
        // Skeptic #1: two same-track clips where the LOWER index has the HIGHER t0. The old
        // descending-index loop (per-clip t0-based shift) over-shifted the survivor; the batch
        // must be order-independent and identical to deleting them as one contiguous block.
        // Track 0: idx0 @ t0=100 len=50 (end 150), idx1 @ t0=0 len=100 (end 100), then a
        // downstream survivor S @ t0=150 len=40 (end 190). Selected block [0,150) removes 150
        // frames before S -> S slides left 150 to t0=0.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 100, 50, 0, "later-but-idx0")); // idx 0, t0=100
        p.clips.push(Clip::video(0, 0, 100, 0, "earlier-idx1"));    // idx 1, t0=0
        p.clips.push(Clip::video(0, 150, 40, 0, "S"));              // idx 2, downstream survivor
        p.ripple_delete_many(&[0, 1]);
        assert_eq!(p.clips.len(), 1, "both block clips removed, S survives");
        assert_eq!(p.clips[0].len, 40, "survivor is S");
        assert_eq!(p.clips[0].t0, 0, "S slid left by the full 150-frame removed block (no double-shift)");
    }

    #[test]
    fn ripple_delete_many_other_track_unmoved() {
        // A same-track ripple must never move clips on another track (ripple-current-track-only).
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A"));   // idx 0, track 0
        p.clips.push(Clip::video(0, 100, 50, 0, "B"));  // idx 1, track 0 downstream
        p.clips.push(Clip::video(0, 50, 100, 1, "D"));  // idx 2, track 1 (other track)
        p.ripple_delete_many(&[0]);
        assert_eq!(p.clips.len(), 2);
        let b = p.clips.iter().find(|c| c.track == 0).unwrap();
        assert_eq!(b.t0, 0, "B rippled left by A.len");
        let d = p.clips.iter().find(|c| c.track == 1).unwrap();
        assert_eq!(d.t0, 50, "other-track clip unmoved");
    }

    #[test]
    fn ripple_trim_end_shifts_downstream() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A")); // idx 0
        p.clips.push(Clip::video(0, 100, 50, 0, "B")); // idx 1 (starts at A.end)
        p.ripple_trim_end(0, 80); // A 100 -> 80, B ripples left by 20 to t0 80
        assert_eq!(p.clips[0].len, 80);
        assert_eq!(p.clips[1].t0, 80, "downstream clip closed the 20-frame gap");
    }

    #[test]
    fn ripple_trim_start_shifts_downstream() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A")); // idx 0
        p.clips.push(Clip::video(0, 100, 50, 0, "B")); // idx 1
        // Ripple head-trim: 20 frames come off A's head (src_in advances), the shortened clip is
        // re-anchored at its original start, and the sequence slides left to stay gapless.
        p.ripple_trim_start(0, 20);
        assert_eq!(p.clips[0].t0, 0, "re-anchored at original start — no front gap");
        assert_eq!(p.clips[0].len, 80, "head trimmed by 20 frames");
        assert_eq!(p.clips[0].src_in, 20, "head trim advances the source in-point");
        // A now ends at 80; downstream B ripples left by 20 to stay tight (100 -> 80).
        assert_eq!(p.clips[1].t0, 80, "downstream rippled left by the head-trim delta");
    }

    #[test]
    fn slip_moves_source_only() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        let mut c = Clip::video(0, 50, 100, 0, "A");
        c.src_in = 30;
        p.clips.push(c);
        p.slip(0, 10);
        assert_eq!(p.clips[0].src_in, 40);
        assert_eq!(p.clips[0].t0, 50, "t0 unchanged by slip");
        assert_eq!(p.clips[0].len, 100, "len unchanged by slip");
        p.slip(0, -1000); // clamp at 0
        assert_eq!(p.clips[0].src_in, 0);
    }

    #[test]
    fn roll_moves_shared_cut_preserving_total() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        let mut a = Clip::video(0, 0, 100, 0, "A");
        a.src_in = 0;
        let mut b = Clip::video(0, 100, 100, 0, "B");
        b.src_in = 200;
        p.clips.push(a); // idx 0
        p.clips.push(b); // idx 1
        let total_before = p.clips[0].len + p.clips[1].len;
        let d = p.roll_edit(0, 1, 15); // cut moves right 15
        assert_eq!(d, 15);
        assert_eq!(p.clips[0].len, 115, "left grew");
        assert_eq!(p.clips[1].t0, 115, "right starts later");
        assert_eq!(p.clips[1].src_in, 215, "right source advanced with the cut");
        assert_eq!(p.clips[1].len, 85, "right shrank");
        assert_eq!(p.clips[0].len + p.clips[1].len, total_before, "combined span unchanged");
        // boundary-keyed convenience: roll the cut at frame 115 back left by 15.
        let d2 = p.roll(0, 115, -15);
        assert_eq!(d2, -15);
        assert_eq!(p.clips[0].len, 100);
        assert_eq!(p.clips[1].t0, 100);
    }

    #[test]
    fn slide_moves_clip_neighbours_absorb_total_invariant() {
        // Three abutting clips on ONE track: P[0,10) C[10,20) N[20,30). Sliding the MIDDLE clip C by
        // +3 shifts C's content in time while P (prev) extends its OUT and N (next) trims its HEAD,
        // so the track total (end of N) is invariant at 30. Sliding past a neighbour is rejected with
        // no mutation (all-or-nothing clamp).
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 10, 0, "P")); // idx 0: [0,10)
        let mut c = Clip::video(0, 10, 10, 0, "C"); // idx 1: [10,20)
        c.src_in = 100; // C's own content marker — must NOT change on a slide
        p.clips.push(c);
        let mut n = Clip::video(0, 20, 10, 0, "N"); // idx 2: [20,30)
        n.src_in = 5; // N's source in-point; head trim advances it by +delta
        p.clips.push(n);
        let total_end_before = p.clips[2].end(); // == 30
        let c_src_before = p.clips[1].src_in; // == 100

        // slide(C=idx1, +3): C.t0 10->13 (content held), P.len 10->13, N head trimmed by +3.
        let ok = p.slide(1, 3);
        assert!(ok, "in-range slide succeeds");
        assert_eq!(p.clips[1].t0, 13, "C shifted +3 in time");
        assert_eq!(p.clips[1].len, 10, "C len (content) unchanged");
        assert_eq!(p.clips[1].src_in, c_src_before, "C src_in (content) unchanged");
        assert_eq!(p.clips[0].len, 13, "prev P extended its OUT by +3");
        assert_eq!(p.clips[2].t0, 23, "next N head moved to C's new end (13+10)");
        assert_eq!(p.clips[2].len, 7, "next N shrank from the head by 3");
        assert_eq!(p.clips[2].src_in, 5 + 3, "next N source advanced by +3 with the head trim");
        // Cuts still abut and total track length is invariant.
        assert_eq!(p.clips[0].end(), p.clips[1].t0, "P still abuts C");
        assert_eq!(p.clips[1].end(), p.clips[2].t0, "C still abuts N");
        assert_eq!(p.clips[2].end(), total_end_before, "total track length unchanged (==30)");

        // Slide PAST the next neighbour: N has len 7 now, so +7 would make N.len == 0 -> reject,
        // mutate NOTHING. Snapshot the whole layout and confirm it is byte-identical afterward.
        let before: Vec<(i64, i64, i64)> =
            p.clips.iter().map(|c| (c.t0, c.len, c.src_in)).collect();
        let bad = p.slide(1, 7);
        assert!(!bad, "slide that collapses a neighbour returns false");
        let after: Vec<(i64, i64, i64)> =
            p.clips.iter().map(|c| (c.t0, c.len, c.src_in)).collect();
        assert_eq!(before, after, "rejected slide mutated nothing");

        // Slide with a MISSING neighbour: a lone clip on a fresh track has neither P nor N -> false.
        let mut q = Project::demo("x".into());
        q.clips.clear();
        q.clips.push(Clip::video(0, 0, 10, 0, "lone"));
        assert!(!q.slide(0, 1), "slide with no abutting neighbours returns false");
        assert_eq!(q.clips[0].t0, 0, "lone clip unmoved");
    }

    #[test]
    fn replace_clip_swaps_media_keeps_everything() {
        // A clip on media 0 at t0=5, len=10, track 0, with a non-default per-clip field (bright=0.5)
        // and group=0. Replacing its media with a VALID index changes ONLY `media`; every other
        // field (t0/len/track/bright/group) is preserved. Replacing with an out-of-range media is a
        // no-op returning false, leaving the clip (incl. media) unchanged.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.media = vec!["zero.mp4".into(), "one.mp4".into()]; // exactly 2 valid media indices: 0, 1
        let mut c = Clip::video(0, 5, 10, 0, "A");
        c.bright = 0.5; // non-default field that must survive the replace
        p.clips.push(c); // idx 0: media 0, t0 5, len 10, track 0, bright 0.5, group 0

        // Replace media 0 -> 1: succeeds, and ONLY media changes.
        let ok = p.replace_clip(0, 1);
        assert!(ok, "replace with a valid media index succeeds");
        assert_eq!(p.clips[0].media, 1, "media swapped to the new index");
        assert_eq!(p.clips[0].t0, 5, "t0 preserved");
        assert_eq!(p.clips[0].len, 10, "len preserved");
        assert_eq!(p.clips[0].track, 0, "track preserved");
        assert!((p.clips[0].bright - 0.5).abs() < 1e-6, "non-default field (bright) preserved");
        assert_eq!(p.clips[0].group, 0, "group preserved");

        // Replace with an OUT-OF-RANGE media (== media.len()): false + nothing mutated.
        let bad = p.replace_clip(0, 2); // media.len() == 2, so 2 is out of range
        assert!(!bad, "replace with an out-of-range media index returns false");
        assert_eq!(p.clips[0].media, 1, "out-of-range replace leaves media unchanged");
        // Out-of-range CLIP index is also a no-op/false.
        assert!(!p.replace_clip(99, 0), "out-of-range clip index returns false");
    }

    #[test]
    fn group_clips_assigns_fresh_id_and_move_together() {
        // Two clips at t0=0 and t0=20 (same track). Grouping them assigns ONE fresh non-zero id,
        // distinct from any prior group. move_group shifts both by the same delta; a large negative
        // move is clamped so the EARLIEST member stays >= 0. ungroup clears both back to 0.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 10, 0, "A"));  // idx 0, t0 0
        p.clips.push(Clip::video(0, 20, 10, 0, "B")); // idx 1, t0 20
        // Pre-existing group on an unrelated clip, to prove the fresh id never collides.
        p.clips.push(Clip::video(0, 50, 10, 1, "PRIOR")); // idx 2
        p.clips[2].group = 7; // a prior group id

        let id = p.group_clips(&[0, 1]);
        assert!(id != 0, "fresh group id is non-zero");
        assert_ne!(id, 7, "fresh id distinct from the prior group (7)");
        assert_eq!(id, 8, "fresh id = 1 + max existing group (7)");
        assert_eq!(p.clips[0].group, id, "clip A joined the group");
        assert_eq!(p.clips[1].group, id, "clip B joined the same group");
        assert_eq!(p.clips[2].group, 7, "the prior clip's group is untouched");

        // Move the group right by 5: both members shift +5 together.
        p.move_group(id, 5);
        assert_eq!(p.clips[0].t0, 5, "A moved +5");
        assert_eq!(p.clips[1].t0, 25, "B moved +5 (layout preserved)");

        // Move the group far LEFT (-100): clamped so the earliest (A at 5) stays >= 0, i.e. dt = -5.
        p.move_group(id, -100);
        assert_eq!(p.clips[0].t0, 0, "earliest member clamped to 0");
        assert_eq!(p.clips[1].t0, 20, "B shifted by the SAME clamped delta (-5)");

        // Ungroup: both members go back to group 0; the prior clip (group 7) is unaffected.
        p.ungroup(id);
        assert_eq!(p.clips[0].group, 0, "A ungrouped");
        assert_eq!(p.clips[1].group, 0, "B ungrouped");
        assert_eq!(p.clips[2].group, 7, "ungroup(id) leaves a different group untouched");
    }

    #[test]
    fn detach_audio_clones_onto_audio_track_and_silences_original() {
        // A single VIDEO clip on track 0 (V1) with per-clip audio gain 0.8 and a distinctive
        // media/src_in/t0/len. There is NO audio track yet, so detach_audio must CREATE one, clone the
        // clip onto it carrying gain 0.8 + the same media/src_in/t0/len, and SILENCE the original
        // (gain -> 0.0) while leaving the original's video footprint (track/t0/len) untouched.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        // Start from the legacy 2-video + 1-audio default set, then REMOVE the audio track so the
        // project has ONLY video tracks → detach_audio must create a fresh audio track.
        p.tracks = vec![Track::new(TrackKind::Video, "V1")];
        let mut c = Clip::video(3, 12, 40, 0, "A");
        c.src_in = 7;
        c.gain = 0.8; // per-clip AUDIO gain to detach
        p.clips.push(c); // idx 0: media 3, src_in 7, t0 12, len 40, track 0, gain 0.8
        assert!(!p.tracks.iter().any(|t| t.kind == TrackKind::Audio), "no audio track to start");

        let ok = p.detach_audio(0);
        assert!(ok, "detach_audio on a valid clip returns true");

        // An Audio-kind track now exists (created because none did).
        let a_idx = p
            .tracks
            .iter()
            .position(|t| t.kind == TrackKind::Audio)
            .expect("an audio track was created") as u8;

        // A NEW clip exists on that audio track with the SAME media/src_in/t0/len and gain 0.8.
        assert_eq!(p.clips.len(), 2, "exactly one detached clip was added");
        let det = &p.clips[1];
        assert_eq!(det.track, a_idx, "detached clip is on the audio track");
        assert_eq!(det.media, 3, "detached clip keeps the source media");
        assert_eq!(det.src_in, 7, "detached clip keeps src_in");
        assert_eq!(det.t0, 12, "detached clip keeps t0 (aligned in time)");
        assert_eq!(det.len, 40, "detached clip keeps len");
        assert!((det.gain - 0.8).abs() < 1e-6, "detached clip carries the original gain 0.8");

        // The ORIGINAL clip is silenced (gain == 0.0) but its VIDEO footprint is unchanged.
        let orig = &p.clips[0];
        assert!((orig.gain - 0.0).abs() < 1e-6, "original clip audio is silenced (gain 0.0)");
        assert_eq!(orig.track, 0, "original clip stays on its video track");
        assert_eq!(orig.t0, 12, "original clip t0 unchanged (video untouched)");
        assert_eq!(orig.len, 40, "original clip len unchanged (video untouched)");
        assert_eq!(orig.media, 3, "original clip media unchanged");

        // Out-of-range clip index is a no-op returning false.
        let before = p.clips.len();
        assert!(!p.detach_audio(99), "out-of-range clip index returns false");
        assert_eq!(p.clips.len(), before, "no clip added on a no-op detach");
    }

    // ----- T4 EXPORT REGION + FREEZE-FRAME -------------------------------------------------------

    #[test]
    fn export_range_region_or_whole_timeline() {
        // export_range(total) returns the marked [in,out) sub-range when VALID (in>=0 && out>in,
        // clamped to [0,total]), else the WHOLE timeline (0,total). Default -1/-1 = whole timeline.
        let total = 100i64;
        let mut p = Project::demo("x".into());

        // (1) Default in=-1 (no region) -> (0, total), byte-identical to "export everything".
        assert_eq!(p.export_in, -1, "default in mark is -1 (no region)");
        assert_eq!(p.export_out, -1, "default out mark is -1 (no region)");
        assert_eq!(p.export_range(total), (0, total), "no region -> whole timeline");

        // (2) Valid region in=10,out=40 -> (10, 40).
        p.export_in = 10;
        p.export_out = 40;
        assert_eq!(p.export_range(total), (10, 40), "valid region honored verbatim");

        // (3) INVALID region in=10,out=5 (out <= in) -> falls back to the WHOLE timeline (0, total).
        p.export_in = 10;
        p.export_out = 5;
        assert_eq!(p.export_range(total), (0, total), "out<=in is invalid -> whole timeline");

        // (4) An out past the end clamps to total; an in still >=0 is honored.
        p.export_in = 80;
        p.export_out = 999;
        assert_eq!(p.export_range(total), (80, total), "out clamps to total");
    }

    #[test]
    fn freeze_frame_inserts_silent_held_still_and_ripples_downstream() {
        // One clip C on track 0 at t0=0 len=100 src_in=5, and a LATER same-track clip D at t0=120.
        // freeze_frame(C, t=40, dur=30):
        //   - splits C at 40 (left [0,40), right starts at 40),
        //   - inserts a 30-frame freeze still at t0=40 holding C's source frame at 40
        //     (src_in = 5 + (40-0) = 45), with speed==0 && gain==0,
        //   - ripples C's right half AND D right by 30,
        //   - total timeline length grows by 30.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        let mut c = Clip::video(0, 0, 100, 0, "C");
        c.src_in = 5;
        p.clips.push(c); // idx 0: track 0, t0 0, len 100, src_in 5, end()=100
        p.clips.push(Clip::video(0, 120, 40, 0, "D")); // idx 1: track 0, t0 120 (later, same track)
        // A clip on ANOTHER track that must NOT move (ripple is current-track-only).
        p.clips.push(Clip::video(0, 50, 20, 1, "OTHER")); // idx 2: track 1

        let total_before = p.total_frames();
        let ok = p.freeze_frame(0, 40, 30);
        assert!(ok, "freeze_frame strictly inside the clip succeeds");

        // Clip count: C split into 2 (left + right) + 1 freeze still = +2 over the original 3.
        assert_eq!(p.clips.len(), 5, "split (+1) and freeze still (+1) added two clips");

        // The FREEZE still: a clip at t0==40 with speed==0 && gain==0 && len==30, src_in==45.
        let freeze = p
            .clips
            .iter()
            .find(|x| (x.speed - 0.0).abs() < 1e-6 && x.t0 == 40)
            .expect("a freeze still exists at t0=40 with speed 0");
        assert_eq!(freeze.len, 30, "freeze still length == dur");
        assert!((freeze.speed - 0.0).abs() < 1e-6, "freeze still speed == 0 (held frame)");
        assert!((freeze.gain - 0.0).abs() < 1e-6, "freeze still gain == 0 (silent)");
        assert_eq!(freeze.src_in, 45, "freeze still holds C's source frame at t (5 + 40)");
        assert_eq!(freeze.track, 0, "freeze still on C's track");

        // C's LEFT half: t0=0, len=40 (shortened to the cut), src_in=5, normal speed.
        let left = p
            .clips
            .iter()
            .find(|x| x.track == 0 && x.t0 == 0 && x.src_in == 5)
            .expect("C left half present");
        assert_eq!(left.len, 40, "C left half shortened to the cut point");
        assert!((left.speed - 1.0).abs() < 1e-6, "C left half keeps normal speed");

        // C's RIGHT half rippled +30: was at t0=40 (split point), now at t0=70. Its source continues
        // from src_in = 5 + 40 = 45, len = 60.
        let right = p
            .clips
            .iter()
            .find(|x| x.track == 0 && x.src_in == 45 && (x.speed - 1.0).abs() < 1e-6)
            .expect("C right half present (normal speed)");
        assert_eq!(right.t0, 70, "C right half rippled +dur (40 -> 70)");
        assert_eq!(right.len, 60, "C right half keeps its remaining length");

        // The later same-track clip D rippled +30: 120 -> 150. Identify D unambiguously by its
        // src_in==0 (C's halves carry src_in 5/45; the freeze carries 45) on track 0.
        let d = p
            .clips
            .iter()
            .find(|x| x.track == 0 && x.len == 40 && x.src_in == 0)
            .expect("D present");
        assert_eq!(d.t0, 150, "later same-track clip D rippled +dur (120 -> 150)");

        // The OTHER-track clip is UNMOVED (ripple is current-track-only).
        let other = p
            .clips
            .iter()
            .find(|x| x.track == 1)
            .expect("OTHER-track clip present");
        assert_eq!(other.t0, 50, "other-track clip unmoved by the ripple");

        // Total timeline length grew by exactly dur.
        assert_eq!(p.total_frames(), total_before + 30, "total length += dur");

        // GUARD CASES: t NOT strictly inside (edge / outside), dur<=0, and out-of-range index are all
        // no-ops returning false with the clip set unchanged.
        let count = p.clips.len();
        // Clip 0 is now C's left half [0,40): t==0 (its t0) is its left edge -> not strictly inside.
        assert!(!p.freeze_frame(0, 0, 30), "t == clip t0 (left edge) is not strictly inside");
        // t==40 == left half end() -> not strictly inside either (the strict `t < end()` guard).
        assert!(!p.freeze_frame(0, 40, 30), "t == clip end() (right edge) is not strictly inside");
        assert!(!p.freeze_frame(99, 10, 30), "out-of-range clip index returns false");
        // dur<=0 is rejected (use the left half, t=10 is strictly inside [0,40)).
        assert!(!p.freeze_frame(0, 10, 0), "dur == 0 returns false");
        assert!(!p.freeze_frame(0, 10, -5), "negative dur returns false");
        assert_eq!(p.clips.len(), count, "no-op freezes mutate nothing");
    }

    #[test]
    fn paste_filters_copies_grade_preserving_dst_identity() {
        // SRC clip carries a distinctive filter stack (bright=0.5, sat=2.0) and sits at one place;
        // DST clip is at a DIFFERENT t0/track/media and starts neutral (bright=0.0, sat=1.0). Pasting
        // SRC's filters onto DST copies bright/sat but PRESERVES dst's identity/position (media, t0,
        // track, len, src_in, group, fades). Covered for BOTH the &Clip API and the index API.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.media = vec!["src.mp4".into(), "dst.mp4".into()];

        // idx 0 = SRC: media 0, t0 0, track 0, with bright 0.5 + sat 2.0 (the filters to copy).
        let mut src = Clip::video(0, 0, 10, 0, "SRC");
        src.bright = 0.5;
        src.sat = 2.0;
        p.clips.push(src);

        // idx 1 = DST: DIFFERENT media 1, t0 100, track 1, len 25, src_in 9, group 4, fades 3/4.
        let mut dst = Clip::video(1, 100, 25, 1, "DST");
        dst.src_in = 9;
        dst.group = 4;
        dst.fade_in = 3;
        dst.fade_out = 4;
        // dst starts neutral so the copy is observable: bright 0.0, sat 1.0 (Clip::video defaults).
        assert!((dst.bright - 0.0).abs() < 1e-6, "dst starts at neutral bright");
        assert!((dst.sat - 1.0).abs() < 1e-6, "dst starts at neutral sat");
        p.clips.push(dst); // idx 1 = DST

        // (a) &Clip API: copy_filters_from(dst_idx, &src_clone).
        let src_clone = p.clips[0].clone();
        p.copy_filters_from(1, &src_clone);
        let d = &p.clips[1];
        // Filter fields copied from src ...
        assert!((d.bright - 0.5).abs() < 1e-6, "bright copied from src (0.5)");
        assert!((d.sat - 2.0).abs() < 1e-6, "sat copied from src (2.0)");
        // ... dst identity/position PRESERVED.
        assert_eq!(d.media, 1, "dst media unchanged");
        assert_eq!(d.t0, 100, "dst t0 unchanged");
        assert_eq!(d.track, 1, "dst track unchanged");
        assert_eq!(d.len, 25, "dst len unchanged");
        assert_eq!(d.src_in, 9, "dst src_in unchanged");
        assert_eq!(d.group, 4, "dst group unchanged");
        assert_eq!(d.fade_in, 3, "dst fade_in unchanged");
        assert_eq!(d.fade_out, 4, "dst fade_out unchanged");

        // (b) index API on a fresh neutral dst proves paste_filters mirrors copy_filters_from.
        let mut dst2 = Clip::video(1, 200, 30, 1, "DST2");
        dst2.src_in = 15;
        p.clips.push(dst2); // idx 2
        p.paste_filters(0, 2); // copy src (idx 0) filters onto dst2 (idx 2)
        let d2 = &p.clips[2];
        assert!((d2.bright - 0.5).abs() < 1e-6, "paste_filters copies bright");
        assert!((d2.sat - 2.0).abs() < 1e-6, "paste_filters copies sat");
        assert_eq!(d2.t0, 200, "paste_filters preserves dst2 t0");
        assert_eq!(d2.src_in, 15, "paste_filters preserves dst2 src_in");
        assert_eq!(d2.media, 1, "paste_filters preserves dst2 media");

        // paste onto SELF and out-of-range indices are no-ops (src filters unchanged, no panic).
        p.paste_filters(0, 0); // self -> no-op
        assert!((p.clips[0].bright - 0.5).abs() < 1e-6, "self-paste leaves src unchanged");
        p.paste_filters(0, 99); // OOR dst -> no-op
        p.paste_filters(99, 1); // OOR src -> no-op
    }

    #[test]
    fn split_all_at_cuts_every_spanning_track_with_src_continuity() {
        // Two tracks, each with ONE clip that STRICTLY spans the cut frame t=10, plus a third clip
        // whose RIGHT EDGE touches t exactly (t==end()) and so must NOT be split. split_all_at(10)
        // returns 2 (only the two spanning clips), clip count grows by exactly 2, and on each cut
        // track the two halves abut at t=10 with source continuity (right.src_in = left.src_in +
        // (t - left.t0)). The edge-touching clip is byte-identical afterwards.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        // idx 0: track 0, t0=0 len=30 src_in=5 -> spans 10 (0 < 10 < 30).
        let mut a = Clip::video(0, 0, 30, 0, "A");
        a.src_in = 5;
        p.clips.push(a);
        // idx 1: track 1, t0=4 len=20 src_in=100 -> spans 10 (4 < 10 < 24).
        let mut b = Clip::video(0, 4, 20, 1, "B");
        b.src_in = 100;
        p.clips.push(b);
        // idx 2: track 0, t0=0 len=10 src_in=0 -> end()==10, touches t at its RIGHT edge -> NOT split.
        let mut c = Clip::video(0, 0, 10, 0, "C_EDGE");
        c.src_in = 0;
        p.clips.push(c);

        let edge_before = p.clips[2].clone();
        let n = p.split_all_at(10);
        assert_eq!(n, 2, "exactly the two strictly-spanning clips are split");
        assert_eq!(p.clips.len(), 5, "clip count grows by 2 (3 -> 5)");

        // Track 0: A's left half (idx 0) kept t0=0,len=10,src_in=5; its right half abuts at t=10
        // with src_in = 5 + (10 - 0) = 15, len = 20.
        assert_eq!(p.clips[0].track, 0);
        assert_eq!((p.clips[0].t0, p.clips[0].len, p.clips[0].src_in), (0, 10, 5), "A left half");
        // split_clip inserts A's right half directly after A (idx 1).
        assert_eq!((p.clips[1].t0, p.clips[1].len, p.clips[1].src_in), (10, 20, 15), "A right half");
        assert_eq!(p.clips[1].track, 0, "A right half stays on track 0");
        assert_eq!(p.clips[0].end(), p.clips[1].t0, "A halves abut at t=10");
        assert_eq!(
            p.clips[1].src_in,
            p.clips[0].src_in + (10 - p.clips[0].t0),
            "A src continuity"
        );

        // The edge-touching clip C survived unchanged. After A's split (insert at idx 1) the
        // original idx-2 C and idx-1 B both shifted up by one; locate C by its identity (track 0,
        // len 10, src_in 0, t0 0) among the remaining clips and assert it equals the pre-split clip.
        let edge = p
            .clips
            .iter()
            .find(|x| x.track == 0 && x.len == 10 && x.t0 == 0 && x.src_in == 0)
            .expect("edge-touching clip still present");
        assert_eq!(edge.t0, edge_before.t0, "edge clip t0 untouched");
        assert_eq!(edge.len, edge_before.len, "edge clip len untouched (not split)");
        assert_eq!(edge.src_in, edge_before.src_in, "edge clip src_in untouched");

        // Track 1: B split into a left half (t0=4,len=6,src_in=100) and a right half abutting at
        // t=10 with src_in = 100 + (10 - 4) = 106, len = 14. Find both B halves on track 1.
        let mut b_halves: Vec<&Clip> = p.clips.iter().filter(|x| x.track == 1).collect();
        b_halves.sort_by_key(|x| x.t0);
        assert_eq!(b_halves.len(), 2, "track 1 now has two clips");
        assert_eq!((b_halves[0].t0, b_halves[0].len, b_halves[0].src_in), (4, 6, 100), "B left half");
        assert_eq!((b_halves[1].t0, b_halves[1].len, b_halves[1].src_in), (10, 14, 106), "B right half");
        assert_eq!(b_halves[0].end(), b_halves[1].t0, "B halves abut at t=10");
        assert_eq!(
            b_halves[1].src_in,
            b_halves[0].src_in + (10 - b_halves[0].t0),
            "B src continuity"
        );
    }

    #[test]
    fn nudge_clip_moves_and_clamps_to_zero() {
        // A single clip on track 0 at t0=20. nudge +5 -> t0=25 (len/track/src_in untouched). A large
        // negative nudge clamps t0 to 0 (never negative). An out-of-range index returns false and
        // mutates nothing.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        let mut a = Clip::video(0, 20, 30, 0, "A");
        a.src_in = 7;
        p.clips.push(a);

        assert!(p.nudge_clip(0, 5), "in-range nudge returns true");
        assert_eq!(p.clips[0].t0, 25, "nudge +5 -> t0+5");
        assert_eq!(p.clips[0].len, 30, "len unchanged by a nudge");
        assert_eq!(p.clips[0].track, 0, "track unchanged by a nudge");
        assert_eq!(p.clips[0].src_in, 7, "src_in unchanged by a nudge");

        assert!(p.nudge_clip(0, -1000), "large negative nudge still returns true");
        assert_eq!(p.clips[0].t0, 0, "t0 clamps to 0, never negative");

        // Out-of-range clip index: false, and the existing clip is untouched.
        assert!(!p.nudge_clip(99, 5), "out-of-range clip index returns false");
        assert_eq!(p.clips[0].t0, 0, "OOR nudge mutates nothing");
    }

    #[test]
    fn copy_paste_offset_preserving() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 40, 30, 0, "A"));  // idx 0
        p.clips.push(Clip::video(0, 90, 20, 1, "B"));  // idx 1 (later, other track)
        let clip = p.copy_clips(&[0, 1]);
        assert_eq!(clip.len(), 2);
        // rebased: earliest (A at 40) -> 0; B keeps its +50 offset.
        assert_eq!(clip[0].t0, 0);
        assert_eq!(clip[1].t0, 50);
        let first = p.paste_clips(&clip, 200).unwrap();
        assert_eq!(p.clips.len(), 4);
        assert_eq!(p.clips[first].t0, 200, "first pasted at the playhead");
        assert_eq!(p.clips[first + 1].t0, 250, "offset preserved");
        assert_eq!(p.clips[first + 1].track, 1, "track preserved");
    }

    #[test]
    fn paste_skips_locked_track() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.tracks[1].locked = true; // V2 locked
        let clips = vec![Clip::video(0, 0, 30, 0, "ok"), Clip::video(0, 10, 30, 1, "locked")];
        let first = p.paste_clips(&clips, 100).unwrap();
        assert_eq!(p.clips.len(), 1, "only the unlocked-track clip pasted");
        assert_eq!(p.clips[first].track, 0);
    }

    // ----- 3-POINT EDITING ops (P4) ---------------------------------------------------------

    #[test]
    fn insert_clip_ripples_downstream_by_len() {
        // Track 0: A [0,100) len 100, B [100,160) len 60. Insert a 40-frame clip at t0=50 (inside
        // A). SHOTCUT PARITY: A straddles the insert point, so it is SPLIT at 50 -> A-left [0,50)
        // len 50 src_in 0, A-right [50,100) len 50 src_in 50. Then every same-track clip with
        // t0 >= 50 (A-right at 50, B at 100) shifts RIGHT by 40; the new clip lands at 50 in the
        // opened hole. B's distinct len (60) keeps it uniquely identifiable past the split.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A")); // idx 0
        p.clips.push(Clip::video(0, 100, 60, 0, "B")); // idx 1
        let src = Clip::video(0, 0, 40, 0, "NEW");
        let new_i = p.insert_clip(0, 50, src).expect("insert lands on unlocked track");
        // A split into two halves (+1) and the new clip pushed (+1): 2 -> 4 clips.
        assert_eq!(p.clips.len(), 4);
        assert_eq!(p.clips[new_i].t0, 50, "new clip at the insert frame");
        assert_eq!(p.clips[new_i].len, 40);
        // B (len 60, the only len-60 clip) rippled right by the inserted length (40): 100 -> 140.
        let b = p.clips.iter().find(|c| c.len == 60).unwrap();
        assert_eq!(b.t0, 140, "downstream clip shifted right by the inserted length");
        // A-left is the [0,50) remnant: len 50, src_in 0, unmoved (t0 0 < 50, not shifted right).
        let a_left = p
            .clips
            .iter()
            .find(|c| c.len == 50 && c.src_in == 0)
            .expect("A-left [0,50) remnant");
        assert_eq!(a_left.t0, 0, "left remnant of the split clip stays put");
        // A-right is the [50,100) remnant: split at 50 (src_in advanced to 50) then rippled +40 to 90.
        let a_right = p
            .clips
            .iter()
            .find(|c| c.len == 50 && c.src_in == 50)
            .expect("A-right [50,100) remnant rides the ripple");
        assert_eq!(a_right.t0, 90, "right remnant of the split clip rippled right by the inserted length");
    }

    #[test]
    fn insert_clip_at_clip_boundary_does_not_split() {
        // Insert exactly at a clip's start (a boundary, not strictly inside a body): no split, the
        // clip just ripples right. Track 0: A [0,100). Insert 30 frames at t0=0 (== A's start).
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A"));
        let new_i = p.insert_clip(0, 0, Clip::video(0, 0, 30, 0, "NEW")).unwrap();
        assert_eq!(p.clips.len(), 2, "no split at a clip boundary — only the new clip is added");
        assert_eq!(p.clips[new_i].t0, 0, "new clip at the insert frame");
        let a = p.clips.iter().find(|c| c.len == 100).unwrap();
        assert_eq!(a.t0, 30, "A (t0 0 >= 0) rippled right by the inserted length, intact (unsplit)");
    }

    #[test]
    fn insert_clip_other_track_unmoved_and_clamps() {
        // Insert must not move clips on a different track; t0 clamps to >= 0.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A")); // track 0, t0 >= 0
        p.clips.push(Clip::video(0, 0, 100, 1, "D")); // track 1 (other)
        let new_i = p.insert_clip(0, -10, Clip::video(0, 0, 30, 0, "NEW")).unwrap();
        assert_eq!(p.clips[new_i].t0, 0, "negative insert frame clamped to 0");
        // A (track 0, t0 0 >= 0) shifted right by 30; D (track 1) unmoved.
        let a = p.clips.iter().find(|c| c.track == 0 && c.len == 100).unwrap();
        assert_eq!(a.t0, 30, "same-track clip rippled");
        let d = p.clips.iter().find(|c| c.track == 1).unwrap();
        assert_eq!(d.t0, 0, "other-track clip unmoved");
    }

    #[test]
    fn overwrite_clip_replaces_covered_range_no_ripple() {
        // Track 0: A [0,100), B [100,200). Overwrite a 60-frame clip at t0=80, covering [80,140):
        // A is tail-trimmed to end at 80; B is head-trimmed to start at 140; nothing ripples (B's
        // tail and the timeline length are unchanged). The new clip occupies [80,140).
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A")); // idx 0
        p.clips.push(Clip::video(0, 100, 100, 0, "B")); // idx 1
        let new_i = p.overwrite_clip(0, 80, Clip::video(0, 0, 60, 0, "OVR")).expect("unlocked");
        assert_eq!(p.clips[new_i].t0, 80);
        assert_eq!(p.clips[new_i].len, 60);
        // A tail-trimmed: now ends at 80 (len 80).
        let a = p.clips.iter().find(|c| c.t0 == 0).unwrap();
        assert_eq!(a.end(), 80, "A tail-trimmed to the overwrite start");
        // B head-trimmed: now starts at 140 (its tail at 200 is untouched -> no ripple).
        let b = p.clips.iter().find(|c| c.end() == 200).unwrap();
        assert_eq!(b.t0, 140, "B head-trimmed to the overwrite end; tail unmoved (no ripple)");
    }

    #[test]
    fn overwrite_clip_straddle_splits_into_two_remnants() {
        // One big clip A [0,300) on track 0. Overwrite [100,200) with a 100-frame clip: A is cut
        // into a left remnant [0,100) and a right remnant [200,300); the new clip fills [100,200).
        let mut p = Project::demo("x".into());
        p.clips.clear();
        let mut a = Clip::video(0, 0, 300, 0, "A");
        a.src_in = 0;
        p.clips.push(a);
        let new_i = p.overwrite_clip(0, 100, Clip::video(0, 0, 100, 0, "OVR")).unwrap();
        // 3 clips now: left remnant, new clip, right remnant.
        assert_eq!(p.clips.len(), 3);
        assert_eq!(p.clips[new_i].t0, 100);
        assert_eq!(p.clips[new_i].end(), 200);
        let left = p.clips.iter().find(|c| c.t0 == 0).unwrap();
        assert_eq!(left.end(), 100, "left remnant ends at overwrite start");
        let right = p.clips.iter().find(|c| c.t0 == 200).unwrap();
        assert_eq!(right.end(), 300, "right remnant starts at overwrite end");
        assert_eq!(right.src_in, 200, "right remnant source advanced past the covered span");
    }

    #[test]
    fn overwrite_clip_fully_covered_is_removed() {
        // A small clip A [50,90) entirely inside the overwrite range [0,200) is removed.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 50, 40, 0, "A")); // [50,90)
        p.clips.push(Clip::video(0, 50, 40, 1, "D")); // other track, must survive
        let new_i = p.overwrite_clip(0, 0, Clip::video(0, 0, 200, 0, "OVR")).unwrap();
        // A removed; new clip + D remain (the new clip is appended LAST).
        assert_eq!(p.clips.len(), 2);
        assert_eq!(p.clips[new_i].t0, 0);
        assert!(p.clips.iter().any(|c| c.track == 1), "other-track clip survived");
        assert!(
            !p.clips.iter().any(|c| c.track == 0 && c.len == 40),
            "fully-covered same-track clip was removed"
        );
    }

    #[test]
    fn append_clip_lands_at_track_end() {
        // Track 0: A [0,100), B [100,150) -> track end 150. Track 1: C [0,80) -> track end 80.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 100, 0, "A"));
        p.clips.push(Clip::video(0, 100, 50, 0, "B"));
        p.clips.push(Clip::video(0, 0, 80, 1, "C"));
        let v0 = p.append_clip(0, Clip::video(0, 0, 30, 0, "AP0")).unwrap();
        assert_eq!(p.clips[v0].t0, 150, "appended at the end of track 0");
        let v1 = p.append_clip(1, Clip::video(0, 0, 30, 1, "AP1")).unwrap();
        assert_eq!(p.clips[v1].t0, 80, "appended at the end of track 1");
    }

    #[test]
    fn append_clip_empty_track_lands_at_zero() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        let i = p.append_clip(2, Clip::video(0, 0, 30, 2, "A1")).unwrap();
        assert_eq!(p.clips[i].t0, 0, "append onto an empty track lands at frame 0");
        assert_eq!(p.track_end(2), 30, "track_end now reflects the appended clip");
    }

    #[test]
    fn three_point_ops_refuse_locked_track() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.tracks[0].locked = true; // V1 locked
        let n0 = p.clips.len();
        assert!(p.insert_clip(0, 0, Clip::video(0, 0, 30, 0, "x")).is_none());
        assert!(p.overwrite_clip(0, 0, Clip::video(0, 0, 30, 0, "x")).is_none());
        assert!(p.append_clip(0, Clip::video(0, 0, 30, 0, "x")).is_none());
        assert_eq!(p.clips.len(), n0, "no clip placed on a locked track");
    }

    #[test]
    fn overwrite_clip_sets_track_and_carries_source_fields() {
        // The placed clip takes its `track`/`t0` from the args but carries its other fields verbatim
        // (here a non-default look/gain), and overwrite onto an EMPTY track is just a placement.
        let mut p = Project::demo("x".into());
        p.clips.clear();
        let mut src = Clip::video(0, 5, 40, 0, "S");
        src.src_in = 5; // a non-default source in-point (Clip::video's 2nd arg is t0, not src_in)
        src.look = 1;
        src.gain = 0.5;
        let i = p.overwrite_clip(1, 20, src).unwrap();
        assert_eq!(p.clips[i].track, 1, "track set from the arg");
        assert_eq!(p.clips[i].t0, 20, "t0 set from the arg");
        assert_eq!(p.clips[i].src_in, 5, "source in-point carried");
        assert_eq!(p.clips[i].look, 1, "look carried");
        assert!((p.clips[i].gain - 0.5).abs() < 1e-6, "gain carried");
    }

    // ----- P5 arbitrary tracks -----
    #[test]
    fn add_track_appends_and_names() {
        let mut p = Project::demo("x".into());
        assert_eq!(p.tracks.len(), 3); // default V1 V2 A1
        let vi = p.add_track(TrackKind::Video);
        assert_eq!(vi, 3);
        assert_eq!(p.tracks[3].name, "V3"); // third video
        assert_eq!(p.tracks[3].kind, TrackKind::Video);
        let ai = p.add_track(TrackKind::Audio);
        assert_eq!(p.tracks[ai].name, "A2"); // second audio
    }

    #[test]
    fn remove_track_drops_clips_and_reindexes() {
        let mut p = Project::demo("x".into());
        p.clips.clear();
        p.clips.push(Clip::video(0, 0, 50, 0, "a")); // track 0 (V1)
        p.clips.push(Clip::video(0, 0, 50, 1, "b")); // track 1 (V2) -> removed
        p.clips.push(Clip::video(0, 0, 50, 2, "c")); // track 2 (A1) -> reindexes to 1
        assert!(p.remove_track(1)); // remove V2
        assert_eq!(p.tracks.len(), 2);
        assert_eq!(p.clips.len(), 2, "the clip on the removed track is gone");
        assert!(p.clips.iter().any(|c| c.track == 0), "V1 clip stays on track 0");
        assert!(p.clips.iter().any(|c| c.track == 1), "A1 clip reindexed 2 -> 1");
        assert!(!p.clips.iter().any(|c| c.track == 2), "no clip left on the old track 2");
    }

    #[test]
    fn is_audio_and_hidden_read_tracks() {
        let mut p = Project::demo("x".into());
        assert!(!p.is_audio(0) && !p.is_audio(1) && p.is_audio(2)); // V1 V2 video, A1 audio
        p.tracks[0].hidden = true;
        p.tracks[2].muted = true;
        assert!(p.is_hidden(0) && !p.is_hidden(1));
        assert!(p.is_muted(2) && !p.is_muted(0));
        assert!(!p.is_hidden(99), "out-of-range track is not hidden");
    }

    #[test]
    fn title_is_empty_and_clip_is_title() {
        // Default title -> empty -> the clip is NOT a title (untitled clips render unchanged).
        let mut t = Title::default();
        assert!(t.is_empty(), "default title is empty");
        // Whitespace-only counts as empty (worker still composites normally).
        t.text = "   \t \n".into();
        assert!(t.is_empty(), "whitespace-only title is empty");
        // Real text -> not empty.
        t.text = "Hello".into();
        assert!(!t.is_empty(), "non-blank title is not empty");

        // Clip::is_title mirrors !title.is_empty(): the demo clips have a default (empty) title.
        let mut c = Clip::video(0, 0, 100, 0, "x");
        assert!(!c.is_title(), "a fresh clip has no title");
        c.title.text = "Lower third".into();
        assert!(c.is_title(), "a clip with text is a title");
        c.title.text = "  ".into();
        assert!(!c.is_title(), "whitespace title is not a title");

        // The lower_third preset is a real (non-empty) title with the expected layout.
        let lt = Title::lower_third("Name / Role");
        assert!(!lt.is_empty(), "lower_third has text");
        assert_eq!(lt.text, "Name / Role");
        assert!(lt.y > 0.5, "lower_third anchors toward the lower part of the frame");
        assert_eq!(lt.rgb, [1.0, 1.0, 1.0], "lower_third defaults to white");
    }

    // T3 — MEDIA BINS. The pre-added bin helpers (add_bin / set_media_bin / bin_of) backing the pool's
    // bin-grouping UI: a 3-media project starts all in bin 0; adding a bin "B" yields index 1; moving
    // media 2 into bin 1 reads back via bin_of while the untouched media stays in bin 0; an
    // out-of-range media move is a no-op; and a blank-named bin is ignored (returns 0, adds nothing).
    // Pure model edits — binning never touches clips / the timeline / the render.
    #[test]
    fn media_bins_add_and_assign() {
        // A project with 3 media (paths + names parallel); default single "Media" bin.
        let mut p = Project::demo("a".into());
        p.media = vec!["a".into(), "b".into(), "c".into()];
        p.names = vec!["a".into(), "b".into(), "c".into()];
        // Fresh single-bin baseline so the test owns its bin state regardless of demo() defaults.
        p.bin_names = vec!["Media".into()];
        p.media_bin = vec![0, 0, 0];

        // Everything starts in bin 0.
        assert_eq!(p.bin_of(0), 0, "media 0 starts in bin 0");
        assert_eq!(p.bin_of(1), 0, "media 1 starts in bin 0");
        assert_eq!(p.bin_of(2), 0, "media 2 starts in bin 0");

        // add_bin("B") -> index 1 (appended after bin 0 "Media").
        let b = p.add_bin("B");
        assert_eq!(b, 1, "add_bin returns the new bin's index (1)");
        assert_eq!(p.bin_names.len(), 2, "two bins now: Media + B");
        assert_eq!(p.bin_names[1], "B", "the new bin is named 'B'");

        // set_media_bin(2, 1) -> bin_of(2)==1, and the untouched media 0 stays in bin 0.
        p.set_media_bin(2, 1);
        assert_eq!(p.bin_of(2), 1, "media 2 moved into bin 1");
        assert_eq!(p.bin_of(0), 0, "media 0 untouched (still bin 0)");

        // set_media_bin on an out-of-range media is a no-op (no panic, no bin change).
        p.set_media_bin(99, 1);
        assert_eq!(p.bin_of(2), 1, "out-of-range move did not disturb media 2");
        assert_eq!(p.bin_of(0), 0, "out-of-range move did not disturb media 0");

        // add_bin("") returns 0 and adds NOTHING (blank-name guard).
        let nbins_before = p.bin_names.len();
        let blank = p.add_bin("");
        assert_eq!(blank, 0, "a blank bin name returns 0");
        assert_eq!(p.bin_names.len(), nbins_before, "a blank bin name adds no bin");
    }

    // T4 — MEDIA RELINK. relink_media swaps a pool media's path in place; clips index by media id, so
    // a clip on media 0 keeps its index (now decoding the new file). Out-of-range index and an empty
    // new path are both no-ops that return false. Pure model edit — never touches clip indices.
    #[test]
    fn relink_media_swaps_path_keeps_clip_index() {
        // A project with one media ["/x/a.mp4"] and a clip on media 0 (demo() places its clips on
        // media 0). Collapse to a single media so the clip references exactly index 0.
        let mut p = Project::demo("/x/a.mp4".into());
        p.media = vec!["/x/a.mp4".into()];
        p.names = vec!["a".into()];
        // Sanity: a clip exists and references media 0 before the relink.
        assert!(!p.clips.is_empty(), "demo has at least one clip");
        assert_eq!(p.clips[0].media, 0, "clip starts on media 0");

        // relink_media(0, "/y/b.mp4") -> true; media[0] is swapped; the clip still references media 0
        // (it now decodes the NEW file via its unchanged index).
        assert!(p.relink_media(0, "/y/b.mp4"), "relink of a valid index + non-empty path succeeds");
        assert_eq!(p.media[0], "/y/b.mp4", "media[0] is now the new path");
        assert_eq!(p.clips[0].media, 0, "the clip still references media 0 after relink");

        // relink_media(9, "z") -> false, NO mutation (index out of range).
        assert!(!p.relink_media(9, "z"), "out-of-range index returns false");
        assert_eq!(p.media[0], "/y/b.mp4", "out-of-range relink did not mutate media[0]");
        assert_eq!(p.media.len(), 1, "out-of-range relink did not add media");

        // relink_media(0, "") -> false, NO mutation (empty new path).
        assert!(!p.relink_media(0, ""), "an empty new path returns false");
        assert_eq!(p.media[0], "/y/b.mp4", "empty-path relink did not mutate media[0]");
    }
}
