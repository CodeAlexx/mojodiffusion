//! Client to the `gcompose` engine worker (separate process; owns OpenCL — the egui process
//! never links/calls OpenCL).
//!
//! P1: a single PERSISTENT `gcompose --serve` process is started once and reused for every
//! frame. `request_frame(project, t)` resolves the program at timeline frame `t` from the model
//! (V1 base + V2 overlay, mirroring MojoMedia's rs0/rs1 logic), sends one request line to the
//! worker's stdin, waits for a `DONE <out>` line, reads the RGBA file back, and returns it.
//! On any I/O / protocol failure it restarts the worker (up to a few times) to absorb the
//! worker's small OpenCL-init flake.

use crate::model::Project;
use ab_glyph::{point, Font, FontVec, PxScale, ScaleFont};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ============================ P5 TITLE TEXT RASTERIZER ============================
//
// A per-clip TEXT overlay (Shotcut "Text: Simple" / dynamictext). A clip whose `Clip.title` is
// non-empty has its text rasterized into a full-frame GVW×GVH transparent RGBA8 layer here, written
// to a cached temp file, and fed to the engine on the existing base/over path as the `RAW:<path>`
// sentinel (see `resolve_frame`). The engine reads that raw buffer straight into the slot (skipping
// decode) and composites it with OVER alpha — so a V2 title clip shows its text over the V1 base,
// and a title-only clip shows text over black. A project with NO titles never produces a `RAW:`
// path, so the render is byte-identical to the pre-P5 output (identity).
//
// FONT: loaded ONCE (lazy `OnceLock`) from the bundled `ui/assets/title_font.ttf` — located by
// mirroring icons.rs's asset-path search (beside the exe, `<exe>/assets`, a few parents up, and the
// dev `ui/assets`). If the bundled asset can't be found/parsed we fall back to the system
// LiberationSans. When NO font can be loaded, `rasterize_title` returns None (the title is dropped;
// the clip composites normally) rather than failing the frame.

/// The engine compose canvas (== gcompose ffi::GVW/GVH and worker PVW/PVH). The title is rasterized
/// into a GVW×GVH×4 RGBA8 buffer so the engine can upload it directly to a slot via the `RAW:` path.
const TITLE_W: usize = PVW; // 1280
const TITLE_H: usize = PVH; // 856

/// System-font fallback when the bundled `title_font.ttf` asset is missing (matches the contract's
/// pinned fallback path). LiberationSans is the same family bundled at ui/assets/title_font.ttf.
const FALLBACK_FONT_PATH: &str = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf";

/// Process-global, lazily-loaded title font. `None` only if neither the bundled asset nor the system
/// fallback could be read/parsed; the rasterizer then yields None (the clip composites without text).
static TITLE_FONT: OnceLock<Option<FontVec>> = OnceLock::new();

/// Candidate filesystem paths for the bundled `title_font.ttf`, in priority order — mirrors
/// `icons.rs::candidate_paths` (beside the running exe, `<exe>/assets`, a couple of parents up into a
/// sibling `assets`/`ui/assets`, then the dev-tree `ui/assets`/`assets` relative to the cwd). The
/// system fallback is appended LAST so a deployed build with no bundled asset still finds a font.
fn title_font_candidates() -> Vec<std::path::PathBuf> {
    const FONT_NAME: &str = "title_font.ttf";
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();

    if let Some(path) = std::env::var_os("SERENITY_GENESIS_ASSETS") {
        dirs.push(std::path::PathBuf::from(path));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            dirs.push(exe_dir.to_path_buf());
            dirs.push(exe_dir.join("assets"));
            let mut up = exe_dir.to_path_buf();
            for _ in 0..3 {
                if let Some(parent) = up.parent() {
                    up = parent.to_path_buf();
                    dirs.push(up.join("assets"));
                    dirs.push(up.join("ui").join("assets"));
                } else {
                    break;
                }
            }
        }
    }
    // Dev fallbacks relative to the working directory.
    dirs.push(std::path::PathBuf::from("ui/assets"));
    dirs.push(std::path::PathBuf::from("assets"));

    let mut paths: Vec<std::path::PathBuf> = dirs.into_iter().map(|d| d.join(FONT_NAME)).collect();
    // System-font fallback last (the pinned LiberationSans path).
    paths.push(std::path::PathBuf::from(FALLBACK_FONT_PATH));
    paths
}

/// Load (once) and return the bundled title font, or None if no font file could be read/parsed.
/// Tries each candidate in turn; the first that both reads AND parses as a valid font wins.
fn title_font() -> Option<&'static FontVec> {
    TITLE_FONT
        .get_or_init(|| {
            for p in title_font_candidates() {
                match std::fs::read(&p) {
                    Ok(bytes) => match FontVec::try_from_vec(bytes) {
                        Ok(font) => {
                            eprintln!("[title] loaded font: {}", p.display());
                            return Some(font);
                        }
                        Err(_) => {
                            eprintln!("[title] font parse failed (trying next): {}", p.display());
                        }
                    },
                    Err(_) => {} // not at this candidate; try the next.
                }
            }
            eprintln!("[title] no title font found (bundled or system); titles will not render");
            None
        })
        .as_ref()
}

/// Stable hash of a title's RENDERED inputs, for the cache filename. Two titles that rasterize to the
/// same pixels share the same `/tmp/genesis_title_<hash>.rgba` file (and the same upload). Uses the
/// same FNV-1a as the thumbnail temp paths. The text + every layout/colour field is folded in so a
/// change to any of them keys a fresh file.
fn title_hash(title: &crate::model::Title) -> u64 {
    let key = format!(
        "{}|{}|{}|{}|{}|{}|{}",
        title.text,
        title.size_frac.to_bits(),
        title.x.to_bits(),
        title.y.to_bits(),
        title.rgb[0].to_bits(),
        title.rgb[1].to_bits(),
        title.rgb[2].to_bits(),
    );
    small_hash(&key)
}

/// Rasterize `title.text` into a full-frame (`TITLE_W`×`TITLE_H`) transparent RGBA8 layer and write
/// it to a CACHED temp file `/tmp/genesis_title_<hash>.rgba`, returning that path. The engine reads
/// the file via the `RAW:` sentinel and composites it (OVER alpha) over the program.
///
/// Layout (Shotcut Text: Simple-ish): the font pixel height is `title.size_frac * TITLE_H`; the pen
/// origin is at `(title.x*TITLE_W, title.y*TITLE_H + ascent)` so `title.y` anchors the TOP of the
/// text. Glyphs advance the pen by their scaled `h_advance`; a `\n` starts a new line (pen x reset,
/// y advanced by the scaled line height). Each glyph is outlined and `draw`n: every covered pixel is
/// written `rgb = title.rgb*255`, `alpha = coverage*255`, MAX-blended so overlapping glyph outlines
/// don't darken the overlap. Pixels outside the frame are clipped.
///
/// Returns None when the text is empty (the caller then composites the clip normally — no `RAW:`
/// path, identity render) or when no font could be loaded.
///
/// CACHING: if the keyed file already exists, the raster is skipped and the existing path returned —
/// a held playhead / a long render on the same title reuses the file (the engine re-reads it cheaply;
/// it never changes for the same inputs). A write failure returns None (the title is dropped, never
/// fails the frame).
fn rasterize_title(title: &crate::model::Title) -> Option<String> {
    if title.is_empty() {
        return None; // no text: clip composites normally (identity).
    }
    let path = format!("/tmp/genesis_title_{:x}.rgba", title_hash(title));
    // Cache hit: the keyed file already holds this exact raster — reuse it (skip re-rasterizing).
    if std::path::Path::new(&path).exists() {
        return Some(path);
    }

    let font = title_font()?; // no font available: drop the title (composite normally).

    // Font pixel height from the normalized size fraction; clamp to a sane positive range so a stray
    // 0/huge size_frac can't produce a zero-area or runaway raster.
    let px = (title.size_frac * TITLE_H as f32).clamp(4.0, TITLE_H as f32);
    let scale = PxScale::from(px);
    let scaled = font.as_scaled(scale);
    let ascent = scaled.ascent();
    let line_height = scaled.height(); // ascent - descent + line_gap (scaled)

    // Full-frame, fully TRANSPARENT RGBA8 (alpha 0 everywhere) — the engine composites it OVER the
    // program, so only the drawn glyph pixels (alpha > 0) show.
    let mut buf = vec![0u8; TITLE_W * TITLE_H * 4];

    let r = (title.rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (title.rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (title.rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8;

    // Pen origin: x = title.x*W (left of the text), y = title.y*H + ascent (so title.y anchors the
    // TEXT TOP, with the baseline one ascent below it). Multi-line: each '\n' resets pen_x and
    // advances pen_y by the scaled line height.
    let origin_x = title.x * TITLE_W as f32;
    let top_y = title.y * TITLE_H as f32;
    let mut pen_x = origin_x;
    let mut pen_y = top_y + ascent;

    for ch in title.text.chars() {
        if ch == '\n' {
            pen_x = origin_x;
            pen_y += line_height;
            continue;
        }
        let glyph_id = scaled.glyph_id(ch);
        let advance = scaled.h_advance(glyph_id);
        // Build a positioned glyph at the current pen, then outline it. Whitespace / glyphs with no
        // outline (e.g. ' ') just advance the pen.
        let mut glyph = glyph_id.with_scale(scale);
        glyph.position = point(pen_x, pen_y);
        if let Some(outline) = font.outline_glyph(glyph) {
            // px_bounds gives the integer pixel rect of the outline in the buffer's coordinate space;
            // draw yields (gx, gy) RELATIVE to that rect's top-left + a coverage in [0,1].
            let bounds = outline.px_bounds();
            let ox = bounds.min.x as i32;
            let oy = bounds.min.y as i32;
            outline.draw(|gx, gy, coverage| {
                if coverage <= 0.0 {
                    return;
                }
                let x = ox + gx as i32;
                let y = oy + gy as i32;
                if x < 0 || y < 0 || x >= TITLE_W as i32 || y >= TITLE_H as i32 {
                    return; // off-frame: clip.
                }
                let idx = (y as usize * TITLE_W + x as usize) * 4;
                let a = (coverage.clamp(0.0, 1.0) * 255.0).round() as u8;
                // MAX-blend the alpha so overlapping glyph outlines never darken the overlap; write
                // the (solid) colour wherever this glyph contributes more alpha than what's there.
                if a > buf[idx + 3] {
                    buf[idx] = r;
                    buf[idx + 1] = g;
                    buf[idx + 2] = b;
                    buf[idx + 3] = a;
                }
            });
        }
        pen_x += advance;
    }

    if std::fs::write(&path, &buf).is_err() {
        eprintln!("[title] failed to write raster cache: {path}");
        return None; // drop the title rather than fail the frame.
    }
    Some(path)
}

// ============================ P48 SUBTITLE RASTERIZER ============================
//
// A timeline-wide TIMED CAPTION (`model::Subtitle`), rendered OVER THE PROGRAM as the top-most
// layer. This REUSES the P5 title raster + the EXISTING `RAW:` composite — there is NO second
// rasterizer and NO hand-rolled compositing. `rasterize_subtitle` packages the cue text into a
// `model::Title` lower-third preset and calls `rasterize_title`, returning the SAME cached
// `RAW:<path>` sentinel the title path uses. The active-subtitle frame then folds that RAW layer
// over the program composite (see `render_program` / `request_frame`) so the caption sits on top.
//
// IDENTITY: a frame with NO active subtitle never calls this and never emits a subtitle RAW: layer,
// so its ENC / fold lines are byte-identical to pre-P48. Empty text or a missing font yields None
// (the frame composites with no caption — it is NEVER failed).

/// Lower-third caption layout for a subtitle, as fractions of the GVW×GVH canvas. Kept here (not on
/// the model) so the render input is a pure worker concern and the model stays raster-agnostic.
///   - size_frac 0.055 of the frame height (a readable caption, smaller than a clip title).
///   - x 0.12 (the existing `rasterize_title` LEFT-ALIGNS at `x`, so a typical single line sits
///     low-CENTRE; a longer line extends right but stays on-canvas — glyphs past the edge are
///     clipped, never failing the frame).
///   - y 0.86 (lower third).
///   - rgb white.
const SUB_SIZE_FRAC: f32 = 0.055;
const SUB_X: f32 = 0.12;
const SUB_Y: f32 = 0.86;

/// P48 — rasterize a subtitle's `text` into the SAME full-frame transparent RGBA8 raster the P5 title
/// path produces, returning its cached `/tmp/genesis_title_<hash>.rgba` path (NO `RAW:` prefix; the
/// caller wraps it). Builds a lower-third `model::Title` and delegates to `rasterize_title` — there is
/// NO new rasterizer and the bundled font + glyph draw are reused verbatim.
///
/// Returns None when the text is empty/whitespace (the `Title::is_empty` guard in `rasterize_title`)
/// or no font could be loaded — the caller then composites the frame with NO caption (never fails the
/// frame). The raster is cached by `title_hash`, so a held caption over many frames rasterizes once.
fn rasterize_subtitle(text: &str) -> Option<String> {
    let title = crate::model::Title {
        text: text.to_string(),
        size_frac: SUB_SIZE_FRAC,
        x: SUB_X,
        y: SUB_Y,
        rgb: [1.0, 1.0, 1.0],
    };
    rasterize_title(&title)
}

/// P48 — a `Resolved` that composites the subtitle RAW raster (`sub_raw`, a bare temp path) OVER a
/// RAW-RGBA base (`base_raw` = the finished program composite). Mirrors `build_layer_resolved` (the
/// N-layer fold step): both base and over are `RAW:` sentinels, the pip rect is FULL-FRAME
/// (0,0,1,1) so the caption raster overlays the program 1:1, opacity 1.0, blend Normal, and EVERY
/// colour/transform/look/transition/key field is IDENTITY — so the program composite underneath
/// keeps its baked pixels and only the caption's alpha>0 glyph pixels show on top. Reuses the
/// `Resolved` struct exactly as the layer fold does, so `format_preview` builds the wire line (no
/// hand-typed line, no second compositor).
fn build_subtitle_resolved(base_raw: &str, sub_raw: &str) -> Resolved {
    Resolved {
        base_path: format!("RAW:{base_raw}"),
        base_frame: 0,
        over_path: format!("RAW:{sub_raw}"),
        over_frame: 0,
        op: 1.0,
        blend: 0,
        px: 0.0,
        py: 0.0,
        pw: 1.0,
        ph: 1.0,
        bright: 0.0,
        contrast: 1.0,
        sat: 1.0,
        cbright: 0.0,
        ccontrast: 1.0,
        csat: 1.0,
        look_kind: 0,
        look_amt: 1.0,
        lut_path: "-".to_string(),
        trans_kind: -1,
        trans_prog: 0.0,
        trans_param: 4.0,
        trans_path: "-".to_string(),
        trans_frame: 0,
        lift: [0.0, 0.0, 0.0],
        gamma: [1.0, 1.0, 1.0],
        gain_rgb: [1.0, 1.0, 1.0],
        rot: 0.0,
        scale: 1.0,
        blur: 0.0,
        ck_on: 0,
        ck_key: [0.0, 1.0, 0.0],
        ck_sim: 0.4,
        ck_smooth: 0.1,
        ck_spill: 0.0,
        curve: [0.0, 0.25, 0.5, 0.75, 1.0],
        vignette: 0.0,
        sharpen: 0.0,
        flip: 0,
        fx: 0,
        hsl: [0.0, 1.0, 0.0],
        levels: [0.0, 1.0, 1.0],
        mosaic: 0,
        gmap_amt: 0.0,
        gmap_lo: [0.0, 0.0, 0.0],
        gmap_hi: [1.0, 1.0, 1.0],
        denoise: 0.0,
        glow_amt: 0.0,
        glow_thr: 0.7,
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
        eq_fov: 90.0,
        mask_shape: 0,
        mask_cx: 0.5,
        mask_cy: 0.5,
        mask_rw: 0.5,
        mask_rh: 0.5,
        mask_feather: 0.0,
        mask_invert: 0,
        mirror_x: 0,
        kaleido: 0,
        dither: 0.0,
        sel_band: 0,
        sel_hshift: 0.0,
        sel_sat: 1.0,
        sol_thr: 0.0,
        temp: 0.0,
        fade: 1.0,
    }
}

// Preview surface resolution. These MUST equal the engine's OpenCL working resolution
// (gcompose ffi::GVW/GVH = 1280×856): the worker always composes at GVW×GVH and returns exactly
// PVW*PVH*4 bytes (see `try_once`'s length check) / GVW*GVH*4 floats per ENC frame. The render
// path (`render_program`) is decoupled — the engine pins the encoder dims to its own GVW/GVH and
// ignores the OPEN wire w/h (finding #7) — but the PREVIEW path still length-checks against
// PVW/PVH, so if PVW/PVH ever change here they MUST be changed to match the engine's GVW/GVH too,
// or every preview frame fails the byte-count check. ffi lives in the gcompose crate and is not
// importable from ui, so this invariant is enforced by convention, not a static assert.
pub const PVW: usize = 1280;
pub const PVH: usize = 856;

/// Scope image dimensions (PINNED, Slice A). The engine's scope kernels always render a fixed
/// 256×256 RGBA8 image (gcompose ffi::SVW/SVH; the histogram is rasterized into the same size), so
/// `scope()` always reads back exactly SW*SH*4 bytes. Team C consumes these for the scopes panel.
pub const SW: usize = 256;
pub const SH: usize = 256;

const PREVIEW_RGBA: &str = "/tmp/genesis_frame.rgba"; // per-request output path
/// Per-request output path for a rendered scope image (256×256 RGBA8). Reused each call — `scope()`
/// holds the worker mutex across its PREVIEW+SCOPE round-trip, so there is never a concurrent writer.
const SCOPE_RGBA: &str = "/tmp/genesis_scope.rgba";
const MAX_ATTEMPTS: usize = 3;

/// After a worker spawn/handshake fails, suppress further (re)spawns for this long. A single
/// egui repaint can miss many thumbnails at once; without this, each cache-miss would pay up to
/// MAX_ATTEMPTS fresh `gcompose --serve` spawns (each re-initing OpenCL) — a spawn/init storm
/// within one frame (finding #6). During the cooldown, `command_with_restart` fails fast (None)
/// instead of re-spawning, so the dead worker is retried at most once per cooldown window.
const SPAWN_COOLDOWN: Duration = Duration::from_millis(750);

/// Unix-millis of the last failed worker spawn/handshake (0 = none yet). Used to gate respawns
/// across the stateless THUMB/ENV/OPEN path so one dead-worker repaint can't storm the OS.
static LAST_SPAWN_FAIL_MS: AtomicU64 = AtomicU64::new(0);

/// True while a `play_program` background thread is assembling + playing the audition WAV
/// (finding #9). Set before the thread is spawned and cleared when it finishes. A second Space
/// press while one playback is already in flight is dropped (no-op) rather than stacking another
/// detached thread that would block on the worker mutex and then spawn a duplicate `paplay`. This
/// dedups rapid presses and bounds the number of concurrent background players to one.
static PLAYBACK_IN_FLIGHT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Stop-edge coordination flag (findings #7 + #8). `stop_playback` sets this true; `play_program`
/// clears it at the moment it dispatches a fresh audition. The detached assembly thread checks it
/// right BEFORE spawning the system player and SKIPS the spawn if a stop arrived during the
/// (seconds-long) WAVE/AUDIO assembly window — so a `stop_playback` fired before `spawn_player` ran
/// no longer leaks a late, unkillable player. Without this, the only stop mechanism was killing an
/// already-spawned child (finding #7), which does nothing for a player that has not been spawned yet.
static STOP_REQUESTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The currently-playing detached audition player child (`paplay`/`aplay`), if any. `spawn_player`
/// stores the spawned `Child` here so `stop_playback` (the PINNED API) can kill it on demand —
/// instead of leaving the audio to run to its natural EOF. `None` when nothing is playing (no child
/// spawned yet, or the previous one was already killed/finished). Guarded by its own mutex,
/// independent of the worker mutex, so `stop_playback` can fire instantly even while the worker is
/// busy composing/assembling. See `stop_playback` for the kill mechanism + pkill fallback.
static PLAYER_CHILD: OnceLock<Mutex<Option<Child>>> = OnceLock::new();

fn player_slot() -> &'static Mutex<Option<Child>> {
    PLAYER_CHILD.get_or_init(|| Mutex::new(None))
}

/// Current Unix time in millis (monotonic enough for a short cooldown; never panics).
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// True if we are still inside the post-failure spawn cooldown (so we should fail fast).
fn in_spawn_cooldown() -> bool {
    let last = LAST_SPAWN_FAIL_MS.load(Ordering::Relaxed);
    last != 0 && now_ms().saturating_sub(last) < SPAWN_COOLDOWN.as_millis() as u64
}

/// Mark a fresh spawn failure (starts/refreshes the cooldown window).
fn mark_spawn_fail() {
    LAST_SPAWN_FAIL_MS.store(now_ms(), Ordering::Relaxed);
}

/// Clear the cooldown after any successful round-trip (worker is healthy again).
fn clear_spawn_cooldown() {
    LAST_SPAWN_FAIL_MS.store(0, Ordering::Relaxed);
}

/// How long a single worker reply may take before we treat the worker as WEDGED (finding #4).
///
/// The ~10% flake the slice targets is a worker process CRASH (the driver segfaults), which the OS
/// surfaces as an immediate EOF on the pipe → `Broken` → retry. But a flaky OpenCL driver can also
/// HANG (deadlock in the queue) rather than crash; a blocking `read_line` would then wait FOREVER,
/// holding the worker mutex and freezing the egui UI thread (which blocks on the same mutex for the
/// whole render). To bound that, every reply is read with this timeout: if no line arrives in time,
/// the read returns `Broken` so the worker is torn down + retried instead of hanging the export.
///
/// The value must comfortably exceed the slowest LEGITIMATE single reply. The heaviest per-command
/// work is one ENC frame (decode + OpenCL composite + one encoded frame) or the CLOSE drain (mux +
/// trailer); both are well under a second on any working GPU. 30 s is generous headroom so a merely
/// slow-but-alive worker is never killed, while a true deadlock is caught in bounded time.
const REPLY_TIMEOUT: Duration = Duration::from_secs(30);

/// A line read from the worker's stdout, or the EOF marker.
enum WorkerLine {
    Line(String),
    Eof,
}

/// A live `gcompose --serve` process plus its piped stdin and a background reader.
///
/// READER THREAD (finding #4): the worker's stdout is drained by a dedicated thread that pushes each
/// line (trimmed) onto `rx` and finally an `Eof` marker when the pipe closes. The request path then
/// reads replies with `recv_timeout(REPLY_TIMEOUT)` so a WEDGED worker (driver deadlock — no crash,
/// no EOF) is detected as `Broken` in bounded time rather than blocking the held worker mutex (and
/// the egui UI) forever. A crash still surfaces promptly as `Eof` (pipe closed). The thread owns the
/// `BufReader<ChildStdout>`; it exits on EOF or when the child is killed in `Drop`.
struct WorkerProc {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<WorkerLine>,
}

impl Drop for WorkerProc {
    fn drop(&mut self) {
        // Best-effort: closing stdin makes the serve loop exit, killing the child closes its stdout
        // which unblocks + ends the reader thread (it then drops its channel sender). We do not join
        // the reader thread (it is detached): once the child's stdout closes, its blocking read
        // returns and the thread exits on its own.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Process-global persistent worker. `None` until first spawned / after a failed restart.
static WORKER: OnceLock<Mutex<Option<WorkerProc>>> = OnceLock::new();

fn worker_slot() -> &'static Mutex<Option<WorkerProc>> {
    WORKER.get_or_init(|| Mutex::new(None))
}

/// Tear down the persistent worker (kills + reaps the gcompose child via WorkerProc::Drop).
/// Call before `std::process::exit`, which otherwise skips destructors and leaks the child.
pub fn shutdown() {
    if let Ok(mut slot) = worker_slot().lock() {
        *slot = None;
    }
}

/// Path to the isolated `gcompose` binary.
///
/// Serenity sets `SERENITY_GENESIS_WORKER` because its web server and the
/// vendored editor engine are built into separate output trees. The sibling
/// lookup remains the upstream/native-UI fallback.
fn worker_path() -> Option<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("SERENITY_GENESIS_WORKER") {
        if !path.is_empty() {
            return Some(std::path::PathBuf::from(path));
        }
    }
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.join("gcompose"))
}

/// Spawn a fresh `gcompose --serve` with piped stdin/stdout, plus a background reader thread that
/// feeds stdout lines over a channel so the request path can read replies WITH a timeout (finding
/// #4: a wedged worker must not block the held mutex / UI forever).
fn spawn_worker() -> Option<WorkerProc> {
    let w = worker_path()?;
    let mut child = Command::new(&w)
        .arg("--serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        // stderr stays inherited so the worker's [gcompose]/[gpu] logs reach the terminal.
        .spawn()
        .ok()?;
    // Piped stdin/stdout are always present after a successful piped spawn, but if `take()` ever
    // returns None we must REAP the child before bailing: `std::process::Child::Drop` does NOT kill
    // the process, so a bare `?` here would orphan a live `gcompose --serve` with no WorkerProc to
    // reap it. Kill + wait, then return None so the caller retries a clean spawn.
    let (stdin, stdout) = match (child.stdin.take(), child.stdout.take()) {
        (Some(si), Some(so)) => (si, so),
        _ => {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
    };

    // Reader thread: owns the BufReader, pushes each line onto the channel, then an Eof marker when
    // the pipe closes (worker exited/crashed). It exits when the child's stdout closes — which Drop
    // forces by killing the child. The send may fail if the receiver was dropped (proc torn down);
    // in that case the thread just stops. Detached: never explicitly joined.
    let (tx, rx) = mpsc::channel::<WorkerLine>();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut buf = String::new();
        loop {
            buf.clear();
            match reader.read_line(&mut buf) {
                Ok(0) => {
                    let _ = tx.send(WorkerLine::Eof);
                    break; // worker closed stdout (crashed/exited).
                }
                Ok(_) => {
                    if tx.send(WorkerLine::Line(buf.trim().to_string())).is_err() {
                        break; // receiver gone (proc dropped): stop reading.
                    }
                }
                Err(_) => {
                    let _ = tx.send(WorkerLine::Eof);
                    break; // read error: treat as EOF.
                }
            }
        }
    });

    Some(WorkerProc { child, stdin, rx })
}

/// Read one reply line from the worker with `REPLY_TIMEOUT` (finding #4). Returns the line, or None
/// if the worker hit EOF (crash/exit), the reader channel disconnected, OR no line arrived in time
/// (a wedged worker — driver deadlock). In every None case the worker is considered Broken and the
/// caller tears it down + retries instead of blocking forever.
fn read_reply(proc: &mut WorkerProc) -> Option<String> {
    match proc.rx.recv_timeout(REPLY_TIMEOUT) {
        Ok(WorkerLine::Line(l)) => Some(l),
        Ok(WorkerLine::Eof) => None,             // worker closed stdout (crashed/exited).
        Err(RecvTimeoutError::Timeout) => {
            eprintln!("gcompose: worker reply timed out after {REPLY_TIMEOUT:?} (wedged worker?)");
            None
        }
        Err(RecvTimeoutError::Disconnected) => None, // reader thread gone.
    }
}

/// One request/response round-trip against an already-running worker. Returns the RGBA bytes.
/// Any failure (write error, EOF, "ERR", short read) returns None so the caller can restart.
fn try_once(proc: &mut WorkerProc, req: &str) -> Option<Vec<u8>> {
    // Send the request line.
    proc.stdin.write_all(req.as_bytes()).ok()?;
    proc.stdin.write_all(b"\n").ok()?;
    proc.stdin.flush().ok()?;

    // Read response lines until DONE/ERR (skip any stray worker chatter that reached stdout). Reads
    // are bounded by REPLY_TIMEOUT (finding #4): a wedged worker yields None instead of blocking.
    loop {
        let r = read_reply(proc)?; // None on EOF/timeout/disconnect -> caller restarts.
        if r.is_empty() {
            continue;
        }
        if let Some(out_path) = r.strip_prefix("DONE ") {
            let bytes = std::fs::read(out_path.trim()).ok()?;
            if bytes.len() == PVW * PVH * 4 {
                return Some(bytes);
            }
            return None;
        }
        if r == "ERR" {
            return None;
        }
        // Unknown line: ignore and keep reading for the real response.
    }
}

/// Outcome of a single command round-trip, distinguishing the worker's two failure shapes:
///   - `Done(payload)`   : worker replied "DONE [payload]" — success.
///   - `Err`             : worker replied "ERR" — the COMMAND failed, but the worker is alive and
///                          the encoder/session is intact (e.g. AUDIO range with no audio). The
///                          caller may continue the session without restarting.
///   - `Broken`          : write error / EOF / protocol break — the worker is in an unknown state
///                          (likely dead) and must be dropped + (maybe) respawned.
enum CmdStatus {
    Done(String),
    Err,
    Broken,
}

/// One request/response round-trip returning the full tri-state status (see `CmdStatus`). This is
/// the lowest-level transport; `try_command` wraps it for the common DONE/None callers, while
/// `render_program` drives it directly on the held proc for the whole OPEN→ENC→AUDIO→CLOSE
/// sequence (finding #1) so it can (a) keep one lock across the render and (b) treat a plain `ERR`
/// on AUDIO as skip-this-clip without tearing the live worker/encoder down mid-render.
fn try_command_status(proc: &mut WorkerProc, req: &str) -> CmdStatus {
    if proc.stdin.write_all(req.as_bytes()).is_err()
        || proc.stdin.write_all(b"\n").is_err()
        || proc.stdin.flush().is_err()
    {
        return CmdStatus::Broken;
    }

    // Reads are bounded by REPLY_TIMEOUT (finding #4): EOF (crash), a read error, OR a reply that
    // never arrives in time (a wedged/deadlocked worker) all map to `Broken` so the render-retry
    // machinery tears the worker down + respawns rather than blocking the held mutex forever.
    loop {
        let r = match read_reply(proc) {
            Some(l) => l,
            None => return CmdStatus::Broken,
        };
        if r.is_empty() {
            continue;
        }
        if r == "DONE" {
            return CmdStatus::Done(String::new());
        }
        if let Some(payload) = r.strip_prefix("DONE ") {
            return CmdStatus::Done(payload.trim().to_string());
        }
        if r == "ERR" {
            return CmdStatus::Err;
        }
        // Unknown chatter on stdout: ignore and keep reading for the real response.
    }
}

/// One request/response round-trip that resolves to the DONE-echoed payload (the text after
/// "DONE ", trimmed) — or an empty String when DONE carries no payload. Returns None on any
/// failure (write error, EOF, "ERR", protocol break) so the caller can restart the worker.
///
/// Unlike `try_once`, this does NOT read or length-check any RGBA file — it is the generic
/// command transport for OPEN/ENC/CLOSE/THUMB/ENV, whose outputs vary in size (or are absent).
fn try_command(proc: &mut WorkerProc, req: &str) -> Option<String> {
    match try_command_status(proc, req) {
        CmdStatus::Done(p) => Some(p),
        CmdStatus::Err | CmdStatus::Broken => None,
    }
}

/// Run one stateless command on the persistent worker WITH auto-restart (up to MAX_ATTEMPTS).
/// Suitable for THUMB / ENV / OPEN (idempotent, no in-flight encoder to lose).
///
/// Respects the post-failure spawn cooldown (finding #6): if a recent spawn failed and there is
/// no live worker to reuse, this fails fast (None) rather than paying another OpenCL-init spawn —
/// so one repaint that misses many thumbnails against a dead worker can't trigger a spawn storm.
fn command_with_restart(req: &str) -> Option<String> {
    let slot = worker_slot();
    let mut guard = slot.lock().ok()?;

    // Entry gate (finding #6): if a previous call already exhausted its retries against a
    // known-dead worker AND there is no live worker to reuse, fail fast for the cooldown window
    // instead of paying another full retry/OpenCL-init storm. A live worker (guard.is_some()) is
    // always tried regardless of cooldown — the cooldown only suppresses fresh respawn attempts.
    if guard.is_none() && in_spawn_cooldown() {
        return None;
    }

    // In-call retry loop: still does the full MAX_ATTEMPTS respawn-and-retry to absorb the
    // worker's known one-off OpenCL-init flake (the whole reason this machinery exists).
    for attempt in 0..MAX_ATTEMPTS {
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            match try_command_status(proc, req) {
                CmdStatus::Done(payload) => {
                    clear_spawn_cooldown(); // healthy round-trip: allow normal respawns again.
                    return Some(payload);
                }
                CmdStatus::Err => {
                    // The worker is ALIVE and replied "ERR": the COMMAND legitimately failed (e.g. a
                    // THUMB for a source frame PAST the media's end — a clip whose len exceeds its
                    // source frame count, or any undecodable frame). This is a SOFT MISS, not a dead
                    // worker: return None WITHOUT tearing the worker down or arming the spawn cooldown.
                    // (Previously ERR and Broken were indistinguishable, so every out-of-range
                    // thumbnail triggered a 3x gcompose+OpenCL reinit storm on each launch/render.)
                    clear_spawn_cooldown(); // a live worker that answers ERR is healthy
                    return None;
                }
                // Broken (write error / EOF / wedged): fall through to drop + restart below.
                CmdStatus::Broken => {}
            }
        }
        // The worker is genuinely BROKEN (dead/unresponsive): drop it so the next attempt spawns
        // clean. Only a real worker death reaches here now — an ERR no longer restarts.
        *guard = None;
        eprintln!("gcompose command attempt {} failed; restarting worker", attempt + 1);
    }

    // All in-call attempts exhausted: arm the cooldown so subsequent misses in this same repaint
    // fail fast (no per-thumbnail respawn storm) until the window elapses and we retry once more.
    mark_spawn_fail();
    None
}

/// Compose the program at timeline frame `t` -> RGBA8 PVW*PVH, via the persistent worker.
/// Restarts the worker (up to MAX_ATTEMPTS) on any failure to absorb its OpenCL-init flake.
pub fn request_frame(project: &Project, t: i64) -> Option<Vec<u8>> {
    // P49 NESTED SEQUENCES: pre-render every visible COMPOUND clip's sub-sequence frame at `t` into
    // its deterministic `subseq_temp_path` BEFORE any frame line is built, so the `RAW:` base/over
    // swaps in `resolve_frame` (and the layer fold below, and the subtitle fold) find the temps on
    // disk. A project with no compound clips returns immediately (no extra work) → the rest of this
    // preview path is BYTE-IDENTICAL to pre-P49.
    prerender_compound_clips(project, t);

    let layers = visible_video_clips(project, t);

    // P48 SUBTITLES (PREVIEW OVERLAY): when a subtitle is active at `t` AND its text rasterizes, the
    // editor preview must show the caption while scrubbing — the SAME overlay the render bakes. We
    // reuse the EXISTING RAW: fold + `run_pipeline` the preview already uses for >2-layer frames:
    // compose the PROGRAM into `SUB_BASE` (the layer fold for >2 layers, else a single PREVIEW), then
    // composite the caption RAW over `SUB_BASE` into `PREVIEW_RGBA` (the bytes the caller reads back),
    // and run the whole list. When NO subtitle is active (or the cue does not rasterize), this branch
    // is skipped entirely and the preview takes the EXISTING byte-identical paths below.
    if let Some(sub) = project.active_subtitle_at(t) {
        if let Some(sub_raster) = rasterize_subtitle(&sub.text) {
            // Step 1: compose the program into SUB_BASE (reusing the layer fold or a single PREVIEW).
            let mut lines: Option<Vec<String>> = if layers.len() > 2 {
                build_layer_pipeline(project, t, &layers, SUB_BASE)
            } else {
                resolve_frame(project, t).map(|r| vec![format_preview(&r, SUB_BASE)])
            };
            if let Some(lines) = lines.as_mut() {
                // Step 2: fold the caption over SUB_BASE into PREVIEW_RGBA (the read-back buffer).
                let cap = build_subtitle_resolved(SUB_BASE, &sub_raster);
                lines.push(format_preview(&cap, PREVIEW_RGBA));
                if let Some(bytes) = run_pipeline(lines, PREVIEW_RGBA) {
                    return Some(bytes);
                }
            }
            // If the overlay pipeline could not be built/run, fall through to the normal preview
            // (no caption) rather than failing the frame.
        }
    }

    // P5 STAGE 2: when MORE than two video layers cover this frame, fold the extras over the
    // base+over composite (each pip'd via the RAW: layer path) and return the final RGBA. The
    // <=2-layer path below is untouched (byte-identical).
    if layers.len() > 2 {
        if let Some(lines) = build_layer_pipeline(project, t, &layers, PREVIEW_RGBA) {
            return run_pipeline(&lines, PREVIEW_RGBA);
        }
        // fall through to the single composite if the pipeline could not be built.
    }

    let req = build_request(project, t)?;

    let slot = worker_slot();
    let mut guard = slot.lock().ok()?;

    for attempt in 0..MAX_ATTEMPTS {
        // Ensure a worker exists.
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            if let Some(bytes) = try_once(proc, &req) {
                clear_spawn_cooldown(); // healthy round-trip: allow normal respawns again.
                return Some(bytes);
            }
        }
        // Failed: drop (kills) the current worker so the next loop spawns a clean one.
        // NOTE (finding #3): unlike `command_with_restart` (which arms the cooldown ONCE, only after
        // exhausting all attempts, so its known one-off OpenCL-init flake can retry cleanly within
        // the same call), the preview path arms it on EVERY failed attempt. That is intentional and
        // harmless: a success returns at the `return Some(bytes)` above before reaching here, so the
        // cooldown is only ever stamped on a genuine per-attempt failure — and stamping it eagerly
        // means a dead-worker preview burst (many cache-miss frames in one repaint) starts failing
        // fast a little sooner.
        *guard = None;
        mark_spawn_fail();
        eprintln!("gcompose serve attempt {} failed; restarting worker", attempt + 1);
    }
    None
}

/// PINNED (Slice A): render a scope of the program frame at timeline frame `t`.
///   kind 0 = histogram, 1 = luma waveform, 2 = vectorscope, 3 = RGB parade (Triad-B P1).
/// Returns the rendered scope as `SW*SH*4` (256×256×4) RGBA8 bytes, or None on any failure.
///
/// Mechanism: send a `PREVIEW` line for frame `t` (composites the frame on the GPU — identical to
/// `request_frame`, leaving the result in the worker's persistent g_buf[OUTB]), then a
/// `SCOPE <kind> <out>` line (runs the scope kernel on that buffer and writes a 256×256 RGBA image),
/// then read the image back. BOTH lines run under ONE hold of the worker mutex (finding #1 style):
/// the lock guarantees no concurrent `request_frame`/`thumbnail`/render can interleave another
/// compose between our PREVIEW and our SCOPE — which would otherwise make the scope read a different
/// frame than the one we composed. On a worker failure the whole PREVIEW+SCOPE pair is retried on a
/// fresh worker (up to MAX_ATTEMPTS), matching the preview path's OpenCL-init-flake absorption.
///
/// The PREVIEW's own RGBA output (PREVIEW_RGBA) is composed-and-discarded here — we only need its
/// side effect (the composed GPU buffer); the bytes we return are the SCOPE image, not the frame.
///
/// NON-BLOCKING LOCK (finding #4): this is called from the egui UI thread (panels::scopes_ui) on
/// every repaint. It takes the worker lock with `try_lock`, NOT a blocking `lock`: while a
/// background audio assembly (`assemble_and_play`) or a render holds the worker for seconds, a
/// blocking acquire here would FREEZE the UI for that whole duration on every repaint. With
/// `try_lock`, a contended scope simply returns None this frame and the caller keeps showing its
/// last scope image — the panel goes momentarily stale instead of the whole UI hanging. The 30 s
/// `REPLY_TIMEOUT` only bounds a wedged WORKER; it does nothing for lock contention, so this is the
/// piece that actually protects the UI thread during a long-running background worker operation.
pub fn scope(project: &Project, t: i64, kind: u8) -> Option<Vec<u8>> {
    if kind > 3 {
        return None; // unknown scope kind: nothing to render (0=hist,1=wave,2=vec,3=parade).
    }
    // P49 NESTED SEQUENCES: the scope reads back a kernel over the SAME composed program frame the
    // preview composes (`build_request` → `resolve_frame`), so a compound clip's `RAW:` temp must
    // exist before that compose — otherwise the engine reads the absent RAW as black and the scope
    // reflects a black frame instead of the sub-sequence's content. Pre-render the visible compound
    // clips at `t` (NO worker lock held — this runs BEFORE the try_lock below; `run_pipeline` takes
    // and releases the lock itself). A project with no compound clips returns immediately, so the
    // scope path is BYTE-IDENTICAL to pre-P49.
    prerender_compound_clips(project, t);
    let preview_req = build_request(project, t)?;
    let scope_req = format!("SCOPE {kind} {SCOPE_RGBA}");

    let slot = worker_slot();
    // try_lock (finding #4): do NOT block the UI thread behind a background audio assembly / render.
    let mut guard = match slot.try_lock() {
        Ok(g) => g,
        Err(_) => return None, // worker busy: skip this frame's scope (caller keeps last image).
    };

    for attempt in 0..MAX_ATTEMPTS {
        // Ensure a worker exists.
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            // 1) PREVIEW: compose frame t (discard its RGBA file — we only want the GPU side effect).
            //    try_once reads PREVIEW's DONE and length-checks PVW*PVH*4; a None means the worker
            //    is broken, so we fall through to the restart below.
            if try_once(proc, &preview_req).is_some() {
                // 2) SCOPE: run the kernel on the just-composed buffer and read back the image.
                if let Some(img) = read_scope(proc, &scope_req) {
                    clear_spawn_cooldown(); // healthy round-trip.
                    return Some(img);
                }
            }
        }
        // Either PREVIEW or SCOPE failed: drop (kills) the worker so the next loop spawns clean.
        *guard = None;
        mark_spawn_fail();
        eprintln!("gcompose scope attempt {} failed; restarting worker", attempt + 1);
    }
    None
}

/// PINNED-adjacent (Slice A, finding #5): render ALL THREE scopes for the program frame at timeline
/// frame `t` in ONE round-trip — one PREVIEW compose followed by three SCOPE reads — returning
/// `[histogram, luma_waveform, vectorscope]`, each `SW*SH*4` RGBA8 bytes, or None on any failure.
///
/// WHY: a scopes panel showing all three scopes that called `scope()` three times per repaint would
/// trigger THREE identical PREVIEW composes of the same frame (plus three PREVIEW_RGBA file
/// write+read round-trips of ~4.4 MB each), since each `scope()` re-composes before its SCOPE. The
/// composed GPU buffer (g_buf[OUTB]) is stable between requests, so the frame only needs composing
/// ONCE; the three scope kernels all read that same buffer. This composes once then issues SCOPE 0,
/// 1, 2 back-to-back under a single lock hold, cutting the per-frame work from 3 composes to 1.
///
/// Uses `try_lock` for the same UI-freeze reason as `scope()` (finding #4): a contended call returns
/// None and the caller keeps its last images. On any worker failure mid-sequence the WHOLE
/// PREVIEW+3×SCOPE is retried on a fresh worker (a respawn re-composes from scratch, so partial
/// progress is never reused). Team C should prefer this over three separate `scope()` calls.
// Retained batched-scope helper: the scopes panel currently draws via per-scope `scope()` calls, so
// this single-lock 3-in-1 fast path is not yet wired (kept as the intended optimization path).
#[allow(dead_code)]
pub fn scope_all(project: &Project, t: i64) -> Option<[Vec<u8>; 3]> {
    // P49 NESTED SEQUENCES: same reason as `scope()` — the three scopes read kernels over the one
    // composed program frame (`build_request` → `resolve_frame`), so a compound clip's `RAW:` temp
    // must exist before that compose, else it composites as black. Pre-render the visible compound
    // clips at `t` BEFORE the try_lock (no lock held; `run_pipeline` manages the lock itself). A
    // project with no compound clips returns immediately → BYTE-IDENTICAL to pre-P49.
    prerender_compound_clips(project, t);
    let preview_req = build_request(project, t)?;
    let scope_reqs = [
        format!("SCOPE 0 {SCOPE_RGBA}"),
        format!("SCOPE 1 {SCOPE_RGBA}"),
        format!("SCOPE 2 {SCOPE_RGBA}"),
    ];

    let slot = worker_slot();
    // try_lock (finding #4): never block the UI thread behind a background assembly / render.
    let mut guard = match slot.try_lock() {
        Ok(g) => g,
        Err(_) => return None, // worker busy: skip this frame's scopes (caller keeps last images).
    };

    for attempt in 0..MAX_ATTEMPTS {
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            // 1) ONE PREVIEW compose of frame t (finding #5): all three scopes read this buffer.
            if try_once(proc, &preview_req).is_some() {
                // 2) Three SCOPE reads on the same composed buffer. Read them into a fixed-size
                //    array; any short/failed read aborts the whole attempt (restart on a fresh
                //    worker rather than returning a torn mix of old/new scope images).
                let mut imgs: [Option<Vec<u8>>; 3] = [None, None, None];
                let mut ok = true;
                for (i, req) in scope_reqs.iter().enumerate() {
                    match read_scope(proc, req) {
                        Some(img) => imgs[i] = Some(img),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    // SAFETY: every slot is Some when ok (the loop only completes without break in
                    // that case). Unwraps are infallible here.
                    let [h, w, v] = imgs;
                    clear_spawn_cooldown(); // healthy round-trip.
                    return Some([h.unwrap(), w.unwrap(), v.unwrap()]);
                }
            }
        }
        // PREVIEW or one of the SCOPE reads failed: drop (kills) the worker so the next loop spawns
        // clean and re-composes from scratch.
        *guard = None;
        mark_spawn_fail();
        eprintln!("gcompose scope_all attempt {} failed; restarting worker", attempt + 1);
    }
    None
}

/// Run one `SCOPE` round-trip on an already-running worker and read back the 256×256 RGBA image.
/// Mirrors `try_once` but length-checks against the SCOPE image size (SW*SH*4) instead of the
/// preview frame size. Returns None on any failure (write error, EOF/timeout, "ERR", short read).
fn read_scope(proc: &mut WorkerProc, req: &str) -> Option<Vec<u8>> {
    proc.stdin.write_all(req.as_bytes()).ok()?;
    proc.stdin.write_all(b"\n").ok()?;
    proc.stdin.flush().ok()?;

    loop {
        let r = read_reply(proc)?; // None on EOF/timeout/disconnect -> caller restarts.
        if r.is_empty() {
            continue;
        }
        if let Some(out_path) = r.strip_prefix("DONE ") {
            let bytes = std::fs::read(out_path.trim()).ok()?;
            if bytes.len() == SW * SH * 4 {
                return Some(bytes);
            }
            return None; // wrong size: treat as a failed render.
        }
        if r == "ERR" {
            return None;
        }
        // Unknown chatter on stdout: ignore and keep reading for the real response.
    }
}

/// The resolved program at timeline frame `t`: base + optional overlay + composite params.
/// Shared by the preview request and the render ENC line so both bake the identical composite.
///
/// `px/py/pw/ph` are the KEYFRAMED over-clip PiP rect (from `project.pip_at`) and `bright/contrast/
/// sat` are the KEYFRAMED grade (from `project.grade_at`) at this timeline frame — both evaluated
/// once here so the preview and the render emit byte-identical composite params (Slice A).
struct Resolved {
    base_path: String,
    base_frame: i32,
    over_path: String, // "-" when no overlay
    over_frame: i32,
    op: f32,
    // P31 BLEND MODE of the V2 OVERLAY clip (0=Normal 1=Multiply 2=Screen 3=Overlay 4=Add 5=Darken
    // 6=Lighten 7=Difference). Rides the wire as ONE integer token IMMEDIATELY AFTER `op` on BOTH the
    // PREVIEW and ENC lines (format_preview / format_enc), matching the engine's k_pip parser position.
    // The engine combines the overlay RGB with the V1 base via this mode BEFORE the alpha-over composite.
    // Only the OVERLAY clip's blend matters; a base/single clip (no overlay) sends 0 (Normal) → plain
    // alpha-over → BYTE-IDENTICAL to pre-P31. Every NEUTRAL/identity Resolved literal sets `blend: 0`.
    blend: i32,
    px: f32,
    py: f32,
    pw: f32,
    ph: f32,
    bright: f32,
    contrast: f32,
    sat: f32,
    // Per-clip COLOR grade from the BASE (visible) clip (Triad-B P1), ADDITIVE on top of the program
    // grade above. Same kernel semantics: `cbright` added (−1..1), `ccontrast`/`csat` multipliers
    // (0..2, 1.0 = identity). DOCUMENTED COMBINE ORDER: gcompose runs the PER-CLIP grade FIRST (right
    // after the PiP composite), then the PROGRAM grade — so a clip with bright +0.3 lifts the picture
    // before the program grade is applied. A timeline gap (no base clip) is neutral (0/1/1). When a
    // transition is active the per-clip grade travels with the OUTGOING clip (matching the look).
    cbright: f32,
    ccontrast: f32,
    csat: f32,
    // Per-clip LOOK from the BASE (track-0/visible) clip (Slice A). `look_kind`: 0=None, 1=VHS,
    // 2=LUT3D. `look_amt` is the look mix (0..1). `lut_path` is the clip's `.cube` path for LUT3D,
    // or "-" when there is no LUT (look != 2 or an empty path). Both the preview (`build_request`)
    // and the render (`build_enc_line`) emit these, so a clip's look animates identically in the
    // preview and the export. Mirrors MojoMedia's playhead-segment look driving the grade pipeline.
    look_kind: i32,
    look_amt: f32,
    lut_path: String,
    // Per-boundary TRANSITION fields (Wave 8). Set by `resolve_frame` from `project.transition_at`
    // on the base track and the incoming clip the model's `boundaries()` pairs with the outgoing
    // (base) clip at this boundary. Forwarded to the engine (via `build_request`/`build_enc_line`)
    // as the 5 trailing wire fields BEFORE the out:
    //   trans_kind: -1 = no transition (engine skips slot 2, track1(-1,..) copies the base);
    //               0..7 = fpx_gpu transition id (0=crossfade..7=dissolve).
    //   trans_prog: progress in [0,1] across the transition window [center-dur/2, center+dur/2).
    //   trans_param: per-transition parameter (default 4.0, mirrors MojoMedia's tt_p default).
    //   trans_path: the INCOMING clip's media path, or "-" when there is no transition/incoming clip.
    //   trans_frame: the incoming clip's source frame at `t`, clamped to the incoming clip's valid
    //                source range; 0 when there is no transition.
    // So preview AND render bake the same animated transition.
    trans_kind: i32,
    trans_prog: f32,
    trans_param: f32,
    trans_path: String,
    trans_frame: i32,
    // ----- Triad-B P2 per-clip COLOR-WHEELS (LIFT/GAMMA/GAIN) + TRANSFORM + BLUR -----
    // Read from the BASE (visible) clip (or, during a transition, the OUTGOING clip — they travel
    // with the look/grade like the per-clip P1 grade). Forwarded to the engine as the 12 TRAILING
    // wire fields appended AFTER csat (PREVIEW: before `out`) in the PINNED order
    //   lift_r lift_g lift_b  gamma_r gamma_g gamma_b  gain_r gain_g gain_b  rot scale blur
    // so the engine's fpx_gpu_lgg / fpx_gpu_transform / fpx_gpu_blur kernels apply them identically
    // in preview and render. The 9 lift/gamma/gain values ALREADY HAVE white balance folded in (see
    // resolve_frame's wb fold) — the engine only ever sees lift/gamma/gain, never wb_temp/wb_tint.
    // A timeline gap (no base clip) sends IDENTITY (lift 0, gamma 1, gain 1, rot 0, scale 1, blur 0)
    // so the engine no-ops and reproduces the current output.
    lift: [f32; 3],
    gamma: [f32; 3],
    gain_rgb: [f32; 3],
    rot: f32,
    scale: f32,
    blur: f32,
    // ----- P4 per-clip CHROMA KEY (green-screen) on the OVER (V2) clip -----
    // Read from the OVER (V2/track-1) overlay clip's `Clip.chroma` (Team A reads it; never edits
    // model.rs). The engine keys the OVER buffer's alpha where the pixel matches the key colour, so
    // pip then shows V1 through the keyed pixels. Forwarded to the engine as the 6 P4 CHROMA-KEY wire
    // fields appended AFTER blur (PREVIEW: before `out`) in the PINNED order
    //   ck_on ck_r ck_g ck_b ck_sim ck_smooth
    // so preview + render key identically. `ck_on` is 1 only when an over clip exists AND its
    // chroma.enabled is true; otherwise IDENTITY (ck_on=0, key green, sim 0.4, smooth 0.1) so the
    // engine no-ops and reproduces the P3 composite byte-for-byte. NB the chroma describes the OVER
    // clip, NOT the base/outgoing clip (it is the green-screen layer being keyed over V1).
    // P37 SPILL SUPPRESSION rides as ONE EXTRA f32 (`ck_spill`) but it is NOT one of the 6 contiguous
    // P4 chroma tokens — to avoid shifting every later wire index it is APPENDED AS THE LAST wire field
    // (ENC: last; PREVIEW: after mask_invert, before the out path). The struct field lives next to the
    // other ck_* fields for readability; only the FORMATTERS (format_preview/format_enc) place it last
    // on the wire. ck_spill 0 (or chroma disabled) → engine no-ops the spill pass → byte-identical to
    // pre-P37. It runs inside the SAME k_chroma kernel on the OVER (V2) buffer, AFTER the alpha key.
    ck_on: i32,
    ck_key: [f32; 3],
    ck_sim: f32,
    ck_smooth: f32,
    // P37 spill-suppression amount (0..1, 0 = off/identity). Sourced from the same OVER (V2) clip's
    // `Clip.chroma.spill` as ck_sim/ck_smooth. Rides the wire LAST (appended after mask_invert).
    ck_spill: f32,
    // P5 master tone curve: 5 outputs at fixed inputs 0/.25/.5/.75/1 (identity = [0,.25,.5,.75,1]).
    curve: [f32; 5],
    // ----- P6 per-clip STYLIZE / UTILITY filters from the BASE (visible) clip -----
    // Read from the BASE clip (the OUTGOING clip during a transition — they travel with the
    // look/grade/curve like every other per-clip effect). Forwarded to the engine as the 4 TRAILING
    // wire fields appended AFTER the 5 curve fields (cv0..cv4) — and, on the PREVIEW line, BEFORE the
    // out path — in the PINNED order `vig sharp flip fx`. The engine applies them on the composited
    // OUTB AFTER the curve and BEFORE the look (simple-fx -> vignette -> sharpen -> flip), each gated
    // off at its no-op default. A timeline gap (no base clip) sends the IDENTITY tuple
    // (vignette 0, sharpen 0, flip 0, fx 0), so the engine no-ops all four and reproduces the P5
    // output byte-for-byte.
    //   vignette: radial edge-darken amount (0 = none .. 1 = full). 0 = skip.
    //   sharpen : unsharp amount (0 = none .. 2). 0 = skip.
    //   flip    : 0 none / 1 horizontal / 2 vertical / 3 both. 0 = skip.
    //   fx      : 0 none / 1 invert / 2 sepia / 3 grayscale / 4 posterize. 0 = skip.
    vignette: f32,
    sharpen: f32,
    flip: u8,
    fx: i32,
    // ----- P7 per-clip COLOR filters from the BASE (visible) clip -----
    // Read from the BASE clip (the OUTGOING clip during a transition — they travel with the
    // look/grade/curve/stylize like every other per-clip effect). Forwarded to the engine as the 6
    // TRAILING wire fields appended AFTER the 4 P6 stylize fields (vig sharp flip fx) — and, on the
    // PREVIEW line, BEFORE the out path — in the PINNED order `hue sat light inb inw gam`. The engine
    // applies them on the composited OUTB AFTER the P6 filters (flip) and BEFORE the look, in the
    // order HSL -> LEVELS, each gated off at its no-op default. A timeline gap (no base clip) sends
    // the IDENTITY tuples (hsl [0,1,0] = hue 0, sat 1, light 0; levels [0,1,1] = in_black 0,
    // in_white 1, gamma 1), so the engine no-ops both and reproduces the P6 output byte-for-byte.
    //   hsl[0]    : hue shift in degrees (0 = none; wraps 360). 0/1/0 = skip.
    //   hsl[1]    : saturation multiplier (1 = none).
    //   hsl[2]    : lightness add (0 = none).
    //   levels[0] : input black point (0 = none).
    //   levels[1] : input white point (1 = none).
    //   levels[2] : gamma (1 = none).
    hsl: [f32; 3],
    levels: [f32; 3],
    // ----- P8 per-clip STYLIZE-2 filters from the BASE (visible) clip -----
    // Read from the BASE clip (the OUTGOING clip during a transition — they travel with the
    // look/grade/curve/stylize/color like every other per-clip effect). Forwarded to the engine as
    // the 8 TRAILING wire fields appended AFTER the 6 P7 color fields (hue sat light inb inw gam) —
    // and, on the PREVIEW line, BEFORE the out path — in the PINNED order
    //   mosaic gmap_amt glo_r glo_g glo_b ghi_r ghi_g ghi_b
    // The engine applies them on the composited OUTB AFTER the P7 color filters (levels) and BEFORE
    // the look, in the order MOSAIC -> GRADIENT-MAP, each gated off at its no-op default. A timeline
    // gap (no base clip) sends the IDENTITY tuple (mosaic 0, gmap_amt 0, lo [0,0,0], hi [1,1,1]),
    // so the engine no-ops both and reproduces the P7 output byte-for-byte.
    //   mosaic   : pixelate block size in px (0/1 = off, no pixelation).
    //   gmap_amt : gradient-map mix (0 = off .. 1 = full luma->ramp replace).
    //   gmap_lo  : shadow ramp colour (luma 0). Identity [0,0,0].
    //   gmap_hi  : highlight ramp colour (luma 1). Identity [1,1,1].
    mosaic: u32,
    gmap_amt: f32,
    gmap_lo: [f32; 3],
    gmap_hi: [f32; 3],
    // ----- P9 per-clip FX FILTERS from the BASE (visible) clip -----
    // Read from the BASE clip (the OUTGOING clip during a transition — they travel with the
    // look/grade/curve/stylize/color like every other per-clip effect). Forwarded to the engine as
    // the 4 TRAILING wire fields appended AFTER the 8 P8 stylize-2 fields (mosaic gmap_amt glo_r
    // glo_g glo_b ghi_r ghi_g ghi_b) — and, on the PREVIEW line, BEFORE the out path — in the PINNED
    // order `denoise glow_amt glow_thr rgbshift`. The engine applies them on the composited OUTB
    // AFTER the P8 gradient-map and BEFORE the look, in the order DENOISE -> GLOW -> RGB-SHIFT, each
    // gated off at its no-op default. A timeline gap (no base clip) sends the IDENTITY tuple
    // (denoise 0, glow_amt 0, glow_thr 0.7, rgbshift 0), so the engine no-ops all three and
    // reproduces the P8 output byte-for-byte.
    //   denoise  : edge-preserving (bilateral) smooth strength (0 = off .. 1 = full). 0 = skip.
    //   glow_amt : bloom mix (0 = off .. 1 = full bright-pass blur added back). 0 = skip.
    //   glow_thr : glow luma threshold (only pixels brighter bloom). Identity 0.7 (only matters
    //              when glow_amt > 0).
    //   rgbshift : chromatic-aberration channel offset in px (R +shift, B −shift). 0 = skip.
    denoise: f32,
    glow_amt: f32,
    glow_thr: f32,
    rgbshift: f32,
    // ----- P10 per-clip STYLIZE-4 filters from the BASE (visible) clip -----
    // Read from the BASE clip (the OUTGOING clip during a transition — they travel with the
    // look/grade/curve/stylize/color like every other per-clip effect). Forwarded to the engine as
    // the 3 TRAILING wire fields appended AFTER the 4 P9 FX fields (denoise glow_amt glow_thr
    // rgbshift) — and, on the PREVIEW line, BEFORE the out path — in the PINNED order
    // `halftone emboss edge`. The engine applies them on the composited OUTB AFTER the P9 RGB-shift
    // and BEFORE the look, in the order HALFTONE -> EMBOSS -> EDGE, each gated off at its no-op
    // default. A timeline gap (no base clip) sends the IDENTITY tuple (halftone 0, emboss 0, edge 0),
    // so the engine no-ops all three and reproduces the P9 output byte-for-byte.
    //   halftone : luma-driven dot-screen cell size in px (0/1 = off, no dots).
    //   emboss   : directional (NW) relief strength (0 = off .. 1 = full). 0 = skip.
    //   edge     : Sobel edge-detect (sketch) mix (0 = off .. 1 = full edge-on-dark). 0 = skip.
    halftone: u32,
    emboss: f32,
    edge: f32,
    // ----- P13 per-clip OLD-FILM / DISTORT filters from the BASE (visible) clip -----
    // Read from the BASE clip (the OUTGOING clip during a transition — they travel with the
    // look/grade/curve/stylize like every other per-clip effect). Forwarded to the engine as the 3
    // TRAILING wire fields appended AFTER the 3 P10 stylize-4 fields (halftone emboss edge) — and, on
    // the PREVIEW line, BEFORE the out path — in the PINNED order `grain scratches diffusion`. The
    // engine applies them on the composited OUTB AFTER the P10 EDGE and BEFORE the look, in the order
    // GRAIN -> SCRATCHES -> DIFFUSION, each gated off at its no-op default. A timeline gap (no base
    // clip) sends the IDENTITY tuple (grain 0, scratches 0, diffusion 0), so the engine no-ops all
    // three and reproduces the P10 output byte-for-byte. The effects are DETERMINISTIC (a coordinate
    // integer hash, not time/RNG) so the same input frame always yields the same output.
    //   grain     : film-noise strength (0 = off .. 1). 0 = skip.
    //   scratches : old-film vertical-scratch density/amount (0 = off .. 1). 0 = skip.
    //   diffusion : frosted-glass jitter radius in px (0 = off .. 16). 0 = skip.
    grain: f32,
    scratches: f32,
    diffusion: f32,
    // ----- P16 per-clip DISTORT filters from the BASE (visible) clip -----
    // Read from the BASE clip (the OUTGOING clip during a transition — they travel with the
    // look/grade/curve/stylize/old-film like every other per-clip effect). Forwarded to the engine
    // as the 3 TRAILING wire fields appended AFTER the 3 P13 old-film fields (grain scratches
    // diffusion) — and, on the PREVIEW line, BEFORE the out path — in the PINNED order
    // `wave swirl threshold`. The engine applies them on the composited OUTB AFTER the P13 DIFFUSION
    // and BEFORE the look, in the order WAVE -> SWIRL -> THRESHOLD, each gated off at its no-op
    // default. A timeline gap (no base clip) sends the IDENTITY tuple (wave 0, swirl 0, threshold 0),
    // so the engine no-ops all three and reproduces the P13 output byte-for-byte.
    //   wave      : sinusoidal horizontal row-displacement amplitude in px (0 = off .. 40). 0 = skip.
    //   swirl     : rotational distortion strength in radians at the centre (0 = off .. ~3.14).
    //               0 = skip.
    //   threshold : luma binarize level (0 = off .. 1; pixels with luma >= level become white, else
    //               black). 0 = skip.
    wave: f32,
    swirl: f32,
    threshold: f32,
    // ----- P17 per-clip GEOMETRIC filters from the BASE (visible) clip -----
    // Read from the BASE clip (the OUTGOING clip during a transition — they travel with the
    // look/grade/curve/stylize/old-film/distort like every other per-clip effect). Forwarded to the
    // engine as the 3 TRAILING wire fields appended AFTER the 3 P16 distort fields (wave swirl
    // threshold) — and, on the PREVIEW line, BEFORE the out path — in the PINNED order
    // `lens crop glitch`. The engine applies them on the composited OUTB AFTER the P16 THRESHOLD
    // and BEFORE the look, in the order LENS -> CROP -> GLITCH, each gated off at its no-op default.
    // A timeline gap (no base clip) sends the IDENTITY tuple (lens 0, crop 0, glitch 0), so the
    // engine no-ops all three and reproduces the P16 output byte-for-byte. GLITCH is DETERMINISTIC
    // (an integer band hash, not time/RNG) so a held frame is stable.
    //   lens   : radial barrel/pincushion distortion (+ barrel, - pincushion; 0 = off). 0 = skip.
    //   crop   : margin fraction 0..0.49 (outside the inner rect -> black; 0 = off). 0 = skip.
    //   glitch : per-band horizontal channel-split shift, max px (0 = off). 0 = skip.
    lens: f32,
    crop: f32,
    glitch: f32,
    // ----- P23 per-clip 360 REFRAME from the BASE (visible) clip -----
    // Read from the BASE clip (the OUTGOING clip during a transition — they travel with the
    // look/grade/curve/stylize/old-film/distort/geometric like every other per-clip effect).
    // Forwarded to the engine as the 4 TRAILING wire fields appended AFTER the 3 P17 GEOMETRIC
    // fields (lens crop glitch) — and, on the PREVIEW line, BEFORE the out path — in the PINNED
    // order `eq360 eq_yaw eq_pitch eq_fov`. When `eq360` is TRUE the engine treats the composited
    // OUTB as a full 360x180 equirectangular panorama and reprojects it to a flat rectilinear view
    // at (eq_yaw, eq_pitch) degrees with horizontal field-of-view eq_fov degrees (the standard
    // pinhole "360 viewer" model). When `eq360` is FALSE the engine returns immediately (no kernel
    // run) so the composited frame is byte-identical to pre-P23 — a timeline gap (no base clip)
    // sends the IDENTITY tuple (eq360 false, eq_yaw 0, eq_pitch 0, eq_fov 90).
    //   eq360    : enable 360 equirectangular -> rectilinear reprojection (false = off). false = skip.
    //   eq_yaw   : view yaw in degrees (0 = forward).
    //   eq_pitch : view pitch in degrees (0 = level).
    //   eq_fov   : view horizontal field of view in degrees (90 = default/identity).
    eq360: bool,
    eq_yaw: f32,
    eq_pitch: f32,
    eq_fov: f32,
    // ----- P34 per-clip SHAPE MASK from the BASE (visible) clip -----
    // Read from the BASE clip (the OUTGOING clip during a transition — they travel with the
    // look/grade/curve/stylize/old-film/distort/geometric/360-reframe like every other per-clip
    // effect). Forwarded to the engine as the 7 TRAILING wire fields appended AFTER the 4 P23 360
    // REFRAME fields (eq360 eq_yaw eq_pitch eq_fov) — and, on the PREVIEW line, BEFORE the out path —
    // in the PINNED order `mask_shape mask_cx mask_cy mask_rw mask_rh mask_feather mask_invert`. The
    // engine zeroes (to black) every pixel OUTSIDE a centred rectangle (mask_shape 1) or ellipse
    // (mask_shape 2) at (mask_cx,mask_cy) with half-extents (mask_rw,mask_rh) in normalized [0,1]
    // frame coords, softening the edge over `mask_feather` and flipping inside/outside when
    // `mask_invert` is set. Applied on the composited OUTB AFTER the P17 geometry (lens/crop/glitch)
    // and the P23 360 reframe, BEFORE the look — the SAME slot the geometry filters use. When
    // mask_shape is 0 (none) the engine returns immediately (no kernel run) so the composited frame
    // is byte-identical to pre-P34 — a timeline gap (no base clip) sends the IDENTITY tuple
    // (mask_shape 0, cx 0.5, cy 0.5, rw 0.5, rh 0.5, feather 0, invert 0).
    //   mask_shape   : 0 = none (off, byte-identical no-op), 1 = rectangle, 2 = ellipse. 0 = skip.
    //   mask_cx      : mask centre X in normalized [0,1] frame coords (0.5 = centred).
    //   mask_cy      : mask centre Y in normalized [0,1] frame coords (0.5 = centred).
    //   mask_rw      : mask half-width  in normalized [0,1] (0.5 = full width).
    //   mask_rh      : mask half-height in normalized [0,1] (0.5 = full height).
    //   mask_feather : edge softness fraction 0..1 (0 = hard edge).
    //   mask_invert  : 1 = keep OUTSIDE / zero INSIDE (flip), 0 = keep inside. Integer flag.
    mask_shape: i32,
    mask_cx: f32,
    mask_cy: f32,
    mask_rw: f32,
    mask_rh: f32,
    mask_feather: f32,
    mask_invert: i32,
    // ----- P38 per-clip DISTORT (MIRROR + KALEIDOSCOPE + DITHER) from the BASE (visible) clip -----
    // Read from the BASE clip (the OUTGOING clip during a transition — they travel with the
    // look/grade/curve/stylize/old-film/distort/geometric/360-reframe/shape-mask like every other
    // per-clip effect). Forwarded to the engine as the 3 TRAILING wire fields appended AFTER the 1
    // P37 chroma-spill field (ck_spill — the current LAST wire field) — on the PREVIEW line BETWEEN
    // ck_spill and the out path; on the ENC line as the final 3 tokens — in the PINNED order
    // `mirror_x kaleido dither`. The engine applies them on the composited OUTB AFTER the P34 shape
    // mask and BEFORE the look — the SAME slot the P17/P23/P34 OUTB filters use, in the order
    // MIRROR -> KALEIDOSCOPE -> DITHER, each gated off at its no-op default. A timeline gap (no base
    // clip) sends the IDENTITY tuple (mirror_x 0, kaleido 0, dither 0), so the engine no-ops all
    // three and reproduces the P37 output byte-for-byte. Pre-P38 projects load these defaults via
    // serde, so they remain byte-identical.
    //   mirror_x : 0 = off (byte-identical no-op), 1 = mirror the LEFT half onto the right. Integer
    //              token (engine parses i32; 0/1).
    //   kaleido  : N-fold radial kaleidoscope segment count. 0 or 1 = off (no-op); >=2 = segments.
    //              Integer token (engine parses i32).
    //   dither   : ordered 4x4 Bayer dither strength (0 = off/identity .. 1 = full). f32 token.
    mirror_x: i32,
    kaleido: i32,
    dither: f32,
    // ----- P39 per-clip SELECTIVE COLOR (one HUE BAND) from the BASE (visible) clip -----
    // Read from the BASE clip (the OUTGOING clip during a transition — travels with the
    // look/grade/curve/stylize/old-film/distort/selective-color/geometric like every other per-clip
    // effect). Forwarded to the engine as the 3 TRAILING wire fields appended AFTER the P38 `dither`
    // field (the current LAST wire field) — on the PREVIEW line BETWEEN dither and the out path; on
    // the ENC line as the final 3 tokens — in the PINNED order `sel_band sel_hshift sel_sat`. The
    // engine applies it on the composited OUTB AFTER the P38 distort and BEFORE the look — the SAME
    // OUTB slot the P17/P23/P34/P38 filters use, gated OFF at its no-op default (sel_band 0). A
    // timeline gap (no base clip) sends the IDENTITY tuple (sel_band 0, sel_hshift 0, sel_sat 1.0), so
    // the engine no-ops the filter and reproduces the P38 output byte-for-byte. Pre-P39 projects load
    // these defaults via serde, so they remain byte-identical.
    //   sel_band  : selected hue band. 0 = off (byte-identical no-op) / 1=Reds 2=Yellows 3=Greens
    //               4=Cyans 5=Blues 6=Magentas. INTEGER token (engine parses i32; 0..6).
    //   sel_hshift: hue rotation applied to the selected band, -1..1 = -180..180 deg. f32 token.
    //   sel_sat   : saturation MULTIPLIER for the selected band, 1.0 = unchanged (the no-op default),
    //               0 = desaturate the band to grey, 2 = double. f32 token.
    sel_band: i32,
    sel_hshift: f32,
    sel_sat: f32,
    // P41 SOLARIZE + COLOUR TEMPERATURE. Two simple per-clip IN-PLACE pixel ops on the composited
    // OUTB (no neighbour sampling, exactly like P36 dither / P39 selective-color), applied AFTER the
    // P39 selective color and BEFORE the look. Forwarded as 2 NEW TRAILING wire fields in the PINNED
    // order `sol_thr temp` (sol_thr first, temp second): on the ENC line APPENDED AFTER sel_sat (the
    // current last token, NO out path); on the PREVIEW line INSERTED BETWEEN sel_sat and the out
    // path. Both are BYTE-IDENTICAL no-ops at their default, so every pre-P41 project (whose Clip
    // loads sol_thr 0.0 / temp 0.0 via serde) renders unchanged.
    //   sol_thr: solarize threshold. 0.0 = OFF (caller skips the kernel — byte-exact no-op); active
    //            range (0,1]. Per channel: if v > sol_thr then v = 1.0 - v, else v unchanged. f32 token.
    //   temp   : colour temperature. 0.0 = neutral/OFF (caller skips the kernel — byte-exact no-op);
    //            range -1..1. temp>0 warms (raises R, lowers B), temp<0 cools (reverse); GREEN
    //            unchanged. r = clamp(r + temp*0.30, 0,1); b = clamp(b - temp*0.30, 0,1). f32 token.
    sol_thr: f32,
    temp: f32,
    // P45 video fade-to-black factor in [0,1] (1.0 = no fade). Multiplies the composed frame LAST.
    fade: f32,
}

/// Fold a clip's white balance (`wb_temp`, `wb_tint`) INTO its 9 lift/gamma/gain values, returning
/// the white-balanced `(lift, gamma, gain_rgb)` triples. White balance is NOT a wire field (PINNED):
/// the engine only ever sees lift/gamma/gain, so the warm/cool + green/magenta bias is baked into the
/// GAIN channels here (multiplicative highlight gains are the natural place for a colour cast, matching
/// how Shotcut's white-balance maps a temperature onto the channel gains). `lift` and `gamma` pass
/// through unchanged — only `gain_rgb` is modulated.
///
/// FORMULA (both biases in [−1,1], 0 = neutral; `K_TEMP`/`K_TINT` keep a full-scale bias to a sane
/// ±gain so a maxed slider tints rather than blowing the channel out):
///   temp > 0 (WARMER): gain_r *= 1 + K_TEMP*temp ; gain_b *= 1 − K_TEMP*temp   (red up, blue down)
///   temp < 0 (COOLER): symmetric (red down, blue up — the same expression with temp negative)
///   tint > 0 (GREENER): gain_g *= 1 + K_TINT*tint ; gain_r *= 1 − 0.5*K_TINT*tint ;
///                       gain_b *= 1 − 0.5*K_TINT*tint   (green up, red+blue down → magenta drops)
///   tint < 0 (MAGENTA): symmetric.
/// Each resulting gain is clamped to [0, 4] (the engine/Shotcut gain range) so a combined
/// temp+tint push can never produce a negative or runaway multiplier.
fn fold_white_balance(
    lift: [f32; 3],
    gamma: [f32; 3],
    gain: [f32; 3],
    wb_temp: f32,
    wb_tint: f32,
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    const K_TEMP: f32 = 0.5; // full warm/cool bias scales a channel gain by up to ±50%.
    const K_TINT: f32 = 0.4; // full green/magenta bias scales the green gain by up to ±40%.
    let temp = wb_temp.clamp(-1.0, 1.0);
    let tint = wb_tint.clamp(-1.0, 1.0);

    let mut gr = gain[0] * (1.0 + K_TEMP * temp);
    let mut gg = gain[1];
    let mut gb = gain[2] * (1.0 - K_TEMP * temp);

    gg *= 1.0 + K_TINT * tint;
    gr *= 1.0 - 0.5 * K_TINT * tint;
    gb *= 1.0 - 0.5 * K_TINT * tint;

    let gain_wb = [gr.clamp(0.0, 4.0), gg.clamp(0.0, 4.0), gb.clamp(0.0, 4.0)];
    // lift + gamma pass through unchanged; only the highlight gains carry the colour cast.
    (lift, gamma, gain_wb)
}

/// P49 — the DETERMINISTIC `/tmp` path the pre-rendered frame of a COMPOUND clip is written to and
/// read back from. Keyed by the clip's index in `project.clips` and the clip's INNER source frame
/// (`src_frame_at(clip, t)`), so `prerender_compound_clips` (which writes the file) and
/// `resolve_frame` (which references it as a `RAW:` base/over path) DERIVE THE SAME PATH from the
/// SAME `(clip_idx, inner_t)` — the load-bearing invariant. Mirrors `thumb_temp_path` (an FNV-1a
/// `small_hash` over the keys), distinct prefix so a compound temp never collides with a thumb/title
/// raster. The temp holds a single GVW×GVH×4 RGBA frame, so resolve_frame always reads it at frame 0.
fn subseq_temp_path(clip_idx: usize, inner_t: i64) -> String {
    format!("/tmp/genesis_subseq_{:x}.rgba", small_hash(&format!("{clip_idx}|{inner_t}")))
}

/// P49 — pre-render every VISIBLE COMPOUND clip's sub-sequence frame at outer frame `t` into its
/// deterministic `subseq_temp_path`, so the RAW: base/over swaps in `resolve_frame` (and the layer
/// fold) find the file already on disk. CALLED at the TOP of the per-frame builders (`request_frame`
/// preview, `render_program`'s per-frame loop) BEFORE any frame line is built — never while the
/// worker lock is held (`run_pipeline` takes that lock itself).
///
/// For each clip with `seq >= 0` covering `t`: the INNER time is `inner = src_frame_at(c, t)` (the
/// SAME mapping resolve_frame uses for the temp key — speed/reverse honoured); `view =
/// project.subseq_view(seq)` is the transient parent-less sub-timeline Project (clips/tracks over the
/// SHARED media pool, NO subseqs of its own → ONE LEVEL of nesting, so a compound clip INSIDE a
/// sub-sequence finds no sub-view to compose and simply leaves no temp = a gap, never recursing). The
/// view's frame at `inner` is composed via the SAME fold the program uses — `build_layer_pipeline` +
/// `run_pipeline` writing the GVW×GVH RGBA to `subseq_temp_path(idx, inner)` — so there is NO new
/// compositor and NO hand-rolled compositing.
///
/// BEST-EFFORT: a None view (seq out of range), an unbuildable pipeline, or a failed compose simply
/// leaves NO file at that path → the clip composites as a black gap (the engine reads the absent RAW
/// as black). It NEVER panics and NEVER fails the parent frame.
///
/// NO-OP FAST PATH (byte-identity): if NO clip in the project is compound (`seq >= 0`), this returns
/// IMMEDIATELY without composing anything, so a project where every clip has `seq == -1` does ZERO
/// extra work and produces the EXACT same ENC/fold/PREVIEW lines as before P49.
fn prerender_compound_clips(project: &Project, t: i64) {
    // Fast path: a project with no compound clips at all does nothing (byte-identical to pre-P49).
    if !project.clips.iter().any(|c| c.seq >= 0) {
        return;
    }
    for (idx, c) in project.clips.iter().enumerate() {
        // Only VISIBLE compound clips at this outer frame (same [t0, end()) cover test resolve_frame
        // uses). A non-compound clip (seq == -1) is skipped → it keeps its existing media path.
        if c.seq < 0 || t < c.t0 || t >= c.end() {
            continue;
        }
        let inner = src_frame_at(c, t); // SAME inner the temp key + resolve_frame swap use.
        // The transient one-level sub-timeline view (None if seq is out of range → leave no file).
        let Some(view) = project.subseq_view(c.seq as usize) else {
            continue;
        };
        let out = subseq_temp_path(idx, inner);
        // Compose the VIEW's frame at `inner` into `out` as GVW×GVH RGBA via the SAME compose-to-file
        // the program uses — branching EXACTLY like `request_frame` so each layer count lands its
        // pixels in `out`:
        //   - >2 video layers → the N-layer fold (`build_layer_pipeline`, whose LAST step writes the
        //     fold's `final_out`),
        //   - <=2 layers (the common case: a sub-sequence with one V1 base, or V1+V2) → a SINGLE
        //     `resolve_frame` PREVIEW into `out` (`build_layer_pipeline` only writes `final_out` for
        //     exactly 2 layers, so the 0/1-layer case must go through this direct PREVIEW to write
        //     `out`). Both reuse `run_pipeline` + the SAME formatter as a normal frame — no new
        //     compositor, no hand-rolled compositing.
        // A failed/empty compose (None pipeline or a worker miss) leaves NO file at `out` → the parent
        // composites the compound clip as a black gap. Never panics, never fails the parent frame.
        let layers = visible_video_clips(&view, inner);
        let lines: Option<Vec<String>> = if layers.len() > 2 {
            build_layer_pipeline(&view, inner, &layers, &out)
        } else {
            resolve_frame(&view, inner).map(|r| vec![format_preview(&r, &out)])
        };
        if let Some(lines) = lines {
            let _ = run_pipeline(&lines, &out);
        }
    }
}

/// Map a TIMELINE frame `t` to the SOURCE frame clip `c` reads, honoring per-clip speed/reverse
/// (P24, Model A). speed scales source consumption (2.0 = 2x faster, 0.5 = slow-mo); reverse plays
/// the consumed range backward. Identity (speed 1.0, reverse false) returns c.src_in + (t - c.t0)
/// EXACTLY (round of an exact integer is exact) — byte-identical to pre-P24.
fn src_frame_at(c: &crate::model::Clip, t: i64) -> i64 {
    // T4 FREEZE-FRAME: speed == 0.0 HOLDS the source frame (a still). Every timeline frame of the
    // clip reads the SAME source frame `src_in` (which freeze_frame set to the frozen source frame),
    // so the still never advances. This also avoids feeding 0 into any rate math below. A clip with
    // a normal speed (>0) is unaffected — pre-T4 projects never store speed 0, so this is a no-op for
    // them (byte-identical), and the general clamp below still guards tiny/huge speeds elsewhere.
    if c.speed == 0.0 {
        return c.src_in;
    }
    let local = t - c.t0;
    let span = if c.reverse { (c.len - 1 - local).max(0) } else { local };
    let s = (c.speed as f64).clamp(0.05, 16.0);
    c.src_in + (span as f64 * s).round() as i64
}

/// Resolve the program at timeline frame `t` from the model.
///
/// Mirrors MojoMedia main_editor.mojo (lines ~592-631 preview; ~1225-1301 render): the base is
/// the topmost clip on track 0 (V1) covering `t`; the overlay is the clip on track 1 (V2)
/// covering `t`. Source frame for a clip is `clip.src_in + (t - clip.t0)`. The PiP composite is
/// only enabled when BOTH a V1 base and a V2 overlay cover `t`. If no clip covers `t` (a timeline
/// gap), the base path is the "-" sentinel, which the engine fills with a black frame (matching
/// MojoMedia's black-gap behavior). Returns None only on a corrupt media index.
///
/// KEYFRAME-AWARE (Slice A, consuming Team C's model API): the grade (bright/contrast/sat) is read
/// from `project.grade_at(t)` — the keyframed grade at timeline frame `t`, mirroring MojoMedia's
/// `kf_eval(...)` at the playhead/output frame (main_editor.mojo ~692-694 preview, ~1302-1304
/// render). The over-clip PiP rect is read from `project.pip_at(over_clip_idx, t - over_clip.t0)` —
/// the keyframed (px,py,pw,ph) at the CLIP-LOCAL frame, mirroring `pip_eval(... pip_lf ...)`
/// (~642-645 preview, ~1293-1296 render). Both accessors fall back to the clip/project STATIC
/// values when no keyframes exist for that track/clip, so a project with no keyframes resolves
/// identically to before. Because both the preview line (`build_request`) and the render line
/// (`build_enc_line`) flow through this one resolver, keyframed grade + PiP animate identically in
/// the preview and the export.
///
/// PER-CLIP LOOK (Slice A): the look is taken from the BASE (track-0/visible) clip — `look_kind =
/// clip.look` (0=None, 1=VHS, 2=LUT3D), `look_amt = clip.look_amt`, and `lut_path = clip.lut` for a
/// LUT3D look (else the "-" sentinel). Mirrors MojoMedia, whose playhead-segment look drives the
/// look pipeline. Flowing through this one resolver means the look applies in BOTH the preview and
/// the export. A timeline gap (no base clip) has no look.
fn resolve_frame(project: &Project, t: i64) -> Option<Resolved> {
    // Topmost (last-wins, matching Mojo's >= track scan) clip on track 0 and track 1 covering t.
    // We also remember the V2 (overlay) clip's INDEX, because `pip_at` keys PiP keyframes by clip
    // index (not by reference); the index mirrors MojoMedia's `s1` segment index fed to pip_eval.
    //
    // `s1` carries the V2 (overlay) clip AND its index together (finding #7): `pip_at` keys PiP
    // keyframes by clip index, so the index must travel with the clip. Storing them as one
    // `Option<(&Clip, usize)>` makes it impossible for the clip and its index to desync (the old
    // two-variable form relied on the `s1`/`s1_idx` assignments staying in lockstep, and used two
    // `.unwrap()`s downstream that a future edit could break).
    // P5 ARBITRARY TRACKS: generalize the old fixed V1=base / V2=over scan. The BASE is the covering
    // clip on the LOWEST visible VIDEO track; the OVERLAY is the covering clip on the HIGHEST visible
    // video track ABOVE the base. For the default V1/V2 layout this is exactly base=track0,
    // over=track1. Hidden video tracks + all audio tracks are skipped. (Engine is single-base +
    // single-over, so video layers strictly between base and over are not composited — true N-layer
    // compositing is a Stage-2 follow-up; bottom+top covers the overwhelmingly common case.)
    // P5 STAGE 2 N-LAYER: BASE = lowest visible video; OVER = the NEXT visible video track ABOVE the
    // base (not the highest). Any FURTHER video layers above the over are folded by request_frame/
    // render_program (each pip'd over the accumulated composite via the RAW: layer path). For the
    // default V1/V2 this is base=V1, over=V2, no extras — byte-identical to before.
    let mut base_tv: Option<(&crate::model::Clip, usize, u8)> = None; // lowest visible video
    for (i, c) in project.clips.iter().enumerate() {
        if t >= c.t0 && t < c.end() && !project.is_audio(c.track) && !project.is_hidden(c.track)
            && base_tv.is_none_or(|(_, _, bt)| c.track <= bt)
        {
            base_tv = Some((c, i, c.track)); // <= => last-wins on a tie (transition overlap)
        }
    }
    let mut over_tv: Option<(&crate::model::Clip, usize, u8)> = None; // lowest visible video ABOVE base
    if let Some((_, _, bt)) = base_tv {
        for (i, c) in project.clips.iter().enumerate() {
            if t >= c.t0 && t < c.end() && c.track > bt
                && !project.is_audio(c.track) && !project.is_hidden(c.track)
                && over_tv.is_none_or(|(_, _, ot)| c.track <= ot)
            {
                over_tv = Some((c, i, c.track));
            }
        }
    }
    let s0: Option<&crate::model::Clip> = base_tv.map(|(c, _, _)| c);
    let s1: Option<(&crate::model::Clip, usize)> = over_tv.map(|(c, i, _)| (c, i));

    // Base = the lowest visible video clip if present, else the overlay shown directly.
    let base_clip = s0.or(s1.map(|(c, _)| c));
    // INDEX of the base clip (P30): the per-clip VIDEO-param keyframe store keys by clip index, so the
    // index must travel alongside `base_clip` to evaluate keyframed bright/contrast/sat/blur/rot/scale.
    // It mirrors `base_clip`'s own `s0.or(s1)` choice: the base-track clip's index when present, else
    // the overlay's index (when the overlay is shown directly as the base). `None` => timeline gap.
    let base_idx: Option<usize> = base_tv.map(|(_, i, _)| i).or(s1.map(|(_, i)| i));

    // Per-clip LOOK from the BASE (visible) clip (Slice A). Mirrors MojoMedia, whose playhead-
    // segment look (seg_look/seg_look_amt) drives the look pipeline. look_kind: 0=None, 1=VHS,
    // 2=LUT3D. For LUT3D we forward the clip's `.cube` path; for every other case (None/VHS, or an
    // empty LUT path) we send the "-" sentinel so the engine never tries to load a LUT. A timeline
    // gap (no base clip) has no look. The amount is the clip's `look_amt`.
    // `mut` because an active transition overrides the look to the OUTGOING clip's (it fades out
    // carrying its own look) — see the transition block below.
    let (mut look_kind, mut look_amt, mut lut_path) = match base_clip {
        Some(c) => {
            // LUT path only travels with a LUT3D look (kind 2) AND a non-empty path; otherwise "-".
            // Whitespace is now WIRE-SAFE: format_preview/format_enc percent-encode the lut token
            // via enc_path and the engine decodes it, so a spaced .cube path no longer shifts the
            // fixed-arity fields. The guard is just a NON-EMPTY check now (the old single-token
            // whitespace reject is removed).
            let lut = if c.look == 2 && !c.lut.is_empty() {
                c.lut.clone()
            } else {
                "-".to_string()
            };
            (c.look, c.look_amt, lut)
        }
        None => (0, 0.0, "-".to_string()), // gap: no look.
    };

    // `mut` because an active transition forces base = OUTGOING clip for the whole window (a
    // symmetric crossfade) rather than the cover-scan winner — see the transition block below.
    let (mut base_path, mut base_frame) = match base_clip {
        Some(c) if c.seq >= 0 => {
            // P49 COMPOUND BASE: this clip plays a SUB-SEQUENCE, not decoded media. Its frame was
            // pre-composed (by `prerender_compound_clips`, called at the top of the per-frame builders)
            // into the deterministic temp `subseq_temp_path(base_idx, inner)` where `inner =
            // src_frame_at(c, t)` — the SAME index+inner this swap uses, so the two always agree on the
            // file. We substitute the EXISTING RAW: path (exactly like a title raster, model.rs:1981):
            // a `RAW:<temp>` base + frame 0 (the temp is a single pre-rendered GVW×GVH RGBA, so there
            // is no seek into it). If the pre-render produced no file (an empty/failed sub-compose, or
            // a compound clip nested inside a sub-sequence whose view carries no subseqs), the engine
            // reads the absent RAW as black → the compound clip degrades to a gap (never fails the
            // parent frame). `base_idx` is the base clip's index in `project.clips`, mirroring
            // `base_clip`'s own s0.or(s1) choice — it cannot be None when `base_clip` is Some.
            let idx = base_idx.unwrap_or(0);
            let inner = src_frame_at(c, t);
            (format!("RAW:{}", subseq_temp_path(idx, inner)), 0)
        }
        Some(c) => {
            let path = project.media.get(c.media)?;
            let frame = src_frame_at(c, t) as i32;
            (path.clone(), frame.max(0))
        }
        None => {
            // No clip covers t (timeline gap): emit the "-" base sentinel so the engine fills
            // the frame with black, matching MojoMedia's black-gap behavior — rather than
            // freeze-framing media[0]@0 (finding #5). The engine treats base "-" as the
            // all-black slot-0 frame; this never returns None for a gap on a valid project.
            ("-".to_string(), 0)
        }
    };

    // Overlay only when V1 is the base AND V2 is present (Mojo: over_v2 = s0>=0 && s1>=0). The
    // clip + its index come from the single `s1` tuple (finding #7), so there are no unwraps and no
    // way for the index to desync from the clip.
    let (over_path, over_frame, op, px, py, pw, ph) = if let (Some(_), Some((c, idx))) = (s0, s1) {
        // P49 COMPOUND OVERLAY: same single-RAW substitution as the base above. When the V2 overlay
        // clip plays a sub-sequence (`c.seq >= 0`), its frame was pre-composed into
        // `subseq_temp_path(idx, src_frame_at(c, t))` (same index+inner the pre-render used), so the
        // overlay path becomes `RAW:<temp>` + frame 0. The PiP rect / opacity keyframes still apply to
        // the composed sub-sequence frame exactly as they would to decoded media (it composites OVER
        // V1 at its PiP position). A missing pre-render temp degrades to a black overlay (a gap), never
        // a frame failure. The `over_path`/`over_frame` we return have the same shape as the media
        // case, so the rest of resolve_frame (chroma/blend/look) is untouched.
        if c.seq >= 0 {
            let t_local = t - c.t0;
            let (px, py, pw, ph) = project.pip_at(idx, t_local);
            let op = project.opacity_at(t).clamp(0.0, 1.0);
            let inner = src_frame_at(c, t);
            (format!("RAW:{}", subseq_temp_path(idx, inner)), 0, op, px, py, pw, ph)
        } else {
        match project.media.get(c.media) {
            Some(p) => {
                let frame = src_frame_at(c, t) as i32;
                // Clip-LOCAL frame for the PiP keyframe track: t - over_clip.t0 (matches Mojo's
                // pip_lf / qlf = the frame offset into the overlay clip). pip_at returns the clip's
                // static px/py/pw/ph when this clip has no PiP keyframes, so this is a drop-in
                // upgrade of the previous static `(c.px, c.py, c.pw, c.ph)`.
                let t_local = t - c.t0;
                let (px, py, pw, ph) = project.pip_at(idx, t_local);
                // OPACITY KEYFRAME (P1): the base overlay opacity (1.0) is scaled by the keyframed
                // opacity_kf at this timeline frame. An empty opacity_kf track → 1.0 (unchanged), so
                // composites without an opacity animation behave exactly as before; a keyframed fade
                // now drops `op` toward 0, fading the V2 overlay out in preview + render. Clamp to a
                // sane [0,1] so a stray keyframe value can't push the composite weight out of range.
                let op = project.opacity_at(t).clamp(0.0, 1.0);
                (p.clone(), frame.max(0), op, px, py, pw, ph)
            }
            None => ("-".to_string(), 0, 0.0, 0.0, 0.0, 1.0, 1.0),
        }
        }
    } else {
        ("-".to_string(), 0, 0.0, 0.0, 0.0, 1.0, 1.0)
    };

    // P31 BLEND MODE of the OVER (V2) overlay clip. Read from `s1` (the overlay clip) when an overlay
    // exists, else 0 (Normal). Mirrors `ck_on`/`ck_key`: it describes the V2 OVERLAY being composited
    // over V1, NOT the base/outgoing clip. A base/single clip with no overlay → 0 (Normal) → the engine
    // does a plain alpha-over → BYTE-IDENTICAL to pre-P31. Cast to i32 for the wire (the engine parses
    // an i32 blend token right after `op`).
    let blend: i32 = match s1 {
        Some((c, _)) => c.blend_mode as i32,
        None => 0,
    };

    // P4 CHROMA KEY (green-screen) from the OVER (V2) overlay clip. The key applies to the OVERLAY
    // (the green-screen layer composited over V1), so it is read from `s1` (the V2 clip), NOT the
    // base/outgoing clip — and only when a PiP composite is actually active (BOTH a V1 base AND a V2
    // overlay cover `t`, i.e. `op > 0`). A disabled chroma (or no overlay) sends the IDENTITY sentinel
    // (ck_on=0 + the default key/sim/smooth/spill), so the engine no-ops and the composite is
    // byte-identical to P3. Team A READS `Clip.chroma` here (pre-added by Team C) but NEVER edits
    // model.rs. P37: ck_spill is sourced from the SAME overlay clip's `chroma.spill`, exactly mirroring
    // how ck_sim/ck_smooth are sourced; the disabled/no-overlay arm sends 0.0 (off) for byte-identity.
    let (ck_on, ck_key, ck_sim, ck_smooth, ck_spill) = match (s0, s1) {
        (Some(_), Some((c, _))) if c.chroma.enabled && op > 0.0 => {
            (1, c.chroma.key, c.chroma.similarity, c.chroma.smoothness, c.chroma.spill)
        }
        // No overlay / chroma disabled: identity (engine skips keying). Defaults mirror ChromaKey.
        _ => (0, [0.0, 1.0, 0.0], 0.4, 0.1, 0.0),
    };

    // Keyframed grade at the timeline frame (falls back to project.bright/contrast/sat when there
    // are no grade keyframes — Team C's grade_at contract). Replaces the old static grade that was
    // previously read directly off the project in build_request/build_enc_line.
    let (bright, contrast, sat) = project.grade_at(t);

    // PER-CLIP COLOR grade (P1) from the BASE (visible) clip — ADDITIVE on top of the program grade
    // (gcompose runs the per-clip grade FIRST, then the program grade). A timeline gap (no base clip)
    // is neutral (0/1/1). `mut` because an active transition overrides this to the OUTGOING clip's
    // grade for the whole window (it travels with the clip as it fades out, like the look).
    // P30: each per-clip grade param is now KEYFRAME-DRIVEN when the base clip has keys for it.
    // `clip_param_at(idx, par, t - c.t0, static)` interpolates the (clip,par) track at the clip-LOCAL
    // frame, falling back to the static field when the clip has no keys → byte-identical for an
    // un-keyframed clip. par 4=bright, 5=contrast, 6=sat. `base_idx` is `Some` exactly when
    // `base_clip` is (they share the `s0.or(s1)` choice), so the `(Some(c), Some(idx))` arm always
    // matches for a real base clip; the gap arm stays neutral (0/1/1).
    let (mut cbright, mut ccontrast, mut csat) = match (base_clip, base_idx) {
        (Some(c), Some(idx)) => {
            let tl = t - c.t0;
            (
                project.clip_param_at(idx, 4, tl, c.bright),
                project.clip_param_at(idx, 5, tl, c.contrast),
                project.clip_param_at(idx, 6, tl, c.sat),
            )
        }
        _ => (0.0, 1.0, 1.0),
    };

    // PER-CLIP COLOR-WHEELS (LIFT/GAMMA/GAIN) + TRANSFORM + BLUR (P2) from the BASE (visible) clip.
    // White balance (wb_temp/wb_tint) is FOLDED into the 9 lift/gamma/gain values here (the engine
    // never sees wb — PINNED), so `lgg` below carries the white-balanced gains. A timeline gap (no
    // base clip) is IDENTITY (lift 0 / gamma 1 / gain 1 / rot 0 / scale 1 / blur 0) so the engine
    // no-ops and reproduces the current output. `mut` because an active transition overrides these
    // to the OUTGOING clip's for the whole window (they travel with the look/grade as it fades out).
    // P30: rot/scale/blur are KEYFRAME-DRIVEN when the base clip has keys for them (the lift/gamma/
    // gain color-wheels stay static — not in the curated keyframeable set). par 7=blur, 8=rot,
    // 9=scale, each falling back to the static field so an un-keyframed clip is byte-identical.
    let (mut lift, mut gamma, mut gain_rgb, mut rot, mut scale, mut blur) = match (base_clip, base_idx)
    {
        (Some(c), Some(idx)) => {
            let (l, g, gn) = fold_white_balance(c.lift, c.gamma, c.gain_rgb, c.wb_temp, c.wb_tint);
            let tl = t - c.t0;
            (
                l,
                g,
                gn,
                project.clip_param_at(idx, 8, tl, c.rot),
                project.clip_param_at(idx, 9, tl, c.scale),
                project.clip_param_at(idx, 7, tl, c.blur),
            )
        }
        _ => ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 0.0, 1.0, 0.0),
    };

    // PER-CLIP master tone CURVE (P5) from the BASE clip; identity ([0,.25,.5,.75,1]) for a gap so an
    // un-curved clip / gap is a no-op (the engine skips the kernel on identity).
    let curve: [f32; 5] = base_clip.map(|c| c.curve).unwrap_or([0.0, 0.25, 0.5, 0.75, 1.0]);

    // PER-CLIP STYLIZE / UTILITY filters (P6) from the BASE clip; IDENTITY (0/0/0/0) for a gap so an
    // un-stylized clip / gap is a no-op (the engine skips each kernel at its no-op default). `mut`
    // because an active transition overrides them to the OUTGOING clip's values (they fade out with
    // the look/grade/curve) in the transition block below.
    let (mut vignette, mut sharpen, mut flip, mut fx) = match base_clip {
        Some(c) => (c.vignette, c.sharpen, c.flip, c.fx),
        None => (0.0_f32, 0.0_f32, 0_u8, 0_i32),
    };

    // PER-CLIP P7 COLOR filters (HSL ADJUST + LEVELS) from the BASE clip; IDENTITY (hsl [0,1,0],
    // levels [0,1,1]) for a gap so an un-color-filtered clip / gap is a no-op (the engine skips each
    // kernel at its no-op default). `mut` because an active transition overrides them to the OUTGOING
    // clip's values (they fade out with the look/grade/curve/stylize) in the transition block below.
    let (mut hsl, mut levels) = match base_clip {
        Some(c) => (c.hsl, c.levels),
        None => ([0.0_f32, 1.0_f32, 0.0_f32], [0.0_f32, 1.0_f32, 1.0_f32]),
    };

    // PER-CLIP P8 STYLIZE-2 filters (MOSAIC + GRADIENT MAP) from the BASE clip; IDENTITY
    // (mosaic 0, gmap_amt 0, lo [0,0,0], hi [1,1,1]) for a gap so an un-stylized clip / gap is a
    // no-op (the engine skips each kernel at its no-op default). `mut` because an active transition
    // overrides them to the OUTGOING clip's values (they fade out with the look/grade/curve/stylize/
    // color) in the transition block below.
    let (mut mosaic, mut gmap_amt, mut gmap_lo, mut gmap_hi) = match base_clip {
        Some(c) => (c.mosaic, c.gmap_amt, c.gmap_lo, c.gmap_hi),
        None => (0_u32, 0.0_f32, [0.0_f32, 0.0_f32, 0.0_f32], [1.0_f32, 1.0_f32, 1.0_f32]),
    };

    // PER-CLIP P9 FX FILTERS (DENOISE + GLOW + RGB-SHIFT) from the BASE clip; IDENTITY
    // (denoise 0, glow_amt 0, glow_thr 0.7, rgbshift 0) for a gap so an un-FX'd clip / gap is a
    // no-op (the engine skips each kernel at its no-op default). `mut` because an active transition
    // overrides them to the OUTGOING clip's values (they fade out with the look/grade/curve/stylize/
    // color/stylize-2) in the transition block below.
    let (mut denoise, mut glow_amt, mut glow_thr, mut rgbshift) = match base_clip {
        Some(c) => (c.denoise, c.glow_amt, c.glow_thr, c.rgbshift),
        None => (0.0_f32, 0.0_f32, 0.7_f32, 0.0_f32),
    };

    // PER-CLIP P10 STYLIZE-4 filters (HALFTONE + EMBOSS + EDGE) from the BASE clip; IDENTITY
    // (halftone 0, emboss 0, edge 0) for a gap so an un-stylized clip / gap is a no-op (the engine
    // skips each kernel at its no-op default). `mut` because an active transition overrides them to
    // the OUTGOING clip's values (they fade out with the look/grade/curve/stylize/color/stylize-2/fx)
    // in the transition block below.
    let (mut halftone, mut emboss, mut edge) = match base_clip {
        Some(c) => (c.halftone, c.emboss, c.edge),
        None => (0_u32, 0.0_f32, 0.0_f32),
    };

    // PER-CLIP P13 OLD-FILM / DISTORT filters (GRAIN + SCRATCHES + DIFFUSION) from the BASE clip;
    // IDENTITY (grain 0, scratches 0, diffusion 0) for a gap so an un-aged clip / gap is a no-op (the
    // engine skips each kernel at its no-op default). `mut` because an active transition overrides
    // them to the OUTGOING clip's values (they fade out with the look/grade/curve/stylize/color/
    // stylize-2/fx/stylize-4) in the transition block below.
    let (mut grain, mut scratches, mut diffusion) = match base_clip {
        Some(c) => (c.grain, c.scratches, c.diffusion),
        None => (0.0_f32, 0.0_f32, 0.0_f32),
    };

    // PER-CLIP P16 DISTORT filters (WAVE + SWIRL + THRESHOLD) from the BASE clip; IDENTITY
    // (wave 0, swirl 0, threshold 0) for a gap so an un-distorted clip / gap is a no-op (the engine
    // skips each kernel at its no-op default). `mut` because an active transition overrides them to
    // the OUTGOING clip's values (they fade out with the look/grade/curve/stylize/color/stylize-2/fx/
    // stylize-4/old-film) in the transition block below.
    let (mut wave, mut swirl, mut threshold) = match base_clip {
        Some(c) => (c.wave, c.swirl, c.threshold),
        None => (0.0_f32, 0.0_f32, 0.0_f32),
    };

    // PER-CLIP P17 GEOMETRIC filters (LENS + CROP + GLITCH) from the BASE clip; IDENTITY
    // (lens 0, crop 0, glitch 0) for a gap so an un-distorted clip / gap is a no-op (the engine
    // skips each kernel at its no-op default). `mut` because an active transition overrides them to
    // the OUTGOING clip's values (they fade out with the look/grade/curve/stylize/color/stylize-2/fx/
    // stylize-4/old-film/distort) in the transition block below.
    let (mut lens, mut crop, mut glitch) = match base_clip {
        Some(c) => (c.lens, c.crop, c.glitch),
        None => (0.0_f32, 0.0_f32, 0.0_f32),
    };

    // PER-CLIP P23 360 REFRAME (EQ360 + YAW/PITCH/FOV) from the BASE clip; IDENTITY
    // (eq360 false, yaw 0, pitch 0, fov 90) for a gap so an un-reframed clip / gap is a byte-exact
    // no-op (the engine returns immediately when eq360 is false). `mut` because an active transition
    // overrides them to the OUTGOING clip's values (they fade out with the look/grade/curve/stylize/
    // color/stylize-2/fx/stylize-4/old-film/distort/geometric) in the transition block below.
    let (mut eq360, mut eq_yaw, mut eq_pitch, mut eq_fov) = match base_clip {
        Some(c) => (c.eq360, c.eq_yaw, c.eq_pitch, c.eq_fov),
        None => (false, 0.0_f32, 0.0_f32, 90.0_f32),
    };

    // PER-CLIP P34 SHAPE MASK (mask_shape + cx/cy/rw/rh/feather/invert) from the BASE clip; IDENTITY
    // (mask_shape 0, cx 0.5, cy 0.5, rw 0.5, rh 0.5, feather 0, invert 0) for a gap so an un-masked
    // clip / gap is a byte-exact no-op (the engine returns immediately when mask_shape is 0).
    // mask_shape is stored u8 on the Clip but rides the wire as an i32 token; mask_invert is stored
    // bool and rides as a 1/0 integer flag. `mut` because an active transition overrides them to the
    // OUTGOING clip's values (they fade out with the look/grade/curve/stylize/color/stylize-2/fx/
    // stylize-4/old-film/distort/geometric/360-reframe) in the transition block below.
    let (mut mask_shape, mut mask_cx, mut mask_cy, mut mask_rw, mut mask_rh, mut mask_feather, mut mask_invert) =
        match base_clip {
            Some(c) => (
                c.mask_shape as i32,
                c.mask_cx,
                c.mask_cy,
                c.mask_rw,
                c.mask_rh,
                c.mask_feather,
                if c.mask_invert { 1 } else { 0 },
            ),
            None => (0_i32, 0.5_f32, 0.5_f32, 0.5_f32, 0.5_f32, 0.0_f32, 0_i32),
        };

    // PER-CLIP P38 DISTORT (MIRROR + KALEIDOSCOPE + DITHER) from the BASE clip; IDENTITY
    // (mirror_x 0, kaleido 0, dither 0) for a gap so an un-distorted clip / gap is a byte-exact
    // no-op (the engine skips each kernel at its no-op default). mirror_x is stored u8 on the Clip
    // but rides the wire as an i32 token; kaleido rides as i32; dither as f32. `mut` because an
    // active transition overrides them to the OUTGOING clip's values (they fade out with the
    // look/grade/curve/stylize/color/stylize-2/fx/stylize-4/old-film/distort/geometric/360-reframe/
    // shape-mask) in the transition block below.
    let (mut mirror_x, mut kaleido, mut dither) = match base_clip {
        Some(c) => (c.mirror_x as i32, c.kaleido, c.dither),
        None => (0_i32, 0_i32, 0.0_f32),
    };

    // PER-CLIP P39 SELECTIVE COLOR (one HUE BAND) from the BASE clip; IDENTITY (sel_band 0,
    // sel_hshift 0, sel_sat 1.0) for a gap so an un-graded clip / gap is a byte-exact no-op (the
    // engine skips the kernel at its no-op default sel_band 0). sel_band is stored u8 on the Clip but
    // rides the wire as an i32 token; sel_hshift / sel_sat ride as f32. NOTE sel_sat's identity is 1.0
    // (a saturation MULTIPLIER), NOT 0.0 — 0.0 would desaturate the band. `mut` because an active
    // transition overrides them to the OUTGOING clip's values (they fade out with the
    // look/grade/curve/stylize/color/stylize-2/fx/stylize-4/old-film/distort/geometric/360-reframe/
    // shape-mask) in the transition block below.
    let (mut sel_band, mut sel_hshift, mut sel_sat) = match base_clip {
        Some(c) => (c.sel_band as i32, c.sel_hshift, c.sel_sat),
        None => (0_i32, 0.0_f32, 1.0_f32),
    };

    // PER-CLIP P41 SOLARIZE + COLOUR TEMPERATURE from the BASE clip; IDENTITY (sol_thr 0.0,
    // temp 0.0) for a gap so an un-graded clip / gap is a byte-exact no-op (the engine skips both
    // kernels at their no-op defaults). Both ride the wire as f32 tokens. NOTE both identities are
    // 0.0 (sol_thr 0 = solarize OFF; temp 0 = neutral). `mut` because an active transition overrides
    // them to the OUTGOING clip's values (they fade out with the
    // look/grade/curve/stylize/color/stylize-2/fx/stylize-4/old-film/distort/geometric/360-reframe/
    // shape-mask/selective-color) in the transition block below.
    let (mut sol_thr, mut temp) = match base_clip {
        Some(c) => (c.sol_thr, c.temp),
        None => (0.0_f32, 0.0_f32),
    };
    // P45 VIDEO FADE factor from the base (visible) clip's fade_in/fade_out at this frame. 1.0 for a
    // gap / a clip with no fades (byte-identical). `mut` because an active transition overrides it to
    // the OUTGOING clip's fade (it darkens as the outgoing clip fades out, like the grade/look).
    let mut fade = base_clip.map(|c| c.fade_factor(t)).unwrap_or(1.0);

    // ----- Per-boundary TRANSITION (Wave 8) -------------------------------------------------------
    // Consult the transition (if any) on the BASE track whose window contains `t`, then blend the
    // OUTGOING clip (slot 0) -> INCOMING clip (slot 2) by `trans_prog` over the CENTERED window
    // [center - dur/2, center + dur/2). The progress ramp + window come from `Transition::progress`
    // so preview/render never desync from the model. `trans_param` defaults to 4.0 (MojoMedia tt_p).
    //
    // SYMMETRIC crossfade (the wave-8 follow-up): when a transition is active we OVERRIDE the base
    // (slot 0) + look to the OUTGOING clip for the WHOLE window — not the cover-scan winner. Without
    // this, once `t` reaches the incoming clip's span the cover scan makes the INCOMING clip the
    // base, so slot 0 == slot 2 and the window's second half degenerates to incoming-vs-incoming
    // (effectively just the incoming clip). Forcing base = outgoing makes prog=0 -> 100% outgoing,
    // prog=1 -> 100% incoming, a true crossfade straddling the seam. Both clips' source frames are
    // clamped into their own `[src_in, src_in+len-1]` range, so past a clip's end it freezes on its
    // last frame — the standard crossfade hold for adjacent, non-overlapping media.
    //
    // base_track is the track the chosen base clip lives on (V1 if present, else V2). A timeline gap
    // (no base clip) has no track to query, so there is no transition.
    let (trans_kind, trans_prog, trans_param, trans_path, trans_frame) = {
        let mut tk: i32 = -1;
        let mut tp: f32 = 0.0;
        let mut tparam: f32 = 4.0;
        let mut tpath = "-".to_string();
        let mut tframe: i32 = 0;

        // A media path is wire-safe whenever it is NON-EMPTY: format_preview/format_enc percent-
        // encode every path token (enc_path) and the engine decodes it, so a spaced path no longer
        // shifts the fixed-arity fields. The old single-token whitespace reject is removed; only the
        // empty/missing-path case still degrades to no transition.
        let clean = |s: &str| !s.is_empty();

        if let Some(bc) = base_clip {
            let base_track = bc.track;
            // `transition_at` copies-out so we don't hold a `&project` borrow across the
            // `project.boundaries(..)` call below (which also borrows `project` immutably).
            if let Some(tr) = project.transition_at(base_track, t).copied() {
                // `boundaries()` pairs each OUTGOING clip with its successor (the INCOMING clip) and
                // reports the seam frame (the overlap midpoint for overlapping pairs); match the pair
                // whose seam == the transition center (finding #1 — never a `t0 >= center` scan that
                // would skip the real incoming clip).
                let pair = project
                    .boundaries(base_track)
                    .into_iter()
                    .find(|&(_outgoing, _incoming, boundary)| boundary == tr.center);
                if let Some((out_idx, in_idx, _boundary)) = pair {
                    if let (Some(out_clip), Some(inc)) =
                        (project.clips.get(out_idx), project.clips.get(in_idx))
                    {
                        match (project.media.get(out_clip.media), project.media.get(inc.media)) {
                            (Some(op), Some(ip)) if clean(op) && clean(ip) => {
                                let prog = tr.progress(t);
                                // OUTGOING -> slot 0 base (forced for the whole window). Clamp into
                                // the outgoing clip's valid source range so past its end it freezes
                                // on its last frame instead of decoding a frame it doesn't have.
                                let raw_out = src_frame_at(out_clip, t);
                                let last_out =
                                    (out_clip.src_in + out_clip.len - 1).max(out_clip.src_in);
                                base_path = op.clone();
                                base_frame = raw_out.clamp(out_clip.src_in, last_out) as i32;
                                // The LOOK travels with the OUTGOING clip while it fades out.
                                look_kind = out_clip.look;
                                look_amt = out_clip.look_amt;
                                lut_path = if out_clip.look == 2 && clean(&out_clip.lut) {
                                    out_clip.lut.clone()
                                } else {
                                    "-".to_string()
                                };
                                // The per-clip GRADE likewise travels with the OUTGOING clip (P1),
                                // now KEYFRAME-DRIVEN (P30): interpolate the outgoing clip's
                                // (out_idx, par) tracks at ITS clip-local frame (t - out_clip.t0),
                                // falling back to its static fields → un-keyframed = byte-identical.
                                let otl = t - out_clip.t0;
                                cbright = project.clip_param_at(out_idx, 4, otl, out_clip.bright);
                                ccontrast = project.clip_param_at(out_idx, 5, otl, out_clip.contrast);
                                csat = project.clip_param_at(out_idx, 6, otl, out_clip.sat);
                                // The P2 color-wheels (white-balanced) + transform + blur ALSO
                                // travel with the OUTGOING clip while it fades out (matching the
                                // look/grade), so a graded/transformed clip keeps its grade through
                                // the whole transition window rather than snapping at the seam.
                                let (ol, og, ogn) = fold_white_balance(
                                    out_clip.lift,
                                    out_clip.gamma,
                                    out_clip.gain_rgb,
                                    out_clip.wb_temp,
                                    out_clip.wb_tint,
                                );
                                lift = ol;
                                gamma = og;
                                gain_rgb = ogn;
                                // rot/scale/blur are KEYFRAME-DRIVEN on the OUTGOING clip too (P30);
                                // par 8=rot, 9=scale, 7=blur, fallback = the static field.
                                rot = project.clip_param_at(out_idx, 8, otl, out_clip.rot);
                                scale = project.clip_param_at(out_idx, 9, otl, out_clip.scale);
                                blur = project.clip_param_at(out_idx, 7, otl, out_clip.blur);
                                // The P6 stylize/utility filters (vignette/sharpen/flip/fx) ALSO
                                // travel with the OUTGOING clip while it fades out (matching the
                                // look/grade/curve), so a stylized clip keeps its filters through the
                                // whole transition window rather than snapping at the seam.
                                vignette = out_clip.vignette;
                                sharpen = out_clip.sharpen;
                                flip = out_clip.flip;
                                fx = out_clip.fx;
                                // The P7 color filters (HSL adjust + levels) ALSO travel with the
                                // OUTGOING clip while it fades out (matching the look/grade/curve/
                                // stylize), so a color-filtered clip keeps its HSL/levels through the
                                // whole transition window rather than snapping at the seam.
                                hsl = out_clip.hsl;
                                levels = out_clip.levels;
                                // The P8 stylize-2 filters (mosaic + gradient map) ALSO travel with
                                // the OUTGOING clip while it fades out (matching the look/grade/curve/
                                // stylize/color), so a mosaic'd / gradient-mapped clip keeps its
                                // P8 filters through the whole transition window rather than snapping
                                // at the seam.
                                mosaic = out_clip.mosaic;
                                gmap_amt = out_clip.gmap_amt;
                                gmap_lo = out_clip.gmap_lo;
                                gmap_hi = out_clip.gmap_hi;
                                // The P9 FX filters (denoise + glow + rgb-shift) ALSO travel with the
                                // OUTGOING clip while it fades out (matching the look/grade/curve/
                                // stylize/color/stylize-2), so a denoised / glowing / rgb-shifted clip
                                // keeps its P9 filters through the whole transition window rather than
                                // snapping at the seam.
                                denoise = out_clip.denoise;
                                glow_amt = out_clip.glow_amt;
                                glow_thr = out_clip.glow_thr;
                                rgbshift = out_clip.rgbshift;
                                // The P10 stylize-4 filters (halftone + emboss + edge) ALSO travel
                                // with the OUTGOING clip while it fades out (matching the look/grade/
                                // curve/stylize/color/stylize-2/fx), so a halftoned / embossed /
                                // edge-detected clip keeps its P10 filters through the whole
                                // transition window rather than snapping at the seam.
                                halftone = out_clip.halftone;
                                emboss = out_clip.emboss;
                                edge = out_clip.edge;
                                // The P13 old-film/distort filters (grain + scratches + diffusion)
                                // ALSO travel with the OUTGOING clip while it fades out (matching the
                                // look/grade/curve/stylize/color/stylize-2/fx/stylize-4), so a grainy /
                                // scratched / diffused clip keeps its P13 filters through the whole
                                // transition window rather than snapping at the seam.
                                grain = out_clip.grain;
                                scratches = out_clip.scratches;
                                diffusion = out_clip.diffusion;
                                // The P16 distort filters (wave + swirl + threshold) ALSO travel
                                // with the OUTGOING clip while it fades out (matching the look/grade/
                                // curve/stylize/color/stylize-2/fx/stylize-4/old-film), so a waved /
                                // swirled / thresholded clip keeps its P16 filters through the whole
                                // transition window rather than snapping at the seam.
                                wave = out_clip.wave;
                                swirl = out_clip.swirl;
                                threshold = out_clip.threshold;
                                // The P17 geometric filters (lens + crop + glitch) ALSO travel
                                // with the OUTGOING clip while it fades out (matching the look/grade/
                                // curve/stylize/color/stylize-2/fx/stylize-4/old-film/distort), so a
                                // lens-distorted / cropped / glitched clip keeps its P17 filters
                                // through the whole transition window rather than snapping at the seam.
                                lens = out_clip.lens;
                                crop = out_clip.crop;
                                glitch = out_clip.glitch;
                                // The P23 360 reframe (eq360 + yaw/pitch/fov) ALSO travels with the
                                // OUTGOING clip while it fades out (matching the look/grade/curve/
                                // stylize/color/stylize-2/fx/stylize-4/old-film/distort/geometric), so
                                // a 360-reframed clip keeps its P23 settings through the whole
                                // transition window rather than snapping at the seam.
                                eq360 = out_clip.eq360;
                                eq_yaw = out_clip.eq_yaw;
                                eq_pitch = out_clip.eq_pitch;
                                eq_fov = out_clip.eq_fov;
                                // The P34 shape mask (mask_shape + cx/cy/rw/rh/feather/invert) ALSO
                                // travels with the OUTGOING clip while it fades out (matching the
                                // look/grade/curve/stylize/color/stylize-2/fx/stylize-4/old-film/
                                // distort/geometric/360-reframe), so a masked clip keeps its P34
                                // settings through the whole transition window rather than snapping at
                                // the seam. mask_shape rides as i32, mask_invert as a 1/0 integer flag.
                                mask_shape = out_clip.mask_shape as i32;
                                mask_cx = out_clip.mask_cx;
                                mask_cy = out_clip.mask_cy;
                                mask_rw = out_clip.mask_rw;
                                mask_rh = out_clip.mask_rh;
                                mask_feather = out_clip.mask_feather;
                                mask_invert = if out_clip.mask_invert { 1 } else { 0 };
                                // The P38 distort (mirror_x + kaleido + dither) ALSO travels with the
                                // OUTGOING clip while it fades out (matching the look/grade/curve/
                                // stylize/color/stylize-2/fx/stylize-4/old-film/distort/geometric/
                                // 360-reframe/shape-mask), so a mirrored/kaleido'd/dithered clip keeps
                                // its P38 settings through the whole transition window rather than
                                // snapping at the seam. mirror_x rides as a 1/0 i32, kaleido as i32,
                                // dither as f32.
                                mirror_x = out_clip.mirror_x as i32;
                                kaleido = out_clip.kaleido;
                                dither = out_clip.dither;
                                // The P39 selective color (sel_band + sel_hshift + sel_sat) ALSO
                                // travels with the OUTGOING clip while it fades out (matching the
                                // look/grade/curve/stylize/color/stylize-2/fx/stylize-4/old-film/
                                // distort/geometric/360-reframe/shape-mask), so a band-graded clip
                                // keeps its P39 settings through the whole transition window rather
                                // than snapping at the seam. sel_band rides as an i32 token,
                                // sel_hshift / sel_sat as f32 (sel_sat identity = 1.0).
                                sel_band = out_clip.sel_band as i32;
                                sel_hshift = out_clip.sel_hshift;
                                sel_sat = out_clip.sel_sat;
                                // The P41 solarize + colour temperature (sol_thr + temp) ALSO travel
                                // with the OUTGOING clip while it fades out (matching the look/grade/
                                // curve/stylize/color/stylize-2/fx/stylize-4/old-film/distort/
                                // geometric/360-reframe/shape-mask/selective-color), so a solarized /
                                // temperature-shifted clip keeps its P41 settings through the whole
                                // transition window rather than snapping at the seam. Both ride as f32
                                // tokens (sol_thr identity = 0.0 / off, temp identity = 0.0 / neutral).
                                sol_thr = out_clip.sol_thr;
                                temp = out_clip.temp;
                                fade = out_clip.fade_factor(t);
                                // INCOMING -> slot 2 (the partner the kernel blends the base toward),
                                // its source frame likewise clamped into its valid range.
                                let raw_in = src_frame_at(inc, t);
                                let last_in = (inc.src_in + inc.len - 1).max(inc.src_in);
                                tk = tr.kind;
                                tp = prog;
                                tparam = 4.0;
                                tpath = ip.clone();
                                tframe = raw_in.clamp(inc.src_in, last_in) as i32;
                            }
                            _ => {
                                // A missing/whitespace media path on either side degrades to no
                                // transition: the base (cover-scan winner) still composes and the
                                // fixed-arity wire line stays intact.
                                eprintln!("gcompose: transition clip media path missing/has whitespace; skipping transition at center {}", tr.center);
                            }
                        }
                    }
                }
                // No matching boundary / clip pair: leave tk = -1 (no transition).
            }
        }
        (tk, tp, tparam, tpath, tframe)
    };

    // ----- P5 TITLE TEXT OVERLAY (Slice A) --------------------------------------------------------
    // When the BASE clip OR the OVER clip carries a non-empty title, rasterize that title to a
    // full-frame transparent RGBA8 layer (cached temp file) and SUBSTITUTE that clip's media path
    // with the `RAW:<raster_path>` sentinel + frame 0 (a title is STATIC — its source frame is
    // ignored). The engine then reads the raw RGBA straight into the slot (skipping decode) and
    // composites it: a V2 title shows its text OVER the V1 base; a title-only clip (no V1) shows the
    // text over black (the "-" base). All other per-clip effects (grade/pip/chroma/look/transition)
    // still apply to the rasterized text exactly as they would to decoded media — by design.
    //
    // IDENTITY: a clip with an empty title yields no raster (rasterize_title -> None), so its path is
    // left untouched and a project with no titles never emits a `RAW:` path → byte-identical render.
    // No new wire field is needed: the title rides on the EXISTING base/over path as the `RAW:`
    // sentinel, so build_request / build_enc_line are unchanged.
    let mut over_path = over_path;
    let mut over_frame = over_frame;
    // BASE clip title (covers both a title-only V1 clip and a V1 clip that itself has a title).
    if let Some(c) = base_clip {
        if !c.title.is_empty() {
            if let Some(raster) = rasterize_title(&c.title) {
                base_path = format!("RAW:{raster}");
                base_frame = 0;
            }
        }
    }
    // OVER (V2) clip title: substitute the overlay path so the title composites over the base. Only
    // meaningful when an overlay actually exists (s1.is_some()); when it does, force op>0 so the
    // composite runs even if the model's opacity happened to resolve to 0 for a pure title layer.
    let mut op = op;
    if let Some((c, _)) = s1 {
        if !c.title.is_empty() {
            if let Some(raster) = rasterize_title(&c.title) {
                over_path = format!("RAW:{raster}");
                over_frame = 0;
                if op <= 0.0 {
                    op = 1.0; // a title overlay with zero opacity would never show; make it visible.
                }
            }
        }
    }

    Some(Resolved {
        base_path,
        base_frame,
        over_path,
        over_frame,
        op,
        blend,
        px,
        py,
        pw,
        ph,
        bright,
        contrast,
        sat,
        cbright,
        ccontrast,
        csat,
        look_kind,
        look_amt,
        lut_path,
        trans_kind,
        trans_prog,
        trans_param,
        trans_path,
        trans_frame,
        lift,
        gamma,
        gain_rgb,
        rot,
        scale,
        blur,
        ck_on,
        ck_key,
        ck_sim,
        ck_smooth,
        ck_spill,
        curve,
        vignette,
        sharpen,
        flip,
        fx,
        hsl,
        levels,
        mosaic,
        gmap_amt,
        gmap_lo,
        gmap_hi,
        denoise,
        glow_amt,
        glow_thr,
        rgbshift,
        halftone,
        emboss,
        edge,
        grain,
        scratches,
        diffusion,
        wave,
        swirl,
        threshold,
        lens,
        crop,
        glitch,
        eq360,
        eq_yaw,
        eq_pitch,
        eq_fov,
        mask_shape,
        mask_cx,
        mask_cy,
        mask_rw,
        mask_rh,
        mask_feather,
        mask_invert,
        mirror_x,
        kaleido,
        dither,
        sel_band,
        sel_hshift,
        sel_sat,
        sol_thr,
        temp,
        fade,
    })
}

/// Resolve frame `t` and format the preview request line: an explicit `PREVIEW` keyword followed
/// by the 102 positional payload fields, the LAST of which is the out path (P41: was 100+out, the 2 new
/// fields sol_thr/temp inserted between sel_sat and out; P39: was 97+out, the 3 new
/// fields sel_band/sel_hshift/sel_sat inserted between dither and out; P38: was 94+out, the 3 new
/// fields mirror_x/kaleido/dither inserted between ck_spill and out; P37: was 93+out, the new
/// field is the single ck_spill token appended AFTER mask_invert, before out; P34: was 86+out, the
/// 7 shape-mask tokens between eq_fov and out; P31: was 85+out, the new field is the
/// over-blend mode right after `op`; P23 was 81+out; P17 was 78+out). The keyword removes the latent
/// dispatch ambiguity where a media path whose first token happened to equal a command keyword
/// (OPEN/ENC/...) could misroute a preview frame to the wrong handler (finding #3); the engine now
/// matches `PREVIEW` explicitly and never falls through to keyword-guessing for a real frame request.
/// Total PREVIEW token count = 103 (keyword + 102 payload fields).
fn build_request(project: &Project, t: i64) -> Option<String> {
    let r = resolve_frame(project, t)?;
    Some(format_preview(&r, PREVIEW_RGBA))
}

/// Format a PREVIEW wire line from a resolved frame spec + an explicit out path. Split out of
/// `build_request` (P5 Stage 2) so the N-layer fold can reuse the EXACT same 103-token format (keyword
/// + 102 payload fields, last = out) for its intermediate composites (a hand-typed wire line is too
/// error-prone — see the reverted attempt).
fn format_preview(r: &Resolved, out: &str) -> String {
    // PREVIEW + 102 space-separated payload fields incl out path (P41: was 100; the 2 new fields are the
    // P41 SOLARIZE / COLOUR-TEMPERATURE tokens `sol_thr temp`, inserted BETWEEN sel_sat and the out
    // path — see below. P39: was 97; the 3 new fields are the
    // P39 SELECTIVE-COLOR tokens `sel_band sel_hshift sel_sat`, inserted BETWEEN dither and the out
    // path — see below. P38: was 94; the 3 new fields are the
    // P38 DISTORT tokens `mirror_x kaleido dither`, inserted BETWEEN ck_spill and the out path — see
    // below. P37: was 93; the new field is the P37
    // CHROMA-SPILL token `ck_spill`, appended AFTER mask_invert and BEFORE the out path — see below.
    // P31 earlier added the V2 overlay BLEND mode as ONE integer token IMMEDIATELY AFTER `op`): the 12
    // composite fields (now 13 incl blend), then the 3 Slice A LOOK fields (look_kind, look_amt,
    // lut_path), then the 5 Wave 8
    // TRANSITION fields (trans_kind, trans_prog, trans_param, trans_path, trans_frame), then the 3
    // Triad-B P1 PER-CLIP GRADE fields (cbright, ccontrast, csat), then the 12 Triad-B P2 fields
    // (lift_r lift_g lift_b  gamma_r gamma_g gamma_b  gain_r gain_g gain_b  rot scale blur) in the
    // PINNED order, then the out path. The program grade (b/c/s) comes from the RESOLVED (keyframed)
    // values, the per-clip grade/wheels/transform/blur from the BASE clip (white-balance already
    // folded into the 9 lift/gamma/gain), the look from the BASE clip, and the transition from the
    // base-track boundary — so the preview reflects keyframed grade, per-clip grade, color-wheels,
    // transform, blur, per-clip look, AND the animated transition, identical to the render
    // (`build_enc_line`), then the 6 Triad-A P4 CHROMA-KEY fields (ck_on ck_r ck_g ck_b ck_sim
    // ck_smooth) describing the OVER (V2) clip — identity (ck_on=0) when there is no overlay / the
    // chroma is disabled, so a project with no chroma renders byte-identically to P3. Then the 5
    // P5 CURVE fields (cv0..cv4), then the 4 P6 STYLIZE/UTILITY fields (vig sharp flip fx) in the
    // PINNED order, then the 6 P7 COLOR fields (hue sat light inb inw gam) in the PINNED order
    // (hue=hsl[0], sat=hsl[1], light=hsl[2], inb=levels[0], inw=levels[1], gam=levels[2]), then the
    // 8 P8 STYLIZE-2 fields (mosaic gmap_amt glo_r glo_g glo_b ghi_r ghi_g ghi_b) in the PINNED
    // order (mosaic=mosaic, gmap_amt=gmap_amt, glo=gmap_lo[0..3], ghi=gmap_hi[0..3]), then the 4 P9
    // FX fields (denoise glow_amt glow_thr rgbshift) in the PINNED order, then the 3 P10 STYLIZE-4
    // fields (halftone emboss edge) in the PINNED order, then the 3 P13 OLD-FILM fields
    // (grain scratches diffusion) in the PINNED order, then the 3 P16 DISTORT fields
    // (wave swirl threshold) in the PINNED order, then the 3 P17 GEOMETRIC fields
    // (lens crop glitch) in the PINNED order, then the 4 P23 360 REFRAME fields
    // (eq360 eq_yaw eq_pitch eq_fov) in the PINNED order, then the 7 P34 SHAPE MASK fields
    // (mask_shape mask_cx mask_cy mask_rw mask_rh mask_feather mask_invert) in the PINNED order, then
    // the 1 P37 CHROMA-SPILL field (ck_spill — APPENDED AS A LATE PAYLOAD FIELD so it does NOT shift
    // any existing ck_* / mask index), then the 3 P38 DISTORT fields
    // (mirror_x kaleido dither — INSERTED between ck_spill and out so they do NOT shift any existing
    // index), then the 3 P39 SELECTIVE-COLOR fields
    // (sel_band sel_hshift sel_sat — INSERTED between dither and out so they do NOT shift any existing
    // index), then the 2 P41 SOLARIZE / COLOUR-TEMPERATURE fields
    // (sol_thr temp — INSERTED between sel_sat and out so they do NOT shift any existing index), then
    // the out
    // path. PREVIEW token count = 103 (the PREVIEW keyword + 102 fields, last = out; P39 was 101, P38 98).
    // P31: the new V2-overlay BLEND token rides at f[5] (right after op f[4]), shifting every later
    // field +1 vs P23. P34: the 7 shape-mask tokens ride at f[85..=91], between eq_fov f[84] and the
    // out path. P37: the 1 ck_spill token is appended at f[92] (after mask_invert f[91]). P38: the 3
    // distort tokens ride at f[93..=95] (after ck_spill f[92]). P39: the 3 selective-color tokens ride
    // at f[96..=98] (after dither f[95], BEFORE the out path which moves from f[96] to f[99]). P41: the
    // 2 solarize/temperature tokens ride at f[99..=100] (after sel_sat f[98], BEFORE the out path which
    // moves from f[99] to f[101]). After
    // the worker strips the PREVIEW keyword the engine reads 102 fields:
    // base f[0], over f[1], bf f[2], of f[3], op f[4], blend f[5], px f[6], py f[7], pw f[8], ph f[9],
    // ... curve at f[42..=46], vig f[47], sharp f[48], flip f[49], fx f[50], hue f[51], sat f[52],
    // light f[53], inb f[54], inw f[55], gam f[56], mosaic f[57], gmap_amt f[58], glo f[59..=61],
    // ghi f[62..=64], denoise f[65], glow_amt f[66], glow_thr f[67], rgbshift f[68], halftone f[69],
    // emboss f[70], edge f[71], grain f[72], scratches f[73], diffusion f[74], wave f[75], swirl
    // f[76], threshold f[77], lens f[78], crop f[79], glitch f[80], eq360 f[81], eq_yaw f[82],
    // eq_pitch f[83], eq_fov f[84], mask_shape f[85], mask_cx f[86], mask_cy f[87], mask_rw f[88],
    // mask_rh f[89], mask_feather f[90], mask_invert f[91], ck_spill f[92], mirror_x f[93],
    // kaleido f[94], dither f[95], sel_band f[96], sel_hshift f[97], sel_sat f[98], sol_thr f[99],
    // temp f[100], out f[101]. blend
    // is emitted
    // as an INTEGER token (0=Normal, 1=Multiply..7=Difference; engine parses i32). blend 0 makes the
    // engine do a plain alpha-over so the frame is byte-identical to pre-P31. eq360 is also an INTEGER
    // token (1 = on, 0 = off; engine parses i32, nonzero = on); eq360 0 returns immediately (no kernel
    // run), byte-identical to pre-P23. mask_shape and mask_invert are INTEGER tokens (engine parses
    // i32); mask_shape 0 (none) returns immediately (no kernel run), byte-identical to pre-P34. ck_spill
    // is a plain f32 token; ck_spill 0 (or ck_on 0) skips the spill pass inside k_chroma, byte-identical
    // to pre-P37. mirror_x and kaleido are INTEGER tokens (engine parses i32), dither is a plain f32;
    // mirror_x 0 / kaleido <2 / dither 0 → engine no-ops each distort kernel → byte-identical to pre-P38.
    // sel_band is an INTEGER token (engine parses i32; 0..6), sel_hshift / sel_sat are plain f32;
    // sel_band 0 (none) → engine no-ops the selective-color kernel → byte-identical to pre-P39. NOTE
    // sel_sat's NEUTRAL value is 1.0 (a saturation MULTIPLIER), NOT 0.0.
    // P41: the 2 solarize / colour-temperature tokens `sol_thr temp` (PINNED order) ride at
    // f[99..=100], INSERTED between sel_sat f[98] and the out path (which moves from f[99] to f[101]).
    // Both are plain f32; sol_thr 0.0 (off) and temp 0.0 (neutral) → caller skips each kernel →
    // byte-identical to pre-P41. PREVIEW post-strip arity 100 → 102 (PREVIEW keyword + 102 fields,
    // last = out).
    format!(
        "PREVIEW {base} {over} {bf} {of} {op} {blend} {px} {py} {pw} {ph} {b} {c} {s} {lk} {la} {lut} \
         {tk} {tp} {tparam} {tpath} {tframe} {cb} {cc} {cs} \
         {lr} {lg} {lb} {gmr} {gmg} {gmb} {gnr} {gng} {gnb} {rot} {scl} {blr} \
         {ckon} {ckr} {ckg} {ckb} {cksim} {cksm} {cv0} {cv1} {cv2} {cv3} {cv4} \
         {vig} {sharp} {flip} {fx} {hue} {sat} {light} {inb} {inw} {gam} \
         {mosaic} {gmapamt} {glor} {glog} {glob} {ghir} {ghig} {ghib} \
         {denoise} {glowamt} {glowthr} {rgbshift} {halftone} {emboss} {edge} \
         {grain} {scratches} {diffusion} {wave} {swirl} {threshold} \
         {lens} {crop} {glitch} {eq360} {eqyaw} {eqpitch} {eqfov} \
         {maskshape} {maskcx} {maskcy} {maskrw} {maskrh} {maskfeather} {maskinvert} {ckspill} \
         {mirrorx} {kaleido} {dither} {selband} {selhshift} {selsat} {solthr} {temp} {fade} {out}",
        base = enc_path(&r.base_path),
        over = enc_path(&r.over_path),
        bf = r.base_frame,
        of = r.over_frame,
        op = r.op,
        // P31 BLEND: the V2 overlay's blend mode as ONE integer token IMMEDIATELY AFTER `op` — the
        // engine's k_pip parser reads it from this exact position. 0 (Normal) => plain alpha-over =>
        // byte-identical to pre-P31.
        blend = r.blend,
        px = r.px,
        py = r.py,
        pw = r.pw,
        ph = r.ph,
        b = r.bright,
        c = r.contrast,
        s = r.sat,
        lk = r.look_kind,
        la = r.look_amt,
        lut = enc_path(&r.lut_path),
        tk = r.trans_kind,
        tp = r.trans_prog,
        tparam = r.trans_param,
        tpath = enc_path(&r.trans_path),
        tframe = r.trans_frame,
        cb = r.cbright,
        cc = r.ccontrast,
        cs = r.csat,
        lr = r.lift[0],
        lg = r.lift[1],
        lb = r.lift[2],
        gmr = r.gamma[0],
        gmg = r.gamma[1],
        gmb = r.gamma[2],
        gnr = r.gain_rgb[0],
        gng = r.gain_rgb[1],
        gnb = r.gain_rgb[2],
        rot = r.rot,
        scl = r.scale,
        blr = r.blur,
        ckon = r.ck_on,
        ckr = r.ck_key[0],
        ckg = r.ck_key[1],
        ckb = r.ck_key[2],
        cksim = r.ck_sim,
        cksm = r.ck_smooth,
        cv0 = r.curve[0],
        cv1 = r.curve[1],
        cv2 = r.curve[2],
        cv3 = r.curve[3],
        cv4 = r.curve[4],
        vig = r.vignette,
        sharp = r.sharpen,
        flip = r.flip,
        fx = r.fx,
        hue = r.hsl[0],
        sat = r.hsl[1],
        light = r.hsl[2],
        inb = r.levels[0],
        inw = r.levels[1],
        gam = r.levels[2],
        mosaic = r.mosaic,
        gmapamt = r.gmap_amt,
        glor = r.gmap_lo[0],
        glog = r.gmap_lo[1],
        glob = r.gmap_lo[2],
        ghir = r.gmap_hi[0],
        ghig = r.gmap_hi[1],
        ghib = r.gmap_hi[2],
        denoise = r.denoise,
        glowamt = r.glow_amt,
        glowthr = r.glow_thr,
        rgbshift = r.rgbshift,
        halftone = r.halftone,
        emboss = r.emboss,
        edge = r.edge,
        grain = r.grain,
        scratches = r.scratches,
        diffusion = r.diffusion,
        wave = r.wave,
        swirl = r.swirl,
        threshold = r.threshold,
        lens = r.lens,
        crop = r.crop,
        glitch = r.glitch,
        // P23 360 reframe: eq360 emitted as an INTEGER flag token (1 = on, 0 = off) to match the
        // engine's i32 parse (NOT a bool literal "true"/"false"); yaw/pitch/fov as plain f32 with the
        // same Display formatting as the neighbouring lens/crop/glitch fields.
        eq360 = if r.eq360 { 1 } else { 0 },
        eqyaw = r.eq_yaw,
        eqpitch = r.eq_pitch,
        eqfov = r.eq_fov,
        // P34 shape mask: mask_shape emitted as an INTEGER token (0 = none/off, 1 = rectangle,
        // 2 = ellipse) and mask_invert as an INTEGER flag (1 = invert, 0 = normal) to match the
        // engine's i32 parse (NOT bool literals "true"/"false"); cx/cy/rw/rh/feather as plain f32 with
        // the same Display formatting as the neighbouring eq_yaw/eq_pitch/eq_fov fields. mask_shape 0
        // → engine no-op → byte-identical to pre-P34. These 7 fields sit BEFORE the out path (out
        // stays LAST on the PREVIEW line).
        maskshape = r.mask_shape,
        maskcx = r.mask_cx,
        maskcy = r.mask_cy,
        maskrw = r.mask_rw,
        maskrh = r.mask_rh,
        maskfeather = r.mask_feather,
        maskinvert = r.mask_invert,
        // P37 chroma SPILL suppression: a plain f32 token APPENDED AS THE LAST PAYLOAD FIELD (after
        // mask_invert, BEFORE the out path — out stays LAST on the PREVIEW line). Appending it here
        // (rather than inserting among the ck_* fields) keeps every existing wire index unchanged.
        // ck_spill 0 (or ck_on 0) → k_chroma skips the spill pass → byte-identical to pre-P37.
        ckspill = r.ck_spill,
        // P38 DISTORT (MIRROR + KALEIDOSCOPE + DITHER): 3 tokens APPENDED AFTER ck_spill and BEFORE the
        // out path (out stays LAST on the PREVIEW line) in the PINNED order `mirror_x kaleido dither`.
        // mirror_x and kaleido are INTEGER tokens (engine parses i32); dither is a plain f32. mirror_x 0
        // / kaleido <2 / dither 0 → engine no-ops each kernel → byte-identical to pre-P38. Appending
        // them here keeps every existing wire index unchanged (post-strip arity 94 → 97).
        mirrorx = r.mirror_x,
        kaleido = r.kaleido,
        dither = r.dither,
        // P39 SELECTIVE COLOR (one HUE BAND): 3 tokens INSERTED AFTER the P38 dither field and BEFORE
        // the out path (out stays LAST on the PREVIEW line) in the PINNED order
        // `sel_band sel_hshift sel_sat`. sel_band is an INTEGER token (engine parses i32; 0..6);
        // sel_hshift / sel_sat are plain f32. sel_band 0 (none) → engine no-ops the kernel →
        // byte-identical to pre-P39. NOTE sel_sat's NEUTRAL value is 1.0 (a saturation MULTIPLIER), so
        // a neutral clip emits `... <dither> 0 0 1 <out>` (NOT sel_sat 0, which would desaturate the
        // band). Inserting them here keeps every existing wire index unchanged (post-strip arity
        // 97 → 100).
        selband = r.sel_band,
        selhshift = r.sel_hshift,
        selsat = r.sel_sat,
        // P41 SOLARIZE + COLOUR TEMPERATURE: 2 tokens INSERTED AFTER the P39 sel_sat field and BEFORE
        // the out path (out stays LAST on the PREVIEW line) in the PINNED order `sol_thr temp`
        // (sol_thr first, temp second). Both are plain f32. sol_thr 0.0 (off) → caller skips the
        // solarize kernel; temp 0.0 (neutral) → caller skips the temperature kernel → byte-identical
        // to pre-P41. Inserting them here keeps every existing wire index unchanged and keeps `out`
        // as the trailing field (post-strip arity 100 → 102).
        solthr = r.sol_thr,
        temp = r.temp,
        fade = r.fade,
        // The out token is a Genesis-chosen /tmp path (no whitespace) → enc_path is identity here;
        // wrapped for symmetry with the engine's dec_path on the trailing field (also identity).
        out = enc_path(out),
    )
}

/// P5 STAGE 2: the visible VIDEO clips covering frame `t`, one per video track, ASCENDING track order
/// (bottom -> top). `[0]` is the base, `[1]` the first overlay (both composited by `build_request`/
/// `build_enc_line` with full effects); `[2..]` are the EXTRA layers folded over the accumulated
/// composite via the RAW: layer path.
fn visible_video_clips(project: &Project, t: i64) -> Vec<usize> {
    let mut per_track: Vec<(u8, usize)> = Vec::new();
    for (i, c) in project.clips.iter().enumerate() {
        if t >= c.t0 && t < c.end() && !project.is_audio(c.track) && !project.is_hidden(c.track) {
            match per_track.iter_mut().find(|(tr, _)| *tr == c.track) {
                Some(slot) => slot.1 = i, // last covering clip on a track wins (transition overlap)
                None => per_track.push((c.track, i)),
            }
        }
    }
    per_track.sort_by_key(|(tr, _)| *tr);
    per_track.into_iter().map(|(_, i)| i).collect()
}

/// P5 STAGE 2: a `Resolved` that composites ONE extra video layer (`idx`) over a RAW-RGBA base
/// (`base_raw` = the accumulated lower layers). Applies ONLY the layer's pip rect (keyframed) + full
/// opacity + its chroma key — every colour/transform/look/transition field is IDENTITY, so the
/// already-composited lower layers keep their baked grade while this layer composites positionally +
/// keyed on top (k_pip's `over.a*op` blend). Reuses `format_preview`, so the wire is built by the
/// SAME formatter as a normal frame (no hand-typed line) — the lesson from the reverted attempt.
fn build_layer_resolved(project: &Project, t: i64, base_raw: &str, idx: usize) -> Option<Resolved> {
    let c = project.clips.get(idx)?;
    let path = project.media.get(c.media)?;
    // Whitespace is now WIRE-SAFE (format_preview percent-encodes the over path via enc_path; the
    // engine decodes it), so the old single-token reject is removed — only an EMPTY path drops the
    // layer.
    if path.is_empty() {
        return None;
    }
    let frame = src_frame_at(c, t).max(0) as i32;
    let (px, py, pw, ph) = project.pip_at(idx, t - c.t0);
    let ck = &c.chroma;
    // P37: ck_spill sourced from THIS layer's `chroma.spill` (mirrors ck_sim/ck_smooth); disabled → 0.0
    // (off), so a no-spill / disabled-chroma layer composites byte-identically.
    let (ck_on, ck_key, ck_sim, ck_smooth, ck_spill) = if ck.enabled {
        (1, ck.key, ck.similarity, ck.smoothness, ck.spill)
    } else {
        (0, [0.0, 1.0, 0.0], 0.4, 0.1, 0.0)
    };
    Some(Resolved {
        base_path: format!("RAW:{base_raw}"),
        base_frame: 0,
        over_path: path.clone(),
        over_frame: frame,
        op: 1.0,
        // P31 IDENTITY: an N-layer extra layer composites with plain alpha-over (k_pip's over.a*op),
        // so blend 0 (Normal) keeps the fold byte-identical to the pre-P31 layer pipeline.
        blend: 0,
        px,
        py,
        pw,
        ph,
        bright: 0.0,
        contrast: 1.0,
        sat: 1.0,
        cbright: 0.0,
        ccontrast: 1.0,
        csat: 1.0,
        look_kind: 0,
        look_amt: 1.0,
        lut_path: "-".to_string(),
        trans_kind: -1,
        trans_prog: 0.0,
        trans_param: 4.0,
        trans_path: "-".to_string(),
        trans_frame: 0,
        lift: [0.0, 0.0, 0.0],
        gamma: [1.0, 1.0, 1.0],
        gain_rgb: [1.0, 1.0, 1.0],
        rot: 0.0,
        scale: 1.0,
        blur: 0.0,
        ck_on,
        ck_key,
        ck_sim,
        ck_smooth,
        ck_spill,
        curve: [0.0, 0.25, 0.5, 0.75, 1.0],
        vignette: 0.0,
        sharpen: 0.0,
        flip: 0,
        fx: 0,
        hsl: [0.0, 1.0, 0.0],
        levels: [0.0, 1.0, 1.0],
        mosaic: 0,
        gmap_amt: 0.0,
        gmap_lo: [0.0, 0.0, 0.0],
        gmap_hi: [1.0, 1.0, 1.0],
        denoise: 0.0,
        glow_amt: 0.0,
        glow_thr: 0.7,
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
        // P23 IDENTITY: eq360 false makes the 360 reframe a byte-exact no-op (engine skips the kernel).
        eq360: false,
        eq_yaw: 0.0,
        eq_pitch: 0.0,
        eq_fov: 90.0,
        // P34 IDENTITY: mask_shape 0 (none) makes the shape mask a byte-exact no-op (engine skips the
        // kernel), so an extra layer composites positionally + keyed with no masking.
        mask_shape: 0,
        mask_cx: 0.5,
        mask_cy: 0.5,
        mask_rw: 0.5,
        mask_rh: 0.5,
        mask_feather: 0.0,
        mask_invert: 0,
        // P38 IDENTITY: mirror_x 0 / kaleido 0 / dither 0 make the distort filters a byte-exact no-op
        // (engine skips each kernel), so an extra layer composites positionally + keyed with no
        // distortion.
        mirror_x: 0,
        kaleido: 0,
        dither: 0.0,
        // P39 IDENTITY: sel_band 0 (none) makes the selective-color filter a byte-exact no-op (engine
        // skips the kernel), so an extra layer composites with no band grading. sel_sat's NEUTRAL is
        // 1.0 (a saturation MULTIPLIER), NOT 0.0.
        sel_band: 0,
        sel_hshift: 0.0,
        sel_sat: 1.0,
        // P41 IDENTITY: sol_thr 0.0 (solarize OFF) + temp 0.0 (neutral) make both filters byte-exact
        // no-ops (caller skips each kernel), so an extra layer composites with no solarize / colour-
        // temperature shift.
        sol_thr: 0.0,
        temp: 0.0,
        fade: 1.0,
    })
}

/// P5 STAGE 2: assemble the PREVIEW line sequence folding >2 video layers into `final_out`. Step 0 =
/// the full base+over composite (`build_request`, redirected to a temp); each extra layer is then
/// pip'd over the accumulated temp, ping-ponging two temp files, the LAST writing `final_out`.
fn build_layer_pipeline(project: &Project, t: i64, layers: &[usize], final_out: &str) -> Option<Vec<String>> {
    const TMP: [&str; 2] = ["/tmp/genesis_layer0.rgba", "/tmp/genesis_layer1.rgba"];
    let mut lines = Vec::new();
    let base = resolve_frame(project, t)?;
    let first_out = if layers.len() == 2 { final_out } else { TMP[0] };
    lines.push(format_preview(&base, first_out));
    let extras = &layers[2..];
    let mut cur = 0usize;
    for (k, &idx) in extras.iter().enumerate() {
        let last = k == extras.len() - 1;
        let prev = if k == 0 { TMP[0] } else { TMP[cur] };
        let out = if last { final_out } else { TMP[1 - cur] };
        let r = build_layer_resolved(project, t, prev, idx)?;
        lines.push(format_preview(&r, out));
        if !last {
            cur = 1 - cur;
        }
    }
    Some(lines)
}

/// P48 — the temp the program composite is written to when a subtitle is active over a SINGLE-ENC
/// (<=2-layer) frame: instead of encoding the composite directly, it is composited to this RGBA
/// (a PREVIEW line), the caption is folded over it into `SUB_FOLD`, and `SUB_FOLD` is encoded. Kept
/// distinct from the `RENDER_FOLD` / layer temps so a subtitle-over-fold frame's last fold output is
/// not clobbered by the subtitle step.
const SUB_BASE: &str = "/tmp/genesis_sub_base.rgba";
/// P48 — the temp holding the program composite WITH the caption folded on top (the buffer the final
/// `build_enc_raw` encodes for a subtitle-active render frame).
const SUB_FOLD: &str = "/tmp/genesis_sub_fold.rgba";

/// P48 — build the RENDER line list for ONE output timeline frame `t`, overlaying the active subtitle
/// (if any) as the TOP-MOST layer. This is the SINGLE new render code path; it wraps the EXISTING
/// per-frame builders so a no-subtitle frame stays byte-identical.
///
/// `prog_lines` is the per-frame line list the existing code already produced for the PROGRAM (the
/// single `vec![enc_line]` for <=2 layers, or the >2-layer fold + `build_enc_raw(RENDER_FOLD)`), and
/// `prog_fold_out` is the RGBA temp that fold wrote to (`Some(RENDER_FOLD)` for the >2-layer case,
/// `None` for the single-ENC case — there is no intermediate RGBA there).
///
/// When `project.active_subtitle_at(t)` is Some AND its text rasterizes (`rasterize_subtitle`):
///   - >2-layer fold frame: the program composite already lives in `RENDER_FOLD` (the fold's final
///     PREVIEW step). We DROP the program's trailing `build_enc_raw(RENDER_FOLD)`, keep every
///     compositing PREVIEW line, then append: a PREVIEW that composites the caption RAW over
///     `RENDER_FOLD` into `SUB_FOLD` (`build_subtitle_resolved`), and `build_enc_raw(SUB_FOLD)`.
///   - single-ENC frame: there is no program RGBA temp, so we RE-EXPRESS the program composite as a
///     PREVIEW into `SUB_BASE` (the SAME `resolve_frame`, via `format_preview` — same pixels the ENC
///     would have produced), then a PREVIEW that composites the caption RAW over `SUB_BASE` into
///     `SUB_FOLD`, then `build_enc_raw(SUB_FOLD)`. The caption is the top layer either way.
///
/// When there is NO active subtitle, or the cue text does not rasterize (empty / no font), the
/// program's own `prog_lines` are returned UNCHANGED — byte-identical to pre-P48. The subtitle branch
/// is the ONLY new line shape.
fn build_frame_with_subtitle(
    project: &Project,
    t: i64,
    prog_lines: Vec<String>,
    prog_fold_out: Option<&str>,
) -> Vec<String> {
    // No active cue → existing path, untouched.
    let Some(sub) = project.active_subtitle_at(t) else {
        return prog_lines;
    };
    // Empty text / no font → composite the frame with no caption (never fail the frame).
    let Some(sub_raster) = rasterize_subtitle(&sub.text) else {
        return prog_lines;
    };

    match prog_fold_out {
        // >2-layer fold: the program composite is already in `fold_out`. Keep the fold's compositing
        // PREVIEW lines, drop its trailing `build_enc_raw(fold_out)`, fold the caption over it, encode.
        Some(fold_out) => {
            let mut lines = prog_lines;
            lines.pop(); // remove the program's final build_enc_raw(fold_out)
            let cap = build_subtitle_resolved(fold_out, &sub_raster);
            lines.push(format_preview(&cap, SUB_FOLD));
            lines.push(build_enc_raw(SUB_FOLD));
            lines
        }
        // Single-ENC (<=2 layers): re-express the program composite as a PREVIEW into SUB_BASE (same
        // resolve_frame → same pixels), fold the caption over it, encode the result.
        None => {
            // resolve_frame here matches what build_enc_line resolved for this frame; if it can't
            // resolve we fall back to the program's own ENC line (no caption) rather than failing.
            let Some(base) = resolve_frame(project, t) else {
                return prog_lines;
            };
            let cap = build_subtitle_resolved(SUB_BASE, &sub_raster);
            vec![
                format_preview(&base, SUB_BASE),
                format_preview(&cap, SUB_FOLD),
                build_enc_raw(SUB_FOLD),
            ]
        }
    }
}

/// P5 STAGE 2: run a sequence of worker command lines under ONE held lock (so no other compose can
/// interleave + overwrite a temp), retrying the WHOLE sequence from scratch on any failure (absorbs
/// the OpenCL-init flake like `command_with_restart`). Returns the bytes of `final_out`, or None.
fn run_pipeline(lines: &[String], final_out: &str) -> Option<Vec<u8>> {
    if lines.is_empty() {
        return None;
    }
    let slot = worker_slot();
    let mut guard = slot.lock().ok()?;
    for attempt in 0..MAX_ATTEMPTS {
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            let mut ok = true;
            for line in lines {
                if try_command(proc, line).is_none() {
                    ok = false;
                    break;
                }
            }
            if ok {
                return std::fs::read(final_out).ok();
            }
        }
        *guard = None;
        eprintln!("gcompose pipeline attempt {} failed; restarting worker", attempt + 1);
    }
    None
}

/// Program render framerate. Matches gcompose's OPEN default and MojoMedia's render config (30).
const RENDER_FPS: i32 = 30;

/// Render the whole program to `out_path` (mp4) via the persistent worker.
///
/// Sequence (mirrors MojoMedia's render loop, driven over the serve protocol):
///   OPEN <out> <out_w> <out_h> <fps_num> <fps_den> <rate_mode> <rate_value> <vcodec> <total_s> <gop> <preset> <abitrate>
///                                                  (config video [scaled to out WxH, gop/preset] + aac [abitrate], alloc audio accumulator)
///   for t in 0..total_frames:  ENC <resolved frame fields>   (composite + feed video encoder)
///   for each audible clip:  AUDIO <media> <src_in/FPS> <len/FPS> <t0/FPS> <gain> <fade_in_s> <fade_out_s> <clip_len_s> <range_local_s>
///   CLOSE                                          (drain accumulator -> encoder; write BOTH)
///
/// CONCURRENCY (finding #1): the ENTIRE OPEN→ENC*→AUDIO*→CLOSE sequence runs under ONE hold of the
/// worker mutex (`worker_slot().lock()`), driving the held `WorkerProc` via `try_command_status`
/// directly — it never re-enters the per-call `command_*` helpers (which each re-lock). This
/// guarantees that no concurrent worker consumer (an egui repaint calling `request_frame` /
/// `thumbnail`) can interleave PREVIEW/THUMB traffic on the worker's stdin/stdout *between* this
/// render's commands — which would otherwise let the render's `read_line` consume a preview's
/// `DONE` as its own ENC ack, or spawn a fresh encoder-less worker mid-render. The whole render is
/// now atomic with respect to the worker; concurrent callers simply block on the mutex until it
/// completes. The trade-off (the UI thread stalls for the render's duration if it shares the lock)
/// is acceptable for this slice — render is an explicit, blocking user action.
///
/// PROGRAM AUDIO (Slice A — TIMELINE-SYNCED): after all video frames are encoded, the program audio
/// is assembled timeline-positioned. For each AUDIBLE timeline clip, an AUDIO command tells the
/// worker to decode the clip's SOURCE range [src_in/FPS, (src_in+len)/FPS) and MIX it into the
/// worker's program-audio accumulator at dst_offset = t0/FPS seconds. The accumulator was sized to
/// the full timeline duration (total_s in OPEN), so on CLOSE the whole buffer is fed to the encoder
/// and the audio stream length MATCHES the video: a clip at t0=70 starts at 70/FPS s, gaps are
/// silence, and overlapping clips mix (sample-add, clamped). This replaces the old back-to-back
/// concatenation (which ignored t0 and made audio duration != video duration).
///
/// SILENT AUDIO-DROP CAVEAT (finding #6, INTEGRATOR NOTE): a render that returns `true` does NOT
/// guarantee an audio stream. On a minimal FFmpeg build with no aac encoder, OPEN's config_audio
/// fails NON-FATALLY in the worker (logged on the worker's stderr as "config_audio failed; rendering
/// video-only") and the render proceeds video-only — every AUDIO line still mixes into the
/// accumulator but CLOSE discards it. The serve protocol has no DONE field to report this back, so
/// `render_program` cannot distinguish a with-audio render from a video-only one; the only signal is
/// the worker's stderr. Acceptable for the minimal-FFmpeg fallback this slice, but the integrator
/// must not assume render-true ⇒ audio-present.
///
/// TRACK POLICY (NEW this slice — NOT a MojoMedia mirror): the AUDIBLE tracks are track 0 (V1, the
/// program base) and track 2 (A1, the dedicated audio track). Track 1 (V2 overlay) is INTENTIONALLY
/// SKIPPED — the assumption is its audio would duplicate the underlying V1 audio. NOTE for the
/// integrator: MojoMedia's own render audio assembly does NOT filter by track (it sums every
/// segment regardless of lane), so this V2-skip is a deliberate editorial decision here, and any
/// legitimately-distinct V2 audio is silently dropped — revisit if a different track model is wanted.
/// ADDITIONALLY (Slice A, pinned by the contract): any clip on a track for which
/// `project.is_muted(track)` is true contributes NO audio (the AUDIO line is not emitted) — that
/// mute-honoring IS required by the contract.
///
/// RETRY-FROM-SCRATCH (Slice A, finding #4 fixed): the WHOLE OPEN→ENC*→AUDIO*→CLOSE sequence is
/// wrapped in a retry loop (up to MAX_ATTEMPTS). A fresh OPEN supersedes any half-open encoder
/// (open_render drops the prior encoder), so re-running the entire render against a freshly spawned
/// worker is safe — this absorbs the intermittent (~10%) worker OpenCL-init/compose flake that used
/// to abort a long export with a partial/empty mp4. Only the worker-death shape (`CmdStatus::Broken`
/// mid-render, or a CLOSE that comes back Broken) triggers a respawn + full retry; a clean
/// `CmdStatus::Err` on OPEN or ENC (a genuine encoder/protocol error, worker still alive) is a clean
/// abort with NO retry (retrying a deterministic error would just burn attempts). Returns true only
/// when one full OPEN..CLOSE attempt completes.
///
/// Per attempt the worker is held under ONE lock for the whole OPEN→CLOSE sequence (finding #1): no
/// concurrent preview/thumbnail can interleave on the worker's pipes mid-render. On a worker death,
/// the dead proc is dropped + mark_spawn_fail'd and a NEW worker is spawned before the next attempt.
/// Keeps every other worker.rs API/behavior (request_frame, play_program, thumbnail, …) intact.
pub fn render_program(project: &Project, out_path: &str) -> bool {
    let total = project.total_frames();
    if total <= 0 {
        return false;
    }
    // T4 EXPORT IN/OUT REGION: render ONLY the marked sub-range [region_start, region_end). The
    // DEFAULT (export_in/out == -1/-1) yields (0, total) — the whole timeline — so a project with no
    // region is BYTE-IDENTICAL to pre-T4 (the loop is `region_start..region_end` == `0..total`, the
    // audio dst offsets shift by -0, and total_s is the whole timeline). When a valid region is set,
    // output frame 0 is `region_start` (the encoder emits a region-length mp4), and the audio
    // accumulator is sized to the REGION length with each clip's dst offset shifted by -region_start.
    let (region_start, region_end) = project.export_range(total);
    if region_end <= region_start {
        return false; // empty region (defensive; export_range never returns this for valid input)
    }

    // WHITESPACE PATHS ARE NOW SUPPORTED (finding #8 resolved): the ENC/AUDIO lines percent-encode
    // every path token via enc_path (worker.rs) and the engine decodes them via dec_path
    // (gcompose), so a media path containing a space no longer shifts the fixed-arity fields. The
    // old up-front abort that rejected any spaced media path is removed — the render now proceeds
    // and emits the encoded path. Space-free paths encode to themselves, so existing renders are
    // byte-identical.

    // Build every request line BEFORE taking the lock so a corrupt media index fails the render
    // up-front without ever opening an encoder (and without holding the worker mutex while we
    // touch the model). The lines are reused across retry attempts (the model is immutable here).
    // If any ENC line can't be resolved, abort before OPEN.
    // P5 STAGE 2: each frame is a LIST of worker lines. For <=2 video layers it is just the single ENC
    // (byte-identical). For >2 it is the fold sequence (PREVIEW composites writing a temp RGBA) ending
    // in an ENC that encodes the RAW folded composite — so the export shows ALL video layers.
    const RENDER_FOLD: &str = "/tmp/genesis_render_fold.rgba";
    // T4 REGION: emit ENC frames ONLY for the region [region_start, region_end). The encoder
    // receives them in order, so the FIRST ENC encodes output frame 0 == region_start (the mp4 is
    // region-length, not whole-timeline). The whole-timeline default has region_start==0 &&
    // region_end==total, so this is the same `0..total` pass as before (byte-identical).
    let region_len = (region_end - region_start) as usize;
    let mut enc_frames: Vec<Vec<String>> = Vec::with_capacity(region_len);
    for t in region_start..region_end {
        // P49 NESTED SEQUENCES: pre-render every visible COMPOUND clip's sub-sequence frame at this
        // output frame `t` into its deterministic `subseq_temp_path` BEFORE the frame's ENC/fold lines
        // are built, so resolve_frame's `RAW:` base/over swaps find the temps on disk. This runs here
        // in the line-BUILDING pass (NO worker lock held — `run_pipeline` takes the lock itself); the
        // actual encode happens later under one lock hold in `render_attempt`, by which point every
        // temp exists. A project with no compound clips returns immediately, so the per-frame line
        // shapes are BYTE-IDENTICAL to pre-P49.
        prerender_compound_clips(project, t);
        let layers = visible_video_clips(project, t);
        // P48 SUBTITLES: build the PROGRAM line list exactly as before, then route it through
        // `build_frame_with_subtitle`, which overlays the active caption as the TOP layer when one is
        // active at `t` and is a no-op (returns the same lines) otherwise. The no-subtitle path is
        // therefore byte-identical to pre-P48; the active-subtitle branch is the only new line shape.
        if layers.len() > 2 {
            match build_layer_pipeline(project, t, &layers, RENDER_FOLD) {
                Some(mut lines) => {
                    lines.push(build_enc_raw(RENDER_FOLD));
                    // fold frame: the program composite is in RENDER_FOLD; the caption folds over it.
                    enc_frames.push(build_frame_with_subtitle(project, t, lines, Some(RENDER_FOLD)));
                }
                None => return false,
            }
        } else {
            match build_enc_line(project, t) {
                // single-ENC frame: no program RGBA temp (None) — the caption path re-expresses the
                // composite as a PREVIEW into SUB_BASE before folding the caption on top.
                Some(r) => enc_frames.push(build_frame_with_subtitle(project, t, vec![r], None)),
                None => return false, // corrupt media index: nothing opened yet, just bail.
            }
        }
    }
    // T4 REGION: the audio covers ONLY the region too. `build_audio_lines` shifts every clip's dst
    // offset by -region_start (so a clip at the region start lands at audio t=0) and drops clips that
    // fall entirely outside [region_start, region_end). With region_start==0 && region_end==total
    // (the default) the offset is -0 and no clip is dropped → the SAME AUDIO lines as before.
    let audio_lines = build_audio_lines(project, region_start, region_end);

    // Program-audio accumulator duration in seconds = the REGION length (sizes the worker buffer so
    // the rendered audio is exactly the output/region length, matching the region-length video).
    // Computed from the TIMELINE fps (RENDER_FPS=30, the rate clips are sampled), NOT the output fps:
    // the audio buffer is wall-clock seconds and each composed frame is stamped at its (region-local)
    // timeline time, so audio/video stay synced regardless of the chosen OUTPUT framerate. For the
    // whole-timeline default this is `total / RENDER_FPS` exactly (byte-identical). Same per attempt.
    let total_s = (region_end - region_start) as f64 / RENDER_FPS as f64;

    // EXPORT CONTROLS (Triad-B P1): the OPEN line now carries the output resolution, fps, rate mode +
    // value, and codec from `project.export` — decoupling the ENCODER dims from the fixed GVW×GVH
    // OpenCL working canvas (gcompose swscales the composed GVW×GVH frame to out_w×out_h). DEFAULTS
    // (1280×856 @ 30/1, mpeg4, 4 Mbit/s bitrate, rate_mode 0) reproduce today's behavior so existing
    // render gates pass unchanged. A vcodec containing whitespace would shift the fixed-arity OPEN
    // parse, so it is sanitized to the default here (codec names are single-token in practice).
    let ex = &project.export;
    let out_w = if ex.out_w == 0 { PVW as u32 } else { ex.out_w };
    let out_h = if ex.out_h == 0 { PVH as u32 } else { ex.out_h };
    let fps_num = if ex.fps_num == 0 { RENDER_FPS as u32 } else { ex.fps_num };
    let fps_den = if ex.fps_den == 0 { 1 } else { ex.fps_den };
    let rate_mode = if ex.rate_mode > 1 { 0 } else { ex.rate_mode };
    // In bitrate mode (0) a non-positive value falls back to the 4 Mbit/s default; in CRF mode (1) the
    // value IS the CRF (0 = lossless is legitimate), so it is passed through clamped to a sane range.
    let rate_value = if rate_mode == 1 {
        ex.rate_value.clamp(0, 51)
    } else if ex.rate_value > 0 {
        ex.rate_value
    } else {
        4_000_000
    };
    let vcodec = if ex.vcodec.split_whitespace().count() == 1 && !ex.vcodec.is_empty() {
        ex.vcodec.as_str()
    } else {
        "mpeg4"
    };
    // EXPORT DEPTH (Triad-B P25): three more encoder controls ride after total_s. DEFAULTS reproduce
    // today's render byte-for-byte: gop 0 leaves the encoder's gop_size untouched, preset "-" sets no
    // x264/x265 preset, abitrate 0 keeps gcompose's hardcoded 128000 audio bitrate. gop/abitrate are
    // clamped non-negative; preset is sanitized to a single token (mirrors vcodec above) and emitted as
    // "-" (NEVER an empty token) when blank, so the fixed-arity OPEN parse stays aligned.
    let gop = ex.gop.max(0);
    let preset_tok = if ex.preset.split_whitespace().count() == 1 && !ex.preset.is_empty() {
        ex.preset.as_str()
    } else {
        "-"
    };
    let abitrate = ex.abitrate.max(0);
    // P29 export depth: audio codec rides after abitrate. Default "-" => gcompose's hardcoded "aac"
    // (byte-identical to pre-P29). Sanitized to a single token (mirrors vcodec) so the fixed-arity
    // OPEN parse stays aligned; a blank/multi-word value degrades to "-".
    let acodec_tok = if ex.acodec.split_whitespace().count() == 1 && !ex.acodec.is_empty() {
        ex.acodec.as_str()
    } else {
        "-"
    };
    // OPEN <out> <out_w> <out_h> <fps_num> <fps_den> <rate_mode> <rate_value> <vcodec> <total_s> <gop> <preset> <abitrate> <acodec> (14 tokens, was 13).
    // WHITESPACE-SAFE WIRE: percent-encode the user-chosen render out path like every other path token
    // (base/over/lut/trans/THUMB/ENV/…). Without this an export destination containing a space (e.g.
    // "/home/alex/My Videos/out.mp4") split OPEN into >14 tokens → the engine's fixed-arity parser
    // rejected it (`bad OPEN`) and the whole export silently aborted. The engine dec_path's f[1].
    let out_enc = enc_path(out_path);
    let open_req = format!(
        "OPEN {out_enc} {out_w} {out_h} {fps_num} {fps_den} {rate_mode} {rate_value} {vcodec} {total_s} {gop} {preset_tok} {abitrate} {acodec_tok}"
    );

    // RETRY-FROM-SCRATCH loop: each iteration runs one full OPEN..CLOSE attempt. A worker-death
    // outcome (Retry) respawns and loops; a Success returns true; a clean Abort returns false.
    for attempt in 0..MAX_ATTEMPTS {
        match render_attempt(&open_req, &enc_frames, &audio_lines) {
            RenderOutcome::Success => return true,
            RenderOutcome::Abort => {
                // Deterministic error (bad OPEN, ENC/CLOSE encoder error) — retrying won't help.
                return false;
            }
            RenderOutcome::Retry => {
                // Worker died mid-render (the flake). render_attempt already dropped the dead proc
                // and armed the cooldown; spawn-from-scratch happens at the top of the next attempt.
                eprintln!(
                    "[render] worker died mid-render (attempt {} of {}); retrying whole render from a fresh OPEN",
                    attempt + 1,
                    MAX_ATTEMPTS
                );
            }
        }
    }
    eprintln!("[render] all {MAX_ATTEMPTS} render attempts hit a worker death; giving up");
    false
}

/// Outcome of one full `render_attempt` (one OPEN..CLOSE pass):
///   - `Success` : the whole sequence completed; the mp4 is written.
///   - `Abort`   : a CLEAN, deterministic failure (corrupt request, OPEN/ENC/CLOSE encoder error)
///                 with the worker STILL ALIVE — retrying the same render would just fail again.
///   - `Retry`   : the worker DIED mid-render (a `Broken` transport break, the ~10% flake). The dead
///                 proc has already been dropped + cooldown-armed; the caller should respawn + retry
///                 the whole render from a fresh OPEN (safe: a new OPEN supersedes any half encoder).
enum RenderOutcome {
    Success,
    Abort,
    Retry,
}

/// One full OPEN→ENC*→AUDIO*→CLOSE render pass under a single worker-lock hold (finding #1). All
/// request lines are pre-built and immutable, so this is safe to call repeatedly (retry-from-scratch,
/// finding #4): a fresh OPEN inside drops any prior half-open encoder on the worker side.
///
/// Failure mapping (drives the retry loop in `render_program`):
///   - OPEN  Broken  → drop+respawn-mark, return Retry  (worker died spawning/initialising: the flake)
///   - OPEN  Err     → return Abort                      (bad out path etc.; worker alive, no encoder)
///   - ENC   Broken  → abort_held (tear half encoder), return Retry  (worker died mid-video pass)
///   - ENC   Err     → abort_held, return Abort          (deterministic encoder error; worker alive)
///   - AUDIO Broken  → drop+mark, return Retry           (worker died feeding audio; encoder gone)
///   - AUDIO Err     → skip just this clip's audio, continue (worker alive)
///   - CLOSE Broken  → drop+mark, return Retry           (worker died finalising)
///   - CLOSE Err     → return Abort                      (encoder finish error; worker alive)
fn render_attempt(open_req: &str, enc_frames: &[Vec<String>], audio_lines: &[String]) -> RenderOutcome {
    // Acquire the worker for the WHOLE attempt (finding #1): one lock hold spanning OPEN→CLOSE, so
    // no concurrent preview/thumbnail can interleave on the worker's pipes mid-render.
    let slot = worker_slot();
    let mut guard = match slot.lock() {
        Ok(g) => g,
        // A poisoned worker mutex is a clean, non-retryable failure (the process is in trouble).
        Err(_) => return RenderOutcome::Abort,
    };

    // OPEN: ensure a live worker, then start the encoder. A respawn here is safe (no encoder in
    // flight yet). Distinguish a worker death (Broken/spawn-fail → Retry) from a deterministic OPEN
    // rejection (Err → Abort). We only try to (re)spawn ONCE inside an attempt; the retry-from-
    // scratch loop in render_program provides the outer attempts.
    if guard.is_none() {
        *guard = spawn_worker();
    }
    let open_status = match guard.as_mut() {
        Some(proc) => try_command_status(proc, open_req),
        None => {
            // Spawn failed outright: treat as a worker death so the caller retries (respawn next).
            *guard = None;
            mark_spawn_fail();
            return RenderOutcome::Retry;
        }
    };
    match open_status {
        CmdStatus::Done(_) => clear_spawn_cooldown(),
        CmdStatus::Err => {
            // OPEN rejected (e.g. bad out path) but the worker is alive: no encoder was created, so
            // there is nothing to tear down — and a retry would deterministically fail. Abort clean.
            return RenderOutcome::Abort;
        }
        CmdStatus::Broken => {
            // Worker died on OPEN (the init flake): drop it + arm cooldown, ask the caller to retry.
            *guard = None;
            mark_spawn_fail();
            return RenderOutcome::Retry;
        }
    }

    // From here an encoder is open: a Broken on any step means the worker died (encoder lost) → the
    // caller respawns + retries the WHOLE render; an Err on ENC/CLOSE is a deterministic encoder
    // error (worker alive) → tear the half-open encoder down and Abort (no retry).

    // ENC every frame, in order, on the held proc. P5 STAGE 2: each frame is a LIST of lines — for
    // >2 video layers the fold PREVIEW composites (writing temp RGBAs) precede the encoding ENC; for
    // <=2 layers it is just the single ENC. All lines run in order on the held worker.
    for frame_lines in enc_frames {
        for req in frame_lines {
        let status = match guard.as_mut() {
            Some(proc) => try_command_status(proc, req),
            None => CmdStatus::Broken, // proc vanished (a prior Broken cleared it): worker died.
        };
        match status {
            CmdStatus::Done(_) => {}
            CmdStatus::Err => {
                // Deterministic encoder error: tear the half-open encoder down NOW (finding #8) so
                // its partial mp4 is dropped immediately, then Abort (retry would just fail again).
                eprintln!("[render] ENC error (worker alive) at: {req}");
                abort_held(&mut *guard);
                return RenderOutcome::Abort;
            }
            CmdStatus::Broken => {
                // Worker died mid-video: try a best-effort teardown on whatever's left (usually a
                // no-op since the proc is gone), then ask the caller to respawn + retry from OPEN.
                eprintln!("[render] ENC worker-death at: {req}");
                abort_held(&mut *guard);
                return RenderOutcome::Retry;
            }
        }
        }
    }

    // PROGRAM AUDIO: feed each AUDIBLE clip's source-audio range, in t0 (timeline) order. A clip
    // with no audio / a decode skip (worker ERR) is dropped but the render continues; only a worker
    // death (Broken / vanished proc) aborts the attempt for retry. This uses MojoMedia's per-segment
    // fpx_decode_audio_range building block, but is NOT a 1:1 mirror of its assembly: MojoMedia's
    // render path concatenates EVERY segment's audio back-to-back with no track filtering, whereas
    // this path positions each clip at its timeline offset (dst_offset) AND applies the track-
    // audibility policy in build_audio_lines (track-2/track-0 only; track_mute honored).
    for line in audio_lines {
        let outcome = match guard.as_mut() {
            Some(proc) => try_command_status(proc, line),
            None => CmdStatus::Broken,
        };
        match outcome {
            CmdStatus::Done(_) => {}
            CmdStatus::Err => {} // worker alive; skip just this clip's audio and continue.
            CmdStatus::Broken => {
                // The worker died feeding audio: the encoder is gone. Drop the dead proc, arm the
                // cooldown, and ask the caller to retry the whole render from a fresh OPEN.
                *guard = None;
                mark_spawn_fail();
                return RenderOutcome::Retry;
            }
        }
    }

    // CLOSE — finish + close the encoder, flush + write BOTH the video and audio trailers.
    match guard.as_mut() {
        Some(proc) => match try_command_status(proc, "CLOSE") {
            CmdStatus::Done(_) => {
                clear_spawn_cooldown();
                RenderOutcome::Success
            }
            CmdStatus::Err => RenderOutcome::Abort, // encoder finish error; worker alive, no retry.
            CmdStatus::Broken => {
                // Worker died finalising: the trailer may be missing. Drop + retry from a fresh OPEN
                // (a re-render produces a clean, complete file rather than a truncated one).
                *guard = None;
                mark_spawn_fail();
                RenderOutcome::Retry
            }
        },
        None => RenderOutcome::Retry, // proc vanished before CLOSE: worker died, retry.
    }
}

/// Temp WAV the worker writes program audio to for playback (see `play_program`).
const PLAY_WAV: &str = "/tmp/genesis_play.wav";

/// Per-request output path for the program-audio LEVELS query (4 little-endian f32: peak_L, peak_R,
/// rms_L, rms_R, all in dBFS). Reused each call — `program_levels` holds the worker mutex across its
/// whole MEAS→AUDIO*→LEVELS round-trip, so there is never a concurrent writer.
const LEVELS_OUT: &str = "/tmp/genesis_levels.f32";

/// Per-request output path for the program-audio SPECTRUM query (`SPECTRUM_BINS` little-endian f32
/// magnitudes, linear over [0, sr/2]). Reused each call — `program_spectrum` holds the worker mutex
/// across its whole MEAS→AUDIO*→SPECTRUM round-trip, so there is never a concurrent writer. Mirrors
/// `LEVELS_OUT` for the level-meter path; the SPECTRUM path is read-only analysis and changes nothing
/// in the render/mix/LEVELS pipeline.
const SPECTRUM_OUT: &str = "/tmp/genesis_spectrum.f32";

/// Number of LINEAR frequency bins the front requests (and the engine therefore returns) for the
/// audio-spectrum scope. The engine writes EXACTLY this many little-endian f32 magnitudes to
/// `SPECTRUM_OUT`; bar `b` covers `[b·(sr/2)/nbins, (b+1)·(sr/2)/nbins)` with `sr = PROG_SR = 48000`,
/// i.e. 93.75 Hz/bar at 256 bins. The wire query the worker sends is `SPECTRUM <nbins> <out>` and the
/// engine returns `nbins` f32 — this const is the single agreed `nbins` on both sides.
pub const SPECTRUM_BINS: usize = 256;

/// Per-request output path for the program-audio SAMPLES query (`SAMPLES_N` little-endian f32 RAW
/// time-domain amplitudes, ~[-1,1]). Reused each call — `program_samples` holds the worker mutex
/// across its whole MEAS→AUDIO*→SAMPLES round-trip, so there is never a concurrent writer. Mirrors
/// `SPECTRUM_OUT` for the spectrum-scope path; the SAMPLES path is read-only analysis and changes
/// nothing in the render/mix/LEVELS/SPECTRUM pipeline.
const SAMPLES_OUT: &str = "/tmp/genesis_samples.f32";

/// Number of RAW time-domain points the front requests (and the engine therefore returns) for the
/// audio-waveform oscilloscope. The engine writes EXACTLY this many little-endian f32 amplitudes
/// (~[-1,1], LEFT channel, decimated over the accumulator window) to `SAMPLES_OUT`. The wire query
/// the worker sends is `SAMPLES <n> <out>` and the engine returns `n` f32 — this const is the single
/// agreed `n` on both sides.
pub const SAMPLES_N: usize = 256;

/// Floor (dBFS) the engine reports for digital silence (and the worst-case the UI meter draws). A
/// peak/RMS of 0 linear maps to this instead of −inf so the meter has a finite bottom of its scale.
pub const LEVELS_FLOOR_DB: f32 = -90.0;

/// Stereo program-audio levels (dBFS) of the assembled mix over a measurement window. `peak` is the
/// per-channel sample peak, `rms` the per-channel root-mean-square, both already in dBFS (0 dBFS =
/// full scale, `LEVELS_FLOOR_DB` = silence). Produced by `program_levels`; drawn by panels' meter.
#[derive(Clone, Copy)]
pub struct AudioLevels {
    pub peak_l: f32,
    pub peak_r: f32,
    pub rms_l: f32,
    pub rms_r: f32,
}

/// Measure the ASSEMBLED program-audio levels (peak + RMS, dBFS, per channel) over a short window of
/// the timeline starting at `start_frame` — the lightweight meter feed for panels.rs. Returns the
/// stereo `AudioLevels`, or None when there's nothing to measure (empty/past-end timeline) OR the
/// worker is busy (a contended `try_lock`, so the meter keeps its last reading rather than the UI
/// freezing — exactly like `scope`).
///
/// MECHANISM (no real-time device capture — we measure the ASSEMBLED mix, per the contract): run a
/// `MEAS <window_s>` / `AUDIO*` / `LEVELS <out>` session on the persistent worker. `MEAS` allocates a
/// playback-style accumulator (no encoder, no WAV) sized to a SHORT window; each AUDIO line mixes a
/// clip's filtered+gained range into it exactly like the render/playback path (so the meter reflects
/// the per-clip gain/fade/FX chain); `LEVELS` then computes peak+RMS over the accumulator, writes the
/// 4 f32 dBFS values to `<out>`, and clears the accumulator. The whole session runs under ONE held
/// mutex so no concurrent compose interleaves on the worker's pipes.
///
/// WINDOW: a fixed `LEVELS_WINDOW_S` slice from the playhead (not the whole tail) keeps the decode +
/// measure cheap enough to call at meter cadence — the meter shows the level "around the playhead".
///
/// NON-BLOCKING LOCK (finding #4 style): `try_lock`, called from the egui UI thread (panels), so a
/// background assembly/render holding the worker just yields None this frame.
pub fn program_levels(project: &Project, start_frame: i64) -> Option<AudioLevels> {
    /// Measurement window length (seconds) from the playhead. Short so the per-repaint decode is cheap.
    const LEVELS_WINDOW_S: f64 = 0.25;

    let total = project.total_frames();
    let start = start_frame.max(0);
    if total <= 0 || start >= total {
        return None; // nothing to measure at/after the timeline end.
    }
    let fps = RENDER_FPS as f64;
    let start_s = start as f64 / fps;
    // The window is the lesser of LEVELS_WINDOW_S and the remaining tail.
    let tail_s = (total - start) as f64 / fps;
    let window_s = LEVELS_WINDOW_S.min(tail_s).max(1.0 / fps);

    // Build the MEAS/AUDIO*/LEVELS lines on the CALLING thread (cheap, no decode/lock). dst offsets
    // are shifted so the playhead is t=0 in the measurement window; only clips overlapping the window
    // are emitted (a clip entirely before/after the window contributes nothing). This reuses the same
    // per-clip gain/fade/FX-chain build as playback so the meter reflects the audible mix.
    let meas_open = format!("MEAS {window_s}");
    let window_end = start_s + window_s;
    let mut audio_lines: Vec<String> = Vec::new();
    // P27: prepend the master gain envelope (element 0) so it is sent right after the MEAS opener and
    // before the AUDIO lines, so the meter reflects the gained mix (empty gain_kf → None → no change).
    if let Some(env) = build_gainenv_line(project) {
        audio_lines.push(env);
    }
    {
        let mut idx: Vec<usize> = (0..project.clips.len()).collect();
        idx.sort_by_key(|&i| project.clips[i].t0);
        for i in idx {
            let c = &project.clips[i];
            if c.len <= 0 || !track_is_audible(project, c.track) {
                continue;
            }
            if c.end() <= start {
                continue; // entirely in the past.
            }
            let clip_t0_s = c.t0 as f64 / fps;
            if clip_t0_s >= window_end {
                continue; // starts after the measurement window: nothing in range.
            }
            let media_path = match project.media.get(c.media) {
                Some(p) => p,
                None => continue,
            };
            // Whitespace is now WIRE-SAFE: the AUDIO line percent-encodes the path token (enc_path)
            // below and the engine decodes it, so a spaced path no longer shifts the fixed-arity
            // fields. The old whitespace skip is removed.
            // Head-trim the same way playback does (clip straddling the playhead plays from the
            // source frame under it), then clamp the decoded duration to the window so a long clip
            // doesn't decode its whole tail just to measure a 0.25 s window.
            let head_skip = (start - c.t0).max(0);
            // (P24: the sent source in-point is speed-scaled, computed after the window cap below; the
            // plain eff_src_in is no longer the sent value — `_`-prefixed to avoid an unused binding.)
            let _eff_src_in = c.src_in + head_skip;
            let mut eff_len = c.len - head_skip;
            if eff_len <= 0 {
                continue;
            }
            let dst_off_s = ((c.t0 + head_skip) as f64 / fps - start_s).max(0.0);
            // Cap the decoded range to what fits inside the window from this clip's dst offset.
            let max_len_frames = (((window_s - dst_off_s).max(0.0)) * fps).ceil() as i64;
            if max_len_frames <= 0 {
                continue;
            }
            eff_len = eff_len.min(max_len_frames);
            // P24 per-clip speed/reverse (Model A): retime the SOURCE window + filter chain so the
            // engine reads more/less source and the atempo/areverse compresses it back to the SAME
            // timeline window (eff_len/fps). speed==1.0 && !rev → identity (byte-identical line).
            let speed = (c.speed as f64).clamp(0.05, 16.0);
            let rev = c.reverse;
            // SOURCE seconds to read = timeline window seconds * speed (atempo compresses back).
            let dur_src_s = (eff_len as f64 * speed) / fps;
            if !(dur_src_s.is_finite()) || dur_src_s <= 0.0 {
                continue;
            }
            let src_in_s = if rev {
                // The window maps to source range [src_in + (len - head_skip - eff_len)*speed, ...),
                // read forward then areversed.
                let lo = c.src_in as f64 + ((c.len - head_skip - eff_len) as f64 * speed);
                lo.max(0.0) / fps
            } else {
                // Source start = src_in + head_skip*speed (the head trim scales with speed).
                let src0 = c.src_in as f64 + (head_skip as f64 * speed);
                src0.max(0.0) / fps
            };
            let dur_s = dur_src_s;
            // P42 MIXER GAIN: fold the per-TRACK fader into this clip's per-clip gain (the scalar that
            // rides the AUDIO line). track_gain default 1.0 → byte-identical for a default project.
            let gain = c.gain * project.track_gain(c.track);
            let fade_in_s = (c.fade_in.max(0)) as f64 / fps;
            let fade_out_s = (c.fade_out.max(0)) as f64 / fps;
            let clip_len_s = c.len as f64 / fps;
            let range_local_s = head_skip as f64 / fps;
            // P24: prepend areverse (if rev) and atempo stages (if speed != 1) BEFORE the AudioFx
            // chain. atempo only accepts [0.5,2.0], so a factor outside that range is staged. The
            // post-atempo buffer IS timeline-length, so fade/clip_len/range_local still line up.
            let base_fx = build_audio_chain(&c.audio_fx); // "-" or a comma chain, SPACE-FREE
            let mut pre: Vec<String> = Vec::new();
            if rev {
                pre.push("areverse".to_string());
            }
            if (speed - 1.0).abs() > 1e-6 {
                for f in atempo_factors(speed) {
                    pre.push(format!("atempo={:.6}", f));
                }
            }
            let fx_chain = if pre.is_empty() {
                base_fx
            } else {
                let mut all = pre;
                if base_fx != "-" {
                    all.push(base_fx);
                }
                all.join(",")
            };
            // P42 MIXER PAN: fold the per-TRACK L/R balance onto the clip's fx_chain (stays space-free,
            // wire arity unchanged). track_pan default 0.0 → chain returned unchanged → byte-identical.
            let fx_chain = apply_track_pan(fx_chain, project.track_pan(c.track));
            // WHITESPACE-SAFE WIRE: percent-encode the media path token (enc_path); the engine
            // dec_path's it before opening the decoder. Space-free paths are byte-identical.
            let media_path = enc_path(media_path);
            audio_lines.push(format!(
                "AUDIO {media_path} {src_in_s} {dur_s} {dst_off_s} {gain} {fade_in_s} {fade_out_s} {clip_len_s} {range_local_s} {fx_chain}"
            ));
        }
    }

    let slot = worker_slot();
    // try_lock (finding #4): never block the UI thread behind a background assembly / render. A
    // contended meter just returns None and the caller keeps its last reading.
    let mut guard = match slot.try_lock() {
        Ok(g) => g,
        Err(_) => return None,
    };

    for _attempt in 0..MAX_ATTEMPTS {
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            // 1) MEAS: open a measurement accumulator (no encoder, no WAV). A clean ERR (bad window)
            //    is non-retryable — bail. Broken falls through to the respawn below.
            match try_command_status(proc, &meas_open) {
                CmdStatus::Done(_) => {
                    // 2) Mix each clip's filtered+gained range (skip ERR clips; Broken aborts).
                    let mut broke = false;
                    for line in &audio_lines {
                        match try_command_status(proc, line) {
                            CmdStatus::Done(_) | CmdStatus::Err => {}
                            CmdStatus::Broken => {
                                broke = true;
                                break;
                            }
                        }
                    }
                    if !broke {
                        // 3) LEVELS: measure + write 4 f32 dBFS, clear the accumulator. Read back.
                        if let Some(levels) = read_levels(proc) {
                            clear_spawn_cooldown();
                            return Some(levels);
                        }
                    }
                }
                CmdStatus::Err => return None, // MEAS rejected (worker alive): nothing to measure.
                CmdStatus::Broken => {}        // worker died: respawn below.
            }
        }
        // MEAS/AUDIO/LEVELS broke mid-session: drop the worker so the next loop spawns clean.
        *guard = None;
        mark_spawn_fail();
    }
    None
}

/// Run the `LEVELS <out>` round-trip on an already-running worker, then read back the 4 little-endian
/// f32 (peak_L, peak_R, rms_L, rms_R; dBFS). Returns None on any failure (write error, EOF/timeout,
/// "ERR", a short/oversized read). The worker clears its measurement accumulator inside LEVELS.
fn read_levels(proc: &mut WorkerProc) -> Option<AudioLevels> {
    let req = format!("LEVELS {LEVELS_OUT}");
    match try_command_status(proc, &req) {
        CmdStatus::Done(payload) => {
            // The worker echoes the out path on DONE; trust our own path if it echoed empty.
            let read_path = if payload.is_empty() { LEVELS_OUT } else { payload.as_str() };
            let bytes = std::fs::read(read_path).ok()?;
            if bytes.len() != 16 {
                return None; // exactly 4 f32 expected.
            }
            let f = |i: usize| {
                f32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]])
            };
            Some(AudioLevels { peak_l: f(0), peak_r: f(4), rms_l: f(8), rms_r: f(12) })
        }
        CmdStatus::Err | CmdStatus::Broken => None,
    }
}

/// Measure the ASSEMBLED program-audio frequency SPECTRUM (`SPECTRUM_BINS` linear magnitudes over
/// [0, sr/2]) of the same short window of the timeline starting at `start_frame` that the level meter
/// measures — the feed for panels.rs' Audio Spectrum scope. Returns the per-bin magnitudes, or None
/// when there's nothing to measure (empty/past-end timeline) OR the worker is busy (a contended
/// `try_lock`, so the scope keeps its last reading rather than the UI freezing — exactly like
/// `program_levels`).
///
/// READ-ONLY ANALYSIS: this MIRRORS `program_levels` EXACTLY — same `MEAS <window_s>` open, same
/// `build_audio_lines` AUDIO* mix (the assembled, filtered+gained program mix), same persistent-worker
/// round-trip and error handling — but the terminating query is `SPECTRUM <nbins> <out>` (NOT LEVELS).
/// The engine computes `ProgAudio::spectrum(nbins)` over the accumulator, writes EXACTLY `nbins`
/// little-endian f32 magnitudes to `<out>`, then clears the accumulator (session terminator, like
/// LEVELS). It changes nothing in the render/mix/LEVELS pipeline.
///
/// MECHANISM: identical to `program_levels` — `MEAS` allocates a playback-style accumulator (no
/// encoder, no WAV) sized to a SHORT window from the playhead; each AUDIO line mixes a clip's
/// filtered+gained range into it exactly like the render/playback path; `SPECTRUM` then computes the
/// linear-binned magnitudes, writes them, and clears the accumulator. The whole session runs under ONE
/// held mutex so no concurrent compose interleaves on the worker's pipes.
///
/// NON-BLOCKING LOCK (finding #4 style): `try_lock`, called from the egui UI thread (panels), so a
/// background assembly/render holding the worker just yields None this frame.
pub fn program_spectrum(project: &Project, start_frame: i64) -> Option<Vec<f32>> {
    /// Measurement window length (seconds) from the playhead. Short so the per-repaint decode is cheap
    /// — the SAME window the level meter uses (mirrors `program_levels`).
    const SPECTRUM_WINDOW_S: f64 = 0.25;

    let total = project.total_frames();
    let start = start_frame.max(0);
    if total <= 0 || start >= total {
        return None; // nothing to measure at/after the timeline end.
    }
    let fps = RENDER_FPS as f64;
    let start_s = start as f64 / fps;
    // The window is the lesser of SPECTRUM_WINDOW_S and the remaining tail.
    let tail_s = (total - start) as f64 / fps;
    let window_s = SPECTRUM_WINDOW_S.min(tail_s).max(1.0 / fps);

    // Build the MEAS/AUDIO*/SPECTRUM lines on the CALLING thread (cheap, no decode/lock). dst offsets
    // are shifted so the playhead is t=0 in the measurement window; only clips overlapping the window
    // are emitted. This is the SAME windowed AUDIO* mix `program_levels` builds (per-clip
    // gain/fade/FX-chain + speed/reverse), so the spectrum reflects the audible mix. (Transcribed
    // verbatim from `program_levels` so the MEAS/AUDIO*/<terminator> lifecycle matches — only the
    // terminating query differs: SPECTRUM, not LEVELS.)
    let meas_open = format!("MEAS {window_s}");
    let window_end = start_s + window_s;
    let mut audio_lines: Vec<String> = Vec::new();
    // P27: prepend the master gain envelope (element 0) so it is sent right after the MEAS opener and
    // before the AUDIO lines, so the spectrum reflects the gained mix (empty gain_kf → None → no change).
    if let Some(env) = build_gainenv_line(project) {
        audio_lines.push(env);
    }
    {
        let mut idx: Vec<usize> = (0..project.clips.len()).collect();
        idx.sort_by_key(|&i| project.clips[i].t0);
        for i in idx {
            let c = &project.clips[i];
            if c.len <= 0 || !track_is_audible(project, c.track) {
                continue;
            }
            if c.end() <= start {
                continue; // entirely in the past.
            }
            let clip_t0_s = c.t0 as f64 / fps;
            if clip_t0_s >= window_end {
                continue; // starts after the measurement window: nothing in range.
            }
            let media_path = match project.media.get(c.media) {
                Some(p) => p,
                None => continue,
            };
            let head_skip = (start - c.t0).max(0);
            let _eff_src_in = c.src_in + head_skip;
            let mut eff_len = c.len - head_skip;
            if eff_len <= 0 {
                continue;
            }
            let dst_off_s = ((c.t0 + head_skip) as f64 / fps - start_s).max(0.0);
            let max_len_frames = (((window_s - dst_off_s).max(0.0)) * fps).ceil() as i64;
            if max_len_frames <= 0 {
                continue;
            }
            eff_len = eff_len.min(max_len_frames);
            let speed = (c.speed as f64).clamp(0.05, 16.0);
            let rev = c.reverse;
            let dur_src_s = (eff_len as f64 * speed) / fps;
            if !(dur_src_s.is_finite()) || dur_src_s <= 0.0 {
                continue;
            }
            let src_in_s = if rev {
                let lo = c.src_in as f64 + ((c.len - head_skip - eff_len) as f64 * speed);
                lo.max(0.0) / fps
            } else {
                let src0 = c.src_in as f64 + (head_skip as f64 * speed);
                src0.max(0.0) / fps
            };
            let dur_s = dur_src_s;
            // P42 MIXER GAIN: fold the per-TRACK fader into this clip's per-clip gain (the scalar that
            // rides the AUDIO line). track_gain default 1.0 → byte-identical for a default project.
            let gain = c.gain * project.track_gain(c.track);
            let fade_in_s = (c.fade_in.max(0)) as f64 / fps;
            let fade_out_s = (c.fade_out.max(0)) as f64 / fps;
            let clip_len_s = c.len as f64 / fps;
            let range_local_s = head_skip as f64 / fps;
            let base_fx = build_audio_chain(&c.audio_fx); // "-" or a comma chain, SPACE-FREE
            let mut pre: Vec<String> = Vec::new();
            if rev {
                pre.push("areverse".to_string());
            }
            if (speed - 1.0).abs() > 1e-6 {
                for f in atempo_factors(speed) {
                    pre.push(format!("atempo={:.6}", f));
                }
            }
            let fx_chain = if pre.is_empty() {
                base_fx
            } else {
                let mut all = pre;
                if base_fx != "-" {
                    all.push(base_fx);
                }
                all.join(",")
            };
            // P42 MIXER PAN: fold the per-TRACK L/R balance onto the clip's fx_chain (stays space-free,
            // wire arity unchanged). track_pan default 0.0 → chain returned unchanged → byte-identical.
            let fx_chain = apply_track_pan(fx_chain, project.track_pan(c.track));
            let media_path = enc_path(media_path);
            audio_lines.push(format!(
                "AUDIO {media_path} {src_in_s} {dur_s} {dst_off_s} {gain} {fade_in_s} {fade_out_s} {clip_len_s} {range_local_s} {fx_chain}"
            ));
        }
    }

    let slot = worker_slot();
    // try_lock (finding #4): never block the UI thread behind a background assembly / render. A
    // contended scope just returns None and the caller keeps its last reading.
    let mut guard = match slot.try_lock() {
        Ok(g) => g,
        Err(_) => return None,
    };

    for _attempt in 0..MAX_ATTEMPTS {
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            // 1) MEAS: open a measurement accumulator (no encoder, no WAV). A clean ERR (bad window)
            //    is non-retryable — bail. Broken falls through to the respawn below.
            match try_command_status(proc, &meas_open) {
                CmdStatus::Done(_) => {
                    // 2) Mix each clip's filtered+gained range (skip ERR clips; Broken aborts).
                    let mut broke = false;
                    for line in &audio_lines {
                        match try_command_status(proc, line) {
                            CmdStatus::Done(_) | CmdStatus::Err => {}
                            CmdStatus::Broken => {
                                broke = true;
                                break;
                            }
                        }
                    }
                    if !broke {
                        // 3) SPECTRUM: compute SPECTRUM_BINS magnitudes + write, clear the
                        //    accumulator. Read back.
                        if let Some(bins) = read_spectrum(proc) {
                            clear_spawn_cooldown();
                            return Some(bins);
                        }
                    }
                }
                CmdStatus::Err => return None, // MEAS rejected (worker alive): nothing to measure.
                CmdStatus::Broken => {}        // worker died: respawn below.
            }
        }
        // MEAS/AUDIO/SPECTRUM broke mid-session: drop the worker so the next loop spawns clean.
        *guard = None;
        mark_spawn_fail();
    }
    None
}

/// Run the `SPECTRUM <nbins> <out>` round-trip on an already-running worker, then read back EXACTLY
/// `SPECTRUM_BINS` little-endian f32 magnitudes (linear over [0, sr/2]). Returns None on any failure
/// (write error, EOF/timeout, "ERR", a short/oversized read). The worker clears its measurement
/// accumulator inside SPECTRUM. The wire query is `SPECTRUM <SPECTRUM_BINS> <SPECTRUM_OUT>` and the
/// engine returns `SPECTRUM_BINS` f32 — a count/format mismatch would read garbage, so the byte length
/// is checked to be EXACTLY `SPECTRUM_BINS * 4`.
fn read_spectrum(proc: &mut WorkerProc) -> Option<Vec<f32>> {
    let req = format!("SPECTRUM {SPECTRUM_BINS} {SPECTRUM_OUT}");
    match try_command_status(proc, &req) {
        CmdStatus::Done(payload) => {
            // The worker echoes the out path on DONE; trust our own path if it echoed empty.
            let read_path = if payload.is_empty() { SPECTRUM_OUT } else { payload.as_str() };
            let bytes = std::fs::read(read_path).ok()?;
            if bytes.len() != SPECTRUM_BINS * 4 {
                return None; // exactly SPECTRUM_BINS f32 expected.
            }
            let bins: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Some(bins)
        }
        CmdStatus::Err | CmdStatus::Broken => None,
    }
}

/// Measure the ASSEMBLED program-audio time-domain WAVEFORM (`SAMPLES_N` raw amplitudes ~[-1,1]) of
/// the same short window of the timeline starting at `start_frame` that the level meter / spectrum
/// measures — the feed for panels.rs' Audio Waveform oscilloscope. Returns the per-point samples, or
/// None when there's nothing to measure (empty/past-end timeline) OR the worker is busy (a contended
/// `try_lock`, so the scope keeps its last reading rather than the UI freezing — exactly like
/// `program_spectrum`).
///
/// READ-ONLY ANALYSIS: this MIRRORS `program_spectrum` EXACTLY — same `MEAS <window_s>` open, same
/// `build_audio_lines` AUDIO* mix (the assembled, filtered+gained program mix), same persistent-worker
/// round-trip and error handling — but the terminating query is `SAMPLES <n> <out>` (NOT SPECTRUM).
/// The engine computes `ProgAudio::samples(n)` over the accumulator, writes EXACTLY `n` little-endian
/// f32 RAW time-domain amplitudes to `<out>`, then clears the accumulator (session terminator, like
/// LEVELS/SPECTRUM). It changes nothing in the render/mix/LEVELS/SPECTRUM pipeline.
///
/// MECHANISM: identical to `program_spectrum` — `MEAS` allocates a playback-style accumulator (no
/// encoder, no WAV) sized to a SHORT window from the playhead; each AUDIO line mixes a clip's
/// filtered+gained range into it exactly like the render/playback path; `SAMPLES` then reads the raw
/// time-domain amplitudes, writes them, and clears the accumulator. The whole session runs under ONE
/// held mutex so no concurrent compose interleaves on the worker's pipes.
///
/// NON-BLOCKING LOCK (finding #4 style): `try_lock`, called from the egui UI thread (panels), so a
/// background assembly/render holding the worker just yields None this frame.
pub fn program_samples(project: &Project, start_frame: i64) -> Option<Vec<f32>> {
    /// Measurement window length (seconds) from the playhead. Short so the per-repaint decode is cheap
    /// — the SAME window the level meter / spectrum use (mirrors `program_spectrum`).
    const SAMPLES_WINDOW_S: f64 = 0.25;

    let total = project.total_frames();
    let start = start_frame.max(0);
    if total <= 0 || start >= total {
        return None; // nothing to measure at/after the timeline end.
    }
    let fps = RENDER_FPS as f64;
    let start_s = start as f64 / fps;
    // The window is the lesser of SAMPLES_WINDOW_S and the remaining tail.
    let tail_s = (total - start) as f64 / fps;
    let window_s = SAMPLES_WINDOW_S.min(tail_s).max(1.0 / fps);

    // Build the MEAS/AUDIO*/SAMPLES lines on the CALLING thread (cheap, no decode/lock). dst offsets
    // are shifted so the playhead is t=0 in the measurement window; only clips overlapping the window
    // are emitted. This is the SAME windowed AUDIO* mix `program_spectrum`/`program_levels` build
    // (per-clip gain/fade/FX-chain + speed/reverse), so the waveform reflects the audible mix.
    // (Transcribed verbatim from `program_spectrum` so the MEAS/AUDIO*/<terminator> lifecycle matches
    // — only the terminating query differs: SAMPLES, not SPECTRUM.)
    let meas_open = format!("MEAS {window_s}");
    let window_end = start_s + window_s;
    let mut audio_lines: Vec<String> = Vec::new();
    // P27: prepend the master gain envelope (element 0) so it is sent right after the MEAS opener and
    // before the AUDIO lines, so the waveform reflects the gained mix (empty gain_kf → None → no change).
    if let Some(env) = build_gainenv_line(project) {
        audio_lines.push(env);
    }
    {
        let mut idx: Vec<usize> = (0..project.clips.len()).collect();
        idx.sort_by_key(|&i| project.clips[i].t0);
        for i in idx {
            let c = &project.clips[i];
            if c.len <= 0 || !track_is_audible(project, c.track) {
                continue;
            }
            if c.end() <= start {
                continue; // entirely in the past.
            }
            let clip_t0_s = c.t0 as f64 / fps;
            if clip_t0_s >= window_end {
                continue; // starts after the measurement window: nothing in range.
            }
            let media_path = match project.media.get(c.media) {
                Some(p) => p,
                None => continue,
            };
            let head_skip = (start - c.t0).max(0);
            let _eff_src_in = c.src_in + head_skip;
            let mut eff_len = c.len - head_skip;
            if eff_len <= 0 {
                continue;
            }
            let dst_off_s = ((c.t0 + head_skip) as f64 / fps - start_s).max(0.0);
            let max_len_frames = (((window_s - dst_off_s).max(0.0)) * fps).ceil() as i64;
            if max_len_frames <= 0 {
                continue;
            }
            eff_len = eff_len.min(max_len_frames);
            let speed = (c.speed as f64).clamp(0.05, 16.0);
            let rev = c.reverse;
            let dur_src_s = (eff_len as f64 * speed) / fps;
            if !(dur_src_s.is_finite()) || dur_src_s <= 0.0 {
                continue;
            }
            let src_in_s = if rev {
                let lo = c.src_in as f64 + ((c.len - head_skip - eff_len) as f64 * speed);
                lo.max(0.0) / fps
            } else {
                let src0 = c.src_in as f64 + (head_skip as f64 * speed);
                src0.max(0.0) / fps
            };
            let dur_s = dur_src_s;
            // P42 MIXER GAIN: fold the per-TRACK fader into this clip's per-clip gain (the scalar that
            // rides the AUDIO line). track_gain default 1.0 → byte-identical for a default project.
            let gain = c.gain * project.track_gain(c.track);
            let fade_in_s = (c.fade_in.max(0)) as f64 / fps;
            let fade_out_s = (c.fade_out.max(0)) as f64 / fps;
            let clip_len_s = c.len as f64 / fps;
            let range_local_s = head_skip as f64 / fps;
            let base_fx = build_audio_chain(&c.audio_fx); // "-" or a comma chain, SPACE-FREE
            let mut pre: Vec<String> = Vec::new();
            if rev {
                pre.push("areverse".to_string());
            }
            if (speed - 1.0).abs() > 1e-6 {
                for f in atempo_factors(speed) {
                    pre.push(format!("atempo={:.6}", f));
                }
            }
            let fx_chain = if pre.is_empty() {
                base_fx
            } else {
                let mut all = pre;
                if base_fx != "-" {
                    all.push(base_fx);
                }
                all.join(",")
            };
            // P42 MIXER PAN: fold the per-TRACK L/R balance onto the clip's fx_chain (stays space-free,
            // wire arity unchanged). track_pan default 0.0 → chain returned unchanged → byte-identical.
            let fx_chain = apply_track_pan(fx_chain, project.track_pan(c.track));
            let media_path = enc_path(media_path);
            audio_lines.push(format!(
                "AUDIO {media_path} {src_in_s} {dur_s} {dst_off_s} {gain} {fade_in_s} {fade_out_s} {clip_len_s} {range_local_s} {fx_chain}"
            ));
        }
    }

    let slot = worker_slot();
    // try_lock (finding #4): never block the UI thread behind a background assembly / render. A
    // contended scope just returns None and the caller keeps its last reading.
    let mut guard = match slot.try_lock() {
        Ok(g) => g,
        Err(_) => return None,
    };

    for _attempt in 0..MAX_ATTEMPTS {
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            // 1) MEAS: open a measurement accumulator (no encoder, no WAV). A clean ERR (bad window)
            //    is non-retryable — bail. Broken falls through to the respawn below.
            match try_command_status(proc, &meas_open) {
                CmdStatus::Done(_) => {
                    // 2) Mix each clip's filtered+gained range (skip ERR clips; Broken aborts).
                    let mut broke = false;
                    for line in &audio_lines {
                        match try_command_status(proc, line) {
                            CmdStatus::Done(_) | CmdStatus::Err => {}
                            CmdStatus::Broken => {
                                broke = true;
                                break;
                            }
                        }
                    }
                    if !broke {
                        // 3) SAMPLES: read SAMPLES_N raw amplitudes + write, clear the accumulator.
                        //    Read back.
                        if let Some(samples) = read_samples(proc) {
                            clear_spawn_cooldown();
                            return Some(samples);
                        }
                    }
                }
                CmdStatus::Err => return None, // MEAS rejected (worker alive): nothing to measure.
                CmdStatus::Broken => {}        // worker died: respawn below.
            }
        }
        // MEAS/AUDIO/SAMPLES broke mid-session: drop the worker so the next loop spawns clean.
        *guard = None;
        mark_spawn_fail();
    }
    None
}

/// Run the `SAMPLES <n> <out>` round-trip on an already-running worker, then read back EXACTLY
/// `SAMPLES_N` little-endian f32 RAW time-domain amplitudes (~[-1,1]). Returns None on any failure
/// (write error, EOF/timeout, "ERR", a short/oversized read). The worker clears its measurement
/// accumulator inside SAMPLES. The wire query is `SAMPLES <SAMPLES_N> <SAMPLES_OUT>` and the engine
/// returns `SAMPLES_N` f32 — a count/format mismatch would read garbage, so the byte length is
/// checked to be EXACTLY `SAMPLES_N * 4`.
fn read_samples(proc: &mut WorkerProc) -> Option<Vec<f32>> {
    let req = format!("SAMPLES {SAMPLES_N} {SAMPLES_OUT}");
    match try_command_status(proc, &req) {
        CmdStatus::Done(payload) => {
            // The worker echoes the out path on DONE; trust our own path if it echoed empty.
            let read_path = if payload.is_empty() { SAMPLES_OUT } else { payload.as_str() };
            let bytes = std::fs::read(read_path).ok()?;
            if bytes.len() != SAMPLES_N * 4 {
                return None; // exactly SAMPLES_N f32 expected.
            }
            let samples: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Some(samples)
        }
        CmdStatus::Err | CmdStatus::Broken => None,
    }
}

/// Best-effort, non-blocking timeline-audio playback from `start_frame` (Slice A).
///
/// CHOSEN PATH (stated for the integrator): rather than vendoring PulseAudio into the worker, this
/// reuses the SAME program-audio mixing as the render path to write a PCM WAV, then spawns the
/// system player (`paplay`, falling back to `aplay`) on it detached. This keeps gcompose free of any
/// new audio-device link (no libpulse), and survives a missing player binary gracefully. The
/// trade-off vs. fpx_aplay is no live position/VU feedback and no instant scrub-restart — acceptable
/// for a best-effort audition this slice.
///
/// THREADING (finding #1): the WAVE→AUDIO*→WAVECLOSE assembly decodes+mixes the ENTIRE timeline tail
/// from the playhead and holds the worker mutex for its full duration — seconds for a long timeline,
/// during which it also blocks any concurrent `request_frame`/`thumbnail` on the same mutex. This
/// function is called from the egui UI thread (app.rs owns the playhead), so doing that work inline
/// would stall the UI. Instead, only the CHEAP, model-touching part — resolving the audible clips
/// into owned `AUDIO ...` command strings — runs on the calling thread (no decode, no lock); the
/// owned lines are then moved onto a detached `std::thread::spawn` that takes the lock, drives the
/// WAVE session, writes the WAV, and spawns the detached player. So `play_program` RETURNS
/// IMMEDIATELY (returning `true` to mean "playback was dispatched", not "audio is already audible");
/// any worker/spawn failure is logged on the background thread, not surfaced to the caller. The
/// background thread owns its own `Vec<String>` + `String` (no borrow of `project`), so there is no
/// lifetime tie to the UI's project.
///
/// Mechanism: run a `WAVE <wav> <dur_s>` / `AUDIO*` / `WAVECLOSE <wav>` session on the persistent
/// worker. The accumulator is sized to the timeline tail from `start_frame` (so the WAV begins at
/// the playhead). Each clip is mapped into playhead-relative time: clips ending at/before the
/// playhead are dropped; a clip straddling the playhead has its source in-point ADVANCED and its
/// duration shortened by the already-played head so it plays from the source frame under the
/// playhead at dst_offset 0; clips after the playhead keep their source range at dst_offset =
/// (t0 - start)/FPS. Returns true if playback was dispatched (a background thread was spawned);
/// false when there is nothing to play (empty timeline / playhead at/after the end) OR when a
/// previous playback is still in flight (finding #9: a Space press while one audition is assembling
/// is dropped rather than stacking a second background thread + duplicate player).
pub fn play_program(project: &Project, start_frame: i64) -> bool {
    let total = project.total_frames();
    let start = start_frame.max(0);
    if total <= 0 || start >= total {
        return false; // nothing to play at/after the timeline end.
    }
    let fps = RENDER_FPS as f64;
    // Duration of the audio to assemble = timeline tail from the playhead.
    let tail_frames = total - start;
    let tail_s = tail_frames as f64 / fps;
    let start_s = start as f64 / fps;

    // Build the WAVE/AUDIO*/WAVECLOSE lines NOW, on the calling (UI) thread — this is the only part
    // that touches `project`, and it is cheap (no decode, no lock). dst_offset is shifted so the
    // playhead is t=0 in the WAV. The resulting owned Vec<String> is moved into the worker thread
    // below, so the heavy decode/mix never borrows `project`.
    let wave_open = format!("WAVE {PLAY_WAV} {tail_s}");
    let mut audio_lines: Vec<String> = Vec::new();
    // P27: prepend the master gain envelope (element 0) so it is sent right after the WAVE opener and
    // before the AUDIO lines (empty gain_kf → None → byte-identical playback).
    if let Some(env) = build_gainenv_line(project) {
        audio_lines.push(env);
    }
    {
        let mut idx: Vec<usize> = (0..project.clips.len()).collect();
        idx.sort_by_key(|&i| project.clips[i].t0);
        for i in idx {
            let c = &project.clips[i];
            if c.len <= 0 {
                continue;
            }
            if !track_is_audible(project, c.track) {
                continue;
            }
            // Skip clips that end at/before the playhead (entirely in the past).
            if c.end() <= start {
                continue;
            }
            let media_path = match project.media.get(c.media) {
                Some(p) => p,
                None => continue,
            };
            // Whitespace is now WIRE-SAFE (finding #8 resolved): the AUDIO line percent-encodes the
            // path token (enc_path) below and the engine decodes it, so a spaced path no longer
            // shifts the fixed-arity fields. The old whitespace skip is removed.
            // For a clip straddling the playhead (t0 < start), skip the already-played head: advance
            // the SOURCE in-point by `start - t0` frames and shorten the duration by the same, so the
            // clip plays from the source frame under the playhead at dst_offset 0. For a clip wholly
            // after the playhead, src_in/len are unchanged and dst_offset is its forward distance.
            let head_skip = (start - c.t0).max(0); // frames of this clip already behind the playhead
            // (P24 makes the source in-point speed-scaled, computed below; the plain eff_src_in is no
            // longer the sent value — kept unprefixed-free to avoid an unused binding.)
            let _eff_src_in = c.src_in + head_skip; // frames
            let eff_len = c.len - head_skip; // frames remaining to play
            if eff_len <= 0 {
                continue;
            }
            // P24 per-clip speed/reverse (Model A): retime the SOURCE window so the engine reads
            // more/less source; the atempo/areverse in the chain (below) compresses it back to the
            // SAME timeline window (eff_len/fps). speed==1.0 && !rev → identity.
            let speed = (c.speed as f64).clamp(0.05, 16.0);
            let rev = c.reverse;
            let dur_src_s = (eff_len as f64 * speed) / fps;
            if !(dur_src_s.is_finite()) || dur_src_s <= 0.0 {
                continue;
            }
            let src_in_s = if rev {
                let lo = c.src_in as f64 + ((c.len - head_skip - eff_len) as f64 * speed);
                lo.max(0.0) / fps
            } else {
                let src0 = c.src_in as f64 + (head_skip as f64 * speed);
                src0.max(0.0) / fps
            };
            let dur_s = dur_src_s;
            // Timeline position relative to the playhead (>= 0 by construction: head_skip clamps it).
            let dst_off_s = ((c.t0 + head_skip) as f64 / fps - start_s).max(0.0);
            // Per-clip gain + fade envelope (P1). The decoded range here is HEAD-TRIMMED by
            // `head_skip` frames, so the first decoded sample is at clip-local `head_skip/FPS` —
            // pass that as `range_local_s` so gcompose ramps the fade against the FULL clip edges
            // (a clip whose fade-in is entirely before the playhead plays at full gain, as it
            // should). fade/clip_len are the clip's own (untrimmed) frame counts in seconds.
            // P42 MIXER GAIN: fold the per-TRACK fader into this clip's per-clip gain (the scalar that
            // rides the AUDIO line). track_gain default 1.0 → byte-identical for a default project.
            let gain = c.gain * project.track_gain(c.track);
            let fade_in_s = (c.fade_in.max(0)) as f64 / fps;
            let fade_out_s = (c.fade_out.max(0)) as f64 / fps;
            let clip_len_s = c.len as f64 / fps;
            let range_local_s = head_skip as f64 / fps;
            // P3 + P24: the per-clip libavfilter chain (space-free) or "-" when neutral, with the
            // P24 areverse (if rev) + atempo stages (if speed != 1) PREPENDED before the AudioFx
            // chain so the post-filter buffer is timeline-length (fade/range_local still line up).
            let base_fx = build_audio_chain(&c.audio_fx);
            let mut pre: Vec<String> = Vec::new();
            if rev {
                pre.push("areverse".to_string());
            }
            if (speed - 1.0).abs() > 1e-6 {
                for f in atempo_factors(speed) {
                    pre.push(format!("atempo={:.6}", f));
                }
            }
            let fx_chain = if pre.is_empty() {
                base_fx
            } else {
                let mut all = pre;
                if base_fx != "-" {
                    all.push(base_fx);
                }
                all.join(",")
            };
            // P42 MIXER PAN: fold the per-TRACK L/R balance onto the clip's fx_chain (stays space-free,
            // wire arity unchanged). track_pan default 0.0 → chain returned unchanged → byte-identical.
            let fx_chain = apply_track_pan(fx_chain, project.track_pan(c.track));
            // WHITESPACE-SAFE WIRE: percent-encode the media path token (enc_path); the engine
            // dec_path's it before opening the decoder. Space-free paths are byte-identical.
            let media_path = enc_path(media_path);
            audio_lines.push(format!(
                "AUDIO {media_path} {src_in_s} {dur_s} {dst_off_s} {gain} {fade_in_s} {fade_out_s} {clip_len_s} {range_local_s} {fx_chain}"
            ));
        }
    }

    // Dedup rapid presses (finding #9): if a previous play_program's background thread is still
    // assembling/playing, do NOT spawn another — it would block on the worker mutex behind the first
    // (and behind any in-progress render), then fire a duplicate, delayed `paplay`. compare_exchange
    // claims the in-flight slot; if it's already taken we drop this press as a no-op.
    if PLAYBACK_IN_FLIGHT
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        )
        .is_err()
    {
        // A playback is already in flight; ignore this press rather than stacking another thread.
        return false;
    }

    // Clear the stop edge for THIS dispatch (findings #7/#8): any stop requested before now belonged
    // to a previous audition; this fresh play_program supersedes it. Ordered AFTER the in-flight
    // claim above so a stop racing the claim is still observed by the assembly thread's pre-spawn
    // check below (it can only make us SKIP a late player, never spuriously spawn one).
    STOP_REQUESTED.store(false, std::sync::atomic::Ordering::Release);

    // Hand the owned command lines to a detached background thread so the UI thread returns at once
    // (finding #1). The thread takes the worker lock, assembles the WAV, and spawns the player; its
    // failures are logged there, not returned here. The in-flight guard is cleared when the thread
    // finishes (success OR failure) so the next Space press can dispatch again.
    std::thread::spawn(move || {
        if assemble_and_play(&wave_open, &audio_lines) {
            // STOP-DURING-ASSEMBLY (findings #7/#8): if `stop_playback` fired while we were
            // assembling the WAV (the seconds-long window before any player exists), do NOT spawn a
            // late player — the user asked for silence. Without this check the WAV would still play a
            // moment later, unkillable (there was no child to kill at stop time), and on a loop wrap
            // it would overlap the next cycle's audio. Acquire pairs with stop_playback's Release.
            if !STOP_REQUESTED.load(std::sync::atomic::Ordering::Acquire) {
                // WAV written and no stop pending: launch the detached system player (best-effort).
                spawn_player(PLAY_WAV);
            }
        }
        PLAYBACK_IN_FLIGHT.store(false, std::sync::atomic::Ordering::Release);
    });

    true // playback dispatched (the background thread owns the rest).
}

/// Worker side of `play_program`, run on a detached background thread (finding #1). Assembles the
/// playback WAV by driving a WAVE→AUDIO*→WAVECLOSE session on the persistent worker under ONE lock
/// hold (no auto-restart mid-session: a restart would lose the accumulator). Any `Broken` tears the
/// worker down + arms the cooldown; a per-clip `ERR` is skipped. Returns true if the WAV was written.
/// Takes only OWNED data (`&str` / `&[String]` into thread-local Strings) so it never borrows the
/// UI's `Project`.
fn assemble_and_play(wave_open: &str, audio_lines: &[String]) -> bool {
    let slot = worker_slot();
    let mut guard = match slot.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };

    // Ensure a live worker for the WAVE (no in-flight session yet, so a respawn here is safe).
    let mut opened = false;
    for _ in 0..MAX_ATTEMPTS {
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            match try_command_status(proc, wave_open) {
                CmdStatus::Done(_) => {
                    clear_spawn_cooldown();
                    opened = true;
                    break;
                }
                CmdStatus::Err => return false, // WAVE rejected (bad dur): worker alive, bail.
                CmdStatus::Broken => {
                    *guard = None;
                    mark_spawn_fail();
                }
            }
        } else {
            *guard = None;
            mark_spawn_fail();
        }
    }
    if !opened {
        return false;
    }

    // Mix each clip (skip ERR clips; a Broken kills the session).
    for line in audio_lines {
        match guard.as_mut() {
            Some(proc) => match try_command_status(proc, line) {
                CmdStatus::Done(_) => {}
                CmdStatus::Err => {} // no audio in range / decode skip: drop this clip.
                CmdStatus::Broken => {
                    *guard = None;
                    mark_spawn_fail();
                    return false;
                }
            },
            None => return false,
        }
    }

    // WAVECLOSE -> write the WAV.
    match guard.as_mut() {
        Some(proc) => match try_command_status(proc, &format!("WAVECLOSE {PLAY_WAV}")) {
            CmdStatus::Done(_) => {
                clear_spawn_cooldown();
                true
            }
            CmdStatus::Err => false,
            CmdStatus::Broken => {
                *guard = None;
                mark_spawn_fail();
                false
            }
        },
        None => false,
    }
}

/// Spawn a detached audio player on `wav`. Tries `paplay` then `aplay`. Returns true if one was
/// launched. The child's stdio is /dev/null and it is not WAITED on (playback runs independently of
/// the UI), but the `Child` handle IS stored in `PLAYER_CHILD` so `stop_playback` can kill it
/// on demand (Slice A pinned API). Any previously-tracked child is replaced here — by the time a new
/// audition is dispatched the old one is either finished or being superseded, so we best-effort kill
/// + reap the prior handle before storing the new one (avoids two players overlapping AND avoids
/// leaking a zombie). Best-effort: false if neither binary exists.
fn spawn_player(wav: &str) -> bool {
    for bin in ["paplay", "aplay"] {
        match Command::new(bin)
            .arg(wav)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => {
                if let Ok(mut slot) = player_slot().lock() {
                    // Replace any prior child: kill+reap it first so we don't stack two players or
                    // leak a zombie. A finished child's kill is a harmless no-op.
                    if let Some(mut old) = slot.take() {
                        let _ = old.kill();
                        let _ = old.wait();
                    }
                    *slot = Some(child);
                }
                return true;
            }
            Err(_) => continue,
        }
    }
    eprintln!("gcompose: no audio player (paplay/aplay) found; playback skipped");
    false
}

/// PINNED (Slice A): stop any in-flight audition player started by `play_program`.
///
/// CHOSEN MECHANISM (stated for the integrator): we track the spawned player `Child` (paplay/aplay)
/// in the process-global `PLAYER_CHILD` and `kill()`+`wait()` it here — the precise path that stops
/// exactly the player THIS process launched and reaps it so no zombie is left. `stop_playback` does
/// NOT touch the worker mutex, so it returns instantly even while a render/assemble holds the worker.
///
/// TWO COORDINATION FLAGS (findings #7/#8):
///   1. `STOP_REQUESTED` is set so the detached assembly thread, if it is still inside the
///      WAVE/AUDIO window (no player spawned yet), SKIPS spawning the late player when it finishes
///      (see `play_program`). This is what makes a stop fired DURING assembly actually take effect —
///      previously there was no child to kill yet, so the WAV played a moment later, unkillable.
///   2. `PLAYBACK_IN_FLIGHT` is force-cleared so an immediately-following `play_program` (the loop
///      wrap: `stop_playback(); play_program(0)`) can re-dispatch instead of being dropped as a
///      no-op by the in-flight guard while the previous assembly thread is still winding down.
///      Clearing it here can transiently allow TWO assembly threads to run concurrently, but the new
///      thread cleared `STOP_REQUESTED` for itself and the OLD thread's pre-spawn check now sees that
///      cleared flag — so the OLD thread might still spawn its (now-stale) player. That is bounded:
///      `spawn_player` kills+reaps any prior tracked child before storing the new one, so at most one
///      player is ever audible. The net effect is the loop wrap re-dispatches reliably rather than
///      going silent (the previous fragility) while never stacking audible players.
///
/// REMOVED (finding #7): the old `pkill -f /tmp/genesis_play.wav` fallback. A broad `pkill -f`
/// matches the pattern against the WHOLE command line of every process, so it could kill unrelated
/// processes that merely reference the path (a text editor, `tail`, a second Genesis instance's
/// player sharing the hard-coded WAV) — collateral damage for a marginal benefit. The tracked-child
/// kill plus the STOP_REQUESTED edge cover the real cases (audible player, and the assembly-window
/// race) precisely, with no risk of killing a bystander.
pub fn stop_playback() {
    // Edge flags first (findings #7/#8): suppress any late player from an in-progress assembly, and
    // free the in-flight slot so a loop-wrap re-dispatch isn't dropped. Release pairs with the
    // assembly thread's Acquire load and play_program's compare_exchange.
    STOP_REQUESTED.store(true, std::sync::atomic::Ordering::Release);
    PLAYBACK_IN_FLIGHT.store(false, std::sync::atomic::Ordering::Release);

    // Primary + only kill path: kill + reap the tracked player child (no broad pkill — finding #7).
    if let Ok(mut slot) = player_slot().lock() {
        if let Some(mut child) = slot.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Best-effort teardown of a half-open render encoder after `render_program` aborts mid-ENC, using
/// the ALREADY-HELD worker guard (finding #1 + #8): we never re-lock and never spawn a worker just
/// to send CLOSE (finding #8). If the proc is still live we send a single CLOSE so the worker drops
/// the encoder (and its partial mp4) immediately; if the worker already died we just clear the slot
/// so the next top-level call spawns clean.
fn abort_held(guard: &mut Option<WorkerProc>) {
    if let Some(proc) = guard.as_mut() {
        match try_command_status(proc, "CLOSE") {
            CmdStatus::Done(_) | CmdStatus::Err => {} // encoder torn down; worker stays alive.
            CmdStatus::Broken => {
                // Worker died during teardown: drop it + arm the cooldown.
                *guard = None;
                mark_spawn_fail();
            }
        }
    }
}

/// Build a libavfilter CHAIN STRING (P3 Triad-B audio depth) from a clip's `AudioFx`, or the "-"
/// sentinel when the FX are neutral. The returned string has NO SPACES — filters are joined with
/// COMMAS, and `=`/`:`/`|` are used inside a filter — so it rides the fixed-arity AUDIO wire line as
/// ONE token. gcompose runs `fpx_au_apply(sr, ch, chain, ...)` on the decoded clip range BEFORE the
/// per-clip gain + fade + dst-offset mix, but ONLY when the chain != "-".
///
/// NEUTRAL ⇒ "-" (PINNED contract): a clip whose `AudioFx::is_neutral()` is true sends "-", so the
/// engine skips `fpx_au_apply` entirely and the mix is BYTE-IDENTICAL to P2. This is the identity
/// guarantee — no FX dialed means the P2 audio path is reproduced exactly.
///
/// MAPPING (each emitted only when it changes the audio; mirrors Shotcut's audio_eq3band / audio_pan
/// / audio_compressor / audio_noisegate / audio_normalize_1p):
///   eq_low_db  (≠0) -> `equalizer=f=100:t=q:w=1:g=<low>`   (low band   @ 100 Hz)
///   eq_mid_db  (≠0) -> `equalizer=f=1000:t=q:w=1:g=<mid>`  (mid band   @ 1 kHz)
///   eq_high_db (≠0) -> `equalizer=f=8000:t=q:w=1:g=<high>` (high band  @ 8 kHz)
///   geq[i]     (≠0) -> `equalizer=f=<Fi>:width_type=o:width=1.0:g=<geq[i]>` (P32 graphic 10-band EQ;
///                       peaking gain dB clamped ±24, {:.2}; one-octave bandwidth; band centres Hz
///                       index 0..9 = 31,62,125,250,500,1000,2000,4000,8000,16000 — only non-zero
///                       bands emit; grouped with the 3-band EQ above)
///   pan        (≠0) -> `stereotools=balance_out=<pan>`     (−1 = full L, +1 = full R; matches the
///                       model's −1..1 pan, which is exactly stereotools' balance_out range)
///   compress (true) -> `acompressor`     (libavfilter sensible defaults)
///   gate     (true) -> `agate`           (libavfilter sensible defaults)
///   delay_ms  (>0)  -> `aecho=0.8:0.9:<ms>:<dec>`            (single echo tap; ms int, dec {:.3})
///   reverb    (>0)  -> `aecho=0.8:0.9:47|97|151|211:<d0..3>` (multi-tap aecho ≈ small-room reverb)
///   pitch    (≠0)  -> `rubberband=pitch=<ratio>`            (tempo-PRESERVING semitone shift)
///   highpass_hz (>0) -> `highpass=f=<hz>`                   (P12; cutoff Hz clamped 10..20000, int)
///   lowpass_hz  (>0) -> `lowpass=f=<hz>`                    (P12; cutoff Hz clamped 10..20000, int)
///   tremolo     (>0) -> `tremolo=f=5:d=<depth>`             (P12; 5 Hz rate, depth 0..0.95 {:.3})
///   bass_db    (≠0)  -> `bass=g=<db>`                       (P15; low-shelf gain, clamp ±30 dB {:.3})
///   treble_db  (≠0)  -> `treble=g=<db>`                     (P15; high-shelf gain, clamp ±30 dB {:.3})
///   notch_hz    (>0) -> `bandreject=f=<hz>`                 (P15; band-reject centre, clamp 20..20000 int)
///   chorus      (>0) -> `chorus=0.5:0.9:50:0.4:0.25:<dep>`  (P15; single-voice, dep = 2*depth ms {:.3})
///   flanger     (>0) -> `flanger=depth=<dep>:speed=0.5`     (P22; dep = 8*depth ms 0..8 {:.3})
///   phaser      (>0) -> `aphaser=speed=<spd>`               (P22; spd = 2*intensity+0.1 Hz, >0 {:.3})
///   limiter     (>0) -> `alimiter=limit=<lim>`              (P22; lim = peak ceiling, clamp 0.05..1 {:.3})
///   normalize(true) -> `loudnorm`        (single-pass EBU R128 loudness normalization)
/// Filter ORDER is EQ → pan → compress → gate → delay → reverb → pitch → highpass → lowpass → tremolo
/// → bass → treble → notch → chorus → flanger → phaser → limiter → normalize (tone-shape first, then
/// dynamics, then time/echo + pitch, then the P12 band/amplitude filters, then the P15 shelves/notch/
/// chorus, then the P22 flanger/phaser/limiter, then a final loudness pass — loudnorm stays LAST).
/// dB / ratio values are formatted WITHOUT a thousands separator / locale, so the `{:.N}` float never
/// contains a space. `pitch` uses `rubberband` (NOT `asetrate`, which would also change tempo).
///
/// SAFETY: `is_neutral()` is the single source of truth for "no FX"; if a future field is added to
/// AudioFx, neutral must keep returning "-". A non-finite slider value (shouldn't occur from the UI
/// sliders) is treated as 0 / off so the chain can never contain "NaN"/"inf" tokens.

/// Decompose a tempo factor s>0 into a list of per-stage atempo factors each within [0.5, 2.0].
/// e.g. 1.0->[1.0] (caller skips when ~1), 4.0->[2.0,2.0], 0.25->[0.5,0.5], 1.5->[1.5].
fn atempo_factors(mut s: f64) -> Vec<f64> {
    let mut out = Vec::new();
    if !(s.is_finite()) || s <= 0.0 { return vec![1.0]; }
    while s > 2.0 { out.push(2.0); s /= 2.0; }
    while s < 0.5 { out.push(0.5); s *= 2.0; }
    out.push(s);
    out
}

fn build_audio_chain(fx: &crate::model::AudioFx) -> String {
    if fx.is_neutral() {
        return "-".to_string();
    }
    let mut parts: Vec<String> = Vec::new();

    // 3-band EQ — only emit a band whose gain is non-zero (and finite). `f` = center freq, `t=q` +
    // `w=1` give a moderate-Q peaking filter per band (Shotcut's 3-band bass/mid/treble equivalent).
    let band = |freq: i32, g: f32| -> Option<String> {
        if g != 0.0 && g.is_finite() {
            Some(format!("equalizer=f={freq}:t=q:w=1:g={:.3}", g))
        } else {
            None
        }
    };
    if let Some(s) = band(100, fx.eq_low_db) {
        parts.push(s);
    }
    if let Some(s) = band(1000, fx.eq_mid_db) {
        parts.push(s);
    }
    if let Some(s) = band(8000, fx.eq_high_db) {
        parts.push(s);
    }

    // P32 GRAPHIC 10-BAND EQ (Shotcut audio_eq15band-style graphic equalizer). `fx.geq[i]` is a
    // peaking gain in dB at the i-th ISO octave band centre frequency. Each NON-ZERO (and finite)
    // band emits ONE peaking `equalizer` part with `width_type=o:width=1.0` (one-octave bandwidth);
    // a flat (0 dB) band adds nothing and is skipped. Grouped here with the 3-band EQ above (filter
    // order is not critical, but the EQ filters stay together). Gain clamped to [-24, 24] dB and
    // formatted {:.2} so the bare float carries NO space — the comma-joined chain stays space-free.
    // The 10 band centres (Hz), index 0..9: 31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000.
    const GEQ_FREQS: [i32; 10] = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
    for i in 0..10 {
        let g = fx.geq[i];
        if g != 0.0 && g.is_finite() {
            let g = g.clamp(-24.0, 24.0);
            parts.push(format!(
                "equalizer=f={}:width_type=o:width=1.0:g={:.2}",
                GEQ_FREQS[i], g
            ));
        }
    }

    // Pan — stereotools balance_out (−1..1). Clamp defensively so a stray value can't exceed the
    // filter's accepted range (which would make the whole graph fail to parse → unfiltered fallback).
    if fx.pan != 0.0 && fx.pan.is_finite() {
        let p = fx.pan.clamp(-1.0, 1.0);
        parts.push(format!("stereotools=balance_out={:.3}", p));
    }

    // Dynamics + loudness — boolean toggles, libavfilter defaults (NO spaces in the bare names).
    if fx.compress {
        parts.push("acompressor".to_string());
    }
    if fx.gate {
        parts.push("agate".to_string());
    }

    // P11 effects — pushed AFTER the dynamics block and BEFORE loudnorm (loudnorm must stay LAST so
    // the loudness pass sees the fully-processed signal). All values clamped to the filters' valid
    // ranges; a non-finite value is skipped so the chain can never carry a "NaN"/"inf" token.

    // Delay (echo) — single-tap aecho. `aecho=in_gain:out_gain:delays:decays`; delay in ms (integer),
    // decay 0..0.95 ({:.3}). `0.8:0.9` are the conventional in/out gains (matches Shotcut's echo).
    if fx.delay_ms > 0.0 && fx.delay_ms.is_finite() {
        let ms = fx.delay_ms.clamp(1.0, 4000.0) as i32;
        let dec = fx.delay_decay.clamp(0.0, 0.95);
        parts.push(format!("aecho=0.8:0.9:{ms}:{:.3}", dec));
    }

    // Reverb — a 4-tap aecho approximates a small room. Tap delays 47/97/151/211 ms are fixed; the
    // per-tap decays scale with the reverb amount (all strictly < 1 so the graph stays stable).
    if fx.reverb > 0.0 && fx.reverb.is_finite() {
        let a = fx.reverb.clamp(0.0, 1.0);
        let d0 = a * 0.5;
        let d1 = a * 0.4;
        let d2 = a * 0.3;
        let d3 = a * 0.2;
        parts.push(format!(
            "aecho=0.8:0.9:47|97|151|211:{:.3}|{:.3}|{:.3}|{:.3}",
            d0, d1, d2, d3
        ));
    }

    // Pitch — rubberband shifts pitch by `pitch` SEMITONES while PRESERVING tempo (asetrate would
    // change both). ratio = 2^(semitones/12); semitones clamped to [-24, 24] (±2 octaves).
    if fx.pitch != 0.0 && fx.pitch.is_finite() {
        let semis = fx.pitch.clamp(-24.0, 24.0);
        let ratio = 2f32.powf(semis / 12.0);
        parts.push(format!("rubberband=pitch={:.4}", ratio));
    }

    // P12 filters (Shotcut High Pass / Low Pass / Tremolo) — pushed AFTER the P11 pitch part and
    // BEFORE loudnorm (loudnorm must stay LAST). All values finite-guarded and clamped to the
    // filters' valid ranges; every emitted token is SPACE-FREE so the comma-joined chain has no
    // spaces. Order: high-pass then low-pass (a band-pass when both set) then tremolo (amplitude mod).

    // High-pass — `highpass=f=<hz>` removes content below the cutoff. Cutoff clamped to [10, 20000] Hz
    // and emitted as an integer (no decimal/space). 0 = off (skipped).
    if fx.highpass_hz > 0.0 && fx.highpass_hz.is_finite() {
        let hz = fx.highpass_hz.clamp(10.0, 20000.0) as i32;
        parts.push(format!("highpass=f={hz}"));
    }

    // Low-pass — `lowpass=f=<hz>` removes content above the cutoff. Same clamp/int handling as above.
    if fx.lowpass_hz > 0.0 && fx.lowpass_hz.is_finite() {
        let hz = fx.lowpass_hz.clamp(10.0, 20000.0) as i32;
        parts.push(format!("lowpass=f={hz}"));
    }

    // Tremolo — `tremolo=f=5:d=<depth>` amplitude-modulates at a fixed 5 Hz rate; depth clamped to
    // [0, 0.95] ({:.3}). libavfilter's tremolo REJECTS d>=1, hence the 0.95 ceiling. 0 = off (skipped).
    if fx.tremolo > 0.0 && fx.tremolo.is_finite() {
        let d = fx.tremolo.clamp(0.0, 0.95);
        parts.push(format!("tremolo=f=5:d={:.3}", d));
    }

    // P15 filters (Shotcut Bass & Treble / Notch / Chorus) — pushed AFTER the P12 tremolo part and
    // BEFORE loudnorm (loudnorm must stay LAST). All values finite-guarded and clamped to the
    // filters' valid ranges; every emitted token is SPACE-FREE so the comma-joined chain has no
    // spaces. Order: bass shelf → treble shelf → notch (band-reject) → chorus.

    // Bass — `bass=g=<db>` is a low-shelf gain. 0 dB = flat (off / skipped). Gain clamped ±30 dB
    // ({:.3}); the bare float never contains a space.
    if fx.bass_db != 0.0 && fx.bass_db.is_finite() {
        let g = fx.bass_db.clamp(-30.0, 30.0);
        parts.push(format!("bass=g={:.3}", g));
    }

    // Treble — `treble=g=<db>` is a high-shelf gain. 0 dB = flat (off / skipped). Same ±30 dB clamp.
    if fx.treble_db != 0.0 && fx.treble_db.is_finite() {
        let g = fx.treble_db.clamp(-30.0, 30.0);
        parts.push(format!("treble=g={:.3}", g));
    }

    // Notch — `bandreject=f=<hz>` is a band-reject (notch) at the centre frequency. Centre clamped
    // to [20, 20000] Hz and emitted as an integer (no decimal/space). 0 = off (skipped).
    if fx.notch_hz > 0.0 && fx.notch_hz.is_finite() {
        let hz = fx.notch_hz.clamp(20.0, 20000.0) as i32;
        parts.push(format!("bandreject=f={hz}"));
    }

    // Chorus — single-voice `chorus=in_gain:out_gain:delays:decays:speeds:depths`. Fixed
    // 0.5:0.9:50:0.4:0.25 with the LAST slot (depths, in ms) = 2 * depth so each list slot carries
    // exactly one value (no spaces). depth clamped to [0, 1] → depths in [0, 2] ms. 0 = off (skipped).
    if fx.chorus > 0.0 && fx.chorus.is_finite() {
        let dep = 2.0 * fx.chorus.clamp(0.0, 1.0);
        parts.push(format!("chorus=0.5:0.9:50:0.4:0.25:{:.3}", dep));
    }

    // P22 filters (Shotcut Flanger / Phaser / Limiter) — pushed AFTER the P15 chorus part and
    // BEFORE loudnorm (loudnorm must stay LAST). All values finite-guarded and clamped to the
    // filters' valid ranges; every emitted token is SPACE-FREE so the comma-joined chain has no
    // spaces. Order: flanger → phaser → limiter.

    // Flanger — `flanger=depth=<ms>:speed=0.5` sweeps a short delay. depth (model 0..1) maps to a
    // 0..8 ms sweep depth ({:.3}); speed is a fixed 0.5 Hz LFO. 0 = off (skipped).
    if fx.flanger > 0.0 && fx.flanger.is_finite() {
        let depth = fx.flanger.clamp(0.0, 1.0) * 8.0;
        parts.push(format!("flanger=depth={:.3}:speed=0.5", depth));
    }

    // Phaser — `aphaser=speed=<hz>` sweeps an all-pass network. intensity (model 0..1) maps to a
    // 0.1..2.1 Hz sweep speed ({:.3}); the +0.1 floor keeps speed strictly > 0. 0 = off (skipped).
    if fx.phaser > 0.0 && fx.phaser.is_finite() {
        let speed = fx.phaser.clamp(0.0, 1.0) * 2.0 + 0.1;
        parts.push(format!("aphaser=speed={:.3}", speed));
    }

    // Limiter — `alimiter=limit=<lin>:level=disabled` is a lookahead peak limiter. The level (model
    // 0..1) is the linear peak ceiling, clamped to [0.0625, 1.0] ({:.3}) — 0.0625 is alimiter's REAL
    // minimum `limit` (a smaller value fails to parse → the whole chain falls back unfiltered).
    // `level=disabled` is REQUIRED: without it alimiter auto-LEVELS (boosts the signal toward the
    // ceiling, the opposite of limiting); disabled makes it cap peaks only. 0 = off (skipped).
    if fx.limiter > 0.0 && fx.limiter.is_finite() {
        // {:.4} not {:.3}: the clamp floor 0.0625 formats as "0.062" at 3 decimals, which is BELOW
        // alimiter's 0.0625 minimum and gets rejected (chain falls back unfiltered). 4 decimals
        // emits "0.0625" — a valid limit.
        let lim = fx.limiter.clamp(0.0625, 1.0);
        parts.push(format!("alimiter=limit={:.4}:level=disabled", lim));
    }

    if fx.normalize {
        parts.push("loudnorm".to_string());
    }

    // Defensive: if every "band" was actually 0/off (is_neutral was false only because of a
    // non-finite that we dropped), fall back to "-" so we never send an empty chain token.
    if parts.is_empty() {
        return "-".to_string();
    }
    parts.join(",")
}

/// P42 MIXER PAN — fold a per-TRACK L/R balance into an already-built clip `fx_chain` (the COMMA-joined,
/// SPACE-FREE token that rides the fixed-arity AUDIO wire line). Mirrors `build_audio_chain`'s per-clip
/// pan stage (`stereotools=balance_out=<p>`, −1 = full L … +1 = full R) but applied AFTER it, so the
/// track pan composes on top of any per-clip pan. A default pan (0.0) — or a non-finite stray — returns
/// the chain UNCHANGED, so a pre-P42 / default project emits a byte-identical AUDIO line. The appended
/// stage uses `{:.3}` (no spaces) and a leading comma into the existing chain, keeping the AUDIO token
/// space-free and the wire arity UNCHANGED (the pan stage rides inside the existing fx_chain token).
/// When the chain is the neutral `-`, the pan stage REPLACES it (it becomes the sole filter).
fn apply_track_pan(chain: String, pan: f32) -> String {
    if pan == 0.0 || !pan.is_finite() {
        return chain;
    }
    let p = pan.clamp(-1.0, 1.0);
    let stage = format!("stereotools=balance_out={:.3}", p);
    if chain == "-" {
        stage
    } else {
        format!("{chain},{stage}")
    }
}

/// True if track `t` contributes to the program audio (P5 arbitrary tracks): EVERY track — video or
/// audio — contributes its clips' audio unless it is MUTED. This replaces the old fixed policy (only
/// V1+A1 audible, V2 never) now that tracks are a typed list: a video clip carries audio that plays
/// like Shotcut, and a muted track (`project.is_muted(t)`) is silent. An out-of-range index is silent.
///
/// P42 MIXER SOLO: when ANY track is soloed (`project.any_solo()`), only soloed tracks stay audible
/// (`project.is_solo(t)`) — standard mixer solo. The `!any_solo()` short-circuit makes this term inert
/// (always true) when no track is soloed, so a pre-P42 / default project is byte-identical. This gates
/// ALL audio-emit loops (render/levels/spectrum/samples/playback) since they all funnel through here;
/// the VIDEO loops never call this (they check `!is_audio && !is_hidden`), so video is untouched.
fn track_is_audible(project: &Project, t: u8) -> bool {
    (t as usize) < project.track_count()
        && !project.is_muted(t)
        && (!project.any_solo() || project.is_solo(t))
}

/// P27 MASTER GAIN ENVELOPE — build the single `GAINENV <packed>` wire line from the project's
/// `gain_kf` automation track, or `None` when the track is EMPTY (→ no line emitted → byte-identical
/// to pre-P27; the engine treats a missing GAINENV as an empty/identity envelope).
///
/// WIRE (pinned, both sides identical):
///   GAINENV <packed>
/// where `<packed>` is a SINGLE space-free token `t0:v0,t1:v1,...` — `t` in SECONDS (f64), `v` the
/// gain multiplier (f32); pairs are comma-separated, `t:v` colon-separated. There is NEVER a space
/// inside the packed token (a space would break the fixed-arity wire). `t` is ABSOLUTE timeline
/// seconds = keyframe frame / RENDER_FPS (NOT frames). The keyframes are stored LINEAR, so the engine
/// linear-interps the same curve the UI graph (eval_track on Linear keys) draws.
///
/// Emitted as element 0 of every audio-line list so it is sent right after the session opener
/// (OPEN/MEAS/WAVE) and BEFORE the AUDIO lines — the per-clip mix can then read the installed envelope.
fn build_gainenv_line(project: &Project) -> Option<String> {
    if project.gain_kf.is_empty() {
        return None; // identity: no envelope → no line.
    }
    // Pack "t_sec:v" pairs, comma-joined, NO spaces. t = frame / RENDER_FPS (seconds), v = gain.
    let packed = project
        .gain_kf
        .iter()
        .map(|k| format!("{}:{}", k.t as f64 / RENDER_FPS as f64, k.v))
        .collect::<Vec<String>>()
        .join(",");
    Some(format!("GAINENV {packed}"))
}

/// Build the timeline-synced AUDIO lines for the program audio, one per AUDIBLE clip (track 0 V1 +
/// track 2 A1, honoring `track_mute`; track 1 V2 skipped). Emitted in timeline (`t0`) order for
/// determinism, though order no longer affects the result now that the worker mixes by destination
/// offset rather than concatenating.
///
/// WIRE (Triad-B P3 — 11 tokens; was 10 in P1, 6 in wave-2):
///   AUDIO <media> <src_in_s> <dur_s> <dst_off_s> <gain> <fade_in_s> <fade_out_s> <clip_len_s>
///         <range_local_s> <fx_chain|->
///
/// `src_in_s = clip.src_in / FPS`, `dur_s = clip.len / FPS`, `dst_off_s = clip.t0 / FPS`. `gain` is
/// the per-clip LINEAR gain (`Clip.gain`, was hardcoded 1.0). The fade fields let gcompose apply the
/// clip's audio fades AT MIX TIME: `fade_in_s`/`fade_out_s` are the clip's fade_in/fade_out frame
/// counts in seconds, `clip_len_s` is the clip's FULL on-timeline length in seconds (so fade-out can
/// be measured from the clip end), and `range_local_s` is the clip-local seconds of the FIRST decoded
/// sample (0 for the full-clip render range; the head-trim for a playback clip straddling the
/// playhead). gcompose computes per-sample clip-local time = range_local_s + k/sr and ramps the gain
/// 0→1 over [0, fade_in_s) and 1→0 over [clip_len_s − fade_out_s, clip_len_s). FPS = RENDER_FPS (30).
///
/// `fx_chain` (P3, the 11th field) is the per-clip libavfilter chain from `build_audio_chain(clip.
/// audio_fx)` — a SPACE-FREE comma-joined filter expression, or "-" when the clip's AudioFx is
/// neutral. gcompose runs `fpx_au_apply` on the decoded range BEFORE the gain+fade+offset mix when
/// `fx_chain != "-"`; a "-" skips the filter entirely (byte-identical to P2). The chain is built
/// from `c.audio_fx` (Team B reads/writes audio_fx; never edits model.rs).
///
/// Clips with non-positive length, a corrupt media index, a non-audible/muted track, or whitespace in
/// the media path are skipped (a whitespace path would break the fixed-arity AUDIO parse).
///
/// T4 EXPORT REGION: `region_start`/`region_end` (frames, half-open) bound the rendered audio to the
/// exported sub-range. Every emitted clip is clipped against [region_start, region_end) — a clip that
/// starts before `region_start` is HEAD-TRIMMED (its first decoded sample is the source frame under
/// `region_start`), a clip that runs past `region_end` is TAIL-TRIMMED, and a clip entirely outside
/// the region is dropped — and every `dst_off_s` is shifted by `-region_start` so the clip at the
/// region start lands at audio t=0 (matching the region-length video, whose output frame 0 is
/// `region_start`). For the WHOLE-TIMELINE default (`region_start == 0 && region_end == total`) no
/// clip is dropped, no clip is head/tail-trimmed (every clip lies inside [0,total)), and the -0 shift
/// is a no-op → the SAME AUDIO lines as before (byte-identical). The head-trim math mirrors the
/// `program_levels` meter path (P24-aware: the sent source in-point + duration are speed-scaled).
fn build_audio_lines(project: &Project, region_start: i64, region_end: i64) -> Vec<String> {
    // Sort clip indices by timeline start for deterministic, readable output (order-independent now
    // that the worker positions by dst_offset). Stable on t0; ties keep the project's clip order.
    let mut idx: Vec<usize> = (0..project.clips.len()).collect();
    idx.sort_by_key(|&i| project.clips[i].t0);

    let fps = RENDER_FPS as f64;
    // P27 MASTER GAIN ENVELOPE: when the project carries a master-gain automation track, the FIRST
    // element of the returned Vec is the single GAINENV line (see `build_gainenv_line`). Every
    // consumer iterates this Vec verbatim AFTER the session opener (OPEN/MEAS/WAVE) and BEFORE the
    // AUDIO lines, so the engine installs the per-sample gain envelope on the active accumulator
    // before any clip is mixed. An EMPTY `gain_kf` prepends NOTHING → byte-identical to pre-P27.
    let mut lines = Vec::new();
    if let Some(env) = build_gainenv_line(project) {
        lines.push(env);
    }
    for i in idx {
        let c = &project.clips[i];
        if c.len <= 0 {
            continue;
        }
        // Track policy + Slice A mute: skip V2 overlay and any muted track.
        if !track_is_audible(project, c.track) {
            continue;
        }
        // T4 EXPORT REGION: clip this clip's timeline footprint [c.t0, c.end()) against the rendered
        // region [region_start, region_end). A clip entirely outside the region contributes nothing.
        // For the whole-timeline default (region_start==0, region_end==total) EVERY clip lies fully
        // inside, so head_skip==0 and eff_len==c.len and the math below collapses to the pre-T4 values.
        if c.end() <= region_start || c.t0 >= region_end {
            continue; // entirely before/after the exported region
        }
        let media_path = match project.media.get(c.media) {
            Some(p) => p,
            None => continue, // corrupt media index: skip this clip's audio (don't abort).
        };
        // Whitespace is now WIRE-SAFE: the AUDIO line percent-encodes the path token (enc_path)
        // below and the engine decodes it, so a spaced path no longer shifts the fixed-arity fields.
        // HEAD-TRIM (region start) + TAIL-TRIM (region end): how many of this clip's frames fall
        // before the region start (skipped), and the surviving on-timeline length inside the region.
        // For the whole-timeline default both are no-ops (head_skip==0, eff_len==c.len).
        let head_skip = (region_start - c.t0).max(0);
        let tail_cut = (c.end() - region_end).max(0);
        let eff_len = c.len - head_skip - tail_cut;
        if eff_len <= 0 {
            continue; // nothing of this clip survives inside the region
        }
        // P24 per-clip speed/reverse (Model A): the SOURCE window to read is eff_len*speed/fps and the
        // atempo/areverse in the chain compresses it back to the eff_len/fps timeline window. For the
        // whole-clip default (head_skip==0, eff_len==c.len) this is identity (speed==1.0 && !rev) →
        // src_in_s == c.src_in/fps, dur_s == c.len/fps, fx_chain == base_fx — byte-identical AUDIO line.
        let speed = (c.speed as f64).clamp(0.05, 16.0);
        let rev = c.reverse;
        let dur_src_s = (eff_len as f64 * speed) / fps;
        if !(dur_src_s.is_finite()) || dur_src_s <= 0.0 {
            continue;
        }
        // Source start of the FIRST decoded (region-clipped) sample. For the whole-clip default
        // head_skip==0, so the forward start collapses to src_in and the reverse start to
        // src_in + (len - 0 - len)*speed == src_in — the pre-T4 value. (Mirrors program_levels.)
        let src_in_s = if rev {
            let lo = c.src_in as f64 + ((c.len - head_skip - eff_len) as f64 * speed);
            lo.max(0.0) / fps
        } else {
            let src0 = c.src_in as f64 + (head_skip as f64 * speed);
            src0.max(0.0) / fps
        };
        let dur_s = dur_src_s;
        // T4: dst offset shifted by -region_start (clamped >=0) so a clip at the region start lands at
        // audio t=0. Whole-timeline default: region_start==0 → dst_off_s == c.t0/fps (byte-identical).
        let dst_off_s = (((c.t0 + head_skip) - region_start).max(0)) as f64 / fps;
        // Per-clip linear gain (P1) + fade envelope (frames → seconds). The first decoded sample is at
        // clip-local `head_skip` (0 for a clip wholly inside the region).
        // P42 MIXER GAIN: fold the per-TRACK fader into this clip's per-clip gain (the scalar that
        // rides the AUDIO line). track_gain default 1.0 → byte-identical for a default project.
        let gain = c.gain * project.track_gain(c.track);
        let fade_in_s = (c.fade_in.max(0)) as f64 / fps;
        let fade_out_s = (c.fade_out.max(0)) as f64 / fps;
        let clip_len_s = c.len as f64 / fps;
        // T4: clip-local seconds of the FIRST decoded sample = head_skip/fps (so the clip's fade-in,
        // measured from clip-local 0, still lines up when the region head-trims into the clip body).
        // Whole-timeline default head_skip==0 → 0.0 (byte-identical to pre-T4).
        let range_local_s = head_skip as f64 / fps;
        // P3 + P24: the per-clip libavfilter chain (space-free) or "-" when neutral, with the P24
        // areverse (if rev) + atempo stages (if speed != 1) PREPENDED before the AudioFx chain so the
        // post-filter buffer is timeline-length (the fade envelope still lines up).
        let base_fx = build_audio_chain(&c.audio_fx);
        let mut pre: Vec<String> = Vec::new();
        if rev {
            pre.push("areverse".to_string());
        }
        if (speed - 1.0).abs() > 1e-6 {
            for f in atempo_factors(speed) {
                pre.push(format!("atempo={:.6}", f));
            }
        }
        let fx_chain = if pre.is_empty() {
            base_fx
        } else {
            let mut all = pre;
            if base_fx != "-" {
                all.push(base_fx);
            }
            all.join(",")
        };
        // P42 MIXER PAN: fold the per-TRACK L/R balance onto the clip's fx_chain (stays space-free,
        // wire arity unchanged). track_pan default 0.0 → chain returned unchanged → byte-identical.
        let fx_chain = apply_track_pan(fx_chain, project.track_pan(c.track));
        // WHITESPACE-SAFE WIRE: percent-encode the media path token (enc_path) so a spaced path
        // stays one token; space-free paths are byte-identical. The engine dec_path's it before
        // opening the decoder.
        let media_path = enc_path(media_path);
        lines.push(format!(
            "AUDIO {media_path} {src_in_s} {dur_s} {dst_off_s} {gain} {fade_in_s} {fade_out_s} {clip_len_s} {range_local_s} {fx_chain}"
        ));
    }
    lines
}

/// Format the `ENC ...` line for timeline frame `t` (101 payload fields, no out path; P41 grew it
/// from 99 by appending the 2 solarize/temperature tokens sol_thr/temp as the new LAST fields; P39
/// grew it from 96 by appending the 3 selective-color tokens sel_band/sel_hshift/sel_sat;
/// P38 grew it from 93 by appending the 3 distort tokens mirror_x/kaleido/dither; P37 grew
/// it from 92 by appending the single ck_spill token; P34 grew it from 85 by
/// appending the 7 shape-mask tokens after eq_fov; P31 grew it from 84 by adding the
/// V2-overlay BLEND token right after `op`), baking the same composite as the preview. ENC total
/// token count = 102 (the `ENC` keyword + 101 payload fields). Returns None when the frame can't be
/// resolved.
fn build_enc_line(project: &Project, t: i64) -> Option<String> {
    Some(format_enc(&resolve_frame(project, t)?))
}

/// P5 STAGE 2: a neutral ENC line that just ENCODES a RAW-RGBA buffer (the folded N-layer composite)
/// — base = RAW:<path>, over = "-", every effect identity. Used by the render fold as the final
/// per-frame step (the fold writes the composite to a temp; this encodes it).
fn build_enc_raw(raw_path: &str) -> String {
    let r = Resolved {
        base_path: format!("RAW:{raw_path}"),
        base_frame: 0,
        over_path: "-".to_string(),
        over_frame: 0,
        op: 0.0,
        // P31 IDENTITY: this neutral ENC just encodes a RAW buffer with no overlay (over = "-"), so
        // blend 0 (Normal) keeps the N-layer render fold's final encode byte-identical to pre-P31.
        blend: 0,
        px: 0.0,
        py: 0.0,
        pw: 1.0,
        ph: 1.0,
        bright: 0.0,
        contrast: 1.0,
        sat: 1.0,
        cbright: 0.0,
        ccontrast: 1.0,
        csat: 1.0,
        look_kind: 0,
        look_amt: 1.0,
        lut_path: "-".to_string(),
        trans_kind: -1,
        trans_prog: 0.0,
        trans_param: 4.0,
        trans_path: "-".to_string(),
        trans_frame: 0,
        lift: [0.0, 0.0, 0.0],
        gamma: [1.0, 1.0, 1.0],
        gain_rgb: [1.0, 1.0, 1.0],
        rot: 0.0,
        scale: 1.0,
        blur: 0.0,
        ck_on: 0,
        ck_key: [0.0, 1.0, 0.0],
        ck_sim: 0.4,
        ck_smooth: 0.1,
        // P37 IDENTITY: ck_spill 0.0 (off) → engine no-ops the spill pass → the N-layer render fold's
        // final encode reproduces the composite byte-for-byte.
        ck_spill: 0.0,
        curve: [0.0, 0.25, 0.5, 0.75, 1.0],
        vignette: 0.0,
        sharpen: 0.0,
        flip: 0,
        fx: 0,
        hsl: [0.0, 1.0, 0.0],
        levels: [0.0, 1.0, 1.0],
        mosaic: 0,
        gmap_amt: 0.0,
        gmap_lo: [0.0, 0.0, 0.0],
        gmap_hi: [1.0, 1.0, 1.0],
        denoise: 0.0,
        glow_amt: 0.0,
        glow_thr: 0.7,
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
        // P23 IDENTITY: eq360 false makes the 360 reframe a byte-exact no-op (engine skips the kernel),
        // so the N-layer render fold's final encode reproduces the composite byte-for-byte.
        eq360: false,
        eq_yaw: 0.0,
        eq_pitch: 0.0,
        eq_fov: 90.0,
        // P34 IDENTITY: mask_shape 0 (none) makes the shape mask a byte-exact no-op (engine skips the
        // kernel), so the N-layer render fold's final encode reproduces the composite byte-for-byte.
        mask_shape: 0,
        mask_cx: 0.5,
        mask_cy: 0.5,
        mask_rw: 0.5,
        mask_rh: 0.5,
        mask_feather: 0.0,
        mask_invert: 0,
        // P38 IDENTITY: mirror_x 0 / kaleido 0 / dither 0 make the distort filters a byte-exact no-op
        // (engine skips each kernel), so the N-layer render fold's final encode reproduces the
        // composite byte-for-byte.
        mirror_x: 0,
        kaleido: 0,
        dither: 0.0,
        // P39 IDENTITY: sel_band 0 (none) makes the selective-color filter a byte-exact no-op (engine
        // skips the kernel), so the N-layer render fold's final encode reproduces the composite
        // byte-for-byte. sel_sat's NEUTRAL is 1.0 (a saturation MULTIPLIER), NOT 0.0.
        sel_band: 0,
        sel_hshift: 0.0,
        sel_sat: 1.0,
        // P41 IDENTITY: sol_thr 0.0 (solarize OFF) + temp 0.0 (neutral) make both filters byte-exact
        // no-ops (caller skips each kernel), so the N-layer render fold's final encode reproduces the
        // composite byte-for-byte.
        sol_thr: 0.0,
        temp: 0.0,
        fade: 1.0,
    };
    format_enc(&r)
}

/// Format an ENC wire line from a resolved frame spec (no out path). Split out of `build_enc_line`
/// (P5 Stage 2) so `build_enc_raw` can reuse the EXACT same 100-token format (keyword + 99 payload
/// fields; ENC has no out path; P39 appended the 3 selective-color tokens
/// sel_band/sel_hshift/sel_sat as the new LAST fields; P38 appended the 3 distort tokens
/// mirror_x/kaleido/dither; P37 appended the ck_spill token; P34 appended
/// the 7 shape-mask tokens after eq_fov; P31 added the V2-overlay BLEND token right after `op`).
fn format_enc(r: &Resolved) -> String {
    // Program grade (b/c/s) comes from the RESOLVED (keyframed) values so the render bakes the SAME
    // keyframed grade the preview shows — not the static project.bright/contrast/sat (Slice A). The
    // 3 LOOK fields (look_kind, look_amt, lut_path), then the 5 Wave 8 TRANSITION fields (trans_kind,
    // trans_prog, trans_param, trans_path, trans_frame), then the 3 Triad-B P1 PER-CLIP GRADE fields
    // (cbright, ccontrast, csat), then the 12 Triad-B P2 fields (lift_r lift_g lift_b  gamma_r gamma_g
    // gamma_b  gain_r gain_g gain_b  rot scale blur) are appended in the PINNED order so the render
    // bakes the SAME per-clip look + grade + color-wheels + transform + blur + animated transition
    // the preview shows (white-balance already folded into the 9 lift/gamma/gain). Then the 6 Triad-A
    // P4 CHROMA-KEY fields (ck_on ck_r ck_g ck_b ck_sim ck_smooth) describing the OVER (V2) clip —
    // identity (ck_on=0) when there is no overlay / chroma disabled, so a no-chroma render is
    // byte-identical to P3. Then the 5 P5 CURVE fields (cv0..cv4), then the 4 P6 STYLIZE/UTILITY
    // fields (vig sharp flip fx) in the PINNED order, then the 6 P7 COLOR fields (hue sat light inb
    // inw gam) in the PINNED order (hue=hsl[0], sat=hsl[1], light=hsl[2], inb=levels[0], inw=levels[1],
    // gam=levels[2]), then the 8 P8 STYLIZE-2 fields (mosaic gmap_amt glo_r glo_g glo_b ghi_r ghi_g
    // ghi_b) in the PINNED order (mosaic=mosaic, gmap_amt=gmap_amt, glo=gmap_lo[0..3],
    // ghi=gmap_hi[0..3]), then the 4 P9 FX fields (denoise glow_amt glow_thr rgbshift) in the PINNED
    // order, then the 3 P10 STYLIZE-4 fields (halftone emboss edge) in the PINNED order, then the 3
    // P13 OLD-FILM fields (grain scratches diffusion) in the PINNED order, then the 3
    // P16 DISTORT fields (wave swirl threshold) in the PINNED order, then the 3
    // P17 GEOMETRIC fields (lens crop glitch) in the PINNED order, then the 4
    // P23 360 REFRAME fields (eq360 eq_yaw eq_pitch eq_fov) in the PINNED order, then the 7
    // P34 SHAPE MASK fields (mask_shape mask_cx mask_cy mask_rw mask_rh mask_feather mask_invert) in
    // the PINNED order, then the 1 P37 CHROMA-SPILL field (ck_spill — appended so
    // it does NOT shift any existing ck_* / mask index), then the 3 P38 DISTORT fields
    // (mirror_x kaleido dither — APPENDED so they do NOT shift any existing index), then the 3 P39
    // SELECTIVE-COLOR fields (sel_band sel_hshift sel_sat — APPENDED so they do
    // NOT shift any existing index), then the 2 P41 SOLARIZE / COLOUR-TEMPERATURE fields
    // (sol_thr temp — APPENDED AS THE NEW LAST TOKENS so they do NOT shift any existing index). ENC has
    // NO out path — temp is the LAST ENC
    // field → ENC is now 102 tokens incl the keyword (P39 was 100, P38 97). P31: the V2-overlay BLEND
    // token rides at f[6] (right after op f[5]), shifting every later field +1 vs P23. P34: the 7
    // shape-mask tokens ride at f[86..=92]. P37: ck_spill is appended at f[93]. P38: the 3 distort
    // tokens ride at f[94..=96]. P39: the 3 selective-color tokens ride at f[97..=99]. P41: the 2
    // solarize/temperature tokens ride at f[100..=101]. The engine reads
    // (keyword = f[0]): base f[1], over f[2],
    // bf f[3], of f[4], op f[5], blend f[6], px f[7] ... curve at f[43..=47],
    // vig f[48], sharp f[49], flip f[50], fx f[51], hue f[52], sat f[53], light f[54], inb f[55],
    // inw f[56], gam f[57], mosaic f[58], gmap_amt f[59], glo f[60..=62], ghi f[63..=65], denoise
    // f[66], glow_amt f[67], glow_thr f[68], rgbshift f[69], halftone f[70], emboss f[71], edge
    // f[72], grain f[73], scratches f[74], diffusion f[75], wave f[76], swirl f[77], threshold f[78],
    // lens f[79], crop f[80], glitch f[81], eq360 f[82], eq_yaw f[83], eq_pitch f[84], eq_fov f[85],
    // mask_shape f[86], mask_cx f[87], mask_cy f[88], mask_rw f[89], mask_rh f[90], mask_feather f[91],
    // mask_invert f[92], ck_spill f[93], mirror_x f[94], kaleido f[95], dither f[96], sel_band f[97],
    // sel_hshift f[98], sel_sat f[99], sol_thr f[100], temp f[101].
    // blend is emitted as an INTEGER token (0=Normal..7=Difference; engine parses i32); blend 0 makes
    // the engine do a plain alpha-over → byte-identical to pre-P31. eq360 is also an INTEGER token
    // (1 = on, 0 = off; engine parses i32, nonzero = on). When
    // eq360 is 0 the engine returns immediately (no kernel run) so the frame is byte-identical to
    // pre-P23. mask_shape and mask_invert are INTEGER tokens (engine parses i32); when mask_shape is 0
    // (none) the engine returns immediately (no kernel run) so the frame is byte-identical to pre-P34.
    // ck_spill is a plain f32 token; ck_spill 0 (or ck_on 0) skips the spill pass inside k_chroma, so
    // the frame is byte-identical to pre-P37. mirror_x and kaleido are INTEGER tokens (engine parses
    // i32), dither is a plain f32; mirror_x 0 / kaleido <2 / dither 0 → engine no-ops each distort
    // kernel → byte-identical to pre-P38. sel_band is an INTEGER token (engine parses i32; 0..6),
    // sel_hshift / sel_sat are plain f32; sel_band 0 (none) → engine no-ops the selective-color kernel
    // → byte-identical to pre-P39. NOTE sel_sat's NEUTRAL value is 1.0 (a saturation MULTIPLIER), NOT 0.0.
    // P41: sol_thr / temp are plain f32 tokens; sol_thr 0.0 (off) and temp 0.0 (neutral) → caller skips
    // each kernel → byte-identical to pre-P41.
    format!(
        "ENC {base} {over} {bf} {of} {op} {blend} {px} {py} {pw} {ph} {b} {c} {s} {lk} {la} {lut} \
         {tk} {tp} {tparam} {tpath} {tframe} {cb} {cc} {cs} \
         {lr} {lg} {lb} {gmr} {gmg} {gmb} {gnr} {gng} {gnb} {rot} {scl} {blr} \
         {ckon} {ckr} {ckg} {ckb} {cksim} {cksm} {cv0} {cv1} {cv2} {cv3} {cv4} \
         {vig} {sharp} {flip} {fx} {hue} {sat} {light} {inb} {inw} {gam} \
         {mosaic} {gmapamt} {glor} {glog} {glob} {ghir} {ghig} {ghib} \
         {denoise} {glowamt} {glowthr} {rgbshift} {halftone} {emboss} {edge} \
         {grain} {scratches} {diffusion} {wave} {swirl} {threshold} \
         {lens} {crop} {glitch} {eq360} {eqyaw} {eqpitch} {eqfov} \
         {maskshape} {maskcx} {maskcy} {maskrw} {maskrh} {maskfeather} {maskinvert} {ckspill} \
         {mirrorx} {kaleido} {dither} {selband} {selhshift} {selsat} {solthr} {temp} {fade}",
        base = enc_path(&r.base_path),
        over = enc_path(&r.over_path),
        bf = r.base_frame,
        of = r.over_frame,
        op = r.op,
        // P31 BLEND: the V2 overlay's blend mode as ONE integer token IMMEDIATELY AFTER `op` — the
        // engine's k_pip parser reads it from this exact position (same offset as the PREVIEW line).
        // 0 (Normal) => plain alpha-over => byte-identical to pre-P31.
        blend = r.blend,
        px = r.px,
        py = r.py,
        pw = r.pw,
        ph = r.ph,
        b = r.bright,
        c = r.contrast,
        s = r.sat,
        lk = r.look_kind,
        la = r.look_amt,
        lut = enc_path(&r.lut_path),
        tk = r.trans_kind,
        tp = r.trans_prog,
        tparam = r.trans_param,
        tpath = enc_path(&r.trans_path),
        tframe = r.trans_frame,
        cb = r.cbright,
        cc = r.ccontrast,
        cs = r.csat,
        lr = r.lift[0],
        lg = r.lift[1],
        lb = r.lift[2],
        gmr = r.gamma[0],
        gmg = r.gamma[1],
        gmb = r.gamma[2],
        gnr = r.gain_rgb[0],
        gng = r.gain_rgb[1],
        gnb = r.gain_rgb[2],
        rot = r.rot,
        scl = r.scale,
        blr = r.blur,
        ckon = r.ck_on,
        ckr = r.ck_key[0],
        ckg = r.ck_key[1],
        ckb = r.ck_key[2],
        cksim = r.ck_sim,
        cksm = r.ck_smooth,
        cv0 = r.curve[0],
        cv1 = r.curve[1],
        cv2 = r.curve[2],
        cv3 = r.curve[3],
        cv4 = r.curve[4],
        vig = r.vignette,
        sharp = r.sharpen,
        flip = r.flip,
        fx = r.fx,
        hue = r.hsl[0],
        sat = r.hsl[1],
        light = r.hsl[2],
        inb = r.levels[0],
        inw = r.levels[1],
        gam = r.levels[2],
        mosaic = r.mosaic,
        gmapamt = r.gmap_amt,
        glor = r.gmap_lo[0],
        glog = r.gmap_lo[1],
        glob = r.gmap_lo[2],
        ghir = r.gmap_hi[0],
        ghig = r.gmap_hi[1],
        ghib = r.gmap_hi[2],
        denoise = r.denoise,
        glowamt = r.glow_amt,
        glowthr = r.glow_thr,
        rgbshift = r.rgbshift,
        halftone = r.halftone,
        emboss = r.emboss,
        edge = r.edge,
        grain = r.grain,
        scratches = r.scratches,
        diffusion = r.diffusion,
        wave = r.wave,
        swirl = r.swirl,
        threshold = r.threshold,
        lens = r.lens,
        crop = r.crop,
        glitch = r.glitch,
        // P23 360 reframe: eq360 emitted as an INTEGER flag token (1 = on, 0 = off) to match the
        // engine's i32 parse (NOT a bool literal "true"/"false"); yaw/pitch/fov as plain f32 with the
        // same Display formatting as the neighbouring lens/crop/glitch fields. eq360 = 0 → engine
        // no-op → byte-identical to pre-P23.
        eq360 = if r.eq360 { 1 } else { 0 },
        eqyaw = r.eq_yaw,
        eqpitch = r.eq_pitch,
        eqfov = r.eq_fov,
        // P34 shape mask: mask_shape emitted as an INTEGER token (0 = none/off, 1 = rectangle,
        // 2 = ellipse) and mask_invert as an INTEGER flag (1 = invert, 0 = normal) to match the
        // engine's i32 parse (NOT bool literals "true"/"false"); cx/cy/rw/rh/feather as plain f32 with
        // the same Display formatting as the neighbouring eq_yaw/eq_pitch/eq_fov fields. mask_shape 0
        // → engine no-op → byte-identical to pre-P34. These 7 fields precede the LAST ENC field
        // (ck_spill); ENC has no out path.
        maskshape = r.mask_shape,
        maskcx = r.mask_cx,
        maskcy = r.mask_cy,
        maskrw = r.mask_rw,
        maskrh = r.mask_rh,
        maskfeather = r.mask_feather,
        maskinvert = r.mask_invert,
        // P37 chroma SPILL suppression: a plain f32 token APPENDED AS THE NEW LAST ENC FIELD (ENC has
        // no out path, so ck_spill is literally the final token). Appending it here (rather than
        // inserting among the ck_* fields) keeps every existing wire index unchanged. ck_spill 0 (or
        // ck_on 0) → k_chroma skips the spill pass → byte-identical to pre-P37.
        ckspill = r.ck_spill,
        // P38 DISTORT (MIRROR + KALEIDOSCOPE + DITHER): 3 tokens APPENDED AS THE FINAL ENC FIELDS
        // (ENC has no out path, so dither is literally the last token) in the PINNED order
        // `mirror_x kaleido dither`. mirror_x and kaleido are INTEGER tokens (engine parses i32);
        // dither is a plain f32. mirror_x 0 / kaleido <2 / dither 0 → engine no-ops each kernel →
        // byte-identical to pre-P38. Appending them here keeps every existing wire index unchanged
        // (ENC arity 93 → 96 payload fields, 97 incl the keyword).
        mirrorx = r.mirror_x,
        kaleido = r.kaleido,
        dither = r.dither,
        // P39 SELECTIVE COLOR (one HUE BAND): 3 tokens APPENDED AS THE FINAL ENC FIELDS (ENC has no out
        // path, so sel_sat is literally the last token) in the PINNED order
        // `sel_band sel_hshift sel_sat`. sel_band is an INTEGER token (engine parses i32; 0..6);
        // sel_hshift / sel_sat are plain f32. sel_band 0 (none) → engine no-ops the kernel →
        // byte-identical to pre-P39. NOTE sel_sat's NEUTRAL value is 1.0 (a saturation MULTIPLIER), so
        // a neutral clip emits `... <dither> 0 0 1` (NOT sel_sat 0, which would desaturate the band).
        // Appending them here keeps every existing wire index unchanged (ENC arity 96 → 99 payload
        // fields, 100 incl the keyword).
        selband = r.sel_band,
        selhshift = r.sel_hshift,
        selsat = r.sel_sat,
        // P41 SOLARIZE + COLOUR TEMPERATURE: 2 tokens APPENDED AFTER sel_sat AS THE FINAL ENC FIELDS
        // (ENC has no out path, so temp is literally the last token) in the PINNED order `sol_thr temp`
        // (sol_thr first, temp second). Both are plain f32. sol_thr 0.0 (off) → caller skips the
        // solarize kernel; temp 0.0 (neutral) → caller skips the temperature kernel → byte-identical
        // to pre-P41. Appending them here keeps every existing wire index unchanged (ENC arity
        // 99 → 101 payload fields, 102 incl the keyword).
        solthr = r.sol_thr,
        temp = r.temp,
        fade = r.fade,
    )
}

/// Decode one frame of `media_path` letterboxed to `w*h` -> RGBA8 (`w*h*4` bytes), via the
/// worker's THUMB command (no composite). Returns None on failure. Used for pool/clip thumbs.
/// Total video frame count of `media_path` (the engine's `NFRAMES` query), or None if the engine is
/// unavailable / the count is unknown (returns Some(0) → None here). Used by the UI to clamp a clip's
/// length to its source so it never references frames past the media end. One stateless round-trip
/// (the count comes back INLINE as the DONE payload — no temp file).
pub fn media_frames(media_path: &str) -> Option<i64> {
    let req = format!("NFRAMES {}", enc_path(media_path));
    let payload = command_with_restart(&req)?;
    let n: i64 = payload.trim().parse().ok()?;
    if n > 0 {
        Some(n)
    } else {
        None
    }
}

pub fn thumbnail(media_path: &str, frame: i64, w: usize, h: usize) -> Option<Vec<u8>> {
    if w == 0 || h == 0 {
        return None;
    }
    let out = thumb_temp_path(media_path, frame, w, h);
    // WHITESPACE-SAFE WIRE: percent-encode the media path token (enc_path) so a spaced path stays
    // one token; the engine dec_path's it before opening the decoder. The out token is a hashed
    // /tmp path (no whitespace) → enc_path is identity, wrapped for symmetry. Space-free paths are
    // byte-identical to before.
    let req = format!("THUMB {} {frame} {w} {h} {}", enc_path(media_path), enc_path(&out));
    let payload = command_with_restart(&req)?;
    // Worker echoes the out path on DONE; trust our own path if it echoes empty.
    let read_path = if payload.is_empty() { out.clone() } else { payload };
    let bytes = std::fs::read(&read_path).ok()?;
    if bytes.len() == w * h * 4 {
        Some(bytes)
    } else {
        None
    }
}

/// Per-track peak audio envelope (`buckets` peaks in 0..1) of `media_path`, via the worker's ENV
/// command. The worker writes `buckets` little-endian f32 to a temp file; we read them back.
/// Returns None if the file has no audio / on any failure. Used for waveform display.
pub fn audio_envelope(media_path: &str, buckets: usize) -> Option<Vec<f32>> {
    if buckets == 0 {
        return None;
    }
    let out = env_temp_path(media_path, buckets);
    // WHITESPACE-SAFE WIRE: percent-encode the media path token (enc_path) so a spaced path stays
    // one token; the engine dec_path's it before opening the decoder. The out token is a hashed
    // /tmp path (no whitespace) → enc_path is identity, wrapped for symmetry. Space-free paths are
    // byte-identical to before.
    let req = format!("ENV {} {buckets} {}", enc_path(media_path), enc_path(&out));
    let payload = command_with_restart(&req)?;
    let read_path = if payload.is_empty() { out.clone() } else { payload };
    let bytes = std::fs::read(&read_path).ok()?;
    if bytes.len() != buckets * 4 {
        return None;
    }
    let mut env = Vec::with_capacity(buckets);
    for chunk in bytes.chunks_exact(4) {
        env.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(env)
}

/// P46 AUDIO ALIGN — decode a clip's source range `[start_s, start_s+dur_s)` to MONO @ `sr`,
/// returning exactly `n` f32 samples (zero-padded), via the worker's CLIPAUD command. None on failure.
pub fn clip_audio(media_path: &str, start_s: f64, dur_s: f64, sr: i32, n: usize) -> Option<Vec<f32>> {
    if n == 0 || sr <= 0 {
        return None;
    }
    let out = format!(
        "/tmp/genesis_clipaud_{:x}.f32",
        small_hash(&format!("{media_path}|{start_s}|{dur_s}|{sr}|{n}"))
    );
    let req = format!(
        "CLIPAUD {} {start_s} {dur_s} {sr} {n} {}",
        enc_path(media_path),
        enc_path(&out)
    );
    let payload = command_with_restart(&req)?;
    let read_path = if payload.is_empty() { out.clone() } else { payload };
    let bytes = std::fs::read(&read_path).ok()?;
    if bytes.len() != n * 4 {
        return None;
    }
    let mut v = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(v)
}

/// P46 AUDIO ALIGN — the t0 DELTA (timeline frames) to ADD to the MOVING clip (`mov_idx`) so its audio
/// syncs to the REFERENCE clip (`ref_idx`). Decodes each clip's current source range to low-rate mono
/// and cross-correlates (`model::cross_correlation_offset`). None if either clip has no decodable audio.
pub fn align_audio_offset_frames(project: &Project, ref_idx: usize, mov_idx: usize) -> Option<i64> {
    // SR low enough to keep the O(N*max_lag) correlation responsive (4000 Hz -> ~133 samples/frame,
    // still frame-accurate); WINDOW caps the decoded length and MAX_OFFSET_S caps the lag search so a
    // pair of long clips can't hang the UI (worst case ~SR*WINDOW * SR*MAX_OFFSET ops).
    const SR: i32 = 4000;
    const WINDOW_S: f64 = 15.0;
    const MAX_OFFSET_S: f64 = 10.0;
    let rc = project.clips.get(ref_idx)?;
    let mc = project.clips.get(mov_idx)?;
    let rpath = project.media.get(rc.media)?.clone();
    let mpath = project.media.get(mc.media)?.clone();
    let dec = |path: &str, src_in: i64, len: i64| -> Option<Vec<f32>> {
        let start = src_in.max(0) as f64 / RENDER_FPS as f64;
        let dur = (len.max(1) as f64 / RENDER_FPS as f64).min(WINDOW_S); // bound the analysis window
        let n = ((dur * SR as f64).ceil() as usize).max(1);
        let v = clip_audio(path, start, dur, SR, n)?;
        if v.iter().all(|&x| x == 0.0) {
            None // no audio content -> can't align
        } else {
            Some(v)
        }
    };
    let a = dec(&rpath, rc.src_in, rc.len)?;
    let b = dec(&mpath, mc.src_in, mc.len)?;
    let max_lag = a.len().max(b.len()).min((SR as f64 * MAX_OFFSET_S) as usize);
    let lag = crate::model::cross_correlation_offset(&a, &b, max_lag); // +lag = b lags a (samples)
    let lag_frames = (lag as f64 / SR as f64 * RENDER_FPS as f64).round() as i64;
    Some(-lag_frames) // shift the moving clip EARLIER by its lag to sync to the reference
}

/// A stable-ish temp path for a thumbnail of `path` @ `frame` at `w×h`. Hashing the inputs keeps
/// concurrent THUMB requests for different media/frames from clobbering each other's output file.
fn thumb_temp_path(path: &str, frame: i64, w: usize, h: usize) -> String {
    format!("/tmp/genesis_thumb_{:x}.rgba", small_hash(&format!("{path}|{frame}|{w}|{h}")))
}

/// A temp path for the envelope of `path` @ `buckets`.
fn env_temp_path(path: &str, buckets: usize) -> String {
    format!("/tmp/genesis_env_{:x}.f32", small_hash(&format!("{path}|{buckets}")))
}

/// Percent-encode whitespace (and the escape char itself) in a single wire PATH TOKEN so a path
/// containing spaces/tabs/newlines stays ONE whitespace-split token on the UI→gcompose wire (the
/// protocol is whitespace-delimited with FIXED arity — an unencoded space would split into extra
/// tokens and shift every following field). The exact INVERSE lives in `gcompose/src/main.rs`
/// (`dec_path`); the two CANNOT share a fn (separate binaries).
///
/// Order matters: encode "%" FIRST (so a real "%" in the path becomes "%25" and survives), then the
/// whitespace bytes. A token with no whitespace AND no "%" is returned UNCHANGED (identity), so:
///   - the "-" sentinel, a "RAW:/tmp/x" raster path, and every space-free pool path encode to
///     THEMSELVES → the wire bytes are byte-identical to the pre-encoding protocol (no regression).
///   - a "RAW:" prefix has no whitespace/% so it passes through untouched; the engine strips "RAW:"
///     AFTER decoding (decode of an unencoded prefix is identity), so the order is fine.
/// The REAL (unencoded) path stays in `project.media`; only the WIRE copy is encoded.
fn enc_path(s: &str) -> String {
    // "%"->"%25" FIRST so real percents survive the whitespace pass; then the whitespace bytes.
    s.replace('%', "%25")
        .replace(' ', "%20")
        .replace('\t', "%09")
        .replace('\n', "%0A")
        .replace('\r', "%0D")
}

/// Tiny FNV-1a hash for building collision-resistant temp filenames (no extra deps).
fn small_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
