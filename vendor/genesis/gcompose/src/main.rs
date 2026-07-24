//! gcompose — the Genesis engine worker (a separate process from the egui UI).
//!
//! Two modes:
//!
//!  1. One-shot (back-compat, unchanged):
//!       `gcompose <base> <over|-> <out.rgba>`
//!     decode `base` frame 60 + `over` frame 0 (letterboxed to GVW×GVH), run the OpenCL
//!     PiP composite + grade, write the GVW×GVH RGBA8 result to `out.rgba`.
//!     Falls back to a plain decoded `base` frame if OpenCL is unavailable.
//!
//!  2. Persistent serve mode (P1):
//!       `gcompose --serve`
//!     call fpx_gpu_init() ONCE, then read one request per line from stdin, compose the
//!     requested program frame, write raw RGBA to the per-request out path, and print
//!     "DONE <out_path>\n" (or "ERR\n") to stdout, flushing after each. Decoders are cached
//!     per media path (HashMap) so a held playhead / repeated frame reuses the open handle.
//!
//! Serve commands (one per line; reply "DONE..."/"ERR\n", always flushed):
//!
//!   Preview frame (PREVIEW keyword + 24 positional fields; a keyword-less line is still
//!   accepted for back-compat with one-shot/older clients):
//!     PREVIEW <base> <over|-> <bf> <of> <op> <px> <py> <pw> <ph> <bright> <contrast> <sat>
//!             <look_kind> <look_amt> <lut_path|->
//!             <trans_kind> <trans_prog> <trans_param> <trans_path|-> <trans_frame>
//!             <cbright> <ccontrast> <csat> <out>
//!     -> compose program frame (incl. the per-clip LOOK + any per-boundary TRANSITION), write RGBA
//!     to <out>; reply "DONE <out>". A "-" base path renders a black frame (timeline gap). look_kind:
//!     0=none, 1=VHS, 2=LUT3D (loads <lut_path> .cube, cached); a missing/failed LUT degrades to no
//!     look. trans_kind: -1 = no transition (no slot-2 upload, track1(-1,0,4) copies base); 0..7 = a
//!     transition kernel (0=crossfade..7=dissolve): decode <trans_path>@<trans_frame> (cached) into
//!     slot 2 and run fpx_gpu_track1(trans_kind, trans_prog, trans_param) at the START of the
//!     pipeline (before pip/grade/look). The PREVIEW also records which buffer (OUTB/look-none vs
//!     LOOKB/look) the frame ended in, so a following SCOPE reads the POST-LOOK frame.
//!
//!   Render/export (Slice A video + TIMELINE-SYNCED audio; Triad-B P1 export controls + P25 depth):
//!     OPEN <out> <out_w> <out_h> <fps_num> <fps_den> <rate_mode> <rate_value> <vcodec> <total_s> <gop> <preset> <abitrate>
//!        -> open + config_video(<vcodec>, in=GVW×GVH, out=out_w×out_h @ fps_num/fps_den; rate_mode
//!           0=avg bitrate (rate_value=bits/s), 1=constant quality (rate_value=CRF via av_opt_set);
//!           P25: <gop>=keyframe interval in frames (<=0 keeps the encoder default gop_size),
//!           <preset>=x264/x265 encoder preset token ("-" => none) applied via av_opt_set)
//!           + config_audio(aac,2ch, 48000, <abitrate> bits/s [<=0 => 128000]) + start; reply
//!           DONE/ERR. ALSO allocates the PROGRAM-AUDIO ACCUMULATOR: an f32 stereo @ 48000 buffer
//!           sized to <total_s> seconds (the timeline duration), zero-filled (silence). The
//!           encoder is ready for BOTH streams: ENC feeds video, AUDIO MIXES into the accumulator
//!           at a destination offset, CLOSE feeds the WHOLE accumulator to the encoder then
//!           finalizes — so the rendered audio is timeline-positioned and its length matches the
//!           video. Gaps stay silent; overlaps mix (sample-add, clamped to [-1,1]).
//!     ENC <base> <over|-> <bf> <of> <op> <px> <py> <pw> <ph> <bright> <contrast> <sat>
//!         <look_kind> <look_amt> <lut_path|->
//!         <trans_kind> <trans_prog> <trans_param> <trans_path|-> <trans_frame>
//!         <cbright> <ccontrast> <csat>
//!        -> decode(cached) + compose(track1->pip->grade_clip(per-clip)->grade(program)->look)
//!           + feed the composited f32 frame to the encoder at ts = enc_count/fps; reply DONE/ERR; no
//!           file. look_kind: 0=none, 1=VHS, 2=LUT3D (loads <lut_path> .cube, cached per path); a
//!           missing/failed LUT degrades to no look (the frame still encodes). trans_kind: -1 = no
//!           transition (track1(-1,0,4) copies base); 0..7 = a transition kernel — decode
//!           <trans_path>@<trans_frame> (cached) into slot 2 and blend base→trans by <trans_prog> at
//!           the START of the pipeline (matching the PREVIEW path). A "-"/failed trans_path degrades
//!           to no transition (the frame still encodes the base).
//!        NOTE: BOTH PREVIEW and ENC carry a long PINNED TAIL of per-clip effect fields after the
//!           base composite/look/transition fields (P1..P41 each appended/inserted theirs). The full
//!           index map lives in the inline comments at each parser. The P41 tail adds 2 fields in the
//!           pinned order `sol_thr temp` (sol_thr = solarize threshold f32, 0=off; temp = colour
//!           temperature f32, 0=neutral): ENC APPENDS them as the new LAST tokens at f[100..=101]
//!           (ENC has no out path) → ENC arity 100→102; PREVIEW INSERTS them between sel_sat and the
//!           out path at f[99..=100] (the out path moves to f[101]) → PREVIEW post-strip arity
//!           100→102. Both default to their no-op (0.0) via unwrap_or, so a pre-P41 project renders
//!           byte-identical.
//!     AUDIO <path> <src_in_s> <dur_s> <dst_offset_s> <gain> <fade_in_s> <fade_out_s> <clip_len_s> <range_local_s> <fx_chain|->
//!        -> decode that SOURCE audio range [src_in_s, src_in_s+dur_s) (fpx_decode_audio_range ->
//!           2ch @ 48000 interleaved f32), apply the per-clip libavfilter <fx_chain> (P3; when != "-"
//!           run fpx_au_apply on the decoded range, replacing it with the filtered buffer — a graph
//!           failure falls back to the unfiltered range so audio never drops), THEN per-clip <gain> +
//!           the fade ENVELOPE (ramp 0→1 over [0,fade_in_s), 1→0 over [clip_len_s−fade_out_s,
//!           clip_len_s) in clip-local time, where the first decoded sample is at clip-local
//!           <range_local_s>), and MIX (sample-add, clamp) into the active accumulator (render OR
//!           playback OR measurement) starting at <dst_offset_s> seconds. Replies DONE/ERR; a range
//!           with no audio (or a decode failure) replies ERR so the client can skip that clip without
//!           aborting. NOTHING is fed to the encoder here (deferred to CLOSE), so AUDIO is also valid
//!           in a playback-WAV / measurement session that has no encoder. <fx_chain> is "-" when the
//!           clip's AudioFx is neutral → byte-identical to the P2 mix (11 tokens; was 10 in P1).
//!     GAINENV <packed>
//!        -> set (or clear) the P27 MASTER GAIN ENVELOPE (a master audio-VOLUME automation curve over
//!           the timeline) on the active accumulator. <packed> is a SINGLE space-free token
//!           "t0:v0,t1:v1,..." (t in SECONDS f64, v the gain multiplier f32; pairs comma-separated,
//!           t:v colon-separated) OR "-" meaning NO envelope (clear). Parsed into a sorted
//!           Vec<(sec,gain)>; malformed pairs are skipped. Sent AFTER the session opener (OPEN/WAVE/
//!           MEAS) and BEFORE the AUDIO lines so each per-clip mix multiplies its samples by the
//!           envelope at each sample's ABSOLUTE timeline time. Empty/"-" → eval_gain_env == 1.0 →
//!           per-sample mix byte-identical to pre-P27. Replies DONE/ERR (ERR if no active accumulator).
//!     CLOSE
//!        -> feed the ENTIRE accumulator to the encoder (fpx_enc_audio_samples_f32 in chunks),
//!           then finish + close (flushes + writes BOTH video and audio); reply DONE.
//!     WAVE <out_wav> <total_s>
//!        -> begin a PLAYBACK accumulator session (no encoder): allocate an f32 stereo @ 48000
//!           accumulator sized to <total_s>; subsequent AUDIO lines mix into it; reply DONE/ERR.
//!     WAVECLOSE <out_wav>
//!        -> write the playback accumulator to <out_wav> as a 16-bit PCM stereo @ 48000 WAV and
//!           clear it; reply DONE/ERR. The UI then spawns a system player (paplay/aplay) on it.
//!     MEAS <window_s>
//!        -> begin a MEASUREMENT-only accumulator session (no encoder, no WAV) for the level meter:
//!           allocate an f32 stereo @ 48000 accumulator sized to <window_s>; subsequent AUDIO lines
//!           mix the filtered+gained ranges into it; reply DONE/ERR.
//!     LEVELS <out>
//!        -> measure the active accumulator's per-channel PEAK + RMS (dBFS), write 4 little-endian
//!           f32 [peak_L, peak_R, rms_L, rms_R] to <out>, then CLEAR the accumulator (session
//!           terminator, mirroring WAVECLOSE); reply DONE <out>/ERR. The UI draws a stereo peak+RMS
//!           meter from these. Reflects the ASSEMBLED mix (no real-time device capture).
//!     SPECTRUM <nbins> <out>
//!        -> compute the active accumulator's magnitude spectrum (Hann-windowed 4096-sample mono mix
//!           -> radix-2 FFT -> first N/2 magnitudes grouped LINEARLY peak-hold into <nbins> bars over
//!           [0, sr/2]), write EXACTLY <nbins> little-endian f32 magnitudes to <out>, then CLEAR the
//!           accumulator (session terminator, mirroring LEVELS); reply DONE <out>/ERR. The UI draws a
//!           frequency-spectrum (audio spectrum scope) from these. Read-only analysis — does NOT touch
//!           the render/mix path. With sr=48000 and nbins=256 each bar spans 93.75 Hz.
//!     SAMPLES <n> <out>
//!        -> read the active accumulator's LEFT channel over the first min(4096, frames) frames,
//!           DECIMATED to <n> raw time-domain amplitude points (~[-1,1]); write EXACTLY <n> little-endian
//!           f32 to <out>, then CLEAR the accumulator (session terminator, mirroring SPECTRUM/LEVELS);
//!           reply DONE <out>/ERR. The UI draws a time-domain oscilloscope (audio waveform scope) from
//!           these. Read-only analysis — does NOT touch the render/mix path. SAMPLES_N=256 is the n the
//!           UI sends.
//!     THUMB <path> <frame> <w> <h> <out>
//!        -> decode <frame> letterboxed to w×h -> write RGBA8 to <out>; reply DONE/ERR.
//!     ENV <path> <buckets> <out>
//!        -> fpx_audio_envelope -> write <buckets> little-endian f32 to <out>; reply DONE/ERR.
//!     SCOPE <kind> <out>
//!        -> run the kind-selected scope kernel on the LAST composed GPU buffer (the frame left in
//!           g_buf[OUTB] by the most recent PREVIEW — NOT re-composed here, so the client sends a
//!           PREVIEW for the wanted frame first), produce a 256×256 RGBA8 image, write it to <out>;
//!           reply DONE/ERR. kind 0 = histogram (the 768 R/G/B bins are RASTERIZED into a 256×256
//!           bar graph on a dark bg, since the histogram kernel returns raw bins not an image),
//!           kind 1 = luma waveform (kernel renders the image directly), kind 2 = vectorscope
//!           (kernel renders the image directly), kind 3 = RGB PARADE (Triad-B P1: three side-by-side
//!           per-channel column waveforms R|G|B, kernel renders the image directly). The scope reads the buffer the most recent
//!           PREVIEW left the frame in — g_buf[OUTB] when that PREVIEW had look=none, g_buf[LOOKB]
//!           when a VHS/LUT look ran — so the scope reflects the POST-LOOK displayed frame.
//!
//! This binary links the C engine (FFmpeg + OpenCL) but NO GUI libraries, so it owns the
//! OpenCL driver init in isolation — the egui process never touches OpenCL (see workspace
//! Cargo.toml for why).

mod ffi;

use std::collections::HashMap;
use std::io::{BufRead, Write};

/// Decode a single wire PATH TOKEN that the UI percent-encoded with `worker.rs::enc_path` — the
/// EXACT inverse of that helper (the two CANNOT share a fn: separate binaries).
///
/// Order is critical and is the mirror of enc_path (which encodes "%"->"%25" FIRST, then the
/// whitespace bytes):
///   dec:  "%20"->" " / "%09"->tab / "%0A"->nl / "%0D"->cr  FIRST, then "%25"->"%"  LAST.
/// Decoding "%25"->"%" LAST is what makes a real percent round-trip: enc turns a literal "%20" path
/// segment into "%2520" on the wire; here the whitespace pass leaves "%2520" untouched (it contains
/// no bare "%20" subsequence — the "%25" guards it), then "%25"->"%" restores "%20".
///
/// A token with NO "%" is returned UNCHANGED (identity): the "-" sentinel decodes to "-", a
/// "RAW:/tmp/x" raster path decodes to itself (its "RAW:" prefix is then stripped by upload_slot,
/// AFTER this decode — the order is fine), and every space-free pool path decodes to itself, so the
/// engine sees byte-identical paths to the pre-encoding protocol (no regression).
fn dec_path(s: &str) -> String {
    if !s.contains('%') {
        return s.to_string(); // fast path + identity for "-", "RAW:...", and space-free paths.
    }
    // Whitespace sequences FIRST, then "%25"->"%" LAST (exact inverse of enc_path).
    s.replace("%20", " ")
        .replace("%09", "\t")
        .replace("%0A", "\n")
        .replace("%0D", "\r")
        .replace("%25", "%")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Persistent serve mode: `gcompose --serve`.
    if args.len() >= 2 && args[1] == "--serve" {
        serve();
        return;
    }

    // One-shot mode (back-compat): `gcompose <base> <over|-> <out.rgba>`.
    if args.len() < 4 {
        eprintln!("usage: gcompose <base> <over|-> <out.rgba>   |   gcompose --serve");
        std::process::exit(2);
    }
    let (base, over, out) = (&args[1], &args[2], &args[3]);

    let buf = compose(base, over).or_else(|| decode_only(base));
    match buf {
        Some(b) => {
            std::fs::write(out, &b).expect("write rgba");
            println!("OK {} {} bytes={}", ffi::GVW, ffi::GVH, b.len());
        }
        None => {
            eprintln!("FAIL: could not decode {base}");
            std::process::exit(3);
        }
    }
}

/// Persistent serve loop: init OpenCL once, then service one request per stdin line.
///
/// The authoritative wire protocol (all commands, with the Wave-8 transition fields) is documented
/// in the module header above; see the `PREVIEW`/`ENC` entries there for the full field list.
fn serve() {
    // Initialize OpenCL exactly once for the lifetime of the process. A SOFT init failure (rc != 0:
    // transient driver/device-busy) is retried a couple of times in-process (init_retry); the HARD
    // flake is a driver segfault that kills the process outright, which only the client's respawn can
    // fix (see worker.rs render retry). If init still fails after the retries the worker is useless;
    // exit non-zero so the client's restart logic can react.
    const GPU_INIT_ATTEMPTS: usize = 3;
    let gpu = match ffi::Gpu::init_retry(GPU_INIT_ATTEMPTS) {
        Some(g) => g,
        None => {
            eprintln!("FAIL: fpx_gpu_init failed in --serve (after {GPU_INIT_ATTEMPTS} attempts)");
            std::process::exit(4);
        }
    };

    // Hardening (Slice A): verify OpenCL is actually USABLE — not merely init'd — before announcing
    // readiness. A tiny end-to-end self-check (upload a black frame + a no-op compose + confirm the
    // download round-trips a full-size buffer) catches a compositor that init'd but can't run the
    // kernels. On failure, exit non-zero IMMEDIATELY so the client respawns a fresh worker rather
    // than us serving a broken one (whose first real PREVIEW/ENC would fail). This converts a SOFT
    // broken-init into a fast, clean respawn; it cannot catch the hard mid-run segfault (that is the
    // client's render-retry job).
    if !gpu.self_check() {
        eprintln!(
            "FAIL: OpenCL self-check failed in --serve (init ok but compose round-trip broken)"
        );
        std::process::exit(5);
    }

    // One open decoder per media path, reused across requests (held playhead / repeated frames).
    let mut decoders: HashMap<String, ffi::Decoder> = HashMap::new();

    // LUT cache (Slice A — per-clip LOOK / LUT3D): one parsed `.cube` per path, reused across
    // frames so a held playhead / a long render with the same look does NOT reparse the file every
    // frame. The value is `Option<Lut>` so a file that FAILED to parse is cached as a negative result
    // (None) too — a broken LUT degrades to no look without re-attempting (and re-logging) the parse
    // every frame. `last_uploaded_lut` tracks which path is currently resident on the GPU so we skip
    // re-uploading the same LUT on consecutive same-look frames (mirrors MojoMedia's lut_loaded_idx).
    let mut lut_cache: HashMap<String, Option<ffi::Lut>> = HashMap::new();
    let mut last_uploaded_lut: Option<String> = None;

    // Which device buffer the MOST RECENT PREVIEW left the composed frame in: false = OUTB (look
    // none), true = LOOKB (a VHS/LUT look ran). SCOPE reads this so its scope kernels run on the
    // POST-LOOK composed buffer — the exact frame the UI is showing — rather than always OUTB
    // (which, after a look, holds the pre-look grade result). Updated by every PREVIEW; SCOPE does
    // not re-compose, so this faithfully tracks the displayed frame's final buffer.
    let mut last_final_is_look: bool = false;

    // Active render encoder (set by OPEN, fed by ENC, torn down by CLOSE). Holds the fps so
    // ENC can stamp ts = enc_count / fps; enc_count is the running frame counter for the job.
    let mut enc: Option<ffi::Encoder> = None;
    let mut enc_fps: f64 = 30.0;
    let mut enc_count: i64 = 0;
    // Whether the current encoder has a usable audio stream (finding #6). OPEN sets this true only
    // if config_audio succeeded; on a minimal FFmpeg build with no aac encoder it stays false and
    // OPEN still succeeds video-only, with AUDIO commands replying ERR instead of failing OPEN.
    let mut enc_audio_ok: bool = false;

    // PROGRAM-AUDIO ACCUMULATOR (Slice A): interleaved stereo f32 @ 48000, the full timeline
    // duration. OPEN (render) and WAVE (playback) allocate+zero it; each AUDIO line MIXES one
    // decoded clip range into it at the clip's destination offset (sample-add, clamped); CLOSE
    // feeds the whole buffer to the encoder, WAVECLOSE writes it as a PCM WAV. `prog_active`
    // gates AUDIO so a stray AUDIO with no open session replies ERR rather than silently dropping.
    let mut prog: ProgAudio = ProgAudio::default();

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    eprintln!("[gcompose] serve ready");

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break, // stdin closed / broken: shut down.
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Dispatch on the leading keyword; a line with no keyword is the legacy preview frame.
        let kw = line.split_whitespace().next().unwrap_or("");
        let reply: Reply = match kw {
            "OPEN" => {
                if open_render(
                    line,
                    &mut enc,
                    &mut enc_fps,
                    &mut enc_count,
                    &mut enc_audio_ok,
                    &mut prog,
                ) {
                    Reply::Done(None)
                } else {
                    Reply::Err
                }
            }
            "ENC" => {
                if enc_frame(
                    &gpu,
                    &mut decoders,
                    &mut lut_cache,
                    &mut last_uploaded_lut,
                    enc.as_mut(),
                    enc_fps,
                    &mut enc_count,
                    line,
                ) {
                    Reply::Done(None)
                } else {
                    Reply::Err
                }
            }
            // AUDIO no longer feeds the encoder directly: it MIXES one decoded clip range into the
            // active program-audio accumulator (render or playback) at a destination offset.
            "AUDIO" => {
                if audio_mix(&mut prog, line) {
                    Reply::Done(None)
                } else {
                    Reply::Err
                }
            }
            // GAINENV sets (or clears with "-") the P27 master gain envelope on the active
            // accumulator. Sent AFTER the opener (OPEN/WAVE/MEAS) and BEFORE the AUDIO lines so the
            // per-clip mix reads it. Empty/"-" → identical to pre-P27.
            "GAINENV" => {
                if gainenv_set(line, &mut prog) {
                    Reply::Done(None)
                } else {
                    Reply::Err
                }
            }
            // WAVE begins a playback-only accumulator session (no encoder).
            "WAVE" => {
                if wave_open(line, &mut prog) {
                    Reply::Done(None)
                } else {
                    Reply::Err
                }
            }
            // WAVECLOSE writes the playback accumulator to a PCM WAV and clears it.
            "WAVECLOSE" => match wave_close(line, &mut prog) {
                Some(out) => Reply::Done(Some(out)),
                None => Reply::Err,
            },
            // MEAS begins a MEASUREMENT-only accumulator session (no encoder, no WAV) — the level
            // meter feed. Subsequent AUDIO lines mix the (filtered+gained) ranges into it exactly
            // like the render/playback path; LEVELS then measures + clears it.
            "MEAS" => {
                if meas_open(line, &mut prog) {
                    Reply::Done(None)
                } else {
                    Reply::Err
                }
            }
            // LEVELS measures the active accumulator (peak + RMS dBFS per channel), writes 4
            // little-endian f32 to the given path, and CLEARS the accumulator (session terminator,
            // mirroring WAVECLOSE but emitting levels instead of a WAV).
            "LEVELS" => match levels_query(line, &mut prog) {
                Some(out) => Reply::Done(Some(out)),
                None => Reply::Err,
            },
            // SPECTRUM computes the active accumulator's magnitude spectrum (read-only FFT analysis),
            // writes <nbins> little-endian f32 to the given path, and CLEARS the accumulator (session
            // terminator, mirroring LEVELS — emits a spectrum instead of peak/RMS levels).
            "SPECTRUM" => match spectrum_query(line, &mut prog) {
                Some(out) => Reply::Done(Some(out)),
                None => Reply::Err,
            },
            // SAMPLES reads the active accumulator's raw time-domain samples (read-only oscilloscope),
            // writes <n> little-endian f32 to the given path, and CLEARS the accumulator (session
            // terminator, mirroring SPECTRUM — emits raw samples instead of an FFT magnitude spectrum).
            "SAMPLES" => match samples_query(line, &mut prog) {
                Some(out) => Reply::Done(Some(out)),
                None => Reply::Err,
            },
            "CLOSE" => {
                // Drain the program-audio accumulator into the encoder (timeline-synced audio),
                // then finish + close. The accumulator is the FULL timeline duration, so the audio
                // stream length matches the video. A video-only encoder (no aac) just skips the
                // drain and finishes video-only.
                let ok = match enc.take() {
                    Some(mut e) => {
                        if enc_audio_ok {
                            prog.drain_into_encoder(&mut e);
                        }
                        e.finish() // drop after this scope closes the encoder
                    }
                    None => false,
                };
                enc_audio_ok = false; // encoder torn down: no audio stream until the next OPEN.
                prog.clear(); // accumulator consumed; next OPEN/WAVE reallocates.
                if ok {
                    Reply::Done(None)
                } else {
                    Reply::Err
                }
            }
            "THUMB" => match thumb(&mut decoders, line) {
                Some(out) => Reply::Done(Some(out)),
                None => Reply::Err,
            },
            // NFRAMES <path> -> total video frame count returned INLINE as the DONE payload ("DONE 121").
            // Lets the UI clamp a clip's length to its source so it never references frames past the end.
            "NFRAMES" => match nframes_cmd(&mut decoders, line) {
                Some(n) => Reply::Done(Some(n)),
                None => Reply::Err,
            },
            "ENV" => match envelope(line) {
                Some(out) => Reply::Done(Some(out)),
                None => Reply::Err,
            },
            // P46 CLIPAUD: decode a clip's source range to mono @ a fixed rate (the UI cross-correlates
            // two of these to AUDIO-ALIGN clips). Stateless decode, like ENV.
            "CLIPAUD" => match clipaud(line) {
                Some(out) => Reply::Done(Some(out)),
                None => Reply::Err,
            },
            // SCOPE runs a scope kernel on the LAST composed buffer left by the most recent PREVIEW
            // (NOT cleared between requests, NO re-compose here). `last_final_is_look` selects OUTB
            // (look none) vs LOOKB (a look ran) so the scope reads the POST-LOOK frame the UI shows.
            "SCOPE" => match scope(&gpu, line, last_final_is_look) {
                Some(out) => Reply::Done(Some(out)),
                None => Reply::Err,
            },
            // Explicit preview-frame keyword (finding #3): a PREVIEW line carries the 21
            // positional fields after the keyword (15 composite+look fields, 5 Wave-8 transition
            // fields, then the out path). The keyword disambiguates it from an ENC line of similar
            // arity, so a media path can never be mistaken for a command.
            //
            // Slice A: a PREVIEW leaves the composed frame in the persistent GPU buffer — g_buf[OUTB]
            // when the clip's look is none, g_buf[LOOKB] when a VHS/LUT look ran (handle_request
            // records which in `last_final_is_look`). That buffer is a static cl_mem and is NEVER
            // cleared between requests, so a subsequent SCOPE reads exactly the (post-look) frame the
            // UI is showing. The only way it changes is another compose (a later PREVIEW/ENC) — which
            // is why the worker.rs `scope()` sends its PREVIEW immediately before SCOPE under one held
            // mutex.
            "PREVIEW" => match handle_request(
                &gpu,
                &mut decoders,
                &mut lut_cache,
                &mut last_uploaded_lut,
                &mut last_final_is_look,
                line,
            ) {
                Some(out_path) => Reply::Done(Some(out_path)),
                None => Reply::Err,
            },
            // Back-compat: a keyword-less line is still treated as a legacy positional preview
            // request (one-shot tools / older clients). New UI clients always send PREVIEW.
            _ => match handle_request(
                &gpu,
                &mut decoders,
                &mut lut_cache,
                &mut last_uploaded_lut,
                &mut last_final_is_look,
                line,
            ) {
                Some(out_path) => Reply::Done(Some(out_path)),
                None => Reply::Err,
            },
        };

        match reply {
            Reply::Done(Some(out)) => {
                let _ = writeln!(stdout, "DONE {out}");
            }
            Reply::Done(None) => {
                let _ = writeln!(stdout, "DONE");
            }
            Reply::Err => {
                let _ = writeln!(stdout, "ERR");
            }
        }
        // Always flush so the client (blocking on a single response line) unblocks promptly.
        let _ = stdout.flush();
    }
}

/// A serve reply: DONE (optionally echoing an out path) or ERR.
enum Reply {
    Done(Option<String>),
    Err,
}

/// `OPEN <out> <w> <h> <fps>` — (re)create the render encoder. Any prior encoder is dropped
/// (closed) without finishing, since a fresh OPEN supersedes it. Configures video (mpeg4) AND, when
/// the local FFmpeg build supports it, audio (aac, 2ch, 48000) so the encoder is ready for both the
/// ENC video feed and the AUDIO program-audio feed, then writes the header. Resets the counter.
///
/// `enc_audio_ok` is set true only if `config_audio` succeeded. config_audio failure (finding #6:
/// a minimal FFmpeg build with no aac encoder) is NON-FATAL — OPEN still succeeds video-only and
/// `enc_audio_ok` stays false, so later AUDIO commands reply ERR (the client skips audio) instead
/// of failing the whole render. This restores wave-2 behavior where a missing aac never broke video.
fn open_render(
    line: &str,
    enc: &mut Option<ffi::Encoder>,
    enc_fps: &mut f64,
    enc_count: &mut i64,
    enc_audio_ok: &mut bool,
    prog: &mut ProgAudio,
) -> bool {
    *enc_audio_ok = false;
    let f: Vec<&str> = line.split_whitespace().collect();
    // OPEN <out> <out_w> <out_h> <fps_num> <fps_den> <rate_mode> <rate_value> <vcodec> <total_s> <gop> <preset> <abitrate>
    // (Triad-B P1 export controls + P25 export depth, 13 tokens — P25 appended <gop> <preset>
    // <abitrate> AFTER <total_s>; P1 was 10 tokens, originally the 6-token `OPEN <out> <w> <h> <fps>
    // <total_s>`). The OUTPUT resolution (out_w×out_h) + fps + rate control + codec ride the line;
    // the encoder INPUT dims stay GVW×GVH (the fixed OpenCL compose canvas — every ENC frame is
    // composed at that size) and the encoder SCALES (swscale, in config_video) to out_w×out_h. So the
    // working canvas and the output resolution are decoupled (the slice's export-controls requirement).
    // P25 adds: <gop>=keyframe interval (frames; <=0 keeps the codec default), <preset>=encoder preset
    // token ("-" => none), <abitrate>=audio bitrate in bits/s (<=0 => the legacy 128000).
    // P29 adds: <acodec>=audio codec name token ("-" => the legacy "aac").
    if f.len() != 14 {
        eprintln!("[gcompose] bad OPEN ({} fields): {line}", f.len());
        return false;
    }
    // WHITESPACE-SAFE WIRE: the UI percent-encodes the render out path (worker.rs enc_path) just like
    // every other path token, so decode it here (dec_path) before opening the container. A space-free
    // path decodes to itself (identity), so a normal export is byte-identical to the pre-fix wire.
    let out = dec_path(f[1]);
    let out_w: usize = match f[2].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let out_h: usize = match f[3].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let fps_num: i32 = match f[4].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let fps_den: i32 = match f[5].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let rate_mode: u8 = match f[6].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let rate_value: i64 = match f[7].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let vcodec = f[8];
    if out_w == 0 || out_h == 0 || fps_num <= 0 || fps_den <= 0 || vcodec.is_empty() {
        eprintln!(
            "[gcompose] bad OPEN dims/fps/codec: {out_w}x{out_h} {fps_num}/{fps_den} {vcodec}"
        );
        return false;
    }
    // Timeline duration in seconds; sizes the program-audio accumulator. A non-finite/negative
    // value is a protocol error; 0 is allowed (empty timeline → empty audio).
    let total_s: f64 = match f[9].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    if !total_s.is_finite() || total_s < 0.0 {
        eprintln!("[gcompose] bad OPEN total_s={total_s}");
        return false;
    }
    // P25 export depth (appended after total_s). Defaults reproduce pre-P25 exactly:
    //   gop<=0       -> config_video leaves the encoder's default gop_size untouched.
    //   preset "-"   -> empty preset string -> config_video sets no preset.
    //   abitrate<=0  -> keep the legacy hardcoded 128000 audio bitrate below.
    // Tolerant parse (a malformed token degrades to the identity default rather than failing OPEN).
    let gop: i32 = f[10].parse().unwrap_or(0);
    let preset_raw = f[11];
    let preset = if preset_raw == "-" { "" } else { preset_raw };
    let abitrate: i64 = f[12].parse().unwrap_or(0);
    // P29: audio codec ("-" => the legacy "aac"). Kept at the fixed 2ch/48000 interleaved layout the
    // program-audio accumulator feeds — only the OUTPUT codec changes (the config_audio swr converts
    // the fed FLT samples to the codec's sample format). The container (out path) must accept it.
    let acodec = if f[13] == "-" { "aac" } else { f[13] };

    // Encoder INPUT dims = the engine's fixed compose resolution (every ENC frame is GVW×GVH);
    // OUTPUT (encoded) dims = the requested out_w×out_h. config_video builds the RGBA(in)→pixfmt
    // sws scaler from in_w/in_h to width/height, so the composed GVW×GVH frame is rescaled to the
    // chosen output resolution by the encoder — no change to the OpenCL compose path.
    let in_w = ffi::GVW;
    let in_h = ffi::GVH;

    // Drop any previous (unfinished) encoder before starting a new job.
    *enc = None;

    let mut e = match ffi::Encoder::open(&out) {
        Some(e) => e,
        None => {
            eprintln!("[gcompose] enc_open failed: {out}");
            return false;
        }
    };
    // Video: the requested codec, INPUT=GVW×GVH, OUTPUT=out_w×out_h, fps_num/den. In rate_mode 0
    // (average bitrate) rate_value is the bit_rate; in rate_mode 1 (constant quality) we pass a 0
    // bit_rate here then set the CRF/qscale via set_quality below.
    let bitrate = if rate_mode == 1 { 0 } else { rate_value };
    // P25: pass <gop>/<preset> through to config_video (applied on the codec context before open;
    // gop<=0 / "" preset are no-ops → identity with pre-P25).
    if !e.config_video(
        vcodec, in_w, in_h, out_w, out_h, fps_num, fps_den, bitrate, gop, preset,
    ) {
        eprintln!("[gcompose] config_video failed (codec={vcodec} out={out_w}x{out_h} fps={fps_num}/{fps_den})");
        return false;
    }
    // Constant-quality (CRF) export: rate_value is the CRF/quality value. Best-effort — a codec that
    // rejects every quality knob keeps the (0) bitrate config; we log but do not fail the OPEN.
    if rate_mode == 1 {
        let crf = rate_value as i32;
        // P25: the CRF path re-opens the codec, so the gop/preset must be re-applied here or they'd
        // be lost (the gate uses CRF mode for the GOP test). gop<=0 / "" preset are no-ops.
        if !e.set_quality(crf, gop, preset) {
            eprintln!(
                "[gcompose] set_quality(crf={crf}) failed; encoding at codec-default quality"
            );
        }
    }
    // Program audio (Slice A): configure an aac stream (2ch @ 48000, 128 kbps) matching
    // MojoMedia's render config. The AUDIO command feeds per-clip ranges decoded at this same
    // 2ch/48000 layout; CLOSE (enc_finish) flushes and writes both streams. config_audio is SAFE
    // to enable because the protocol has a real audio-feed command (no more zero-sample track): if
    // every AUDIO clip happens to fail/skip, enc_finish still produces a valid (empty) aac stream
    // rather than a malformed one.
    //
    // Finding #6: a config_audio failure (e.g. an FFmpeg build with no aac encoder) is NON-FATAL.
    // We log it and continue VIDEO-ONLY rather than failing OPEN — otherwise a minimal-FFmpeg
    // environment would lose the ability to render video at all (a regression vs wave-2). The
    // encoder header is then written without an audio stream, and AUDIO commands reply ERR.
    // P25: <abitrate> bits/s drives the aac stream; <=0 keeps the legacy hardcoded 128000 (identity).
    *enc_audio_ok = e.config_audio(
        acodec,
        2,
        48_000,
        if abitrate > 0 { abitrate } else { 128_000 },
    );
    if !*enc_audio_ok {
        eprintln!("[gcompose] config_audio failed; rendering video-only (no aac stream)");
    }
    if !e.start() {
        eprintln!("[gcompose] enc_start failed");
        return false;
    }

    *enc = Some(e);
    // Serenity sends one ENC for each browser-timeline frame and uses the same fps in OPEN. Stamp
    // those frames at the declared rate. The old fixed 30 fps stamp produced duplicate PTS values
    // (and silently dropped frames) whenever a 16/24/25 fps project was exported.
    *enc_fps = fps_num as f64 / fps_den as f64;
    *enc_count = 0;

    // Allocate the program-audio accumulator for this render's full timeline duration (silence).
    // Each AUDIO line mixes a clip range into it; CLOSE drains it into the encoder.
    prog.alloc(total_s);
    true
}

/// Resolve the three wire LOOK fields (`<look_kind> <look_amt> <lut_path|->`) into the
/// `(look_kind, look_amt, lut_n)` triple the compose pipeline passes to `fpx_gpu_look`, performing
/// any LUT load + upload as a SIDE EFFECT (Slice A).
///
/// Semantics (mirrors MojoMedia main_editor.mojo ~673-703):
///   - kind 0 (none): returns (0, 0.0, 0); no LUT touched. The compose then runs `fpx_gpu_look(0,..)`
///     (a no-op that leaves the frame in OUTB).
///   - kind 1 (VHS): returns (1, amt, 0); the procedural VHS kernel needs no LUT (lut_n=0).
///   - kind 2 (LUT3D): parse `lut_path` (CACHED per path in `lut_cache` — including NEGATIVE results,
///     so a broken .cube isn't reparsed every frame), upload it to the GPU only when it isn't already
///     resident (`last_uploaded_lut`, mirroring MojoMedia's lut_loaded_idx), and return (2, amt, N).
///     A MISSING / unparsable / failed-upload LUT DEGRADES TO NO LOOK: returns (0, 0.0, 0) so the
///     frame still composes (never fails) — exactly the contract's "missing/failed LUT degrades to no
///     look (do not fail the frame)".
///
/// `lut_path` is the "-" sentinel (or empty) when there is no LUT; only kind==2 ever consults it.
fn resolve_look(
    gpu: &ffi::Gpu,
    lut_cache: &mut HashMap<String, Option<ffi::Lut>>,
    last_uploaded_lut: &mut Option<String>,
    look_kind: i32,
    look_amt: f32,
    lut_path: &str,
) -> (i32, f32, i32) {
    match look_kind {
        1 => (1, look_amt, 0), // VHS: procedural, no LUT.
        2 => {
            if lut_path.is_empty() || lut_path == "-" {
                return (0, 0.0, 0); // LUT3D requested but no path: degrade to none.
            }
            // Parse the .cube once per path (cache positive AND negative results). The mutable
            // borrow is confined to the insert; we then re-borrow `lut_cache` immutably to read the
            // cached Lut (no overlapping borrows — the insert's borrow ends before the get).
            if !lut_cache.contains_key(lut_path) {
                let loaded = ffi::load_cube(lut_path);
                if loaded.is_none() {
                    eprintln!("[gcompose] LUT load failed (degrading to no look): {lut_path}");
                }
                lut_cache.insert(lut_path.to_string(), loaded);
            }
            let lut = match lut_cache.get(lut_path).and_then(|o| o.as_ref()) {
                Some(l) => l,
                None => return (0, 0.0, 0), // cached negative: no look.
            };
            // Upload only when this path isn't already the resident GPU LUT (skip the re-upload on
            // consecutive same-look frames). On a fresh/changed path, upload and record it; an upload
            // failure also degrades to no look and forgets the resident path so a later retry re-tries.
            if last_uploaded_lut.as_deref() != Some(lut_path) {
                if gpu.upload_lut(lut) {
                    *last_uploaded_lut = Some(lut_path.to_string());
                } else {
                    eprintln!("[gcompose] LUT upload failed (degrading to no look): {lut_path}");
                    *last_uploaded_lut = None;
                    return (0, 0.0, 0);
                }
            }
            (2, look_amt, lut.n as i32)
        }
        _ => (0, 0.0, 0), // 0 / unknown: no look.
    }
}

/// `ENC <base> <over|-> <bf> <of> <op> <px> <py> <pw> <ph> <bright> <contrast> <sat> <look_kind>
/// <look_amt> <lut_path|-> <trans_kind> <trans_prog> <trans_param> <trans_path|-> <trans_frame>` —
/// decode the base (and optional overlay + optional transition partner), run the same OpenCL
/// composite + transition + look the preview uses, and feed the composited RGBA f32 frame to the
/// active encoder at ts = enc_count / fps. No file is written.
fn enc_frame(
    gpu: &ffi::Gpu,
    decoders: &mut HashMap<String, ffi::Decoder>,
    lut_cache: &mut HashMap<String, Option<ffi::Lut>>,
    last_uploaded_lut: &mut Option<String>,
    enc: Option<&mut ffi::Encoder>,
    fps: f64,
    enc_count: &mut i64,
    line: &str,
) -> bool {
    let e = match enc {
        Some(e) => e,
        None => {
            eprintln!("[gcompose] ENC with no open encoder");
            return false;
        }
    };

    let f: Vec<&str> = line.split_whitespace().collect();
    // ENC + 12 composite + 1 P31 BLEND + 3 LOOK + 5 TRANSITION + 3 PER-CLIP GRADE + 12 P2 + 6 P4
    // CHROMA + 5 P5 CURVE + 4 P6 + 6 P7 + 8 P8 + 4 P9 + 3 P10 + 3 P13 + 3 P16 + 3 P17 + 4 P23 + 7 P34
    // + 1 P37 SPILL = 94 tokens (was 93 pre-P37). P31 inserted ONE field `over_blend` at f[6],
    // IMMEDIATELY AFTER `op` (f[5]); every field after op shifted +1. f[25..=36] are
    //   per-clip color/transform lift_r lift_g lift_b gamma_r gamma_g gamma_b gain_r gain_g gain_b rot scale blur,
    // f[37..=42] are the P4 chroma-key fields ck_on ck_r ck_g ck_b ck_sim ck_smooth, f[43..=47] are
    // the P5 curve, f[48..=51] are the P6 fields vig sharp flip fx, f[52..=57] are the P7 color fields
    // hue sat light inb inw gam, f[58..=65] are the P8 stylize-2 fields mosaic gmap_amt glo_r glo_g
    // glo_b ghi_r ghi_g ghi_b, f[66..=69] are the P9 fx fields denoise glow_amt glow_thr rgbshift,
    // f[70..=72] are the P10 stylize-4 fields halftone emboss edge, f[73..=75] are the P13
    // old-film/distort fields grain scratches diffusion, f[76..=78] are the P16 distort fields
    // wave swirl threshold, f[79..=81] are the P17 geometric fields lens crop glitch, f[82..=85]
    // are the P23 360-reframe fields eq360 eq_yaw eq_pitch eq_fov, f[86..=92] are the P34 shape-mask
    // fields mask_shape mask_cx mask_cy mask_rw mask_rh mask_feather mask_invert, and f[93] is the
    // P37 chroma green-spill field ck_spill (APPENDED as the new LAST token, AFTER the P34 mask
    // fields — so the ck_*/mask indices are unchanged). P38 appends the 3 distortion fields
    // mirror_x kaleido dither at f[94..=96] (AFTER ck_spill), so the ck_spill/mask indices stay
    // unchanged. P39 appends the 3 selective-color fields sel_band sel_hshift sel_sat at f[97..=99]
    // (AFTER dither), so the P38/ck_spill/mask indices stay unchanged. P41 appends the 2 filter fields
    // sol_thr temp at f[100..=101] (the new LAST tokens, AFTER sel_sat, pinned order `sol_thr temp`),
    // so the P39/P38/ck_spill/mask indices stay unchanged. ENC has NO out path (temp is the LAST token).
    if f.len() != 103 {
        eprintln!("[gcompose] bad ENC ({} fields): {line}", f.len());
        return false;
    }
    // WHITESPACE-SAFE WIRE: the UI percent-encodes every path token (enc_path); decode it here
    // (dec_path) BEFORE it is used to open a file. A space-free path / "-" / "RAW:..." decodes to
    // itself (identity). upload_slot strips the "RAW:" prefix AFTER this decode.
    let base_path = dec_path(f[1]);
    let over_path = dec_path(f[2]); // "-" means no overlay
    let parsed = (|| {
        Some((
            f[3].parse::<i32>().ok()?,  // base_frame
            f[4].parse::<i32>().ok()?,  // over_frame
            f[5].parse::<f32>().ok()?,  // op
            f[7].parse::<f32>().ok()?,  // px
            f[8].parse::<f32>().ok()?,  // py
            f[9].parse::<f32>().ok()?,  // pw
            f[10].parse::<f32>().ok()?, // ph
            f[11].parse::<f32>().ok()?, // bright
            f[12].parse::<f32>().ok()?, // contrast
            f[13].parse::<f32>().ok()?, // sat
            f[14].parse::<i32>().ok()?, // look_kind
            f[15].parse::<f32>().ok()?, // look_amt
        ))
    })();
    let (base_frame, over_frame, op, px, py, pw, ph, bright, contrast, sat, look_kind, look_amt) =
        match parsed {
            Some(v) => v,
            None => return false,
        };
    // P31 BLEND MODE of the OVER (V2) clip, riding the wire IMMEDIATELY AFTER `op` (f[5]) at f[6]:
    // 0=Normal 1=Multiply 2=Screen 3=Overlay 4=Add 5=Darken 6=Lighten 7=Difference. Tolerant — a
    // bad/absent token degrades to 0 (Normal) which is byte-identical to the pre-P31 plain composite.
    let over_blend: i32 = f[6].parse().unwrap_or(0);
    let lut_path = dec_path(f[16]); // "-" / empty when no LUT (only used by LUT3D look_kind==2)

    // Transition fields (Wave 8): kind (-1 none, 0..7 kernel), progress, param, partner path+frame.
    let trans_parsed = (|| {
        Some((
            f[17].parse::<i32>().ok()?, // trans_kind
            f[18].parse::<f32>().ok()?, // trans_prog
            f[19].parse::<f32>().ok()?, // trans_param
            f[21].parse::<i32>().ok()?, // trans_frame
        ))
    })();
    let (trans_kind, trans_prog, trans_param, trans_frame) = match trans_parsed {
        Some(v) => v,
        None => return false,
    };
    let trans_path = dec_path(f[20]); // "-" when no transition partner

    // PER-CLIP GRADE fields (Triad-B P1): cbright/ccontrast/csat, applied BEFORE the program grade.
    let clip_grade = (|| {
        Some((
            f[22].parse::<f32>().ok()?, // cbright
            f[23].parse::<f32>().ok()?, // ccontrast
            f[24].parse::<f32>().ok()?, // csat
        ))
    })();
    let (cbright, ccontrast, csat) = match clip_grade {
        Some(v) => v,
        None => return false,
    };

    // P2 per-clip color/transform effects (f[25..=36]), pinned order: lift3, gamma3, gain3, rot,
    // scale, blur. Identity defaults: lift_*=0, gamma_*=1, gain_*=1, rot=0, scale=1, blur=0.
    let p2 = (|| {
        Some((
            f[25].parse::<f32>().ok()?, // lift_r
            f[26].parse::<f32>().ok()?, // lift_g
            f[27].parse::<f32>().ok()?, // lift_b
            f[28].parse::<f32>().ok()?, // gamma_r
            f[29].parse::<f32>().ok()?, // gamma_g
            f[30].parse::<f32>().ok()?, // gamma_b
            f[31].parse::<f32>().ok()?, // gain_r
            f[32].parse::<f32>().ok()?, // gain_g
            f[33].parse::<f32>().ok()?, // gain_b
            f[34].parse::<f32>().ok()?, // rot (degrees)
            f[35].parse::<f32>().ok()?, // scale
            f[36].parse::<f32>().ok()?, // blur (sigma)
        ))
    })();
    let (
        lift_r,
        lift_g,
        lift_b,
        gamma_r,
        gamma_g,
        gamma_b,
        gain_r,
        gain_g,
        gain_b,
        rot,
        scale,
        blur,
    ) = match p2 {
        Some(v) => v,
        None => return false,
    };

    // P4 per-clip CHROMA-KEY fields (f[37..=42]), pinned order: ck_on, ck_r, ck_g, ck_b, ck_sim,
    // ck_smooth. Identity defaults: ck_on=0 (disabled → OVER alpha untouched, byte-identical to P3),
    // key=green [0,1,0], sim=0.4, smooth=0.1. These describe the OVER (V2) clip.
    let p4 = (|| {
        Some((
            f[37].parse::<i32>().ok()?, // ck_on (1/0)
            f[38].parse::<f32>().ok()?, // ck_r
            f[39].parse::<f32>().ok()?, // ck_g
            f[40].parse::<f32>().ok()?, // ck_b
            f[41].parse::<f32>().ok()?, // ck_sim
            f[42].parse::<f32>().ok()?, // ck_smooth
        ))
    })();
    let (ck_on, ck_r, ck_g, ck_b, ck_sim, ck_smooth) = match p4 {
        Some(v) => v,
        None => return false,
    };

    // P5 master tone CURVE: 5 outputs at fixed inputs 0/.25/.5/.75/1 (f[43..=47]). Identity
    // [0,.25,.5,.75,1] is skipped engine-side, so an un-curved clip is byte-identical.
    let curve = match (|| {
        Some([
            f[43].parse::<f32>().ok()?,
            f[44].parse::<f32>().ok()?,
            f[45].parse::<f32>().ok()?,
            f[46].parse::<f32>().ok()?,
            f[47].parse::<f32>().ok()?,
        ])
    })() {
        Some(v) => v,
        None => return false,
    };

    // P6 STYLIZE/UTILITY fields (f[48..=51]), pinned order: vig sharp flip fx. Identity defaults
    // vig=0, sharp=0, flip=0, fx=0 are skipped engine-side, so an unfiltered clip is byte-identical.
    let p6 = (|| {
        Some((
            f[48].parse::<f32>().ok()?, // vig (vignette amount)
            f[49].parse::<f32>().ok()?, // sharp (unsharp amount)
            f[50].parse::<i32>().ok()?, // flip (0 none/1 H/2 V/3 both)
            f[51].parse::<i32>().ok()?, // fx (0 none/1 invert/2 sepia/3 grayscale/4 posterize)
        ))
    })();
    let (vig, sharp, flip, fx) = match p6 {
        Some(v) => v,
        None => return false,
    };

    // P7 COLOR fields (f[52..=57]), pinned order: hue sat light inb inw gam. Identity defaults
    // hue=0, sat=1, light=0, inb=0, inw=1, gam=1 are skipped engine-side, so an unfiltered clip is
    // byte-identical. hue=HSL hue shift (deg), sat=HSL saturation mult, light=HSL lightness add;
    // inb/inw=levels input black/white, gam=levels gamma.
    let p7 = (|| {
        Some((
            f[52].parse::<f32>().ok()?, // hue (HSL hue shift, degrees)
            f[53].parse::<f32>().ok()?, // sat (HSL saturation multiplier)
            f[54].parse::<f32>().ok()?, // light (HSL lightness add)
            f[55].parse::<f32>().ok()?, // inb (levels input black)
            f[56].parse::<f32>().ok()?, // inw (levels input white)
            f[57].parse::<f32>().ok()?, // gam (levels gamma)
        ))
    })();
    let (hue, sat_hsl, light, inb, inw, gam) = match p7 {
        Some(v) => v,
        None => return false,
    };

    // P8 STYLIZE-2 fields (f[58..=65]), pinned order: mosaic gmap_amt glo_r glo_g glo_b ghi_r ghi_g
    // ghi_b. Identity defaults mosaic=0 (0/1 = off), gmap_amt=0 are skipped engine-side, so an
    // unfiltered clip is byte-identical. mosaic=block size in px (parsed as i32 — the wire carries a
    // plain integer; the model's u32 is printed as a plain decimal that round-trips to i32); gmap_amt
    // =gradient-map mix 0..1; glo=shadow colour, ghi=highlight colour.
    let p8 = (|| {
        Some((
            f[58].parse::<i32>().ok()?, // mosaic (block size px; 0/1 = off)
            f[59].parse::<f32>().ok()?, // gmap_amt (gradient-map mix 0..1)
            f[60].parse::<f32>().ok()?, // glo_r (shadow colour r)
            f[61].parse::<f32>().ok()?, // glo_g (shadow colour g)
            f[62].parse::<f32>().ok()?, // glo_b (shadow colour b)
            f[63].parse::<f32>().ok()?, // ghi_r (highlight colour r)
            f[64].parse::<f32>().ok()?, // ghi_g (highlight colour g)
            f[65].parse::<f32>().ok()?, // ghi_b (highlight colour b)
        ))
    })();
    let (mosaic, gmap_amt, glo_r, glo_g, glo_b, ghi_r, ghi_g, ghi_b) = match p8 {
        Some(v) => v,
        None => return false,
    };

    // P9 FX fields (f[66..=69]), pinned order: denoise glow_amt glow_thr rgbshift. Identity defaults
    // denoise=0, glow_amt=0, rgbshift=0 are skipped engine-side (glow_thr only matters when glow_amt>0),
    // so an unfiltered clip is byte-identical. denoise=bilateral strength 0..1; glow_amt=bloom mix 0..1;
    // glow_thr=bloom luma threshold; rgbshift=chromatic-aberration channel offset in px.
    let p9 = (|| {
        Some((
            f[66].parse::<f32>().ok()?, // denoise (bilateral strength 0..1)
            f[67].parse::<f32>().ok()?, // glow_amt (bloom mix 0..1)
            f[68].parse::<f32>().ok()?, // glow_thr (bloom luma threshold)
            f[69].parse::<f32>().ok()?, // rgbshift (channel offset in px)
        ))
    })();
    let (denoise, glow_amt, glow_thr, rgbshift) = match p9 {
        Some(v) => v,
        None => return false,
    };

    // P10 STYLIZE-4 fields (f[70..=72]), pinned order: halftone emboss edge. Identity defaults
    // halftone=0 (0/1 = off), emboss=0, edge=0 are skipped engine-side, so an unfiltered clip is
    // byte-identical. halftone=dot cell size in px (parsed as i32 — the wire carries a plain integer;
    // the model's u32 is printed as a plain decimal that round-trips to i32); emboss=relief strength
    // 0..1; edge=Sobel edge/sketch mix 0..1.
    let p10 = (|| {
        Some((
            f[70].parse::<i32>().ok()?, // halftone (dot cell size px; 0/1 = off)
            f[71].parse::<f32>().ok()?, // emboss (relief strength 0..1)
            f[72].parse::<f32>().ok()?, // edge (Sobel edge/sketch mix 0..1)
        ))
    })();
    let (halftone, emboss, edge) = match p10 {
        Some(v) => v,
        None => return false,
    };

    // P13 OLD-FILM/DISTORT fields (f[73..=75]), pinned order: grain scratches diffusion. Identity
    // defaults grain=0, scratches=0, diffusion=0 are skipped engine-side, so an unfiltered clip is
    // byte-identical. grain=film-noise strength 0..1; scratches=scratch density/amount 0..1;
    // diffusion=frosted-glass jitter radius in px (0..16). The pseudo-randomness is a deterministic
    // integer hash of the pixel coords (same input frame => same output), so the gates are stable.
    let p13 = (|| {
        Some((
            f[73].parse::<f32>().ok()?, // grain (film-noise strength 0..1)
            f[74].parse::<f32>().ok()?, // scratches (scratch density/amount 0..1)
            f[75].parse::<f32>().ok()?, // diffusion (jitter radius px; 0..16)
        ))
    })();
    let (grain, scratches, diffusion) = match p13 {
        Some(v) => v,
        None => return false,
    };

    // P16 DISTORT fields (f[76..=78]), pinned order: wave swirl threshold. Identity defaults wave=0,
    // swirl=0, threshold=0 are skipped engine-side, so an unfiltered clip is byte-identical. wave=
    // sinusoidal displacement amplitude in px; swirl=rotation strength in radians at the centre;
    // threshold=luma binarize level 0..1.
    let p16 = (|| {
        Some((
            f[76].parse::<f32>().ok()?, // wave (sinusoidal amplitude px)
            f[77].parse::<f32>().ok()?, // swirl (rotation strength radians at centre)
            f[78].parse::<f32>().ok()?, // threshold (luma binarize level 0..1)
        ))
    })();
    let (wave, swirl, threshold) = match p16 {
        Some(v) => v,
        None => return false,
    };

    // P17 GEOMETRIC fields (f[79..=81]), pinned order: lens crop glitch. Identity defaults lens=0,
    // crop=0, glitch=0 are skipped engine-side, so an unfiltered clip is byte-identical. lens=radial
    // barrel(+)/pincushion(-) coefficient (0=off); crop=border-to-black fraction 0..0.49; glitch=max
    // per-band horizontal channel shift in px (deterministic band hash). These are the LAST 3 tokens
    // (ENC has no out path).
    let p17 = (|| {
        Some((
            f[79].parse::<f32>().ok()?, // lens (radial coefficient: +barrel / -pincushion)
            f[80].parse::<f32>().ok()?, // crop (border-to-black fraction 0..0.49)
            f[81].parse::<f32>().ok()?, // glitch (max per-band horizontal shift px)
        ))
    })();
    let (lens, crop, glitch) = match p17 {
        Some(v) => v,
        None => return false,
    };

    // P23 360-REFRAME fields (f[82..=85]), pinned order: eq360 eq_yaw eq_pitch eq_fov. Identity
    // eq360=0 (off) is skipped engine-side (the FFI returns immediately, OUTB untouched) so an
    // un-reframed clip is byte-identical to pre-P23. eq360 is an INTEGER flag (1=on / 0=off, parsed
    // as i32, nonzero=on); eq_yaw/eq_pitch = view yaw/pitch in degrees (identity 0/0); eq_fov =
    // horizontal field of view in degrees (default 90). These are the LAST 4 tokens (ENC has no out
    // path). A bad token → return false (same fallible style as the other fields).
    let p23 = (|| {
        Some((
            f[82].parse::<i32>().ok()?, // eq360 (flag: nonzero = on)
            f[83].parse::<f32>().ok()?, // eq_yaw (degrees)
            f[84].parse::<f32>().ok()?, // eq_pitch (degrees)
            f[85].parse::<f32>().ok()?, // eq_fov (degrees, default 90)
        ))
    })();
    let (eq360, eq_yaw, eq_pitch, eq_fov) = match p23 {
        Some(v) => v,
        None => return false,
    };

    // P34 SHAPE-MASK fields (f[86..=92]), pinned order: mask_shape mask_cx mask_cy mask_rw mask_rh
    // mask_feather mask_invert. These are the LAST 7 tokens (ENC has no out path). Identity mask_shape=0
    // (none) is skipped engine-side (the FFI returns immediately, OUTB untouched) so an unmasked clip is
    // byte-identical to pre-P34. mask_shape is an INTEGER (0=none 1=rect 2=ellipse); mask_cx/mask_cy =
    // mask centre (0..1, identity 0.5/0.5); mask_rw/mask_rh = half-extents (0..1, default 0.5/0.5);
    // mask_feather = soft-edge band width (default 0); mask_invert is an INTEGER flag (1=on / 0=off).
    // TOLERANT (gate awareness): a bad/absent mask_shape/mask_invert token degrades to 0 (none / not
    // inverted — a true no-op), and a bad geometry token degrades to its natural default, so a malformed
    // P34 tail can never flip a shape-0 clip into a masked one.
    let mask_shape: i32 = f[86].parse().unwrap_or(0);
    let mask_cx: f32 = f[87].parse().unwrap_or(0.5);
    let mask_cy: f32 = f[88].parse().unwrap_or(0.5);
    let mask_rw: f32 = f[89].parse().unwrap_or(0.5);
    let mask_rh: f32 = f[90].parse().unwrap_or(0.5);
    let mask_feather: f32 = f[91].parse().unwrap_or(0.0);
    let mask_invert: i32 = f[92].parse().unwrap_or(0);

    // P37 CHROMA GREEN-SPILL field (f[93]), the new LAST token (APPENDED after the P34 mask fields).
    // ck_spill = green-spill suppression strength (0..1). Identity ck_spill=0 leaves the OVER green
    // untouched (the kernel's spill if is skipped) → byte-identical to pre-P37. TOLERANT: a bad/absent
    // token degrades to 0.0 (a true no-op), so a malformed tail can never introduce spill. Only matters
    // when chroma is enabled (the spill code lives inside k_chroma, run only when eff_ck_on != 0).
    let ck_spill: f32 = f[93].parse().unwrap_or(0.0);

    // P38 DISTORTION fields (f[94..=96]), pinned order: mirror_x kaleido dither. These are the new LAST
    // 3 tokens (APPENDED after the P37 ck_spill; ENC has no out path). Each is a no-op at its default
    // (mirror_x 0 / kaleido <2 / dither 0) → engine skips → byte-identical to pre-P38. mirror_x and
    // kaleido are INTEGERS (mirror_x 0=off/1=on; kaleido 0/1=off, >=2 segment count); dither is an
    // f32 strength (0=off, 0..1). TOLERANT (gate awareness): a bad/absent token degrades to its no-op
    // default (0/0/0.0), so a malformed P38 tail can never enable a distortion on a default clip.
    let mirror_x: i32 = f[94].parse().unwrap_or(0);
    let kaleido: i32 = f[95].parse().unwrap_or(0);
    let dither: f32 = f[96].parse().unwrap_or(0.0);

    // P39 SELECTIVE COLOR fields (f[97..=99]), pinned order: sel_band sel_hshift sel_sat. These are the
    // new LAST 3 tokens (APPENDED after the P38 dither; ENC has no out path). No-op at its default
    // (sel_band==0) → engine skips → byte-identical to pre-P39. sel_band is an INTEGER (0=off 1=Red
    // 2=Yellow 3=Green 4=Cyan 5=Blue 6=Magenta); sel_hshift is an f32 hue rotation (-1..1); sel_sat is
    // an f32 saturation multiplier (default 1.0). TOLERANT (gate awareness): a bad/absent token degrades
    // to its no-op default (0/0.0/1.0), so a malformed P39 tail can never select a band on a default clip.
    let sel_band: i32 = f[97].parse().unwrap_or(0);
    let sel_hshift: f32 = f[98].parse().unwrap_or(0.0);
    let sel_sat: f32 = f[99].parse().unwrap_or(1.0);

    // P41 SOLARIZE + COLOUR TEMPERATURE fields (f[100..=101]), pinned order: sol_thr temp. These are the
    // new LAST 2 tokens (APPENDED after the P39 sel_sat; ENC has no out path). Each is a no-op at its
    // default (sol_thr<=0 / temp==0) → engine skips → byte-identical to pre-P41. sol_thr is the solarize
    // threshold (f32, 0=off, active (0,1]); temp is the warm/cool shift (f32, 0=neutral, -1..1). TOLERANT:
    // a bad/absent token degrades to its no-op default (0.0), so a malformed P41 tail can never activate a
    // filter on a default clip.
    let sol_thr: f32 = f[100].parse().unwrap_or(0.0);
    let temp: f32 = f[101].parse().unwrap_or(0.0);
    // P45 VIDEO FADE factor (f[102], the new LAST ENC token after temp). 1.0 = no fade (byte-identical
    // to pre-P45). TOLERANT: a bad/absent token degrades to 1.0 so a malformed tail can't darken a frame.
    let fade: f32 = f[102].parse().unwrap_or(1.0);

    // Decode base @ base_frame (cached), upload to slot 0. A "-" base is an explicit timeline
    // gap (finding #5): fill slot 0 with black (matching MojoMedia's black-gap behavior) and
    // skip decoding entirely. A `RAW:<path>` base is a P5 rasterized TITLE layer (a raw GVW*GVH*4
    // RGBA8 file): read it straight into the slot, SKIPPING decode (see `upload_slot`). A black
    // frame also keeps timing if a real base can't be decoded.
    if base_path == "-" {
        gpu.upload(0, &vec![0u8; ffi::GVW * ffi::GVH * 4]);
    } else {
        upload_slot(gpu, decoders, 0, &base_path, base_frame);
    }

    // Decode overlay if present and op>0; otherwise disable the composite. A `RAW:<path>` overlay is
    // a P5 rasterized TITLE layer uploaded directly (skip decode); a decode/raw-read failure disables
    // the composite (eff_op=0) rather than failing the frame.
    let mut eff_op = op;
    if over_path != "-" && op > 0.0 {
        if !upload_slot(gpu, decoders, 1, &over_path, over_frame) {
            eff_op = 0.0;
        }
    } else {
        eff_op = 0.0;
    }

    // Resolve the per-boundary TRANSITION (Wave 8): when active (kind 0..7 AND a real partner path),
    // decode the INCOMING clip's frame into slot 2 so track1 can blend base→trans; a "-"/failed
    // partner degrades to no transition (the base still encodes). Mirrors MojoMedia's render loop
    // (~1286-1300): decode the boundary partner, upload slot 2, then track1(tt_id, rtt, tt_p).
    let eff_tt = resolve_trans(gpu, decoders, trans_kind, &trans_path, trans_frame);

    // Resolve the per-clip LOOK (load + upload the .cube for LUT3D, cached; VHS needs no LUT; a
    // missing/failed LUT degrades to no look). Then run the same transition→composite→look the
    // preview uses, downloading f32 for the encoder.
    let (lk, la, ln) = resolve_look(
        gpu,
        lut_cache,
        last_uploaded_lut,
        look_kind,
        look_amt,
        &lut_path,
    );
    // P4: chroma key only matters with an active overlay (it keys the OVER buffer); force ck_on=0 when
    // the overlay is disabled (no over clip / failed decode → eff_op==0) so we never key a stale slot.
    let eff_ck_on = if eff_op > 0.0 { ck_on } else { 0 };
    let (frame, _fin) = gpu.compose_trans_f32(
        eff_tt,
        trans_prog,
        trans_param,
        eff_op,
        over_blend,
        px,
        py,
        pw,
        ph,
        cbright,
        ccontrast,
        csat,
        bright,
        contrast,
        sat,
        lk,
        la,
        ln,
        lift_r,
        lift_g,
        lift_b,
        gamma_r,
        gamma_g,
        gamma_b,
        gain_r,
        gain_g,
        gain_b,
        rot,
        scale,
        blur,
        eff_ck_on,
        ck_r,
        ck_g,
        ck_b,
        ck_sim,
        ck_smooth,
        ck_spill,
        curve,
        vig,
        sharp,
        flip,
        fx,
        hue,
        sat_hsl,
        light,
        inb,
        inw,
        gam,
        mosaic,
        gmap_amt,
        glo_r,
        glo_g,
        glo_b,
        ghi_r,
        ghi_g,
        ghi_b,
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
    );
    let ts = (*enc_count as f64) / fps;
    if !e.video_frame(&frame, ts) {
        eprintln!("[gcompose] enc video_frame failed @ {}", *enc_count);
        return false;
    }
    *enc_count += 1;
    true
}

/// Resolve the wire TRANSITION fields into the effective transition kind for `compose_trans*`,
/// performing the slot-2 upload as a SIDE EFFECT (Wave 8).
///
/// Returns the kind to pass to `track1`/`compose_trans*`:
///   - `trans_kind == -1` (or out of the 0..=10 range, or a "-"/empty partner path): returns -1 so
///     the pipeline runs `track1(-1, ..)` (copy the slot-0 base — today's no-transition behavior).
///     Slot 2 is NOT touched.
///   - `trans_kind` in 0..=10 with a real partner path: decode `trans_path`@`trans_frame` (cached
///     decoder) and upload it to slot 2, then return `trans_kind`. If the decode FAILS, degrade to
///     no transition (return -1) so the frame still composes the base rather than failing.
///
/// Mirrors MojoMedia (main_editor.mojo ~1286-1300): `fpx_decode_frame_letterbox(...rgba8_trans...)`
/// → `fpx_gpu_upload_u8(2, rgba8_trans)` → the kind fed to `fpx_gpu_track1`. Unlike MojoMedia we do
/// NOT track a "currently-resident slot-2 frame" to skip re-uploads: each transition frame's partner
/// frame differs (the incoming clip advances with the playhead), so a cache key would rarely hit;
/// the decoder cache already avoids re-opening the file.
fn resolve_trans(
    gpu: &ffi::Gpu,
    decoders: &mut HashMap<String, ffi::Decoder>,
    trans_kind: i32,
    trans_path: &str,
    trans_frame: i32,
) -> i32 {
    if !(0..=10).contains(&trans_kind) || trans_path == "-" || trans_path.is_empty() {
        return -1; // no transition: track1(-1,..) copies the base. (0..=10: P36 added iris/clock/barndoor.)
    }
    match decode_cached(decoders, trans_path, trans_frame) {
        Some(rgba) => {
            gpu.upload(2, &rgba);
            trans_kind
        }
        None => {
            // Partner frame couldn't be decoded: degrade to no transition (don't fail the frame).
            eprintln!("[gcompose] transition partner decode failed (degrading to no transition): {trans_path}@{trans_frame}");
            -1
        }
    }
}

/// Program-audio sample rate / channel layout. The accumulator and every decoded clip range use
/// this fixed interleaved-stereo-48k layout (matches OPEN's `config_audio("aac", 2, 48000, ...)`
/// and MojoMedia's render config). `SR*CH` floats == one second of program audio.
const PROG_SR: usize = 48_000;
const PROG_CH: usize = 2;

/// In-place iterative radix-2 Cooley-Tukey FFT. `re`/`im` must have a power-of-two length.
fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    if n <= 1 {
        return;
    }
    // bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    // butterflies
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0usize;
        while i < n {
            let (mut cur_r, mut cur_i) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let a = i + k;
                let b = i + k + len / 2;
                let tr = cur_r * re[b] - cur_i * im[b];
                let ti = cur_r * im[b] + cur_i * re[b];
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let nr = cur_r * wr - cur_i * wi;
                cur_i = cur_r * wi + cur_i * wr;
                cur_r = nr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// The timeline-synced program-audio accumulator (Slice A).
///
/// `buf` is interleaved stereo f32 @ 48000 sized to the timeline duration; `active` is true between
/// an OPEN/WAVE (alloc) and its CLOSE/WAVECLOSE (clear). AUDIO lines MIX one decoded clip range into
/// `buf` at a sample offset (sample-add, clamped to [-1,1]); positions with no clip stay silent and
/// overlapping clips sum — this is the fix for the old back-to-back concatenation (a clip at t0=70
/// now starts at 70/FPS s, gaps are silence, overlaps mix).
struct ProgAudio {
    buf: Vec<f32>, // interleaved L,R,L,R... ; len = frames * PROG_CH
    active: bool,
    // P27 MASTER GAIN ENVELOPE: a sorted (timeline-seconds, gain-multiplier) curve set by the
    // GAINENV line. Empty (the default + every reset path) → eval_gain_env() returns 1.0 → the
    // per-sample mix is byte-identical to pre-P27. RESET to empty in alloc() and clear() so a stale
    // envelope can never leak across sessions.
    gain_env: Vec<(f64, f32)>, // (sec, gain), sorted ascending by sec
}

impl Default for ProgAudio {
    fn default() -> Self {
        ProgAudio {
            buf: Vec::new(),
            active: false,
            gain_env: Vec::new(),
        }
    }
}

impl ProgAudio {
    /// (Re)allocate the accumulator to `total_s` seconds of silence and mark it active. A prior
    /// (unconsumed) buffer is dropped. `total_s` is clamped to a sane ceiling so a pathological
    /// duration can't blow up the allocation.
    fn alloc(&mut self, total_s: f64) {
        // Ceiling on the WHOLE-program accumulator (24h stereo 48k ≈ 33 GB floats — far past any
        // real timeline; this only guards against a corrupt/huge total_s, not normal use).
        const MAX_FRAMES: usize = 24 * 3600 * PROG_SR;
        let frames = if total_s.is_finite() && total_s > 0.0 {
            (total_s * PROG_SR as f64).ceil() as usize
        } else {
            0
        };
        let frames = frames.min(MAX_FRAMES);
        self.buf = vec![0.0f32; frames.saturating_mul(PROG_CH)];
        self.active = true;
        // P27: a fresh session starts with NO master envelope (GAINENV sets it after the opener);
        // resetting here stops a previous render's envelope from leaking into the next session.
        self.gain_env.clear();
    }

    /// Drop the accumulator and mark inactive (after CLOSE/WAVECLOSE consumes it).
    fn clear(&mut self) {
        self.buf = Vec::new();
        self.active = false;
        // P27: drop the master envelope too, so the next session never sees a stale curve.
        self.gain_env.clear();
    }

    /// Mix `samples` (interleaved stereo @ 48000, already gain-applied) into the accumulator at
    /// frame offset `dst_frame` (samples-per-channel offset). Sample-adds and clamps to [-1,1].
    /// Samples that would land past the end of the accumulator are dropped (the accumulator is the
    /// authoritative timeline length, so a clip running slightly long is truncated to it).
    ///
    /// SUB-SAMPLE BOUNDARY (finding #5, INTEGRATOR NOTE): `alloc` sizes the accumulator with `.ceil()`
    /// while `dst_frame` here is `.round()`ed and the C decoder trims each range to
    /// `(int)(dur_sec*sr+0.5)` — three independent roundings. A clip whose end lands a fraction above
    /// the ceil'd accumulator end therefore has its final ≤1 interleaved frame clamped off by `room`.
    /// This is inaudible (≤1/48000 s ≈ 21 µs) and intentional (the accumulator is authoritative), but
    /// it means the rendered AUDIO duration can be ≤ the video by up to one sample. Any gate that
    /// compares audio-vs-video duration must allow ≥1 sample / ~21 µs of slack, NOT bit-exact equality.
    fn mix(&mut self, samples: &[f32], dst_frame: usize) {
        if samples.is_empty() || self.buf.is_empty() {
            return;
        }
        let start = dst_frame.saturating_mul(PROG_CH);
        if start >= self.buf.len() {
            return; // entirely past the timeline end.
        }
        // Whole interleaved frames only (drop a stray odd tail float).
        let n = (samples.len() / PROG_CH) * PROG_CH;
        let room = self.buf.len() - start;
        let take = n.min(room);
        for i in 0..take {
            let v = self.buf[start + i] + samples[i];
            self.buf[start + i] = v.clamp(-1.0, 1.0);
        }
    }

    /// Feed the WHOLE accumulator to the encoder's audio stream in chunks. The encoder packs the
    /// interleaved floats into proper codec frames internally (fpx_enc_audio_samples_f32 buffers
    /// across calls), so chunking is purely to bound the per-call slice — the muxed result is the
    /// single continuous timeline-length program audio. Best-effort: a rejected chunk is logged and
    /// the drain stops (the already-fed audio still muxes on finish).
    fn drain_into_encoder(&self, e: &mut ffi::Encoder) {
        if self.buf.is_empty() {
            return;
        }
        // ~0.25 s of stereo @ 48k per chunk (12000 frames). Aligned to whole interleaved samples.
        const CHUNK_FRAMES: usize = 12_000;
        let total_frames = self.buf.len() / PROG_CH;
        let mut f = 0usize;
        while f < total_frames {
            let nb = CHUNK_FRAMES.min(total_frames - f);
            let off = f * PROG_CH;
            let slice = &self.buf[off..off + nb * PROG_CH];
            if !e.audio_samples(slice, nb) {
                eprintln!("[gcompose] enc audio_samples failed draining accumulator @ frame {f}");
                return;
            }
            f += nb;
        }
    }

    /// Per-channel PEAK + RMS of the accumulator, in dBFS (P3 level meter). Returns
    /// `(peak_L, peak_R, rms_L, rms_R)`. The accumulator is interleaved stereo (PROG_CH==2); a
    /// channel index beyond the layout (shouldn't happen) is ignored. An EMPTY accumulator (no clips
    /// mixed, or a zero-length window) reports the silence floor on every channel. Peak is the max
    /// |sample|; RMS is `sqrt(mean(sample^2))` over that channel's samples. Both are mapped to dBFS
    /// (0 dBFS = full scale, `LEVELS_FLOOR_DB` = silence) by `lin_to_dbfs`.
    fn measure(&self) -> (f32, f32, f32, f32) {
        if self.buf.is_empty() {
            let s = LEVELS_FLOOR_DB;
            return (s, s, s, s);
        }
        let frames = self.buf.len() / PROG_CH;
        let mut peak = [0.0f32; PROG_CH];
        let mut sumsq = [0.0f64; PROG_CH];
        for fr in 0..frames {
            let base = fr * PROG_CH;
            for ch in 0..PROG_CH {
                let s = self.buf[base + ch];
                let a = s.abs();
                if a > peak[ch] {
                    peak[ch] = a;
                }
                sumsq[ch] += (s as f64) * (s as f64);
            }
        }
        let rms = |ch: usize| -> f32 {
            if frames == 0 {
                0.0
            } else {
                (sumsq[ch] / frames as f64).sqrt() as f32
            }
        };
        // PROG_CH is 2 (stereo). Map L=0, R=1; if a future mono layout is used, R mirrors L.
        let pl = lin_to_dbfs(peak[0]);
        let pr = lin_to_dbfs(if PROG_CH > 1 { peak[1] } else { peak[0] });
        let rl = lin_to_dbfs(rms(0));
        let rr = lin_to_dbfs(rms(if PROG_CH > 1 { 1 } else { 0 }));
        (pl, pr, rl, rr)
    }

    /// Magnitude spectrum of the accumulator: Hann-windowed 4096-sample mono mix → radix-2 FFT →
    /// magnitude of the first N/2 bins, grouped LINEARLY (peak-hold) into `nbins` bars over [0, sr/2].
    fn spectrum(&self, nbins: usize) -> Vec<f32> {
        if nbins == 0 {
            return Vec::new();
        }
        let mut out = vec![0.0f32; nbins];
        if self.buf.is_empty() {
            return out;
        }
        const N: usize = 4096; // FFT window (power of two)
        let frames = self.buf.len() / PROG_CH;
        let take = frames.min(N);
        let mut re = vec![0.0f32; N];
        let mut im = vec![0.0f32; N];
        for i in 0..take {
            let l = self.buf[i * PROG_CH];
            let r = if PROG_CH > 1 {
                self.buf[i * PROG_CH + 1]
            } else {
                l
            };
            let s = 0.5 * (l + r);
            // Hann window
            let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (N as f32 - 1.0)).cos();
            re[i] = s * w;
        }
        fft(&mut re, &mut im);
        let half = N / 2;
        for k in 0..half {
            let mag = (re[k] * re[k] + im[k] * im[k]).sqrt();
            let bar = (k * nbins) / half; // linear FFT-bin → display-bar
            if bar < nbins && mag > out[bar] {
                out[bar] = mag;
            }
        }
        out
    }

    /// Time-domain oscilloscope: the LEFT channel of the first min(4096, frames) accumulator frames,
    /// DECIMATED to `n` points (raw amplitude ~[-1,1]). Empty/zero -> zeros.
    fn samples(&self, n: usize) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        let mut out = vec![0.0f32; n];
        if self.buf.is_empty() {
            return out;
        }
        let frames = self.buf.len() / PROG_CH;
        let win = frames.min(4096);
        if win == 0 {
            return out;
        }
        for k in 0..n {
            let fr = (k * win) / n; // decimate the window to n points
            out[k] = self.buf[fr * PROG_CH]; // LEFT channel (interleaved)
        }
        out
    }
}

/// Linear-interpolated master gain at absolute timeline time `t` seconds from a sorted (sec,gain)
/// envelope. Empty → 1.0; clamp to the first/last key value outside the range; linear between keys.
fn eval_gain_env(env: &[(f64, f32)], t: f64) -> f32 {
    if env.is_empty() {
        return 1.0;
    }
    if t <= env[0].0 {
        return env[0].1;
    }
    if t >= env[env.len() - 1].0 {
        return env[env.len() - 1].1;
    }
    for w in env.windows(2) {
        let (t0, v0) = w[0];
        let (t1, v1) = w[1];
        if t >= t0 && t <= t1 {
            let r = if t1 > t0 {
                ((t - t0) / (t1 - t0)) as f32
            } else {
                0.0
            };
            return v0 + (v1 - v0) * r;
        }
    }
    env[env.len() - 1].1
}

/// `GAINENV <packed>` — set (or clear) the P27 MASTER GAIN ENVELOPE on the active accumulator. The
/// worker sends this AFTER the session opener (OPEN/WAVE/MEAS) and BEFORE the AUDIO lines, so the
/// per-clip mix can read it. `<packed>` is a single space-free token "t0:v0,t1:v1,..." (t in SECONDS
/// f64, v the gain multiplier f32, pairs comma-separated, t:v colon-separated) OR "-" meaning NO
/// envelope (clear). Malformed pairs are skipped; the parsed keys are SORTED ascending by sec before
/// being stored. Returns false (→ ERR) if there is no active accumulator. An empty/"-" envelope makes
/// the per-sample mix byte-identical to pre-P27 (eval_gain_env → 1.0).
fn gainenv_set(line: &str, prog: &mut ProgAudio) -> bool {
    // No active accumulator: a stray GAINENV outside an OPEN/WAVE/MEAS session. ERR.
    if !prog.active {
        eprintln!("[gcompose] GAINENV with no active accumulator (no OPEN/WAVE/MEAS)");
        return false;
    }
    let f: Vec<&str> = line.split_whitespace().collect();
    // GAINENV <packed> = exactly 2 tokens (the packed token is space-free by construction).
    if f.len() != 2 {
        eprintln!("[gcompose] bad GAINENV ({} fields): {line}", f.len());
        return false;
    }
    // "-" clears the envelope (no master gain → identical to pre-P27).
    if f[1] == "-" {
        prog.gain_env.clear();
        return true;
    }
    // Parse "t0:v0,t1:v1,..." — split on ',', each piece split on ':' into (sec f64, gain f32).
    // Malformed pieces are skipped (a robust parse never aborts the session over one bad pair).
    let mut env: Vec<(f64, f32)> = Vec::new();
    for piece in f[1].split(',') {
        if piece.is_empty() {
            continue;
        }
        let mut it = piece.split(':');
        let t = match it.next().and_then(|s| s.parse::<f64>().ok()) {
            Some(v) if v.is_finite() => v,
            _ => continue, // malformed / missing time: skip this pair.
        };
        let g = match it.next().and_then(|s| s.parse::<f32>().ok()) {
            Some(v) if v.is_finite() => v,
            _ => continue, // malformed / missing gain: skip this pair.
        };
        env.push((t, g));
    }
    // SORT ascending by sec so eval_gain_env's windows() interpolation is correct.
    env.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    prog.gain_env = env;
    true
}

/// Per-clip audio-decode capacity ceiling (in FLOATS), shared by AUDIO mixing. Mirrors MojoMedia's
/// AUF (180 s stereo 48k + headroom) used as a per-decode cap. Bounds the temp decode buffer and
/// guarantees the `cap as c_int` narrowing in ffi::decode_audio_range is lossless & positive.
const AUDIO_CAP_MAX: usize = 180 * PROG_SR * PROG_CH + 8192;

/// `AUDIO <path> <src_in_s> <dur_s> <dst_offset_s> <gain> <fade_in_s> <fade_out_s> <clip_len_s>
/// <range_local_s> <fx_chain|->` — decode the SOURCE audio range [src_in_s, src_in_s+dur_s) of
/// `path` to interleaved 2ch @ 48000 f32, apply the per-clip libavfilter `fx_chain` (P3), THEN the
/// per-clip linear `gain` AND the per-clip fade ENVELOPE, and MIX it into the active program-audio
/// accumulator starting at `dst_offset_s` seconds (sample-add, clamped). The fade fields (Triad-B P1)
/// ramp the gain 0→1 over the clip's first `fade_in_s` and 1→0 over its last `fade_out_s` in
/// CLIP-LOCAL time, where the first decoded sample sits at clip-local `range_local_s` and the clip's
/// full length is `clip_len_s`. This is the timeline-sync fix plus per-clip volume/fades/FX: the clip
/// is positioned at its timeline offset, not concatenated.
///
/// `fx_chain` (P3, the 11th field) is a SPACE-FREE libavfilter chain string (commas between filters,
/// `=`/`:`/`|` inside) or "-" when the clip's AudioFx is neutral. When `!= "-"` we run
/// `fpx_au_apply(sr, ch, chain, decoded, nin, &out, cap)` on the decoded range BEFORE gain/fade/mix,
/// REPLACING the decoded buffer with the filtered one. On a filter-graph FAILURE we fall back to the
/// UNFILTERED range so the clip's audio never drops (the FX are just skipped for that clip). A "-"
/// chain skips the filter entirely → byte-identical to the P2 mix. The filter can change the sample
/// count (loudnorm/acompressor latency); the fade envelope below is then measured against the FILTERED
/// length's clip-local timeline, which is the intended post-FX gain ramp.
///
/// Returns false (-> ERR) if there is no active accumulator (no OPEN/WAVE/MEAS), the line is
/// malformed, or the range has no decodable audio — the client treats ERR as "skip this clip" and
/// continues.
fn audio_mix(prog: &mut ProgAudio, line: &str) -> bool {
    // No active accumulator: a stray AUDIO outside an OPEN/WAVE session. ERR (client skips).
    if !prog.active {
        eprintln!("[gcompose] AUDIO with no active accumulator (no OPEN/WAVE)");
        return false;
    }

    let f: Vec<&str> = line.split_whitespace().collect();
    // AUDIO <path> <src_in_s> <dur_s> <dst_offset_s> <gain> <fade_in_s> <fade_out_s> <clip_len_s>
    // <range_local_s> <fx_chain|-> = 11 tokens (Triad-B P3; was 10 in P1, 6 in wave-2). Both the path
    // AND the fx_chain are whitespace-free (the UI builds a comma-joined, space-free filter string;
    // the path is a pool media path like ENC/THUMB), so a fixed-arity split is safe. The trailing 4
    // numeric fields carry the per-clip AUDIO FADE envelope (applied per-sample below); the FINAL
    // token is the per-clip libavfilter chain ("-" when the AudioFx is neutral).
    if f.len() != 11 {
        eprintln!("[gcompose] bad AUDIO ({} fields): {line}", f.len());
        return false;
    }
    // WHITESPACE-SAFE WIRE: the UI percent-encodes the media path token (enc_path); decode it here
    // (dec_path) BEFORE opening the decoder. A space-free path decodes to itself (identity).
    let path = dec_path(f[1]);
    let src_in_s: f64 = match f[2].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let dur_s: f64 = match f[3].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let dst_off_s: f64 = match f[4].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let gain: f32 = match f[5].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    // Fade envelope fields (P1): fade_in/out seconds, the clip's FULL length (for the fade-out anchor),
    // and the clip-local seconds of the first decoded sample (head-trim for a playback clip).
    let fade_in_s: f64 = match f[6].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let fade_out_s: f64 = match f[7].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let clip_len_s: f64 = match f[8].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let range_local_s: f64 = match f[9].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    // P3: the per-clip libavfilter chain ("-" = neutral, skip the filter). Whitespace-free by
    // construction (the UI joins filters with commas, no spaces), so the fixed-arity split kept it
    // as a single token.
    let fx_chain = f[10];
    if !(src_in_s.is_finite() && dur_s.is_finite() && dst_off_s.is_finite())
        || dur_s <= 0.0
        || src_in_s < 0.0
        || dst_off_s < 0.0
        || !gain.is_finite()
        || !(fade_in_s.is_finite()
            && fade_out_s.is_finite()
            && clip_len_s.is_finite()
            && range_local_s.is_finite())
    {
        eprintln!("[gcompose] bad AUDIO src_in={src_in_s} dur={dur_s} dst={dst_off_s} gain={gain} fi={fade_in_s} fo={fade_out_s} cl={clip_len_s} rl={range_local_s}");
        return false;
    }

    let sr = PROG_SR as i32;
    let ch = PROG_CH as i32;
    // Capacity: dur seconds * sr * ch, + a codec-frame of slack, clamped to AUDIO_CAP_MAX (finding
    // #4: saturating math + clamp keeps the `as c_int` narrowing lossless and positive).
    let want = (dur_s * PROG_SR as f64).ceil();
    let want = if want.is_finite() && want >= 0.0 {
        want as usize
    } else {
        0
    };
    let cap = want
        .saturating_mul(PROG_CH)
        .saturating_add(8192)
        .min(AUDIO_CAP_MAX);

    let mut samples = match ffi::decode_audio_range(&path, src_in_s, dur_s, sr, ch, cap) {
        Some(s) => s,
        None => {
            eprintln!("[gcompose] AUDIO decode failed: {path} @ {src_in_s}+{dur_s}");
            return false; // hard decode error -> ERR (client skips this clip).
        }
    };
    // No audio in the range: nothing to mix. ERR so the client logs it; accumulator is unchanged.
    if samples.is_empty() {
        return false;
    }

    // P3 AUDIO FX: when a real chain is present, run it on the decoded range BEFORE gain/fade/mix and
    // REPLACE the decoded buffer with the filtered output. A "-" (or empty) chain skips this entirely,
    // keeping the no-FX path byte-identical to P2. On a filter-graph FAILURE we KEEP the unfiltered
    // range (fall back) so the clip's audio is never silently dropped — the FX are skipped, the audio
    // still mixes. fpx_au_apply may return a different sample count (loudnorm/acompressor latency);
    // the gain/fade loop below re-derives `frames` from the (possibly new) length, so it stays correct.
    if fx_chain != "-" && !fx_chain.is_empty() {
        match ffi::au_apply(fx_chain, &samples, sr, ch) {
            Some(filtered) if !filtered.is_empty() => samples = filtered,
            Some(_) => {
                // Filter produced no output (e.g. all-trimmed by a gate at the head): nothing to mix
                // for this clip. ERR so the client just skips it; the accumulator is unchanged.
                return false;
            }
            None => {
                // Hard filter-graph error: degrade to the UNFILTERED range so audio never drops.
                eprintln!("[gcompose] AUDIO fx chain failed (mixing unfiltered): {fx_chain}");
            }
        }
    }

    // Apply per-clip GAIN + FADE envelope in place (Triad-B P1). The gain is a flat linear multiplier;
    // the fades ramp the gain 0→1 over [0, fade_in_s) and 1→0 over [clip_len_s − fade_out_s,
    // clip_len_s) in CLIP-LOCAL time. Sample k (per channel) of this decoded range is at clip-local
    // time `range_local_s + k/sr` (range_local_s = head-trim for a playback clip; 0 for a render clip).
    // When there is no fade AND gain == 1.0 the common case skips the loop entirely.
    let has_fade = fade_in_s > 0.0 || fade_out_s > 0.0;
    // P27: snapshot the master gain envelope ONCE before the loop. Cloning (it is tiny — a handful
    // of keys) frees `prog` for the `prog.mix(&samples, ...)` mutable borrow below; an EMPTY env
    // makes eval_gain_env return 1.0 so the multiply is a no-op (byte-identical to pre-P27).
    let menv = prog.gain_env.clone();
    // Run the per-sample loop when the per-clip gain isn't unity, OR there's a fade, OR a master
    // envelope is present (the last clause is the GATE case: a flat-gain, fade-less clip still needs
    // the master envelope applied). When all three are absent the loop is skipped — identical to today.
    if gain != 1.0 || has_fade || !menv.is_empty() {
        let sr_f = PROG_SR as f64;
        let fade_out_start = clip_len_s - fade_out_s; // clip-local time where fade-out begins
        let frames = samples.len() / PROG_CH;
        for fr in 0..frames {
            // Clip-local time of this interleaved frame.
            let tl = range_local_s + (fr as f64) / sr_f;
            let mut env = 1.0f64;
            if fade_in_s > 0.0 && tl < fade_in_s {
                let r = tl / fade_in_s;
                env *= if r < 0.0 { 0.0 } else { r }; // 0→1 ramp in
            }
            if fade_out_s > 0.0 && tl >= fade_out_start {
                let r = (clip_len_s - tl) / fade_out_s;
                env *= r.clamp(0.0, 1.0); // 1→0 ramp out
            }
            // P27: multiply by the MASTER gain envelope at this frame's ABSOLUTE timeline time
            // (dst_off_s = the clip's destination offset in the program; fr/sr = frames into the
            // range). Empty menv → eval_gain_env == 1.0 → unchanged.
            let abs_t = dst_off_s + (fr as f64) / sr_f;
            let g = (gain * env as f32) * eval_gain_env(&menv, abs_t);
            let base = fr * PROG_CH;
            for ch in 0..PROG_CH {
                samples[base + ch] *= g;
            }
        }
    }

    // Destination sample-per-channel offset. Round to nearest frame; the accumulator's mix() drops
    // anything past the end so an offset at/after the timeline end is a harmless no-op.
    let dst_frame = (dst_off_s * PROG_SR as f64).round();
    let dst_frame = if dst_frame.is_finite() && dst_frame >= 0.0 {
        dst_frame as usize
    } else {
        0
    };
    prog.mix(&samples, dst_frame);
    true
}

/// `WAVE <out_wav> <total_s>` — begin a PLAYBACK-ONLY accumulator session (no encoder). Allocates
/// the program-audio accumulator to `total_s` seconds of silence so subsequent AUDIO lines mix into
/// it exactly like the render path; `WAVECLOSE` then writes it to `<out_wav>`. The out path is
/// parsed here only for arity validation (WAVECLOSE carries the real write target).
fn wave_open(line: &str, prog: &mut ProgAudio) -> bool {
    let f: Vec<&str> = line.split_whitespace().collect();
    // WAVE <out_wav> <total_s>
    if f.len() != 3 {
        eprintln!("[gcompose] bad WAVE ({} fields): {line}", f.len());
        return false;
    }
    let total_s: f64 = match f[2].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    if !total_s.is_finite() || total_s < 0.0 {
        return false;
    }
    prog.alloc(total_s);
    true
}

/// `WAVECLOSE <out_wav>` — write the playback accumulator to `<out_wav>` as a 16-bit PCM stereo @
/// 48000 WAV, then clear it. Returns the out path on success. The UI spawns a system player on the
/// file. ERR if there is no active accumulator or the write fails.
fn wave_close(line: &str, prog: &mut ProgAudio) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    // WAVECLOSE <out_wav>
    if f.len() != 2 {
        eprintln!("[gcompose] bad WAVECLOSE ({} fields): {line}", f.len());
        return None;
    }
    let out = f[1];
    if !prog.active {
        eprintln!("[gcompose] WAVECLOSE with no active accumulator");
        return None;
    }
    let ok = write_wav_pcm16(out, &prog.buf, PROG_SR as u32, PROG_CH as u16);
    prog.clear();
    if ok {
        Some(out.to_string())
    } else {
        eprintln!("[gcompose] WAVECLOSE write failed: {out}");
        None
    }
}

/// `MEAS <window_s>` — begin a MEASUREMENT-ONLY accumulator session (no encoder, no WAV) for the
/// audio level meter. Allocates the program-audio accumulator to `window_s` seconds of silence so
/// subsequent AUDIO lines mix the filtered+gained clip ranges into it exactly like the render/playback
/// path; `LEVELS` then measures peak+RMS over it and clears it. Distinct from WAVE only in intent —
/// it shares the ProgAudio accumulator — but kept as its own verb so the protocol reads clearly and a
/// future change (e.g. a smaller fixed measurement layout) doesn't disturb the playback path.
fn meas_open(line: &str, prog: &mut ProgAudio) -> bool {
    let f: Vec<&str> = line.split_whitespace().collect();
    // MEAS <window_s>
    if f.len() != 2 {
        eprintln!("[gcompose] bad MEAS ({} fields): {line}", f.len());
        return false;
    }
    let window_s: f64 = match f[1].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    if !window_s.is_finite() || window_s < 0.0 {
        return false;
    }
    prog.alloc(window_s);
    true
}

/// `LEVELS <out>` — measure the active accumulator's per-channel PEAK + RMS (dBFS), write the 4
/// little-endian f32 [peak_L, peak_R, rms_L, rms_R] to `<out>`, then CLEAR the accumulator. Returns
/// the out path on success. ERR if there is no active accumulator or the write fails. This is the
/// session terminator for a MEAS (or any active) session — it consumes the accumulator like
/// WAVECLOSE, but emits levels instead of a WAV. The values are computed over the WHOLE accumulator
/// (the measurement window MEAS sized), so they reflect the assembled, filtered, gained mix — no
/// real-time device capture.
fn levels_query(line: &str, prog: &mut ProgAudio) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    // LEVELS <out>
    if f.len() != 2 {
        eprintln!("[gcompose] bad LEVELS ({} fields): {line}", f.len());
        return None;
    }
    let out = f[1];
    if !prog.active {
        eprintln!("[gcompose] LEVELS with no active accumulator");
        return None;
    }
    let (peak_l, peak_r, rms_l, rms_r) = prog.measure();
    prog.clear(); // accumulator consumed (session terminator).

    let mut bytes = Vec::with_capacity(16);
    for v in [peak_l, peak_r, rms_l, rms_r] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if std::fs::write(out, &bytes).is_err() {
        eprintln!("[gcompose] LEVELS write failed: {out}");
        return None;
    }
    Some(out.to_string())
}

/// `SPECTRUM <nbins> <out>` — compute the active accumulator's magnitude spectrum (Hann-windowed
/// 4096-sample mono mix → radix-2 FFT → first N/2 magnitudes grouped LINEARLY peak-hold into `<nbins>`
/// bars over [0, sr/2]), write EXACTLY `<nbins>` little-endian f32 magnitudes to `<out>`, then CLEAR
/// the accumulator. Returns the out path on success. ERR if there is no active accumulator or the
/// write fails. This is the session terminator for a MEAS (or any active) session — it consumes the
/// accumulator like LEVELS/WAVECLOSE, but emits a frequency spectrum instead of levels/a WAV. The bins
/// are computed over the WHOLE accumulator (read-only), so they reflect the assembled, filtered,
/// gained mix — no real-time device capture. Mirrors `levels_query` exactly; touches NO render/mix path.
fn spectrum_query(line: &str, prog: &mut ProgAudio) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    // SPECTRUM <nbins> <out>
    if f.len() != 3 {
        eprintln!("[gcompose] bad SPECTRUM ({} fields): {line}", f.len());
        return None;
    }
    let nbins: usize = match f[1].parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("[gcompose] bad SPECTRUM nbins: {line}");
            return None;
        }
    };
    let out = f[2];
    if !prog.active {
        eprintln!("[gcompose] SPECTRUM with no active accumulator");
        return None;
    }
    let bins = prog.spectrum(nbins);
    prog.clear(); // accumulator consumed (session terminator).

    let bytes: Vec<u8> = bins.iter().flat_map(|v| v.to_le_bytes()).collect();
    if std::fs::write(out, &bytes).is_err() {
        eprintln!("[gcompose] SPECTRUM write failed: {out}");
        return None;
    }
    Some(out.to_string())
}

/// `SAMPLES <n> <out>` — compute the active accumulator's time-domain oscilloscope (the LEFT channel
/// of the first min(4096, frames) accumulator frames, decimated to `<n>` raw-amplitude points ~[-1,1]),
/// write EXACTLY `<n>` little-endian f32 to `<out>`, then CLEAR the accumulator. Returns the out path on
/// success. ERR if there is no active accumulator or the write fails. This is the session terminator for
/// a MEAS (or any active) session — it consumes the accumulator like LEVELS/SPECTRUM/WAVECLOSE, but
/// emits raw time-domain samples instead of levels/a spectrum/a WAV. The samples are read over the WHOLE
/// accumulator (read-only), so they reflect the assembled, filtered, gained mix — no real-time device
/// capture. Mirrors `spectrum_query` exactly; touches NO render/mix path.
fn samples_query(line: &str, prog: &mut ProgAudio) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    // SAMPLES <n> <out>
    if f.len() != 3 {
        eprintln!("[gcompose] bad SAMPLES ({} fields): {line}", f.len());
        return None;
    }
    let n: usize = match f[1].parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("[gcompose] bad SAMPLES n: {line}");
            return None;
        }
    };
    let out = f[2];
    if !prog.active {
        eprintln!("[gcompose] SAMPLES with no active accumulator");
        return None;
    }
    let s = prog.samples(n);
    prog.clear(); // accumulator consumed (session terminator).

    let bytes: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
    if std::fs::write(out, &bytes).is_err() {
        eprintln!("[gcompose] SAMPLES write failed: {out}");
        return None;
    }
    Some(out.to_string())
}

/// dBFS floor for digital silence (matches worker.rs `LEVELS_FLOOR_DB`). A linear peak/RMS of 0 maps
/// to this instead of −inf so the meter has a finite bottom.
const LEVELS_FLOOR_DB: f32 = -90.0;

/// Convert a linear amplitude (0..1, 1.0 = full scale) to dBFS, flooring digital silence (and any
/// non-finite input) at `LEVELS_FLOOR_DB`. `20*log10(x)` for x>0.
fn lin_to_dbfs(x: f32) -> f32 {
    if x > 0.0 && x.is_finite() {
        (20.0 * x.log10()).max(LEVELS_FLOOR_DB)
    } else {
        LEVELS_FLOOR_DB
    }
}

/// Write interleaved f32 [-1,1] `samples` as a 16-bit PCM WAV (`sr` Hz, `ch` channels). Standard
/// 44-byte canonical WAV header + little-endian s16 samples. Returns true on success. No external
/// deps — paplay/aplay both play canonical PCM WAV. (Best-effort playback path for Slice A.)
fn write_wav_pcm16(path: &str, samples: &[f32], sr: u32, ch: u16) -> bool {
    let bits_per_sample: u16 = 16;
    let block_align: u16 = ch * bits_per_sample / 8;
    let byte_rate: u32 = sr * block_align as u32;
    let data_bytes: u32 = (samples.len() as u32).saturating_mul(2); // 2 bytes per s16 sample
    let riff_size: u32 = 36u32.saturating_add(data_bytes);

    let mut out: Vec<u8> = Vec::with_capacity(44 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&ch.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in samples {
        // Clamp + scale to s16. (Mix already clamps, but re-clamp defensively.)
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i32;
        out.extend_from_slice(&(v as i16).to_le_bytes());
    }
    std::fs::write(path, &out).is_ok()
}

/// `THUMB <path> <frame> <w> <h> <out>` — decode one frame letterboxed to w×h and write the
/// RGBA8 buffer to <out>. Uses the cached decoders (a thumbnail of an already-open media reuses
/// the handle). Returns the out path on success.
fn thumb(decoders: &mut HashMap<String, ffi::Decoder>, line: &str) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    // THUMB <path> <frame> <w> <h> <out>
    if f.len() != 6 {
        eprintln!("[gcompose] bad THUMB ({} fields): {line}", f.len());
        return None;
    }
    // WHITESPACE-SAFE WIRE: decode the percent-encoded media path token (dec_path) BEFORE opening
    // the decoder. A space-free path decodes to itself (identity). The out token is a hashed /tmp
    // path (no whitespace) → dec_path is identity, applied for symmetry.
    let path = dec_path(f[1]);
    let frame: i32 = f[2].parse().ok()?;
    let w: usize = f[3].parse().ok()?;
    let h: usize = f[4].parse().ok()?;
    let out = dec_path(f[5]);
    if w == 0 || h == 0 {
        return None;
    }

    if !decoders.contains_key(&path) {
        let d = ffi::Decoder::open(&path)?;
        decoders.insert(path.clone(), d);
    }
    let dec = decoders.get_mut(&path)?;
    let buf = dec.decode_rgba(frame.max(0), w, h)?;
    if std::fs::write(&out, &buf).is_err() {
        eprintln!("[gcompose] THUMB write failed: {out}");
        return None;
    }
    Some(out)
}

/// `NFRAMES <path>` — total video frame count of <path>, returned INLINE as the DONE payload (e.g.
/// "DONE 121"; "DONE 0" when unknown). Reuses the cached decoders map (mirrors `thumb`), so a clip
/// already on the timeline shares its open handle. The UI clamps clip lengths against this.
fn nframes_cmd(decoders: &mut HashMap<String, ffi::Decoder>, line: &str) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    if f.len() != 2 {
        eprintln!("[gcompose] bad NFRAMES ({} fields): {line}", f.len());
        return None;
    }
    let path = dec_path(f[1]);
    if !decoders.contains_key(&path) {
        let d = ffi::Decoder::open(&path)?;
        decoders.insert(path.clone(), d);
    }
    let dec = decoders.get(&path)?;
    Some(dec.nframes().to_string())
}

/// `ENV <path> <buckets> <out>` — compute the whole-track peak envelope and write <buckets>
/// little-endian f32 to <out>. Returns the out path on success.
fn envelope(line: &str) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    // ENV <path> <buckets> <out>
    if f.len() != 4 {
        eprintln!("[gcompose] bad ENV ({} fields): {line}", f.len());
        return None;
    }
    // WHITESPACE-SAFE WIRE: decode the percent-encoded media path token (dec_path) BEFORE opening
    // the decoder. A space-free path decodes to itself (identity). The out token is a hashed /tmp
    // path (no whitespace) → dec_path is identity, applied for symmetry.
    let path = dec_path(f[1]);
    let buckets: usize = f[2].parse().ok()?;
    let out = dec_path(f[3]);
    if buckets == 0 {
        return None;
    }

    let env = ffi::audio_envelope(&path, buckets)?;
    // Serialize as little-endian f32 (the UI reads it back the same way).
    let mut bytes = Vec::with_capacity(buckets * 4);
    for v in &env {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if std::fs::write(&out, &bytes).is_err() {
        eprintln!("[gcompose] ENV write failed: {out}");
        return None;
    }
    Some(out)
}

/// `CLIPAUD <path> <start_s> <dur_s> <sr> <n> <out>` (P46 AUDIO ALIGN) — decode `[start_s,
/// start_s+dur_s)` of `path`'s audio to MONO @ `sr`, write EXACTLY `n` little-endian f32
/// (zero-padded/truncated) to `out`. The UI cross-correlates two of these (same `sr`) to recover the
/// time offset between two clips' recordings. Returns the out path on success. Stateless decode.
fn clipaud(line: &str) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    if f.len() != 7 {
        eprintln!("[gcompose] bad CLIPAUD ({} fields): {line}", f.len());
        return None;
    }
    let path = dec_path(f[1]);
    let start_s: f64 = f[2].parse().ok()?;
    let dur_s: f64 = f[3].parse().ok()?;
    let sr: i32 = f[4].parse().ok()?;
    let n: usize = f[5].parse().ok()?;
    let out = dec_path(f[6]);
    if n == 0 || sr <= 0 {
        return None;
    }
    let mut s = ffi::decode_audio_range(&path, start_s, dur_s, sr, 1, n)?;
    s.resize(n, 0.0); // pad/truncate to exactly n so the UI reads a fixed count.
    let mut bytes = Vec::with_capacity(n * 4);
    for v in &s {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if std::fs::write(&out, &bytes).is_err() {
        eprintln!("[gcompose] CLIPAUD write failed: {out}");
        return None;
    }
    Some(out)
}

/// `SCOPE <kind> <out>` — run the kind-selected scope kernel on the LAST composed GPU buffer (the
/// frame left in g_buf[OUTB] by the most recent PREVIEW; it is NOT re-composed or cleared here),
/// produce a 256×256 RGBA8 image, write it to <out>, and return the out path. Returns None (-> ERR)
/// on a malformed line, an unknown kind, or a write failure.
///
/// `final_is_look` selects which composed buffer the scope reads: false = OUTB (the most recent
/// PREVIEW had look=none), true = LOOKB (a VHS/LUT look ran). It is passed by the serve loop from
/// the PREVIEW that composed the displayed frame (Slice A), so the scope ALWAYS reads the POST-LOOK
/// frame the UI is showing — not the pre-look grade buffer. kinds 1/2 (waveform/vectorscope) come
/// back as a rendered RGBA8 image from the kernel; kind 0 (histogram) comes back as 768 R/G/B int
/// bins which `render_histogram` rasterizes into a 256×256 RGBA bar graph.
fn scope(gpu: &ffi::Gpu, line: &str, final_is_look: bool) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    // SCOPE <kind> <out>
    if f.len() != 3 {
        eprintln!("[gcompose] bad SCOPE ({} fields): {line}", f.len());
        return None;
    }
    let kind: u8 = f[1].parse().ok()?;
    let out = f[2];

    // Read the buffer the last PREVIEW left the displayed frame in (OUTB when look=none, LOOKB when
    // a look ran) so the scope reflects the post-look picture.
    let img: Vec<u8> = match kind {
        0 => render_histogram(&gpu.histogram(final_is_look)),
        1 => gpu.waveform(final_is_look),
        2 => gpu.vectorscope(final_is_look),
        3 => gpu.parade(final_is_look), // Triad-B P1: RGB parade (3 side-by-side per-channel panels)
        _ => {
            eprintln!("[gcompose] bad SCOPE kind: {kind}");
            return None;
        }
    };

    // Every scope image is exactly SVW×SVH×4 bytes (the UI length-checks SW*SH*4 on read-back).
    debug_assert_eq!(img.len(), ffi::SVW * ffi::SVH * 4);
    if std::fs::write(out, &img).is_err() {
        eprintln!("[gcompose] SCOPE write failed: {out}");
        return None;
    }
    Some(out.to_string())
}

/// Rasterize the 768 R/G/B histogram bins into a 256×256 RGBA8 bar graph on a dark background
/// (the histogram kernel returns raw counts, not an image — unlike waveform/vectorscope). Mirrors
/// MojoMedia main_editor.mojo (~880-894): scale to the tallest NON-black bin (bin index 0 is the
/// pure-black/letterbox spike and is excluded so it doesn't flatten everything else), then draw one
/// column per bin bottom-up, with the three channels overlaid translucently so overlap blends.
///
/// `bins` is `R[0..256] | G[256..512] | B[512..768]`. Output is row-major top-to-bottom RGBA8
/// (row 0 = top of the image), so a taller bar fills MORE rows toward the bottom — matching the
/// "bars rise from the baseline" look of a standard histogram scope.
fn render_histogram(bins: &[i32]) -> Vec<u8> {
    const W: usize = 256; // == ffi::SVW
    const H: usize = 256; // == ffi::SVH
    let mut img = vec![0u8; W * H * 4];

    // Dark background fill (matches MojoMedia's Col4(0.06,0.07,0.10)).
    const BG: [u8; 4] = [15, 18, 26, 255];
    for px in img.chunks_exact_mut(4) {
        px.copy_from_slice(&BG);
    }
    // A defensive guard: a short/empty bins slice (kernel no-op on a not-ready GPU) yields the
    // plain dark image rather than indexing out of bounds.
    if bins.len() < 768 {
        return img;
    }

    // Scale to the tallest non-black bin across all three channels (skip bin 0 of each channel:
    // the pure-black letterbox spike, which would otherwise dwarf the real content).
    let mut hmax: i32 = 1;
    for i in 1..256 {
        hmax = hmax.max(bins[i]).max(bins[256 + i]).max(bins[512 + i]);
    }
    let hmaxf = hmax as f32;

    // Per-column bar heights (in rows) for each channel, then composite the three translucent bars.
    // alpha ≈ 0.5 over the bg, additive-ish so overlapping channels brighten toward white.
    //
    // COLUMN 0 (finding #2): bin index 0 of each channel (`bins[0]`/`bins[256]`/`bins[512]`) is the
    // per-channel VALUE-0 count — the pure-black/letterbox spike. It is excluded from `hmax` above so
    // it doesn't flatten the real content; if we then DREW it, that spike would still paint column 0
    // at full height (its count ≫ hmax → clamped to H), just relocating the artifact instead of
    // removing it. So column 0 is drawn at zero height to match the hmax exclusion — the histogram
    // shows only the non-black tonal distribution. (The black-level information is intentionally
    // dropped; a scope reads tonal SHAPE, not the absolute black count.)
    for x in 0..W {
        let (rh, gh, bh) = if x == 0 {
            (0usize, 0usize, 0usize)
        } else {
            let rh = ((bins[x] as f32 / hmaxf) * H as f32).round() as usize;
            let gh = ((bins[256 + x] as f32 / hmaxf) * H as f32).round() as usize;
            let bh = ((bins[512 + x] as f32 / hmaxf) * H as f32).round() as usize;
            (rh.min(H), gh.min(H), bh.min(H))
        };
        for y in 0..H {
            // Row y counts from the top; a bar of height `hb` fills the bottom `hb` rows, i.e. rows
            // with (H - y) <= hb  ⇔  y >= H - hb.
            let from_bottom = H - y; // 1..=H
            let r_on = from_bottom <= rh;
            let g_on = from_bottom <= gh;
            let b_on = from_bottom <= bh;
            if !(r_on || g_on || b_on) {
                continue; // leave the dark bg
            }
            let off = (y * W + x) * 4;
            // Start from bg and blend each active channel bar (src-over at a=0.5) so overlapping
            // channels read as a brighter mixed color (R+G -> yellowish, etc.).
            let mut c = [BG[0] as f32, BG[1] as f32, BG[2] as f32];
            if r_on {
                c = blend(c, [235.0, 64.0, 64.0], 0.5);
            }
            if g_on {
                c = blend(c, [64.0, 230.0, 90.0], 0.5);
            }
            if b_on {
                c = blend(c, [90.0, 140.0, 247.0], 0.5);
            }
            img[off] = c[0].round().clamp(0.0, 255.0) as u8;
            img[off + 1] = c[1].round().clamp(0.0, 255.0) as u8;
            img[off + 2] = c[2].round().clamp(0.0, 255.0) as u8;
            img[off + 3] = 255;
        }
    }
    img
}

/// Src-over blend of `src` (RGB 0..255) onto `dst` (RGB 0..255) at alpha `a` (0..1).
fn blend(dst: [f32; 3], src: [f32; 3], a: f32) -> [f32; 3] {
    [
        dst[0] * (1.0 - a) + src[0] * a,
        dst[1] * (1.0 - a) + src[1] * a,
        dst[2] * (1.0 - a) + src[2] * a,
    ]
}

/// Parse + execute one serve request line. Returns the out_path on success, None on any failure.
/// Also UPDATES `last_final_is_look` to the buffer the composed frame ended up in (OUTB=false /
/// LOOKB=true), so a subsequent SCOPE reads the POST-LOOK frame this PREVIEW just produced.
fn handle_request(
    gpu: &ffi::Gpu,
    decoders: &mut HashMap<String, ffi::Decoder>,
    lut_cache: &mut HashMap<String, Option<ffi::Lut>>,
    last_uploaded_lut: &mut Option<String>,
    last_final_is_look: &mut bool,
    line: &str,
) -> Option<String> {
    let mut f: Vec<&str> = line.split_whitespace().collect();
    // Accept both the new explicit form (`PREVIEW` + the positional fields) and the legacy
    // keyword-less form. Strip a leading PREVIEW keyword so the positional indices below are
    // identical for both (finding #3). The fields are the 12 composite fields + the 1 P31 BLEND field
    // (over_blend, inserted IMMEDIATELY AFTER op at f[5]) + the 3 Slice A
    // LOOK fields (look_kind, look_amt, lut_path) + the 5 Wave 8 TRANSITION fields (trans_kind,
    // trans_prog, trans_param, trans_path, trans_frame) + the 3 Triad-B P1 PER-CLIP GRADE fields
    // (cbright, ccontrast, csat) + the 12 P2 per-clip color/transform fields (lift3, gamma3, gain3,
    // rot, scale, blur) + the 6 P4 CHROMA-KEY fields (ck_on, ck_r, ck_g, ck_b, ck_sim, ck_smooth)
    // + the 5 P5 CURVE fields + the 4 P6 STYLIZE/UTILITY fields (vig, sharp, flip, fx) + the 6 P7
    // COLOR fields (hue, sat, light, inb, inw, gam) + the 8 P8 STYLIZE-2 fields (mosaic, gmap_amt,
    // glo_r, glo_g, glo_b, ghi_r, ghi_g, ghi_b) + the 4 P9 FX fields (denoise, glow_amt, glow_thr,
    // rgbshift) + the 3 P10 STYLIZE-4 fields (halftone, emboss, edge) + the 3 P13 OLD-FILM/DISTORT
    // fields (grain, scratches, diffusion) + the 3 P16 DISTORT fields (wave, swirl, threshold) + the
    // 3 P17 GEOMETRIC fields (lens, crop, glitch) + the 4 P23 360-REFRAME fields (eq360, eq_yaw,
    // eq_pitch, eq_fov) + the 7 P34 SHAPE-MASK fields (mask_shape, mask_cx, mask_cy, mask_rw, mask_rh,
    // mask_feather, mask_invert) + the 1 P37 CHROMA SPILL field (ck_spill) + the 3 P38 DISTORTION
    // fields (mirror_x, kaleido, dither) + the 3 P39 SELECTIVE-COLOR fields (sel_band, sel_hshift,
    // sel_sat) + the 2 P41 FILTER fields (sol_thr, temp) + the out path (which stays LAST).
    // (P7 was 57 post-strip; P8 added the 8 mosaic/gmap_amt/glo3/ghi3 → 65; P9 added the 4
    // denoise/glow_amt/glow_thr/rgbshift → 69; P10 added the 3 halftone/emboss/edge → 72; P13 added
    // the 3 grain/scratches/diffusion → 75; P16 added the 3 wave/swirl/threshold → 78; P17 added the 3
    // lens/crop/glitch → 81; P23 adds the 4 eq360/eq_yaw/eq_pitch/eq_fov → 85. P31 inserts the 1
    // over_blend after op (every later field +1) → 86. P34 inserts the 7 mask fields BETWEEN eq_fov and
    // the out path (the out path index shifts +7) → 93. P37 appends the 1 ck_spill field AFTER the P34
    // mask fields (mask_invert) and BEFORE the out path (the out path index shifts +1) → 94. P38 inserts
    // the 3 distortion fields mirror_x/kaleido/dither BETWEEN ck_spill and the out path (the out path
    // index shifts +3) → 97. P39 inserts the 3 selective-color fields sel_band/sel_hshift/sel_sat
    // BETWEEN dither and the out path (the out path index shifts +3) → 100. P41 inserts the 2 filter
    // fields sol_thr/temp (pinned order `sol_thr temp`) BETWEEN sel_sat and the out path (the out path
    // index shifts +2) → 102. The out path stays LAST, now f[101]; ck_spill is f[92]; the P38 fields are
    // f[93..=95]; the P39 fields are f[96..=98]; the P41 fields are f[99..=100].)
    if f.first() == Some(&"PREVIEW") {
        f.remove(0);
    }
    if f.len() != 103 {
        eprintln!("[gcompose] bad request ({} fields): {line}", f.len());
        return None;
    }

    // WHITESPACE-SAFE WIRE: the UI percent-encodes every path token (enc_path); decode each here
    // (dec_path) BEFORE it is used to open a file. A space-free path / "-" / "RAW:..." decodes to
    // itself (identity). upload_slot strips the "RAW:" prefix AFTER this decode.
    let base_path = dec_path(f[0]);
    let over_path = dec_path(f[1]); // "-" means no overlay
    let base_frame: i32 = f[2].parse().ok()?;
    let over_frame: i32 = f[3].parse().ok()?;
    let op: f32 = f[4].parse().ok()?;
    // P31 BLEND MODE of the OVER (V2) clip, riding the wire IMMEDIATELY AFTER `op` (f[4]) at f[5]:
    // 0=Normal 1=Multiply 2=Screen 3=Overlay 4=Add 5=Darken 6=Lighten 7=Difference. Tolerant — a
    // bad/absent token degrades to 0 (Normal) which is byte-identical to the pre-P31 plain composite.
    let over_blend: i32 = f[5].parse().unwrap_or(0);
    let px: f32 = f[6].parse().ok()?;
    let py: f32 = f[7].parse().ok()?;
    let pw: f32 = f[8].parse().ok()?;
    let ph: f32 = f[9].parse().ok()?;
    let bright: f32 = f[10].parse().ok()?;
    let contrast: f32 = f[11].parse().ok()?;
    let sat: f32 = f[12].parse().ok()?;
    let look_kind: i32 = f[13].parse().ok()?;
    let look_amt: f32 = f[14].parse().ok()?;
    let lut_path = dec_path(f[15]); // "-" / empty when no LUT (only used by LUT3D look_kind==2)
                                    // Wave 8 TRANSITION fields.
    let trans_kind: i32 = f[16].parse().ok()?;
    let trans_prog: f32 = f[17].parse().ok()?;
    let trans_param: f32 = f[18].parse().ok()?;
    let trans_path = dec_path(f[19]); // "-" when no transition partner
    let trans_frame: i32 = f[20].parse().ok()?;
    // Triad-B P1 PER-CLIP GRADE fields.
    let cbright: f32 = f[21].parse().ok()?;
    let ccontrast: f32 = f[22].parse().ok()?;
    let csat: f32 = f[23].parse().ok()?;
    // P2 per-clip color/transform effects (f[24..=35]), pinned order: lift3, gamma3, gain3, rot,
    // scale, blur. Identity defaults: lift_*=0, gamma_*=1, gain_*=1, rot=0, scale=1, blur=0.
    let lift_r: f32 = f[24].parse().ok()?;
    let lift_g: f32 = f[25].parse().ok()?;
    let lift_b: f32 = f[26].parse().ok()?;
    let gamma_r: f32 = f[27].parse().ok()?;
    let gamma_g: f32 = f[28].parse().ok()?;
    let gamma_b: f32 = f[29].parse().ok()?;
    let gain_r: f32 = f[30].parse().ok()?;
    let gain_g: f32 = f[31].parse().ok()?;
    let gain_b: f32 = f[32].parse().ok()?;
    let rot: f32 = f[33].parse().ok()?;
    let scale: f32 = f[34].parse().ok()?;
    let blur: f32 = f[35].parse().ok()?;
    // P4 per-clip CHROMA-KEY fields (f[36..=41]), pinned order: ck_on, ck_r, ck_g, ck_b, ck_sim,
    // ck_smooth. Identity defaults: ck_on=0 (disabled → OVER alpha untouched, byte-identical to P3),
    // key=green [0,1,0], sim=0.4, smooth=0.1. These describe the OVER (V2) clip.
    let ck_on: i32 = f[36].parse().ok()?;
    let ck_r: f32 = f[37].parse().ok()?;
    let ck_g: f32 = f[38].parse().ok()?;
    let ck_b: f32 = f[39].parse().ok()?;
    let ck_sim: f32 = f[40].parse().ok()?;
    let ck_smooth: f32 = f[41].parse().ok()?;
    // P5 master tone CURVE (f[42..=46]): 5 outputs at fixed inputs 0/.25/.5/.75/1. Identity skipped.
    let curve: [f32; 5] = [
        f[42].parse().ok()?,
        f[43].parse().ok()?,
        f[44].parse().ok()?,
        f[45].parse().ok()?,
        f[46].parse().ok()?,
    ];
    // P6 STYLIZE/UTILITY fields (f[47..=50]), pinned order: vig sharp flip fx. Identity defaults
    // vig=0, sharp=0, flip=0, fx=0 are skipped engine-side, so an unfiltered clip is byte-identical.
    let vig: f32 = f[47].parse().ok()?;
    let sharp: f32 = f[48].parse().ok()?;
    let flip: i32 = f[49].parse().ok()?;
    let fx: i32 = f[50].parse().ok()?;
    // P7 COLOR fields (f[51..=56]), pinned order: hue sat light inb inw gam. Identity defaults
    // hue=0, sat=1, light=0, inb=0, inw=1, gam=1 are skipped engine-side, so an unfiltered clip is
    // byte-identical. hue=HSL hue shift (deg), sat=HSL saturation mult, light=HSL lightness add;
    // inb/inw=levels input black/white, gam=levels gamma.
    let hue: f32 = f[51].parse().ok()?;
    let sat_hsl: f32 = f[52].parse().ok()?;
    let light: f32 = f[53].parse().ok()?;
    let inb: f32 = f[54].parse().ok()?;
    let inw: f32 = f[55].parse().ok()?;
    let gam: f32 = f[56].parse().ok()?;
    // P8 STYLIZE-2 fields (f[57..=64]), pinned order: mosaic gmap_amt glo_r glo_g glo_b ghi_r ghi_g
    // ghi_b. Identity defaults mosaic=0 (0/1 = off), gmap_amt=0 are skipped engine-side, so an
    // unfiltered clip is byte-identical. mosaic=block size in px (parsed as i32 — the wire carries a
    // plain integer; the model's u32 is printed as a plain decimal that round-trips to i32);
    // gmap_amt=gradient-map mix 0..1; glo=shadow colour, ghi=highlight colour.
    let mosaic: i32 = f[57].parse().ok()?;
    let gmap_amt: f32 = f[58].parse().ok()?;
    let glo_r: f32 = f[59].parse().ok()?;
    let glo_g: f32 = f[60].parse().ok()?;
    let glo_b: f32 = f[61].parse().ok()?;
    let ghi_r: f32 = f[62].parse().ok()?;
    let ghi_g: f32 = f[63].parse().ok()?;
    let ghi_b: f32 = f[64].parse().ok()?;
    // P9 FX fields (f[65..=68]), pinned order: denoise glow_amt glow_thr rgbshift. Identity defaults
    // denoise=0, glow_amt=0, rgbshift=0 are skipped engine-side (glow_thr only matters when
    // glow_amt>0), so an unfiltered clip is byte-identical. denoise=bilateral strength 0..1;
    // glow_amt=bloom mix 0..1; glow_thr=bloom luma threshold; rgbshift=chromatic-aberration offset (px).
    let denoise: f32 = f[65].parse().ok()?;
    let glow_amt: f32 = f[66].parse().ok()?;
    let glow_thr: f32 = f[67].parse().ok()?;
    let rgbshift: f32 = f[68].parse().ok()?;
    // P10 STYLIZE-4 fields (f[69..=71]), pinned order: halftone emboss edge. Identity defaults
    // halftone=0 (0/1 = off), emboss=0, edge=0 are skipped engine-side, so an unfiltered clip is
    // byte-identical. halftone=dot cell size in px (parsed as i32 — the wire carries a plain integer;
    // the model's u32 round-trips to i32); emboss=relief strength 0..1; edge=Sobel edge/sketch mix 0..1.
    let halftone: i32 = f[69].parse().ok()?;
    let emboss: f32 = f[70].parse().ok()?;
    let edge: f32 = f[71].parse().ok()?;
    // P13 OLD-FILM/DISTORT fields (f[72..=74]), pinned order: grain scratches diffusion. Identity
    // defaults grain=0, scratches=0, diffusion=0 are skipped engine-side, so an unfiltered clip is
    // byte-identical. grain=film-noise strength 0..1; scratches=scratch density/amount 0..1;
    // diffusion=frosted-glass jitter radius in px (0..16). The pseudo-randomness is a deterministic
    // integer hash of the pixel coords (same input frame => same output), so the gates are stable.
    let grain: f32 = f[72].parse().ok()?;
    let scratches: f32 = f[73].parse().ok()?;
    let diffusion: f32 = f[74].parse().ok()?;
    // P16 DISTORT fields (f[75..=77]), pinned order: wave swirl threshold. Identity defaults wave=0,
    // swirl=0, threshold=0 are skipped engine-side, so an unfiltered clip is byte-identical. wave=
    // sinusoidal displacement amplitude in px; swirl=rotation strength in radians at the centre;
    // threshold=luma binarize level 0..1. Applied on OUTB AFTER the P13 diffusion, BEFORE the look.
    let wave: f32 = f[75].parse().ok()?;
    let swirl: f32 = f[76].parse().ok()?;
    let threshold: f32 = f[77].parse().ok()?;
    // P17 GEOMETRIC fields (f[78..=80]), pinned order: lens crop glitch. Identity defaults lens=0,
    // crop=0, glitch=0 are skipped engine-side, so an unfiltered clip is byte-identical. lens=radial
    // barrel(+)/pincushion(-) coefficient (0=off); crop=border-to-black fraction 0..0.49; glitch=max
    // per-band horizontal channel shift in px (deterministic band hash). Applied on OUTB AFTER the P16
    // threshold, BEFORE the look.
    let lens: f32 = f[78].parse().ok()?;
    let crop: f32 = f[79].parse().ok()?;
    let glitch: f32 = f[80].parse().ok()?;
    // P23 360-REFRAME fields (f[81..=84]), pinned order: eq360 eq_yaw eq_pitch eq_fov. Slotted
    // BETWEEN the P17 glitch and the out path. Identity eq360=0 (off) is skipped engine-side (the FFI
    // returns immediately, OUTB untouched) so an un-reframed clip is byte-identical to pre-P23. eq360
    // is an INTEGER flag (1=on / 0=off, parsed as i32, nonzero=on); eq_yaw/eq_pitch = view yaw/pitch in
    // degrees (identity 0/0); eq_fov = horizontal field of view in degrees (default 90). Only the out
    // path is percent-decoded — the numeric fields are parsed as-is. Applied on OUTB AFTER the P17
    // glitch, BEFORE the look.
    let eq360: i32 = f[81].parse().ok()?;
    let eq_yaw: f32 = f[82].parse().ok()?;
    let eq_pitch: f32 = f[83].parse().ok()?;
    let eq_fov: f32 = f[84].parse().ok()?;
    // P34 SHAPE-MASK fields (f[85..=91]), pinned order: mask_shape mask_cx mask_cy mask_rw mask_rh
    // mask_feather mask_invert. Slotted BETWEEN the P23 eq_fov and the out path (the out path index
    // shifts +7). Identity mask_shape=0 (none) is skipped engine-side (the FFI returns immediately, OUTB
    // untouched) so an unmasked clip is byte-identical to pre-P34. mask_shape is an INTEGER (0=none
    // 1=rect 2=ellipse); mask_cx/mask_cy = mask centre (0..1, identity 0.5/0.5); mask_rw/mask_rh =
    // half-extents (0..1, default 0.5/0.5); mask_feather = soft-edge band width (default 0); mask_invert
    // is an INTEGER flag (1=on / 0=off). TOLERANT (gate awareness): a bad/absent mask_shape/mask_invert
    // token degrades to 0 (none / not inverted — a true no-op), and a bad geometry token degrades to its
    // natural default, so a malformed P34 tail can never flip a shape-0 clip into a masked one. Applied
    // on OUTB AFTER the P23 reframe, BEFORE the look.
    let mask_shape: i32 = f[85].parse().unwrap_or(0);
    let mask_cx: f32 = f[86].parse().unwrap_or(0.5);
    let mask_cy: f32 = f[87].parse().unwrap_or(0.5);
    let mask_rw: f32 = f[88].parse().unwrap_or(0.5);
    let mask_rh: f32 = f[89].parse().unwrap_or(0.5);
    let mask_feather: f32 = f[90].parse().unwrap_or(0.0);
    let mask_invert: i32 = f[91].parse().unwrap_or(0);
    // P37 CHROMA GREEN-SPILL field (f[92]), slotted AFTER the P34 mask fields (mask_invert) and BEFORE
    // the out path (the out path index shifts +1). ck_spill = green-spill suppression strength (0..1).
    // Identity ck_spill=0 leaves the OVER green untouched (the kernel's spill if is skipped) →
    // byte-identical to pre-P37. TOLERANT: a bad/absent token degrades to 0.0 (a true no-op), so a
    // malformed tail can never introduce spill. Only matters when chroma is enabled (the spill code
    // lives inside k_chroma, run only when eff_ck_on != 0).
    let ck_spill: f32 = f[92].parse().unwrap_or(0.0);
    // P38 DISTORTION fields (f[93..=95]), pinned order: mirror_x kaleido dither. Slotted BETWEEN the P37
    // ck_spill and the out path (the out path index shifts +3). Each is a no-op at its default
    // (mirror_x 0 / kaleido <2 / dither 0) → engine skips → byte-identical to pre-P38. mirror_x and
    // kaleido are INTEGERS (mirror_x 0=off/1=on; kaleido 0/1=off, >=2 segment count); dither is an
    // f32 strength (0=off, 0..1). TOLERANT (gate awareness): a bad/absent token degrades to its no-op
    // default (0/0/0.0), so a malformed P38 tail can never enable a distortion on a default clip.
    // Applied on OUTB AFTER the P34 mask, BEFORE the look.
    let mirror_x: i32 = f[93].parse().unwrap_or(0);
    let kaleido: i32 = f[94].parse().unwrap_or(0);
    let dither: f32 = f[95].parse().unwrap_or(0.0);
    // P39 SELECTIVE-COLOR fields (f[96..=98]), pinned order: sel_band sel_hshift sel_sat. Slotted
    // BETWEEN the P38 dither and the out path (the out path index shifts +3). No-op at its default
    // (sel_band==0) → engine skips → byte-identical to pre-P39. sel_band is an INTEGER (0=off 1=Red
    // 2=Yellow 3=Green 4=Cyan 5=Blue 6=Magenta); sel_hshift is an f32 hue rotation (-1..1); sel_sat is
    // an f32 saturation multiplier (default 1.0). TOLERANT (gate awareness): a bad/absent token degrades
    // to its no-op default (0/0.0/1.0), so a malformed P39 tail can never select a band on a default
    // clip. Applied on OUTB AFTER the P38 dither, BEFORE the look.
    let sel_band: i32 = f[96].parse().unwrap_or(0);
    let sel_hshift: f32 = f[97].parse().unwrap_or(0.0);
    let sel_sat: f32 = f[98].parse().unwrap_or(1.0);
    // P41 SOLARIZE + COLOUR TEMPERATURE fields (f[99..=100]), pinned order: sol_thr temp. Slotted BETWEEN
    // the P39 sel_sat and the out path (the out path index shifts +2). Each is a no-op at its default
    // (sol_thr<=0 / temp==0) → engine skips → byte-identical to pre-P41. sol_thr is the solarize threshold
    // (f32, 0=off, active (0,1]); temp is the warm/cool shift (f32, 0=neutral, -1..1). TOLERANT (gate
    // awareness): a bad/absent token degrades to its no-op default (0.0), so a malformed P41 tail can never
    // activate a filter on a default clip. Applied on OUTB AFTER the P39 selective color, BEFORE the look.
    let sol_thr: f32 = f[99].parse().unwrap_or(0.0);
    let temp: f32 = f[100].parse().unwrap_or(0.0);
    // P45 VIDEO FADE factor (f[101], INSERTED between temp and the out path). 1.0 = no fade
    // (byte-identical). TOLERANT: a bad/absent token degrades to 1.0 so a malformed tail can't darken.
    let fade: f32 = f[101].parse().unwrap_or(1.0);
    // The out path stays LAST (now f[102], shifted by the 4 P23 fields + the 7 P34 fields + the 1 P37
    // ck_spill field + the 3 P38 distortion fields + the 3 P39 selective-color fields + the 2 P41 filter
    // fields + the 1 P45 fade field). It is a Genesis-chosen /tmp path (no whitespace) → dec_path is
    // identity here, applied for symmetry with the encoded emit side.
    let out_path = dec_path(f[102]);

    // Decode base @ base_frame (cached decoder per path), upload to slot 0. A "-" base is an
    // explicit timeline gap (finding #5): fill slot 0 with black, matching the ENC path and
    // MojoMedia's black-gap behavior, rather than failing the frame. A `RAW:<path>` base is a P5
    // rasterized TITLE layer (a raw GVW*GVH*4 RGBA8 file): read it straight into the slot, SKIPPING
    // decode (see `upload_slot`). A raw-read failure uploads black (the title just doesn't show).
    if base_path == "-" {
        gpu.upload(0, &vec![0u8; ffi::GVW * ffi::GVH * 4]);
    } else {
        upload_slot(gpu, decoders, 0, &base_path, base_frame);
    }

    // Decode over @ over_frame (if any), upload to slot 1. A `RAW:<path>` overlay is a P5 rasterized
    // TITLE layer uploaded directly (skip decode). A failed/missing over just disables the composite
    // (op forced to 0) rather than failing the whole frame.
    let mut eff_op = op;
    if over_path != "-" && op > 0.0 {
        if !upload_slot(gpu, decoders, 1, &over_path, over_frame) {
            eff_op = 0.0;
        }
    } else {
        eff_op = 0.0;
    }

    // Resolve the per-boundary TRANSITION (Wave 8): decode the incoming partner into slot 2 when
    // active (kind 0..7 + a real path), else -1 (track1 copies the base). Same side-effecting helper
    // the ENC path uses, so the preview and the export animate the transition identically.
    let eff_tt = resolve_trans(gpu, decoders, trans_kind, &trans_path, trans_frame);

    // Resolve the per-clip LOOK (load + upload the .cube for LUT3D, cached; VHS needs no LUT; a
    // missing/failed LUT degrades to no look). Then run the OpenCL pipeline (transition or base-copy
    // first; PiP over; grade; LOOK). `fin` tells us which buffer the frame ended in (OUTB / LOOKB).
    let (lk, la, ln) = resolve_look(
        gpu,
        lut_cache,
        last_uploaded_lut,
        look_kind,
        look_amt,
        &lut_path,
    );
    // P4: the chroma key only matters when there IS an active overlay (it keys the OVER buffer). If
    // the overlay was disabled (no over clip / failed decode → eff_op==0), force ck_on=0 so we never
    // key a stale/irrelevant slot-1 buffer — identical output either way (pip ignores over at op=0).
    let eff_ck_on = if eff_op > 0.0 { ck_on } else { 0 };
    let (out, fin) = gpu.compose_trans(
        eff_tt,
        trans_prog,
        trans_param,
        eff_op,
        over_blend,
        px,
        py,
        pw,
        ph,
        cbright,
        ccontrast,
        csat,
        bright,
        contrast,
        sat,
        lk,
        la,
        ln,
        lift_r,
        lift_g,
        lift_b,
        gamma_r,
        gamma_g,
        gamma_b,
        gain_r,
        gain_g,
        gain_b,
        rot,
        scale,
        blur,
        eff_ck_on,
        ck_r,
        ck_g,
        ck_b,
        ck_sim,
        ck_smooth,
        ck_spill,
        curve,
        vig,
        sharp,
        flip,
        fx,
        hue,
        sat_hsl,
        light,
        inb,
        inw,
        gam,
        mosaic,
        gmap_amt,
        glo_r,
        glo_g,
        glo_b,
        ghi_r,
        ghi_g,
        ghi_b,
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
    );
    // Record the final buffer so a following SCOPE reads the POST-LOOK frame the UI is showing.
    *last_final_is_look = fin;

    if std::fs::write(&out_path, &out).is_err() {
        eprintln!("[gcompose] write failed: {out_path}");
        return None;
    }
    Some(out_path)
}

/// Decode `path` @ `frame` to GVW×GVH RGBA8, reusing (or opening + caching) the decoder.
/// Returns None if the file can't be opened or the frame can't be decoded.
fn decode_cached(
    decoders: &mut HashMap<String, ffi::Decoder>,
    path: &str,
    frame: i32,
) -> Option<Vec<u8>> {
    if !decoders.contains_key(path) {
        let d = ffi::Decoder::open(path)?;
        decoders.insert(path.to_string(), d);
    }
    let dec = decoders.get_mut(path)?;
    let f = if frame < 0 { 0 } else { frame }; // never seek a negative frame (C-side guard).
    dec.decode_rgba(f, ffi::GVW, ffi::GVH)
}

/// Upload a frame to GPU `slot` from a wire path, handling the P5 `RAW:` sentinel (Slice A).
///
/// Two shapes:
///   - `RAW:<file>` : a RASTERIZED layer (e.g. a P5 title). `<file>` is a RAW `GVW*GVH*4` RGBA8 dump
///     (no container, no codec): `std::fs::read` it and upload DIRECTLY — NO decode. On a read error
///     OR a length mismatch (truncated/corrupt/size-changed file), upload a fully TRANSPARENT/black
///     `GVW*GVH*4` buffer and return false, so the caller can disable the composite (an over title
///     that failed to load simply doesn't show; a base falls back to black) rather than fail.
///   - anything else : a normal MEDIA path → `decode_cached(path, frame)` and upload, or upload black
///     + return false on a decode failure (matching the pre-P5 fallback behavior).
///
/// Returns true when a USABLE (non-fallback) frame was uploaded. Callers that need a base frame
/// ignore the bool (black is uploaded either way); the OVER caller uses false to drop the composite.
///
/// IDENTITY: a project with no titles never sends a `RAW:` path, so this routes every frame through
/// `decode_cached` exactly as before — byte-identical to the pre-P5 engine.
fn upload_slot(
    gpu: &ffi::Gpu,
    decoders: &mut HashMap<String, ffi::Decoder>,
    slot: i32,
    path: &str,
    frame: i32,
) -> bool {
    const FRAME_BYTES: usize = ffi::GVW * ffi::GVH * 4;
    if let Some(raw_path) = path.strip_prefix("RAW:") {
        match std::fs::read(raw_path) {
            Ok(bytes) if bytes.len() == FRAME_BYTES => {
                gpu.upload(slot, &bytes);
                true
            }
            Ok(bytes) => {
                eprintln!(
                    "[gcompose] RAW layer wrong size ({} bytes, expected {FRAME_BYTES}): {raw_path}",
                    bytes.len()
                );
                gpu.upload(slot, &vec![0u8; FRAME_BYTES]); // transparent/black fallback.
                false
            }
            Err(_) => {
                eprintln!("[gcompose] RAW layer read failed: {raw_path}");
                gpu.upload(slot, &vec![0u8; FRAME_BYTES]);
                false
            }
        }
    } else {
        match decode_cached(decoders, path, frame) {
            Some(rgba) => {
                gpu.upload(slot, &rgba);
                true
            }
            None => {
                gpu.upload(slot, &vec![0u8; FRAME_BYTES]);
                false
            }
        }
    }
}

/// One-shot: decode base+over, run the OpenCL composite (demo PiP inset + grade). None if no GPU.
fn compose(base: &str, over: &str) -> Option<Vec<u8>> {
    let gpu = ffi::Gpu::init()?;
    let mut b = ffi::Decoder::open(base)?;
    let base_rgba = b.decode_rgba(60, ffi::GVW, ffi::GVH)?;
    gpu.upload(0, &base_rgba);

    let mut has_over = false;
    if over != "-" {
        if let Some(mut o) = ffi::Decoder::open(over) {
            if let Some(ov) = o.decode_rgba(0, ffi::GVW, ffi::GVH) {
                gpu.upload(1, &ov);
                has_over = true;
            }
        }
    }
    let op = if has_over { 1.0 } else { 0.0 };
    // One-shot demo: no look (kind 0). compose now returns (rgba, final_is_look); we only want the
    // pixels here, so discard the buffer-selection flag.
    let (buf, _fin) = gpu.compose(op, 0.6, 0.1, 0.3, 0.3, 0.08, 1.1, 1.25, 0, 0.0, 0);
    Some(buf)
}

/// CPU/FFmpeg-only fallback: just the decoded base frame.
fn decode_only(base: &str) -> Option<Vec<u8>> {
    let mut b = ffi::Decoder::open(base)?;
    b.decode_rgba(60, ffi::GVW, ffi::GVH)
}
