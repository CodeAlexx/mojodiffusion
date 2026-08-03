# INDEPENDENT derivation of the keyframe (I2VA / FL2VA / L2VA) packed layout.
#
# numpy ONLY — no torch, no diffusers, no GPU, and deliberately NOT an import of
# the reference module: the formulas below are transcribed from
# minimax_h3_ref/diffusers/packing.py by hand so that the Mojo port and this
# derivation are two independent readings of the same source. Emits Mojo
# literals for the probe.
import numpy as np

ROPE_FRAME_RESCALE = 5.0 / 3.0
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
ROPE_SPATIAL_SCALE = 32
AUDIO_CHANNELS = 2
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2


def spatial_grid(dim, patch, sqrt_area):                      # packing.py:331
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE


def temporal_grid(n, origin):                                  # packing.py:344
    spans = np.array([ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % 5] for i in range(n)])
    return origin + np.concatenate([[0.0], np.cumsum(spans[:-1])])


def temporal_span(n):                                          # packing.py:356
    spans = np.ones(n, dtype=np.float64) * ROPE_FRAME_RESCALE
    for i in range(len(ROPE_FRAMES_PER_LATENT)):
        spans[i::len(ROPE_FRAMES_PER_LATENT)] *= ROPE_FRAMES_PER_LATENT[i]
    return float(spans.sum())


def build(text_tags, n_lat, lat_h, lat_w, n_audio, patch_h, patch_w, anchors):
    rows_per_frame = (lat_h // patch_h) * (lat_w // patch_w)
    L = len(text_tags)
    C = len(anchors) * rows_per_frame
    A = n_audio * AUDIO_CHANNELS
    V = n_lat * rows_per_frame
    S = L + C + A + V
    cond0, aud0, vid0 = L, L + C, L + C + A

    pos = np.zeros((S, 3), dtype=np.float64)
    pos[:L, 0] = np.arange(L, dtype=np.float64)

    sqrt_area = np.sqrt(lat_h * lat_w)
    hg = spatial_grid(lat_h, patch_h, sqrt_area)
    wg = spatial_grid(lat_w, patch_w, sqrt_area)
    frame = np.stack([g.reshape(-1) for g in np.meshgrid(hg, wg, indexing="ij")], -1)

    for i, a in enumerate(anchors):
        if a == "first":
            t = float(L)
        elif a == "last":
            t = float(L) + temporal_span(n_lat) - ROPE_FRAME_RESCALE
        else:
            raise ValueError(a)
        sl = slice(cond0 + i * rows_per_frame, cond0 + (i + 1) * rows_per_frame)
        pos[sl, 0] = t
        pos[sl, 1:] = frame

    at = float(L) + np.arange(n_audio, dtype=np.float64)
    pos[aud0:vid0, 0] = np.tile(at, AUDIO_CHANNELS)
    pos[aud0:vid0, 2] = np.concatenate([
        np.full(n_audio, float(wg[0])), np.full(A - n_audio, float(wg[-1]))])

    vp = np.empty((n_lat, rows_per_frame, 3), dtype=np.float64)
    vp[:, :, 0] = temporal_grid(n_lat, float(L))[:, None]
    vp[:, :, 1:] = frame[None]
    pos[vid0:] = vp.reshape(-1, 3)

    video_idx = np.concatenate([np.arange(cond0, aud0), np.arange(vid0, S)]).astype(int)
    audio_idx = np.arange(aud0, vid0).astype(int)
    tags = np.empty(S, dtype=int)
    tags[:L] = text_tags
    tags[audio_idx] = AUDIO_TAG
    tags[video_idx] = VIDEO_TAG
    return dict(S=S, pos=pos, tags=tags, video_idx=video_idx, audio_idx=audio_idx,
                C=C, cond0=cond0, aud0=aud0, vid0=vid0, rows_per_frame=rows_per_frame)


def row_timesteps(S, video_idx, audio_idx, C, vt, at_):        # packing.py:469 + before_denoise.py:413-417
    rt = np.full(S, vt, dtype=np.float32)
    rt[video_idx[:C]] = np.float32(max(vt, 0.999))
    rt[audio_idx] = np.float32(at_)
    vals, inv = np.unique(rt, return_inverse=True)
    return vals, inv


def emit(name, values, fmt="{!r}"):
    print(f"        # {name}")
    print("        " + ", ".join(fmt.format(v) for v in values))


TEXT_TAGS = [1, 0, 0, 1]        # a vision block inside the text run: tagged VIDEO
N_LAT, LAT_H, LAT_W, N_AUDIO, PH, PW = 2, 4, 4, 3, 2, 2

for label, anchors in (("I2VA", ["first"]), ("L2VA", ["last"]), ("FL2VA", ["first", "last"])):
    b = build(TEXT_TAGS, N_LAT, LAT_H, LAT_W, N_AUDIO, PH, PW, anchors)
    print(f"# ==== {label}: anchors={anchors} ====")
    print(f"#   S={b['S']} rows_per_frame={b['rows_per_frame']} cond_rows={b['C']} "
          f"cond0={b['cond0']} aud0={b['aud0']} vid0={b['vid0']}")
    flat = b["pos"].reshape(-1)
    print(f"#   position_ids ({len(flat)} float64):")
    for r in range(b["S"]):
        print(f"        {b['pos'][r,0]!r}, {b['pos'][r,1]!r}, {b['pos'][r,2]!r},")
    print(f"#   token_tags: {list(b['tags'])}")
    vals, inv = row_timesteps(b["S"], b["video_idx"], b["audio_idx"], b["C"], 0.75, 0.60)
    print(f"#   ts values @ (v=0.75,a=0.60): {[float(v) for v in vals]}")
    print(f"#   ts indices: {list(int(i) for i in inv.reshape(-1))}")
    vals1, _ = row_timesteps(b["S"], b["video_idx"], b["audio_idx"], b["C"], 1.0, 0.5)
    print(f"#   ts values @ (v=1.0,a=0.5): {[float(v) for v in vals1]}  "
          f"<- cond_video = max(1.0, 0.999) = 1.0, collapses onto the video row")
    print()

# The "last" anchor's meaning, at a REAL geometry (124 frames -> 37 latent frames).
n = 37
print(f"# real geometry: {n} latent frames, span={temporal_span(n)!r}, "
      f"last_frame_t(origin 0)={temporal_grid(n,0.0)[-1]!r}, "
      f"last_anchor_t(origin 0)={temporal_span(n) - ROPE_FRAME_RESCALE!r}")
print(f"# 5/3 * 123 = {5.0/3.0*123!r}  <- the anchor sits on the LAST PIXEL FRAME (0-indexed), "
      f"not the last latent frame")
