// fpx_decode.c — thin libav C shim for faithful, frame-accurate video decode to RGBA8,
// callable from Mojo via external_call (mirrors how serenitymojo uses cshim libraries).
// Mirrors FramePFX's decode/scale path (FFmpegMedia: av_read_frame -> send/receive ->
// sws_scale SWS_BICUBIC), output AV_PIX_FMT_RGBA (MojoGUI draw_image wants RGBA8).
//
// Build .so:  cc -O2 -fPIC -shared fpx_decode.c $(pkg-config --cflags --libs \
//                 libavformat libavcodec libswscale libavutil) -o libfpxdecode.so
// Build test: cc -DFPX_TEST -O2 fpx_decode.c $(pkg-config --cflags --libs \
//                 libavformat libavcodec libswscale libavutil) -o fpxtest
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    AVFormatContext* fmt;
    AVCodecContext*  dec;
    struct SwsContext* sws;
    struct SwsContext* sws2;     // scaled-output context (lazy, for fpx_decode_frame_scaled)
    int sw, sh;                  // current scaled target size
    struct SwsContext* sws_lb;   // letterbox context (aspect-preserving, lazy)
    int lb_dw, lb_dh;            // current letterbox inner (fit) size
    int vindex;
    int width, height;
    AVRational time_base;
    AVRational frame_rate;
    AVPacket* pkt;
    AVFrame*  frame;
    // Sequential-decode position tracking (fpx_decode_frame_letterbox): pts of the last frame
    // returned, and whether that position is known. Lets a forward in-window request decode forward
    // from here instead of seeking to a keyframe + redecoding (the per-frame playback/render cost).
    // calloc-zeroed at open => have_pos=0 (unknown) until the first decode. ANY seek (here or in the
    // sibling _scaled/_rgba decoders, which share the handle) clears have_pos so we never decode
    // forward from a stale position.
    int64_t cur_pts;
    int have_pos;
} FpxCtx;

void* fpx_open(const char* path) {
    FpxCtx* c = (FpxCtx*)calloc(1, sizeof(FpxCtx));
    if (!c) return NULL;
    if (avformat_open_input(&c->fmt, path, NULL, NULL) < 0) { free(c); return NULL; }
    if (avformat_find_stream_info(c->fmt, NULL) < 0) { avformat_close_input(&c->fmt); free(c); return NULL; }
    const AVCodec* codec = NULL;
    int idx = av_find_best_stream(c->fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (idx < 0 || !codec) { avformat_close_input(&c->fmt); free(c); return NULL; }
    c->vindex = idx;
    AVStream* st = c->fmt->streams[idx];
    c->dec = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(c->dec, st->codecpar);
    if (avcodec_open2(c->dec, codec, NULL) < 0) { avcodec_free_context(&c->dec); avformat_close_input(&c->fmt); free(c); return NULL; }
    c->width = c->dec->width;
    c->height = c->dec->height;
    c->time_base = st->time_base;
    c->frame_rate = av_guess_frame_rate(c->fmt, st, NULL);
    // SWS_BICUBIC like FramePFX; at native size this is pure pixel-format conversion (YUV->RGBA).
    c->sws = sws_getContext(c->width, c->height, c->dec->pix_fmt,
                            c->width, c->height, AV_PIX_FMT_RGBA,
                            SWS_BICUBIC, NULL, NULL, NULL);
    c->pkt = av_packet_alloc();
    c->frame = av_frame_alloc();
    return c;
}

int fpx_width(void* h)  { return h ? ((FpxCtx*)h)->width  : 0; }
int fpx_height(void* h) { return h ? ((FpxCtx*)h)->height : 0; }

// Frame-accurate seek + decode of frame `frame_index` -> RGBA8 into out (width*height*4 bytes).
// Returns 0 on success, negative on error.
int fpx_decode_frame_rgba(void* h, int frame_index, unsigned char* out, int out_size) {
    if (!h || !out) return -1;
    FpxCtx* c = (FpxCtx*)h;
    if (out_size < c->width * c->height * 4) return -2;

    // target PTS in stream time_base = frame_index * (1/frame_rate) / time_base
    int64_t target = av_rescale_q(frame_index, av_inv_q(c->frame_rate), c->time_base);
    // seek to keyframe at or before target, then decode forward (scrubbing path)
    if (av_seek_frame(c->fmt, c->vindex, target, AVSEEK_FLAG_BACKWARD) < 0) return -3;
    avcodec_flush_buffers(c->dec);
    c->have_pos = 0; // sibling seek invalidates the _letterbox forward-decode position

    int got = -1;
    while (got != 0 && av_read_frame(c->fmt, c->pkt) >= 0) {
        if (c->pkt->stream_index == c->vindex) {
            if (avcodec_send_packet(c->dec, c->pkt) == 0) {
                while (avcodec_receive_frame(c->dec, c->frame) == 0) {
                    int64_t pts = c->frame->best_effort_timestamp;
                    if (pts == AV_NOPTS_VALUE || pts >= target) {
                        uint8_t* dst[4] = { out, NULL, NULL, NULL };
                        int dstln[4] = { c->width * 4, 0, 0, 0 };
                        sws_scale(c->sws, (const uint8_t* const*)c->frame->data, c->frame->linesize,
                                  0, c->height, dst, dstln);
                        got = 0;
                        break;
                    }
                }
            }
        }
        av_packet_unref(c->pkt);
        if (got == 0) break;
    }
    // EOF with the target still buffered: codecs with reorder delay (H.264 B-frames) hold the final
    // frames of a stream until a flush, so the loop above (which stops at av_read_frame EOF) never
    // drains them and the clip's last ~GOP decodes BLACK. Send a NULL packet to flush, then drain the
    // remaining frames and take the first at/after target. Mirrors ffmpeg's end-of-stream drain.
    if (got != 0) {
        avcodec_send_packet(c->dec, NULL);
        while (avcodec_receive_frame(c->dec, c->frame) == 0) {
            int64_t pts = c->frame->best_effort_timestamp;
            if (pts == AV_NOPTS_VALUE || pts >= target) {
                uint8_t* dst[4] = { out, NULL, NULL, NULL };
                int dstln[4] = { c->width * 4, 0, 0, 0 };
                sws_scale(c->sws, (const uint8_t* const*)c->frame->data, c->frame->linesize,
                          0, c->height, dst, dstln);
                got = 0;
                break;
            }
        }
    }
    return got;
}

// Like fpx_decode_frame_rgba but SCALES the frame to ow x oh RGBA8 (editor preview /
// fixed processing resolution; lets clips of any native size load into one buffer).
int fpx_decode_frame_scaled(void* h, int frame_index, unsigned char* out, int ow, int oh) {
    if (!h || !out || ow <= 0 || oh <= 0) return -1;
    FpxCtx* c = (FpxCtx*)h;
    if (!c->sws2 || c->sw != ow || c->sh != oh) {
        if (c->sws2) sws_freeContext(c->sws2);
        c->sws2 = sws_getContext(c->width, c->height, c->dec->pix_fmt,
                                 ow, oh, AV_PIX_FMT_RGBA, SWS_BICUBIC, NULL, NULL, NULL);
        c->sw = ow; c->sh = oh;
    }
    if (!c->sws2) return -4;
    int64_t target = av_rescale_q(frame_index, av_inv_q(c->frame_rate), c->time_base);
    if (av_seek_frame(c->fmt, c->vindex, target, AVSEEK_FLAG_BACKWARD) < 0) return -3;
    avcodec_flush_buffers(c->dec);
    c->have_pos = 0; // sibling seek invalidates the _letterbox forward-decode position
    int got = -1;
    while (got != 0 && av_read_frame(c->fmt, c->pkt) >= 0) {
        if (c->pkt->stream_index == c->vindex) {
            if (avcodec_send_packet(c->dec, c->pkt) == 0) {
                while (avcodec_receive_frame(c->dec, c->frame) == 0) {
                    int64_t pts = c->frame->best_effort_timestamp;
                    if (pts == AV_NOPTS_VALUE || pts >= target) {
                        uint8_t* dst[4] = { out, NULL, NULL, NULL };
                        int dstln[4] = { ow * 4, 0, 0, 0 };
                        sws_scale(c->sws2, (const uint8_t* const*)c->frame->data, c->frame->linesize,
                                  0, c->height, dst, dstln);
                        got = 0; break;
                    }
                }
            }
        }
        av_packet_unref(c->pkt);
        if (got == 0) break;
    }
    // EOF flush-drain (see fpx_decode_frame_rgba): drain the decoder's buffered tail frames so the
    // last ~GOP of a clip decodes instead of returning black.
    if (got != 0) {
        avcodec_send_packet(c->dec, NULL);
        while (avcodec_receive_frame(c->dec, c->frame) == 0) {
            int64_t pts = c->frame->best_effort_timestamp;
            if (pts == AV_NOPTS_VALUE || pts >= target) {
                uint8_t* dst[4] = { out, NULL, NULL, NULL };
                int dstln[4] = { ow * 4, 0, 0, 0 };
                sws_scale(c->sws2, (const uint8_t* const*)c->frame->data, c->frame->linesize,
                          0, c->height, dst, dstln);
                got = 0; break;
            }
        }
    }
    return got;
}

// Decode the NEXT frame in sequence (no seek) scaled to ow x oh RGBA8 — for playback.
// Returns 0 on success, negative at end-of-stream (caller should seek back to 0).
int fpx_decode_next_scaled(void* h, unsigned char* out, int ow, int oh) {
    if (!h || !out || ow <= 0 || oh <= 0) return -1;
    FpxCtx* c = (FpxCtx*)h;
    if (!c->sws2 || c->sw != ow || c->sh != oh) {
        if (c->sws2) sws_freeContext(c->sws2);
        c->sws2 = sws_getContext(c->width, c->height, c->dec->pix_fmt,
                                 ow, oh, AV_PIX_FMT_RGBA, SWS_BICUBIC, NULL, NULL, NULL);
        c->sw = ow; c->sh = oh;
    }
    if (!c->sws2) return -4;
    int got = -1;
    while (got != 0 && av_read_frame(c->fmt, c->pkt) >= 0) {
        if (c->pkt->stream_index == c->vindex) {
            if (avcodec_send_packet(c->dec, c->pkt) == 0) {
                while (avcodec_receive_frame(c->dec, c->frame) == 0) {
                    uint8_t* dst[4] = { out, NULL, NULL, NULL };
                    int dstln[4] = { ow * 4, 0, 0, 0 };
                    sws_scale(c->sws2, (const uint8_t* const*)c->frame->data, c->frame->linesize,
                              0, c->height, dst, dstln);
                    got = 0; break;
                }
            }
        }
        av_packet_unref(c->pkt);
        if (got == 0) break;
    }
    return got;
}

// Scale the current decoded c->frame into `out` (ow*oh RGBA8) PRESERVING ASPECT: the source
// is fit (by width or height) into ow x oh and centered, with opaque-black bars filling the
// rest. This is how a portrait/mobile clip lands in the fixed editor buffer without stretching.
static int fpx_lb_blit(FpxCtx* c, unsigned char* out, int ow, int oh) {
    int dw, dh;
    if ((long)c->width * oh >= (long)ow * c->height) {        // source wider than target -> fit width
        dw = ow; dh = (int)((long)ow * c->height / c->width);
    } else {                                                  // taller -> fit height
        dh = oh; dw = (int)((long)oh * c->width / c->height);
    }
    if (dw < 2) dw = 2;
    if (dh < 2) dh = 2;
    dw &= ~1; dh &= ~1;
    int ox = (ow - dw) / 2, oy = (oh - dh) / 2;
    if (!c->sws_lb || c->lb_dw != dw || c->lb_dh != dh) {
        if (c->sws_lb) sws_freeContext(c->sws_lb);
        c->sws_lb = sws_getContext(c->width, c->height, c->dec->pix_fmt,
                                   dw, dh, AV_PIX_FMT_RGBA, SWS_BICUBIC, NULL, NULL, NULL);
        c->lb_dw = dw; c->lb_dh = dh;
    }
    if (!c->sws_lb) return -4;
    for (long i = 0; i < (long)ow * oh; i++) {                // opaque-black background
        out[i*4] = 0; out[i*4+1] = 0; out[i*4+2] = 0; out[i*4+3] = 255;
    }
    uint8_t* dst[4] = { out + ((long)oy * ow + ox) * 4, NULL, NULL, NULL };
    int dstln[4] = { ow * 4, 0, 0, 0 };
    sws_scale(c->sws_lb, (const uint8_t* const*)c->frame->data, c->frame->linesize,
              0, c->height, dst, dstln);
    return 0;
}

// Like fpx_decode_frame_scaled but ASPECT-PRESERVING (letterbox/pillarbox into ow x oh).
// Forward-decode window (frames): a request this many frames AHEAD of the current decoder position
// (or fewer) decodes forward WITHOUT seeking — the sequential playback/render case. A larger forward
// jump (or any backward jump) seeks to the nearest keyframe instead, which is cheaper than decoding
// across multiple GOPs. Correctness is independent of this value (the pts>=target check below always
// returns the exact frame); it only trades a seek for a forward decode. ~1 GOP is the sweet spot.
#define FPX_SEQ_FWD_WINDOW 300

int fpx_decode_frame_letterbox(void* h, int frame_index, unsigned char* out, int ow, int oh) {
    if (!h || !out || ow <= 0 || oh <= 0) return -1;
    FpxCtx* c = (FpxCtx*)h;
    int64_t target = av_rescale_q(frame_index, av_inv_q(c->frame_rate), c->time_base);

    // Decide seek vs forward-decode. We can decode FORWARD from the current position (no seek) only
    // when the position is known AND the target is strictly ahead of it AND within FPX_SEQ_FWD_WINDOW
    // frames. Everything else (first decode, backward, far-forward) seeks to a keyframe. This is the
    // whole optimization: sequential access no longer re-seeks + re-decodes from a keyframe per frame
    // (which was O(GOP^2) over a GOP — ~345ms/frame at 1440p — vs O(GOP) decoding forward once).
    int64_t window_pts = av_rescale_q(FPX_SEQ_FWD_WINDOW, av_inv_q(c->frame_rate), c->time_base);
    int forward = c->have_pos && target > c->cur_pts && (target - c->cur_pts) <= window_pts;
    if (!forward) {
        if (av_seek_frame(c->fmt, c->vindex, target, AVSEEK_FLAG_BACKWARD) < 0) return -3;
        avcodec_flush_buffers(c->dec);
        c->have_pos = 0; // position reset by the flush; the loop re-establishes it on the next frame
    }

    // Canonical DRAIN-FIRST decode loop: pull any already-buffered frame first (this is what makes a
    // no-seek forward decode CORRECT — after a prior call broke out mid-GOP, the decoder may hold
    // reordered frames that must be received before feeding new packets), feed a packet only when the
    // decoder asks (EAGAIN), and at end-of-packets flush (send NULL) to drain the tail GOP so a
    // clip's last frames decode instead of returning black (preserves the b223433 EOF-drain fix).
    // Return the FIRST frame with pts >= target (frame-accurate; identical selection to before).
    int eof_sent = 0;
    for (;;) {
        int r = avcodec_receive_frame(c->dec, c->frame);
        if (r == 0) {
            int64_t pts = c->frame->best_effort_timestamp;
            if (pts != AV_NOPTS_VALUE) { c->cur_pts = pts; c->have_pos = 1; }
            if (pts == AV_NOPTS_VALUE || pts >= target) {
                if (pts == AV_NOPTS_VALUE) c->have_pos = 0; // can't track an unknown-pts position
                if (fpx_lb_blit(c, out, ow, oh) < 0) { c->have_pos = 0; return -4; }
                return 0;
            }
            continue; // frame is before target: drop it and keep receiving
        }
        if (r == AVERROR_EOF) { c->have_pos = 0; return -3; } // drained, target past end
        // r == AVERROR(EAGAIN): the decoder needs more input.
        if (eof_sent) { c->have_pos = 0; return -3; } // already flushing and still nothing → done
        int sent = 0;
        while (av_read_frame(c->fmt, c->pkt) >= 0) {
            if (c->pkt->stream_index == c->vindex) {
                avcodec_send_packet(c->dec, c->pkt);
                av_packet_unref(c->pkt);
                sent = 1;
                break;
            }
            av_packet_unref(c->pkt);
        }
        if (!sent) {
            avcodec_send_packet(c->dec, NULL); // end of packets → enter draining mode
            eof_sent = 1;
        }
    }
}

// Aspect-preserving sequential decode (no seek) — for playback.
int fpx_decode_next_letterbox(void* h, unsigned char* out, int ow, int oh) {
    if (!h || !out || ow <= 0 || oh <= 0) return -1;
    FpxCtx* c = (FpxCtx*)h;
    int got = -1;
    while (got != 0 && av_read_frame(c->fmt, c->pkt) >= 0) {
        if (c->pkt->stream_index == c->vindex) {
            if (avcodec_send_packet(c->dec, c->pkt) == 0) {
                while (avcodec_receive_frame(c->dec, c->frame) == 0) {
                    if (fpx_lb_blit(c, out, ow, oh) < 0) { av_packet_unref(c->pkt); return -4; }
                    got = 0; break;
                }
            }
        }
        av_packet_unref(c->pkt);
        if (got == 0) break;
    }
    return got;
}

// Decode the NEXT frame in sequence (no seek) at NATIVE resolution -> RGBA8 (w*h*4).
// Fast sequential export after one fpx_decode_frame_rgba seek. Negative at EOS.
int fpx_decode_next_rgba(void* h, unsigned char* out, int out_size) {
    if (!h || !out) return -1;
    FpxCtx* c = (FpxCtx*)h;
    if (out_size < c->width * c->height * 4) return -2;
    int got = -1;
    while (got != 0 && av_read_frame(c->fmt, c->pkt) >= 0) {
        if (c->pkt->stream_index == c->vindex) {
            if (avcodec_send_packet(c->dec, c->pkt) == 0) {
                while (avcodec_receive_frame(c->dec, c->frame) == 0) {
                    uint8_t* dst[4] = { out, NULL, NULL, NULL };
                    int dstln[4] = { c->width * 4, 0, 0, 0 };
                    sws_scale(c->sws, (const uint8_t* const*)c->frame->data, c->frame->linesize,
                              0, c->height, dst, dstln);
                    got = 0; break;
                }
            }
        }
        av_packet_unref(c->pkt);
        if (got == 0) break;
    }
    return got;
}

// Video stream frame rate as num/den (e.g. 30000/1001 or 30/1).
int fpx_fps_num(void* h) { return h ? ((FpxCtx*)h)->frame_rate.num : 0; }
int fpx_fps_den(void* h) { return h ? ((FpxCtx*)h)->frame_rate.den : 1; }

// Total frame count of the video stream (nb_frames, or estimated from duration*fps).
int fpx_nframes(void* h) {
    if (!h) return 0;
    FpxCtx* c = (FpxCtx*)h;
    AVStream* st = c->fmt->streams[c->vindex];
    if (st->nb_frames > 0) return (int)st->nb_frames;
    if (c->fmt->duration > 0 && c->frame_rate.num > 0) {
        double secs = (double)c->fmt->duration / AV_TIME_BASE;
        return (int)(secs * av_q2d(c->frame_rate));
    }
    return 0;
}

void fpx_close(void* h) {
    if (!h) return;
    FpxCtx* c = (FpxCtx*)h;
    if (c->sws2) sws_freeContext(c->sws2);
    if (c->sws) sws_freeContext(c->sws);
    if (c->sws_lb) sws_freeContext(c->sws_lb);   // letterbox scaler (fpx_lb_blit) — was leaked on close
    if (c->frame) av_frame_free(&c->frame);
    if (c->pkt) av_packet_free(&c->pkt);
    if (c->dec) avcodec_free_context(&c->dec);
    if (c->fmt) avformat_close_input(&c->fmt);
    free(c);
}

#ifdef FPX_TEST
#include <stdio.h>
// usage: fpxtest <file> <frame_index> <out.raw>
int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s file frame out.raw\n", argv[0]); return 2; }
    void* h = fpx_open(argv[1]);
    if (!h) { fprintf(stderr, "open failed\n"); return 1; }
    int w = fpx_width(h), ht = fpx_height(h);
    int sz = w * ht * 4;
    unsigned char* buf = (unsigned char*)malloc(sz);
    int r = fpx_decode_frame_rgba(h, atoi(argv[2]), buf, sz);
    if (r != 0) { fprintf(stderr, "decode failed %d\n", r); return 1; }
    FILE* f = fopen(argv[3], "wb"); fwrite(buf, 1, sz, f); fclose(f);
    fprintf(stderr, "ok %dx%d -> %s (%d bytes)\n", w, ht, argv[3], sz);
    free(buf); fpx_close(h);
    return 0;
}
#endif
