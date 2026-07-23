// fpx_encode.c — thin libav C ENCODE/MUX shim (the engine MediaEditor's MediaEncoder uses).
// Faithful re-implementation of MediaCore/src/MediaEncoder.cpp's encode spine:
//   Open_Internal          (cpp:485-511)  avformat_alloc_output_context2 + avio_open
//   ConfigureVideoStream   (cpp:513-700)  find encoder, pick pix_fmt, new stream, sws to encoder pix_fmt
//   ConfigureAudioStream   (cpp:898-1010) find encoder, alloc ctx, swr to encoder sample_fmt, new stream
//   Start                  (cpp:163-196)  avformat_write_header
//   EncodeVideoFrame(ImMat)(cpp:288-294,1080-1200,1300-1390) ImMat RGBA->AVFrame->send/recv->mux
//   EncodeAudioSamples     (cpp:296-430,1220-1390)           interleaved f32->frame chunk->swr->send/recv->mux
//   FinishEncoding         (cpp:198-236)  flush encoders, av_write_trailer
// Callable from Mojo via external_call (mirrors fpx_decode.c / fpx_audio.c). Verified by
// encoding RGBA frames -> mp4 and re-probing/decoding with the ffmpeg/ffprobe CLI.
//
// The C++ engine buffers frames in worker-thread queues; the OBSERVABLE contract is
// identical when driven synchronously (configure -> per-frame encode in pts order -> mux
// -> finish), so this shim does the send/receive/mux inline (no threads).
//
// Build .so:  cc -O2 -fPIC -shared fpx_encode.c $(pkg-config --cflags --libs \
//                 libavformat libavcodec libswscale libswresample libavutil) -o libfpxencode.so
// Build test: cc -DFPX_TEST -O2 fpx_encode.c $(pkg-config --cflags --libs \
//                 libavformat libavcodec libswscale libswresample libavutil) -o fpxenctest
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// MILLISEC_TIMEBASE (FFUtils.cpp): {1,1000} — vmat.time_stamp is SECONDS, pts feed is ms.
static const AVRational MILLISEC_TIMEBASE = {1, 1000};

typedef struct {
    AVFormatContext* fmt;
    char url[1024];

    // ---- video ----
    const AVCodec*   venc;
    AVCodecContext*  vctx;
    AVStream*        vstm;
    struct SwsContext* vsws;     // RGBA(input) -> encoder pix_fmt
    AVFrame*         vfrm;       // reusable encoder input frame (encoder pix_fmt)
    int              vidx;       // video stream index, -1 if none
    int              vw, vh;
    enum AVPixelFormat vpixfmt;  // encoder pixel format
    int64_t          vnext_pts;  // fallback pts counter (frames)

    // ---- audio ----
    const AVCodec*   aenc;
    AVCodecContext*  actx;
    AVStream*        astm;
    SwrContext*      swr;        // input fmt -> encoder sample_fmt (NULL if same)
    AVFrame*         afrm;       // current accumulation frame (encoder sample_fmt)
    int              aidx;       // audio stream index, -1 if none
    int              ach;
    int              asr;
    enum AVSampleFormat ainfmt;  // input sample format (interleaved float by default)
    enum AVSampleFormat aencfmt; // encoder sample format
    int              aframe_samples; // encoder frame_size (samples/ch)
    int              afrm_off;   // samples already filled in afrm
    int64_t          anext_pts;  // accumulated audio pts (samples)

    AVPacket*        pkt;
    int              header_written;
} FpxEnc;

// ---- drain encoded packets from `ctx`/`stm` to the muxer ----
static int fpx__drain(FpxEnc* e, AVCodecContext* ctx, AVStream* stm) {
    int ret = 0;
    while (1) {
        ret = avcodec_receive_packet(ctx, e->pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { return 0; }
        if (ret < 0) return ret;
        e->pkt->stream_index = stm->index;
        // rescale codec time_base -> stream time_base (cpp:1311,1347)
        av_packet_rescale_ts(e->pkt, ctx->time_base, stm->time_base);
        ret = av_interleaved_write_frame(e->fmt, e->pkt);   // cpp:1373
        av_packet_unref(e->pkt);
        if (ret < 0) return ret;
    }
}

// Open output container + io (cpp:485-511). Returns handle or NULL.
void* fpx_enc_open(const char* url) {
    FpxEnc* e = (FpxEnc*)calloc(1, sizeof(FpxEnc));
    if (!e) return NULL;
    e->vidx = -1; e->aidx = -1;
    e->vpixfmt = AV_PIX_FMT_NONE;
    e->ainfmt = AV_SAMPLE_FMT_FLT;     // Mojo passes interleaved float (matches AudioMat)
    e->aencfmt = AV_SAMPLE_FMT_NONE;
    snprintf(e->url, sizeof(e->url), "%s", url);

    if (avformat_alloc_output_context2(&e->fmt, NULL, NULL, url) < 0 || !e->fmt) {
        free(e); return NULL;
    }
    // Open the output IO unless the muxer manages files itself (cpp:496-509).
    if ((e->fmt->oformat->flags & AVFMT_NOFILE) == 0) {
        if (avio_open(&e->fmt->pb, url, AVIO_FLAG_WRITE) < 0) {
            avformat_free_context(e->fmt);
            free(e);
            return NULL;
        }
    }
    e->pkt = av_packet_alloc();
    return e;
}

// ConfigureVideoStream (cpp:513-700). codecName e.g. "libx264"/"mpeg4"/"libx265".
// in_w/in_h = source RGBA dims; width/height = encoded dims. fps = num/den.
// Returns video stream index (>=0) or negative on error.
int fpx_enc_config_video(void* h, const char* codecName,
                         int in_w, int in_h, int width, int height,
                         int fps_num, int fps_den, long long bit_rate,
                         int gop, const char* preset) {
    if (!h) return -1;
    FpxEnc* e = (FpxEnc*)h;

    const AVCodec* venc = avcodec_find_encoder_by_name(codecName);
    if (!venc || venc->type != AVMEDIA_TYPE_VIDEO) return -2;

    // pick first non-hwaccel pixel format the encoder supports (cpp:711-729)
    enum AVPixelFormat pixfmt = AV_PIX_FMT_NONE;
    if (venc->pix_fmts) {
        for (int i = 0; venc->pix_fmts[i] != AV_PIX_FMT_NONE; i++) {
            const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(venc->pix_fmts[i]);
            if (!d) continue;
            if (d->flags & AV_PIX_FMT_FLAG_HWACCEL) continue;
            pixfmt = venc->pix_fmts[i];
            break;
        }
    }
    if (pixfmt == AV_PIX_FMT_NONE) pixfmt = AV_PIX_FMT_YUV420P;

    AVCodecContext* vctx = avcodec_alloc_context3(venc);
    if (!vctx) return -3;
    vctx->pix_fmt = pixfmt;                 // cpp:850
    vctx->width = width;                    // cpp:851
    vctx->height = height;                  // cpp:852
    vctx->time_base = (AVRational){ fps_den, fps_num };  // cpp:853
    vctx->framerate = (AVRational){ fps_num, fps_den };  // cpp:854
    vctx->bit_rate = bit_rate;              // cpp:855
    vctx->sample_aspect_ratio = (AVRational){ 1, 1 };    // cpp:856
    if (e->fmt->oformat->flags & AVFMT_GLOBALHEADER)     // cpp:870-871 (global-header path)
        vctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    // P25 export depth: GOP / keyframe interval (frames) + x264/x265 encoder PRESET. Both are read
    // by the encoder at avcodec_open2, so set them on vctx BEFORE the open below. gop<=0 leaves the
    // encoder's default gop_size untouched and an empty preset sets nothing — so the default export
    // (gop0/preset"") is byte-identical to pre-P25. A codec without a "preset" private option
    // returns <0 from av_opt_set harmlessly (e.g. mpeg4).
    if (gop > 0) vctx->gop_size = gop;
    if (preset && preset[0]) av_opt_set(vctx->priv_data, "preset", preset, 0);

    if (avcodec_open2(vctx, venc, NULL) < 0) { avcodec_free_context(&vctx); return -4; }

    AVStream* vstm = avformat_new_stream(e->fmt, venc);   // cpp:689
    if (!vstm) { avcodec_free_context(&vctx); return -5; }
    vstm->id = e->fmt->nb_streams - 1;                    // cpp:695
    vstm->time_base = vctx->time_base;                    // cpp:696
    vstm->avg_frame_rate = vctx->framerate;               // cpp:697
    avcodec_parameters_from_context(vstm->codecpar, vctx);// cpp:698

    // RGBA(input) -> encoder pix_fmt, BICUBIC (matches decode shim's scaler family).
    e->vsws = sws_getContext(in_w, in_h, AV_PIX_FMT_RGBA,
                             width, height, pixfmt,
                             SWS_BICUBIC, NULL, NULL, NULL);
    if (!e->vsws) { return -6; }

    e->vfrm = av_frame_alloc();
    e->vfrm->format = pixfmt;
    e->vfrm->width = width;
    e->vfrm->height = height;
    if (av_frame_get_buffer(e->vfrm, 0) < 0) return -7;

    e->venc = venc; e->vctx = vctx; e->vstm = vstm;
    e->vidx = vstm->index; e->vw = in_w; e->vh = in_h;
    e->vpixfmt = pixfmt; e->vnext_pts = 0;
    return e->vidx;
}

// Constant-quality (CRF) rate control for the video encoder (Triad-B P1 export controls).
// Called AFTER fpx_enc_config_video and BEFORE fpx_enc_start when the export uses rate_mode=1
// (constant quality) instead of an average bitrate. `crf` is the quality value (lower = better;
// x264/x265 ~18-28 typical). Because the video AVCodecContext was already opened in config_video
// (avcodec_open2), we cannot change codec private options after the fact for every codec — so this
// must run on the still-open context's private data via av_opt_set, which libx264/libx265 honor at
// open time only. To make CRF take effect we therefore re-open the codec context here with the crf
// option set and the bitrate cleared. For codecs without a "crf" private option (e.g. mpeg4) we fall
// back to a global_quality / qscale (FF_QP2LAMBDA-scaled) so "constant quality" still has an effect.
// Returns 0 on success, negative on error. Best-effort: a codec that rejects every quality knob
// leaves the average-bitrate config in place (still a valid encode).
int fpx_enc_set_quality(void* h, int crf, int gop, const char* preset) {
    if (!h) return -1;
    FpxEnc* e = (FpxEnc*)h;
    if (e->vidx < 0 || !e->vctx || !e->venc) return -2;
    if (e->header_written) return -3; // must be set before start (avformat_write_header)

    // Re-create + re-open the video codec context with the quality knob set (the prior context was
    // already opened with a bitrate; codec private options like "crf" are read at avcodec_open2).
    AVCodecContext* nv = avcodec_alloc_context3(e->venc);
    if (!nv) return -4;
    nv->pix_fmt = e->vctx->pix_fmt;
    nv->width = e->vctx->width;
    nv->height = e->vctx->height;
    nv->time_base = e->vctx->time_base;
    nv->framerate = e->vctx->framerate;
    nv->sample_aspect_ratio = e->vctx->sample_aspect_ratio;
    nv->bit_rate = 0; // CRF mode: let the quality knob drive the rate.
    if (e->fmt->oformat->flags & AVFMT_GLOBALHEADER)
        nv->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    char crfbuf[16];
    snprintf(crfbuf, sizeof crfbuf, "%d", crf);
    // x264/x265 honor "crf" as a private option; harmless (returns <0) for codecs without it.
    int have_crf = (av_opt_set(nv->priv_data, "crf", crfbuf, 0) == 0);
    if (!have_crf) {
        // Fallback for mpeg4 et al.: constant qscale via global_quality (FF_QP2LAMBDA-scaled) +
        // the CODEC_FLAG_QSCALE flag. Clamp crf into a sane qscale range (1=best..31=worst).
        int q = crf; if (q < 1) q = 1; if (q > 31) q = 31;
        nv->flags |= AV_CODEC_FLAG_QSCALE;
        nv->global_quality = q * FF_QP2LAMBDA;
    }

    // P25 export depth: the GOP/preset config_video set on the OLD vctx is LOST in this rebuilt nv
    // unless we carry it forward (the CRF path re-opens the codec, and both are read at open time).
    // Carry the gop_size value config_video stamped (gop>0 here overrides it; gop<=0 keeps it), and
    // re-apply the preset. gop<=0 + the carried-forward default => identity with the bitrate path.
    nv->gop_size = (gop > 0) ? gop : e->vctx->gop_size;
    if (preset && preset[0]) av_opt_set(nv->priv_data, "preset", preset, 0);

    if (avcodec_open2(nv, e->venc, NULL) < 0) { avcodec_free_context(&nv); return -5; }

    // Swap the new context in, rebuild the stream codecpar, and re-point the reusable input frame's
    // format (dims/pixfmt are unchanged, so vsws/vfrm stay valid).
    avcodec_free_context(&e->vctx);
    e->vctx = nv;
    avcodec_parameters_from_context(e->vstm->codecpar, nv);
    e->vstm->time_base = nv->time_base;
    e->vstm->avg_frame_rate = nv->framerate;
    return 0;
}

// ConfigureAudioStream (cpp:898-1010). codecName e.g. "aac"/"pcm_s16le".
// Input is interleaved float (AV_SAMPLE_FMT_FLT). Returns audio stream index or negative.
int fpx_enc_config_audio(void* h, const char* codecName,
                         int channels, int sample_rate, long long bit_rate) {
    if (!h) return -1;
    FpxEnc* e = (FpxEnc*)h;

    const AVCodec* aenc = avcodec_find_encoder_by_name(codecName);
    if (!aenc || aenc->type != AVMEDIA_TYPE_AUDIO) return -2;

    enum AVSampleFormat encfmt = aenc->sample_fmts ? aenc->sample_fmts[0]
                                                   : AV_SAMPLE_FMT_FLT;   // cpp:930-933

    AVCodecContext* actx = avcodec_alloc_context3(aenc);
    if (!actx) return -3;
    actx->sample_fmt = encfmt;                              // cpp:943
    av_channel_layout_default(&actx->ch_layout, channels);  // cpp:948
    actx->sample_rate = sample_rate;                        // cpp:950
    actx->bit_rate = bit_rate;                              // cpp:951
    actx->time_base = (AVRational){ 1, sample_rate };       // cpp:952
    if (e->fmt->oformat->flags & AVFMT_GLOBALHEADER)        // cpp:982-983
        actx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(actx, aenc, NULL) < 0) { avcodec_free_context(&actx); return -4; }

    int fs = actx->frame_size > 0 ? actx->frame_size : 1024; // cpp:964-967

    AVStream* astm = avformat_new_stream(e->fmt, aenc);      // cpp:975
    if (!astm) { avcodec_free_context(&actx); return -5; }
    astm->id = e->fmt->nb_streams - 1;                       // cpp:981
    astm->time_base = actx->time_base;                       // cpp:984
    avcodec_parameters_from_context(astm->codecpar, actx);   // cpp:985

    // input interleaved float -> encoder sample_fmt (cpp:987-1010)
    SwrContext* swr = NULL;
    if (encfmt != AV_SAMPLE_FMT_FLT) {
        if (swr_alloc_set_opts2(&swr, &actx->ch_layout, encfmt, sample_rate,
                                &actx->ch_layout, AV_SAMPLE_FMT_FLT, sample_rate,
                                0, NULL) < 0 || swr_init(swr) < 0) {
            if (swr) swr_free(&swr);
            return -6;
        }
    }

    e->aenc = aenc; e->actx = actx; e->astm = astm; e->swr = swr;
    e->aidx = astm->index; e->ach = channels; e->asr = sample_rate;
    e->ainfmt = AV_SAMPLE_FMT_FLT; e->aencfmt = encfmt;
    e->aframe_samples = fs; e->afrm_off = 0; e->anext_pts = 0;
    return e->aidx;
}

// Start: write container header (cpp:186-191).
int fpx_enc_start(void* h) {
    if (!h) return -1;
    FpxEnc* e = (FpxEnc*)h;
    if (e->header_written) return 0;
    int r = avformat_write_header(e->fmt, NULL);
    if (r < 0) return r;
    e->header_written = 1;
    return 0;
}

// EncodeVideoFrame: input is packed RGBA f32 in [0,1] (Mojo ImageMat), in_w*in_h*4 floats.
// ts_sec = ImMat.time_stamp (seconds). Reproduces ConvertImMatToAVFrame (cpp:1080-1092):
//   pts = rescale(ts_sec*1000, MILLISEC_TIMEBASE, vctx->time_base).
// rgba8 is built from the f32 input, sws_scale to encoder pix_fmt, then send/recv/mux.
int fpx_enc_video_frame_f32(void* h, const float* rgba_f32, int in_w, int in_h, double ts_sec) {
    if (!h) return -1;
    FpxEnc* e = (FpxEnc*)h;
    if (e->vidx < 0 || !e->vctx) return -2;
    if (in_w != e->vw || in_h != e->vh) return -3;

    // f32[0,1] -> RGBA8 (the input the encoder's sws expects).
    int n = in_w * in_h * 4;
    uint8_t* rgba8 = (uint8_t*)malloc((size_t)n);
    if (!rgba8) return -4;
    for (int i = 0; i < n; i++) {
        float v = rgba_f32[i] * 255.0f + 0.5f;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        rgba8[i] = (uint8_t)v;
    }

    if (av_frame_make_writable(e->vfrm) < 0) { free(rgba8); return -5; }
    const uint8_t* src[4] = { rgba8, NULL, NULL, NULL };
    int srcln[4] = { in_w * 4, 0, 0, 0 };
    sws_scale(e->vsws, src, srcln, 0, in_h, e->vfrm->data, e->vfrm->linesize);
    free(rgba8);

    int64_t pts = av_rescale_q((int64_t)(ts_sec * 1000.0), MILLISEC_TIMEBASE, e->vctx->time_base);
    e->vfrm->pts = pts;
    e->vnext_pts = pts + 1;

    int ret = avcodec_send_frame(e->vctx, e->vfrm);   // cpp:1179
    if (ret < 0) return ret;
    return fpx__drain(e, e->vctx, e->vstm);
}

// internal: send the accumulated audio frame `afrm` to the encoder + drain.
static int fpx__send_audio_frame(FpxEnc* e) {
    e->afrm->pts = e->anext_pts;
    int ret = avcodec_send_frame(e->actx, e->afrm);   // cpp:1259
    if (ret < 0) return ret;
    e->anext_pts += e->afrm->nb_samples;
    av_frame_free(&e->afrm);
    e->afrm_off = 0;
    return fpx__drain(e, e->actx, e->astm);
}

// EncodeAudioSamples: input is interleaved float, `nb` samples/ch (cpp:296-430).
// Accumulates into encoder-frame-size chunks, swr-converts as needed, sends + muxes.
int fpx_enc_audio_samples_f32(void* h, const float* in, int nb) {
    if (!h) return -1;
    FpxEnc* e = (FpxEnc*)h;
    if (e->aidx < 0 || !e->actx) return -2;
    if (nb <= 0) return 0;

    int ch = e->ach;
    int consumed = 0;   // input samples/ch consumed
    while (consumed < nb) {
        if (!e->afrm) {
            e->afrm = av_frame_alloc();
            e->afrm->format = e->aencfmt;
            e->afrm->sample_rate = e->asr;
            av_channel_layout_copy(&e->afrm->ch_layout, &e->actx->ch_layout);
            e->afrm->nb_samples = e->aframe_samples;       // cpp:379
            if (av_frame_get_buffer(e->afrm, 0) < 0) return -3;
            e->afrm_off = 0;
        }
        int space = e->aframe_samples - e->afrm_off;       // room in current frame
        int avail = nb - consumed;
        int take = avail < space ? avail : space;

        if (e->swr) {
            // swr_convert into afrm at offset afrm_off (planar/packed handled by lavu)
            const uint8_t* sp[1] = { (const uint8_t*)(in + (size_t)consumed * ch) };
            uint8_t* dp[AV_NUM_DATA_POINTERS];
            for (int p = 0; p < AV_NUM_DATA_POINTERS; p++) dp[p] = e->afrm->data[p];
            int bps = av_get_bytes_per_sample(e->aencfmt);
            int planar = av_sample_fmt_is_planar(e->aencfmt);
            if (planar) {
                for (int c = 0; c < ch; c++) dp[c] = e->afrm->data[c] + (size_t)e->afrm_off * bps;
            } else {
                dp[0] = e->afrm->data[0] + (size_t)e->afrm_off * bps * ch;
            }
            int got = swr_convert(e->swr, dp, space, sp, take);  // cpp:391
            if (got < 0) return got;
            e->afrm_off += got;
            consumed += take;
        } else {
            // direct interleaved-float copy (cpp:404-411)
            int bps = av_get_bytes_per_sample(e->aencfmt); // == sizeof(float)
            memcpy(e->afrm->data[0] + (size_t)e->afrm_off * bps * ch,
                   in + (size_t)consumed * ch,
                   (size_t)take * bps * ch);
            e->afrm_off += take;
            consumed += take;
        }

        if (e->afrm_off >= e->aframe_samples) {            // cpp:414
            int r = fpx__send_audio_frame(e);
            if (r < 0) return r;
        }
    }
    return consumed;
}

// FinishEncoding (cpp:198-236): flush video + audio encoders, write trailer, close io.
int fpx_enc_finish(void* h) {
    if (!h) return -1;
    FpxEnc* e = (FpxEnc*)h;
    int ret = 0;

    // flush partial audio frame, then send EOF (cpp:325-339,1234)
    if (e->aidx >= 0 && e->actx) {
        if (e->afrm && e->afrm_off > 0) {
            // zero the tail of the partial frame (cpp:330) then send
            int bps = av_get_bytes_per_sample(e->aencfmt);
            int planar = av_sample_fmt_is_planar(e->aencfmt);
            int ch = e->ach;
            if (planar) {
                for (int c = 0; c < ch; c++)
                    memset(e->afrm->data[c] + (size_t)e->afrm_off * bps, 0,
                           (size_t)(e->aframe_samples - e->afrm_off) * bps);
            } else {
                memset(e->afrm->data[0] + (size_t)e->afrm_off * bps * ch, 0,
                       (size_t)(e->aframe_samples - e->afrm_off) * bps * ch);
            }
            int r = fpx__send_audio_frame(e);
            if (r < 0) ret = r;
        }
        if (e->afrm) { av_frame_free(&e->afrm); }
        avcodec_send_frame(e->actx, NULL);                 // EOF
        fpx__drain(e, e->actx, e->astm);
    }

    // flush video encoder (cpp:1155)
    if (e->vidx >= 0 && e->vctx) {
        avcodec_send_frame(e->vctx, NULL);                 // EOF
        fpx__drain(e, e->vctx, e->vstm);
    }

    if (e->header_written) {
        int r = av_write_trailer(e->fmt);                  // cpp:223
        if (r < 0) ret = r;
    }
    return ret;
}

void fpx_enc_close(void* h) {
    if (!h) return;
    FpxEnc* e = (FpxEnc*)h;
    if (e->vsws) sws_freeContext(e->vsws);
    if (e->vfrm) av_frame_free(&e->vfrm);
    if (e->afrm) av_frame_free(&e->afrm);
    if (e->swr) swr_free(&e->swr);
    if (e->vctx) avcodec_free_context(&e->vctx);
    if (e->actx) avcodec_free_context(&e->actx);
    if (e->fmt) {
        if (!(e->fmt->oformat->flags & AVFMT_NOFILE) && e->fmt->pb)
            avio_closep(&e->fmt->pb);
        avformat_free_context(e->fmt);
    }
    if (e->pkt) av_packet_free(&e->pkt);
    free(e);
}

#ifdef FPX_TEST
#include <math.h>
// usage: fpxenctest <out.mp4> <w> <h> <nframes> [codec]
// writes a moving-gradient RGBA test clip and reports per ffprobe externally.
int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s out.mp4 w h nframes [codec]\n", argv[0]); return 2; }
    const char* out = argv[1];
    int w = atoi(argv[2]), h = atoi(argv[3]), nf = atoi(argv[4]);
    const char* codec = argc > 5 ? argv[5] : "mpeg4";
    void* e = fpx_enc_open(out);
    if (!e) { fprintf(stderr, "open failed\n"); return 1; }
    int vi = fpx_enc_config_video(e, codec, w, h, w, h, 25, 1, 2000000, 0, NULL);
    if (vi < 0) { fprintf(stderr, "config video failed %d\n", vi); return 1; }
    if (fpx_enc_start(e) < 0) { fprintf(stderr, "start failed\n"); return 1; }
    float* buf = (float*)malloc((size_t)w * h * 4 * sizeof(float));
    for (int f = 0; f < nf; f++) {
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int o = (y * w + x) * 4;
                buf[o+0] = (float)((x + f * 7) % w) / (float)w;
                buf[o+1] = (float)((y + f * 3) % h) / (float)h;
                buf[o+2] = (float)(f) / (float)nf;
                buf[o+3] = 1.0f;
            }
        double ts = (double)f / 25.0;
        int r = fpx_enc_video_frame_f32(e, buf, w, h, ts);
        if (r < 0) { fprintf(stderr, "encode frame %d failed %d\n", f, r); return 1; }
    }
    if (fpx_enc_finish(e) < 0) { fprintf(stderr, "finish failed\n"); return 1; }
    fpx_enc_close(e);
    free(buf);
    fprintf(stderr, "ok %s %dx%d %d frames (%s)\n", out, w, h, nf, codec);
    return 0;
}
#endif
