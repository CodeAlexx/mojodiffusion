// fpx_audio.c — libavfilter audio C shim (the engine MediaEditor's AudioEffectFilter_FFImpl uses).
// Applies an arbitrary libavfilter chain (volume/pan/aeq/acompressor/alimiter/agate/...) and a
// 2-track amix, to interleaved-float stereo audio. Callable from Mojo via external_call. Verified
// bit-for-bit against `ffmpeg -af "<same chain>"` (libavfilter is the shared engine).
//
// Build .so:  cc -O2 -fPIC -shared fpx_audio.c $(pkg-config --cflags --libs \
//                 libavfilter libavformat libavcodec libavutil) -o libfpxaudio.so
// Build test: cc -DFPX_TEST ... -o fpxautest
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersrc.h>
#include <libavfilter/buffersink.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <libavutil/frame.h>
#include <libavutil/opt.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// One-shot: apply `chain` to `in` (nb samples/ch, interleaved float, stereo `ch` channels),
// write up to out_cap floats into `out`. Returns output samples-per-channel, or negative on error.
int fpx_au_apply(int sr, int ch, const char* chain,
                 const float* in, int nb, float* out, int out_cap) {
    AVFilterGraph* g = avfilter_graph_alloc();
    if (!g) return -1;
    const AVFilter* src  = avfilter_get_by_name("abuffer");
    const AVFilter* sink = avfilter_get_by_name("abuffersink");
    AVFilterContext *srcc = NULL, *sinkc = NULL;

    AVChannelLayout layout; av_channel_layout_default(&layout, ch);
    char lbuf[64]; av_channel_layout_describe(&layout, lbuf, sizeof(lbuf));
    char args[256];
    snprintf(args, sizeof(args),
        "time_base=1/%d:sample_rate=%d:sample_fmt=flt:channel_layout=%s", sr, sr, lbuf);
    if (avfilter_graph_create_filter(&srcc, src, "in", args, NULL, g) < 0) { avfilter_graph_free(&g); return -2; }
    if (avfilter_graph_create_filter(&sinkc, sink, "out", NULL, NULL, g) < 0) { avfilter_graph_free(&g); return -3; }

    // force interleaved-float stereo output so IO stays simple + comparable.
    // PIN sample_rate too: loudnorm (and any future rate-changing filter) otherwise resamples
    // (loudnorm -> 192kHz), returning 4x samples at the wrong rate. With sample_rates=<sr> the
    // graph downsamples back to <sr>, so output samples/ch == input for all our length-preserving
    // filters. Verified: `loudnorm` 4800 in -> 4800 out with this pinned.
    char full[2048];
    snprintf(full, sizeof(full), "%s%saformat=sample_fmts=flt:sample_rates=%d:channel_layouts=%s",
             (chain && chain[0]) ? chain : "anull", (chain && chain[0]) ? "," : "", sr, lbuf);

    AVFilterInOut* outputs = avfilter_inout_alloc();   // the buffersrc's output = graph input feed
    AVFilterInOut* inputs  = avfilter_inout_alloc();   // the buffersink's input  = graph output
    outputs->name = av_strdup("in");  outputs->filter_ctx = srcc;  outputs->pad_idx = 0; outputs->next = NULL;
    inputs->name  = av_strdup("out"); inputs->filter_ctx  = sinkc; inputs->pad_idx  = 0; inputs->next  = NULL;

    int rc = avfilter_graph_parse_ptr(g, full, &inputs, &outputs, NULL);
    avfilter_inout_free(&inputs); avfilter_inout_free(&outputs);
    if (rc < 0) { avfilter_graph_free(&g); return -4; }
    if (avfilter_graph_config(g, NULL) < 0) { avfilter_graph_free(&g); return -5; }

    AVFrame* fr = av_frame_alloc();
    fr->format = AV_SAMPLE_FMT_FLT; fr->sample_rate = sr; fr->nb_samples = nb;
    av_channel_layout_copy(&fr->ch_layout, &layout);
    if (av_frame_get_buffer(fr, 0) < 0) { av_frame_free(&fr); avfilter_graph_free(&g); return -6; }
    memcpy(fr->data[0], in, (size_t)nb * ch * sizeof(float));
    if (av_buffersrc_add_frame(srcc, fr) < 0) { av_frame_free(&fr); avfilter_graph_free(&g); return -7; }
    av_buffersrc_add_frame(srcc, NULL);   // EOF flush

    int total = 0;
    AVFrame* of = av_frame_alloc();
    while (av_buffersink_get_frame(sinkc, of) >= 0) {
        int n = of->nb_samples;
        if ((total + n) * ch <= out_cap)
            memcpy(out + (size_t)total * ch, of->data[0], (size_t)n * ch * sizeof(float));
        total += n;
        av_frame_unref(of);
    }
    av_frame_free(&of); av_frame_free(&fr);
    av_channel_layout_uninit(&layout);
    avfilter_graph_free(&g);
    return total;
}

// Load a raw interleaved-f32 PCM file into out (returns total floats read, <= cap).
int fpx_load_pcm_f32(const char* path, float* out, int cap) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    size_t n = fread(out, sizeof(float), (size_t)cap, f);
    fclose(f);
    return (int)n;
}

// 2-track mixer: each track through its own chain, then `[a0][a1]amix=inputs=2`, then `mixchain`.
int fpx_au_mix2(int sr, int ch,
                const char* chainA, const float* a, int na,
                const char* chainB, const float* b, int nb,
                const char* mixchain, float* out, int out_cap) {
    AVFilterGraph* g = avfilter_graph_alloc();
    if (!g) return -1;
    const AVFilter* abuf = avfilter_get_by_name("abuffer");
    const AVFilter* asink = avfilter_get_by_name("abuffersink");
    AVFilterContext *s0=NULL, *s1=NULL, *snk=NULL;
    AVChannelLayout layout; av_channel_layout_default(&layout, ch);
    char lbuf[64]; av_channel_layout_describe(&layout, lbuf, sizeof(lbuf));
    char args[256];
    snprintf(args, sizeof(args), "time_base=1/%d:sample_rate=%d:sample_fmt=flt:channel_layout=%s", sr, sr, lbuf);
    if (avfilter_graph_create_filter(&s0, abuf, "in0", args, NULL, g) < 0 ||
        avfilter_graph_create_filter(&s1, abuf, "in1", args, NULL, g) < 0 ||
        avfilter_graph_create_filter(&snk, asink, "out", NULL, NULL, g) < 0) { avfilter_graph_free(&g); return -2; }

    char desc[3072];
    // chain attaches directly to its link labels: [in0]chain[a0] (NO comma around labels).
    // PIN sample_rate in the output aformat (same reason as fpx_au_apply: a rate-changing filter
    // like loudnorm in any of the chains would otherwise leave the mix at the wrong rate).
    snprintf(desc, sizeof(desc),
        "[in0]%s[a0];[in1]%s[a1];[a0][a1]amix=inputs=2:normalize=0,%s,aformat=sample_fmts=flt:sample_rates=%d:channel_layouts=%s[out]",
        (chainA&&chainA[0])?chainA:"anull",
        (chainB&&chainB[0])?chainB:"anull",
        (mixchain&&mixchain[0])?mixchain:"anull", sr, lbuf);

    AVFilterInOut *o0=avfilter_inout_alloc(), *o1=avfilter_inout_alloc(), *ins=avfilter_inout_alloc();
    o0->name=av_strdup("in0"); o0->filter_ctx=s0; o0->pad_idx=0; o0->next=o1;
    o1->name=av_strdup("in1"); o1->filter_ctx=s1; o1->pad_idx=0; o1->next=NULL;
    ins->name=av_strdup("out"); ins->filter_ctx=snk; ins->pad_idx=0; ins->next=NULL;
    int rc = avfilter_graph_parse_ptr(g, desc, &ins, &o0, NULL);
    avfilter_inout_free(&ins); avfilter_inout_free(&o0);
    if (rc < 0) { avfilter_graph_free(&g); return -4; }
    if (avfilter_graph_config(g, NULL) < 0) { avfilter_graph_free(&g); return -5; }

    // push both tracks
    for (int t = 0; t < 2; t++) {
        AVFilterContext* sc = t==0 ? s0 : s1;
        const float* buf = t==0 ? a : b; int n = t==0 ? na : nb;
        AVFrame* fr = av_frame_alloc();
        fr->format = AV_SAMPLE_FMT_FLT; fr->sample_rate = sr; fr->nb_samples = n;
        av_channel_layout_copy(&fr->ch_layout, &layout);
        av_frame_get_buffer(fr, 0);
        memcpy(fr->data[0], buf, (size_t)n*ch*sizeof(float));
        av_buffersrc_add_frame(sc, fr);
        av_buffersrc_add_frame(sc, NULL);
        av_frame_free(&fr);
    }
    int total=0; AVFrame* of=av_frame_alloc();
    while (av_buffersink_get_frame(snk, of) >= 0) {
        int n=of->nb_samples;
        if ((total+n)*ch <= out_cap) memcpy(out+(size_t)total*ch, of->data[0], (size_t)n*ch*sizeof(float));
        total+=n; av_frame_unref(of);
    }
    av_frame_free(&of); av_channel_layout_uninit(&layout); avfilter_graph_free(&g);
    return total;
}

#ifdef FPX_TEST
// usage: fpxautest <sr> <ch> <chain> <in.raw f32> <out.raw f32>
int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s sr ch chain in.raw out.raw\n", argv[0]); return 2; }
    int sr = atoi(argv[1]), ch = atoi(argv[2]);
    FILE* fi = fopen(argv[4], "rb"); if (!fi) return 1;
    fseek(fi, 0, SEEK_END); long bytes = ftell(fi); fseek(fi, 0, SEEK_SET);
    int nb = (int)(bytes / sizeof(float) / ch);
    float* in = malloc(bytes); fread(in, 1, bytes, fi); fclose(fi);
    int cap = (nb + sr) * ch;
    float* out = malloc((size_t)cap * sizeof(float));
    int n = fpx_au_apply(sr, ch, argv[3], in, nb, out, cap);
    if (n < 0) { fprintf(stderr, "apply failed %d\n", n); return 1; }
    FILE* fo = fopen(argv[5], "wb"); fwrite(out, sizeof(float), (size_t)n*ch, fo); fclose(fo);
    fprintf(stderr, "ok: in=%d out=%d samples/ch\n", nb, n);
    free(in); free(out); return 0;
}
#endif
