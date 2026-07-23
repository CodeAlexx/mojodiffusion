// fpx_aread.c — decode a TIME RANGE of a media file's audio -> interleaved f32.
// For dataset clip export (per-range audio that matches the trimmed video).
//   cc -O2 -fPIC -shared fpx_aread.c $(pkg-config --cflags --libs \
//        libavformat libavcodec libswresample libavutil) -o libfpxaread.so
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <libavutil/mathematics.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>

// Spawn `path a1 a2` as a detached child (parent returns immediately). For launching a
// sibling tool window (e.g. the editor opening the audio mixer). Returns child pid or neg.
int fpx_spawn(const char* path, const char* a1, const char* a2) {
    if (!path) return -1;
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        execl(path, path, a1, a2, (char*)0);
        _exit(127);
    }
    return (int)pid;
}

// ---- timeline primitives (kept in C so they stay OUT of the Mojo compile unit) ----
// Snap `frame` to the nearest of edges[n] within max_dist frames; else return frame.
int fpx_snap_frame(int frame, const int* edges, int n, int max_dist) {
    if (!edges || n <= 0) return frame;
    int best = frame, bestd = max_dist + 1;
    for (int i = 0; i < n; i++) {
        int d = edges[i] - frame; if (d < 0) d = -d;
        if (d <= max_dist && d < bestd) { bestd = d; best = edges[i]; }
    }
    return best;
}
// Index of the marker nearest `frame` within max_dist; else -1.
int fpx_nearest_marker(const int* mframes, int n, int frame, int max_dist) {
    if (!mframes || n <= 0) return -1;
    int best = -1, bestd = max_dist + 1;
    for (int i = 0; i < n; i++) {
        int d = mframes[i] - frame; if (d < 0) d = -d;
        if (d <= max_dist && d < bestd) { bestd = d; best = i; }
    }
    return best;
}
// Per-clip fade opacity (0..1) at local_frame: ramp up over fade_in, down over fade_out.
float fpx_fade_opacity(int local_frame, int clip_len, int fade_in, int fade_out) {
    float op = 1.0f;
    if (fade_in > 0 && local_frame < fade_in) op = (float)local_frame / (float)fade_in;
    if (fade_out > 0) {
        int from_end = clip_len - 1 - local_frame;
        if (from_end >= 0 && from_end < fade_out) {
            float o2 = (float)from_end / (float)fade_out;
            if (o2 < op) op = o2;
        }
    }
    if (op < 0.0f) op = 0.0f;
    if (op > 1.0f) op = 1.0f;
    return op;
}

// Show a native file picker (zenity) and write the chosen path to `out_path` (UTF-8).
// Blocks until the user picks or cancels. Returns 0 if a file was chosen (path written),
// non-zero if cancelled / unavailable. The caller reads `out_path` back as a String.
int fpx_pick_file(const char* out_path) {
    if (!out_path) return -1;
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "zenity --file-selection --title='Add media to project' > '%s' 2>/dev/null",
             out_path);
    int rc = system(cmd);               // blocks; zenity exits 0 on pick, 1 on cancel
    return (rc == 0) ? 0 : 1;
}

// Parse a .cube 3D LUT into `out` (N*N*N*3 floats, RED-fastest order). Returns the grid
// size N, or a negative error. DOMAIN is assumed [0,1] (the Film-Luts collection's domain).
int fpx_load_cube(const char* path, float* out, int max_floats) {
    if (!path || !out) return -1;
    FILE* f = fopen(path, "r");
    if (!f) return -2;
    int N = 0, count = 0;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        char* p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == 0) continue;
        if (strncmp(p, "LUT_3D_SIZE", 11) == 0) { N = atoi(p + 11); continue; }
        if (strncmp(p, "LUT_1D_SIZE", 11) == 0) { fclose(f); return -3; }   // 1D unsupported
        if (strncmp(p, "TITLE", 5) == 0) continue;
        if (strncmp(p, "DOMAIN_MIN", 10) == 0) continue;
        if (strncmp(p, "DOMAIN_MAX", 10) == 0) continue;
        float r, g, b;
        if (sscanf(p, "%f %f %f", &r, &g, &b) == 3) {
            if (count + 3 > max_floats) { fclose(f); return -4; }
            out[count++] = r; out[count++] = g; out[count++] = b;
        }
    }
    fclose(f);
    if (N <= 0) return -5;
    if (count != N * N * N * 3) return -6;     // incomplete / mismatched
    return N;
}

// Read up to `max_bytes` raw bytes from a binary file into `buf`. Used to load the
// pre-baked icon RGBA8 blob (assets/icons_dark_32.rgba). Returns bytes read, or negative.
int fpx_read_blob(const char* path, unsigned char* buf, int max_bytes) {
    if (!path || !buf || max_bytes <= 0) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -2;
    size_t n = fread(buf, 1, (size_t)max_bytes, f);
    fclose(f);
    return (int)n;
}

// List files ending in `ext` (e.g. ".cube") in `dir`, sorted, newline-separated, into the
// file `out_path`. Returns the count, or negative on error. (Mojo reads the file back.)
static int cmp_str(const void* a, const void* b) { return strcmp(*(const char**)a, *(const char**)b); }
int fpx_list_dir(const char* dir, const char* ext, const char* out_path) {
    if (!dir || !ext || !out_path) return -1;
    DIR* d = opendir(dir);
    if (!d) return -2;
    char* names[4096];
    int n = 0;
    size_t el = strlen(ext);
    struct dirent* de;
    while ((de = readdir(d)) && n < 4096) {
        size_t nl = strlen(de->d_name);
        if (nl > el && strcmp(de->d_name + nl - el, ext) == 0) {
            names[n] = strdup(de->d_name);
            if (names[n]) n++;
        }
    }
    closedir(d);
    qsort(names, n, sizeof(char*), cmp_str);
    FILE* f = fopen(out_path, "w");
    if (!f) { for (int i = 0; i < n; i++) free(names[i]); return -3; }
    for (int i = 0; i < n; i++) { fprintf(f, "%s\n", names[i]); free(names[i]); }
    fclose(f);
    return n;
}

// Create a directory (mode 0777). Returns 0 on success or if it already exists.
// (A dedicated shim symbol so callers avoid a libc `mkdir` external_call clash.)
int fpx_mkdir(const char* path) {
    if (!path) return -1;
    mkdir(path, 0777);
    return 0;
}

// Write `n` interleaved float32 samples to a raw PCM file. Returns samples written or neg.
int fpx_write_f32(const char* path, const float* buf, int n) {
    if (!path || !buf || n < 0) return -1;
    FILE* f = fopen(path, "wb");
    if (!f) return -2;
    size_t w = fwrite(buf, sizeof(float), (size_t)n, f);
    fclose(f);
    return (int)w;
}

// Decode audio of `path` for [start_sec, start_sec+dur_sec) and resample to
// out_sr / out_ch interleaved float32 into `out` (capacity `cap` floats).
// Returns the number of FLOATS written (frames*out_ch), 0 if the file has no
// audio, or negative on error. Trim is frame-granular at the start (the leading
// partial audio frame after the seek may be included; ~<=1 codec frame ~20ms).
int fpx_decode_audio_range(const char* path, double start_sec, double dur_sec,
                           int out_sr, int out_ch, float* out, int cap) {
    if (!path || !out || out_sr <= 0 || out_ch <= 0 || cap <= 0) return -1;

    AVFormatContext* fmt = NULL;
    if (avformat_open_input(&fmt, path, NULL, NULL) < 0) return -2;
    if (avformat_find_stream_info(fmt, NULL) < 0) { avformat_close_input(&fmt); return -3; }

    const AVCodec* codec = NULL;
    int aidx = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
    if (aidx < 0 || !codec) { avformat_close_input(&fmt); return 0; }   // no audio stream
    AVStream* st = fmt->streams[aidx];

    AVCodecContext* dec = avcodec_alloc_context3(codec);
    if (!dec) { avformat_close_input(&fmt); return -4; }
    avcodec_parameters_to_context(dec, st->codecpar);
    if (avcodec_open2(dec, codec, NULL) < 0) {
        avcodec_free_context(&dec); avformat_close_input(&fmt); return -4;
    }

    AVChannelLayout out_layout;
    av_channel_layout_default(&out_layout, out_ch);

    SwrContext* swr = NULL;
    if (swr_alloc_set_opts2(&swr, &out_layout, AV_SAMPLE_FMT_FLT, out_sr,
                            &dec->ch_layout, dec->sample_fmt, dec->sample_rate,
                            0, NULL) < 0 || swr_init(swr) < 0) {
        if (swr) swr_free(&swr);
        av_channel_layout_uninit(&out_layout);
        avcodec_free_context(&dec); avformat_close_input(&fmt); return -5;
    }

    int64_t target = (int64_t)(start_sec / av_q2d(st->time_base));
    av_seek_frame(fmt, aidx, target, AVSEEK_FLAG_BACKWARD);
    avcodec_flush_buffers(dec);

    AVPacket* pkt = av_packet_alloc();
    AVFrame* fr = av_frame_alloc();
    double end_sec = start_sec + dur_sec;
    int total_floats = 0;
    int done = 0;
    double first_t = -1.0;     // time of the first sample collected (for sample-accurate trim)

    while (!done && av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index == aidx) {
            if (avcodec_send_packet(dec, pkt) == 0) {
                while (avcodec_receive_frame(dec, fr) == 0) {
                    double t = (fr->best_effort_timestamp == AV_NOPTS_VALUE)
                                 ? start_sec
                                 : fr->best_effort_timestamp * av_q2d(st->time_base);
                    double flen = (double)fr->nb_samples / (double)dec->sample_rate;
                    if (t + flen < start_sec) continue;   // frame fully before range
                    if (t >= end_sec) { done = 1; break; } // frame at/after range end
                    int out_count = (int)av_rescale_rnd(
                        swr_get_delay(swr, dec->sample_rate) + fr->nb_samples,
                        out_sr, dec->sample_rate, AV_ROUND_UP);
                    if (total_floats + out_count * out_ch > cap)
                        out_count = (cap - total_floats) / out_ch;
                    if (out_count <= 0) { done = 1; break; }
                    uint8_t* outp = (uint8_t*)(out + total_floats);
                    int got = swr_convert(swr, &outp, out_count,
                                          (const uint8_t**)fr->data, fr->nb_samples);
                    if (got > 0) {
                        if (first_t < 0.0) first_t = t;   // time of the very first collected sample
                        total_floats += got * out_ch;
                    }
                    if (total_floats >= cap) { done = 1; break; }
                }
            }
        }
        av_packet_unref(pkt);
    }

    av_frame_free(&fr);
    av_packet_free(&pkt);
    swr_free(&swr);
    av_channel_layout_uninit(&out_layout);
    avcodec_free_context(&dec);
    avformat_close_input(&fmt);

    // sample-accurate trim: drop leading samples before start_sec, cap to exactly dur_sec.
    if (total_floats > 0 && first_t >= 0.0) {
        int drop = (int)((start_sec - first_t) * out_sr + 0.5);
        if (drop < 0) drop = 0;
        int drop_f = drop * out_ch;
        int want_f = (int)(dur_sec * out_sr + 0.5) * out_ch;
        if (drop_f >= total_floats) {
            total_floats = 0;
        } else {
            int avail = total_floats - drop_f;
            int keep = avail < want_f ? avail : want_f;
            if (drop_f > 0) memmove(out, out + drop_f, (size_t)keep * sizeof(float));
            total_floats = keep;
        }
    }
    return total_floats;
}

// Build a peak-amplitude ENVELOPE of the whole audio track for a waveform display:
// resample to mono 8 kHz f32, fill out[nbuckets] with the peak |sample| (0..1) in each
// equal time-bucket across the file. Returns nbuckets on success, 0 if no audio, neg on error.
int fpx_audio_envelope(const char* path, int nbuckets, float* out) {
    if (!path || !out || nbuckets <= 0) return -1;
    for (int i = 0; i < nbuckets; i++) out[i] = 0.0f;

    AVFormatContext* fmt = NULL;
    if (avformat_open_input(&fmt, path, NULL, NULL) < 0) return -2;
    if (avformat_find_stream_info(fmt, NULL) < 0) { avformat_close_input(&fmt); return -3; }
    const AVCodec* codec = NULL;
    int aidx = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
    if (aidx < 0 || !codec) { avformat_close_input(&fmt); return 0; }
    AVStream* st = fmt->streams[aidx];
    AVCodecContext* dec = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(dec, st->codecpar);
    if (avcodec_open2(dec, codec, NULL) < 0) { avcodec_free_context(&dec); avformat_close_input(&fmt); return -4; }

    const int ESR = 8000;                 // envelope sample rate (mono, downsampled)
    AVChannelLayout monolay;
    av_channel_layout_default(&monolay, 1);
    SwrContext* swr = NULL;
    if (swr_alloc_set_opts2(&swr, &monolay, AV_SAMPLE_FMT_FLT, ESR,
                            &dec->ch_layout, dec->sample_fmt, dec->sample_rate, 0, NULL) < 0
        || swr_init(swr) < 0) {
        if (swr) swr_free(&swr);
        av_channel_layout_uninit(&monolay);
        avcodec_free_context(&dec); avformat_close_input(&fmt); return -5;
    }

    // total output samples (for sample->bucket mapping)
    double dur = (fmt->duration > 0) ? (double)fmt->duration / AV_TIME_BASE : 0.0;
    int64_t total = (int64_t)(dur * ESR);
    if (total <= 0) total = 1;

    AVPacket* pkt = av_packet_alloc();
    AVFrame* fr = av_frame_alloc();
    float buf[16384];
    int64_t scount = 0;
    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index == aidx) {
            if (avcodec_send_packet(dec, pkt) == 0) {
                while (avcodec_receive_frame(dec, fr) == 0) {
                    int cap = 16384;
                    uint8_t* op = (uint8_t*)buf;
                    int got = swr_convert(swr, &op, cap, (const uint8_t**)fr->data, fr->nb_samples);
                    for (int j = 0; j < got; j++) {
                        int b = (int)(scount * (int64_t)nbuckets / total);
                        if (b < 0) b = 0;
                        if (b >= nbuckets) b = nbuckets - 1;
                        float a = buf[j] < 0 ? -buf[j] : buf[j];
                        if (a > out[b]) out[b] = a;
                        scount++;
                    }
                }
            }
        }
        av_packet_unref(pkt);
    }
    av_frame_free(&fr);
    av_packet_free(&pkt);
    swr_free(&swr);
    av_channel_layout_uninit(&monolay);
    avcodec_free_context(&dec);
    avformat_close_input(&fmt);
    return nbuckets;
}
