// cumem_counter.c — LD_PRELOAD CUDA driver-API allocation counter (MJ-1090).
//
// The nsys 2023.4.4 capture->import pair on this box dies at ~25-60% on
// trainer-size captures (5/6 failures, quiet box included), which blocks the
// item-5 scratch-ring question: "how many synchronous cuMemAlloc calls does a
// training step make, and how many are >=128MB (the MAX cached-bin miss
// class, MJ-1083)?" This shim answers exactly that with zero external tools.
//
// Usage:
//   gcc -O2 -shared -fPIC -o libcumem_counter.so cumem_counter.c -ldl
//   LD_PRELOAD=$PWD/libcumem_counter.so <trainer> <args>
//   [CUMEM_THRESH_MB=128] [CUMEM_LOG=/path/summary.txt]
//
// Prints (stderr + optional file) at exit: call counts for cuMemAlloc_v2 /
// cuMemAllocAsync / cuMemAllocPitch_v2 / frees, bytes totals, >=threshold
// counts, max single alloc, and a coarse size histogram.
//
// MEASURED 2026-07-07: plain symbol interposition IS bypassed here (all-zero
// counters on a run that allocates GBs) — MAX resolves the driver via
// dlopen+dlsym. So this shim ALSO interposes dlsym itself (real dlsym
// obtained via dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5") — the classic
// bootstrap) and hands back our wrappers for the five counted entry points;
// the wrappers call the real functions captured from the app's own handle.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

static atomic_ullong n_alloc = 0, n_alloc_async = 0, n_pitch = 0;
static atomic_ullong n_free = 0, n_free_async = 0;
static atomic_ullong bytes_total = 0, big_count = 0, max_alloc = 0;
// histogram buckets: <1MB, 1-16MB, 16-64MB, 64-128MB, 128-256MB, 256-512MB, >=512MB
static atomic_ullong hist[7];
static unsigned long long thresh_bytes = 128ull * 1024 * 1024;

static void note_alloc(size_t sz) {
    atomic_fetch_add(&bytes_total, (unsigned long long)sz);
    if (sz >= thresh_bytes) atomic_fetch_add(&big_count, 1);
    unsigned long long cur = atomic_load(&max_alloc);
    while (sz > cur && !atomic_compare_exchange_weak(&max_alloc, &cur, sz)) {}
    int b = sz < (1ull<<20) ? 0 : sz < (16ull<<20) ? 1 : sz < (64ull<<20) ? 2
          : sz < (128ull<<20) ? 3 : sz < (256ull<<20) ? 4 : sz < (512ull<<20) ? 5 : 6;
    atomic_fetch_add(&hist[b], 1);
}

// real fns: captured either from RTLD_NEXT (static-link path) or from the
// app's dlsym lookup (dlopen path — see our dlsym interpose below).
static void *real_fn[5];  // 0 alloc, 1 allocAsync, 2 pitch, 3 free, 4 freeAsync

static void *bootstrap_real_dlsym(void) {
    static void *(*rd)(void *, const char *) = NULL;
    if (!rd)
        rd = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    return (void *)rd;
}

#define REAL(name, idx) \
    if (!real_fn[idx]) { \
        void *(*rd)(void *, const char *) = (void *(*)(void *, const char *))bootstrap_real_dlsym(); \
        real_fn[idx] = rd ? rd(RTLD_NEXT, #name) : NULL; \
        if (!real_fn[idx]) { fprintf(stderr, "[cumem] MISSING real " #name "\n"); abort(); } }

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    REAL(cuMemAlloc_v2, 0);
    atomic_fetch_add(&n_alloc, 1);
    note_alloc(bytesize);
    return ((CUresult (*)(CUdeviceptr *, size_t))real_fn[0])(dptr, bytesize);
}

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, void *hStream) {
    REAL(cuMemAllocAsync, 1);
    atomic_fetch_add(&n_alloc_async, 1);
    note_alloc(bytesize);
    return ((CUresult (*)(CUdeviceptr *, size_t, void *))real_fn[1])(dptr, bytesize, hStream);
}

CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t w, size_t h, unsigned eb) {
    REAL(cuMemAllocPitch_v2, 2);
    atomic_fetch_add(&n_pitch, 1);
    note_alloc(w * h);
    return ((CUresult (*)(CUdeviceptr *, size_t *, size_t, size_t, unsigned))real_fn[2])(dptr, pPitch, w, h, eb);
}

CUresult cuMemFree_v2(CUdeviceptr dptr) {
    REAL(cuMemFree_v2, 3);
    atomic_fetch_add(&n_free, 1);
    return ((CUresult (*)(CUdeviceptr))real_fn[3])(dptr);
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, void *hStream) {
    REAL(cuMemFreeAsync, 4);
    atomic_fetch_add(&n_free_async, 1);
    return ((CUresult (*)(CUdeviceptr, void *))real_fn[4])(dptr, hStream);
}

// ── dlsym interpose: hand out our wrappers when the app looks up the driver ──
void *dlsym(void *handle, const char *symbol) {
    void *(*rd)(void *, const char *) = (void *(*)(void *, const char *))bootstrap_real_dlsym();
    if (!rd) abort();
    if (symbol) {
        if (!strcmp(symbol, "cuMemAlloc_v2")) {
            if (!real_fn[0]) real_fn[0] = rd(handle, symbol);
            return (void *)cuMemAlloc_v2;
        }
        if (!strcmp(symbol, "cuMemAllocAsync")) {
            if (!real_fn[1]) real_fn[1] = rd(handle, symbol);
            return (void *)cuMemAllocAsync;
        }
        if (!strcmp(symbol, "cuMemAllocPitch_v2")) {
            if (!real_fn[2]) real_fn[2] = rd(handle, symbol);
            return (void *)cuMemAllocPitch_v2;
        }
        if (!strcmp(symbol, "cuMemFree_v2")) {
            if (!real_fn[3]) real_fn[3] = rd(handle, symbol);
            return (void *)cuMemFree_v2;
        }
        if (!strcmp(symbol, "cuMemFreeAsync")) {
            if (!real_fn[4]) real_fn[4] = rd(handle, symbol);
            return (void *)cuMemFreeAsync;
        }
    }
    return rd(handle, symbol);
}

__attribute__((constructor)) static void cumem_init(void) {
    const char *t = getenv("CUMEM_THRESH_MB");
    if (t) thresh_bytes = strtoull(t, NULL, 10) * 1024 * 1024;
    fprintf(stderr, "[cumem] counter active (thresh %llu MB)\n",
            thresh_bytes >> 20);
}

__attribute__((destructor)) static void cumem_report(void) {
    static const char *names[7] = {"<1MB", "1-16MB", "16-64MB", "64-128MB",
                                   "128-256MB", "256-512MB", ">=512MB"};
    char buf[1024];
    int off = snprintf(buf, sizeof buf,
        "[cumem] cuMemAlloc_v2=%llu allocAsync=%llu pitch=%llu free=%llu freeAsync=%llu\n"
        "[cumem] bytes_total=%.2fGB max_alloc=%.1fMB >=thresh(%lluMB)=%llu\n[cumem] hist:",
        (unsigned long long)n_alloc, (unsigned long long)n_alloc_async,
        (unsigned long long)n_pitch, (unsigned long long)n_free,
        (unsigned long long)n_free_async,
        bytes_total / 1073741824.0, max_alloc / 1048576.0,
        thresh_bytes >> 20, (unsigned long long)big_count);
    for (int i = 0; i < 7; i++)
        off += snprintf(buf + off, sizeof buf - off, " %s=%llu",
                        names[i], (unsigned long long)hist[i]);
    snprintf(buf + off, sizeof buf - off, "\n");
    fputs(buf, stderr);
    const char *log = getenv("CUMEM_LOG");
    if (log) { FILE *f = fopen(log, "w"); if (f) { fputs(buf, f); fclose(f); } }
}
