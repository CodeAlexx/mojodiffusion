/* http_shim.c — C gap-fillers for the Mojo HTTP server.
 *
 * Filling capability gaps Mojo's stdlib doesn't cover, called from Mojo via
 * external_call. Built to an object and linked into the Mojo binary:
 *     cc -c -fPIC -O2 cshim/http_shim.c -o cshim/http_shim.o
 *     mojo build ... -Xlinker cshim/http_shim.o -Xlinker -lz
 *
 * gzip (zlib): real gzip framing via deflateInit2 with the gzip window flag,
 * which is fiddly to drive from Mojo (the z_stream struct), so it lives here.
 */
#include <zlib.h>
#include <string.h>

/* Safe upper bound on the gzip output size for `srclen` input bytes. */
unsigned long gz_bound(unsigned long srclen) {
    return srclen + (srclen >> 12) + (srclen >> 14) + (srclen >> 25) + 64;
}

/* gzip-compress src[0:srclen] into dst (capacity *dstlen). On return *dstlen is
 * the compressed size. Returns 0 on success, non-zero on error.
 * windowBits = 15 + 16 selects the gzip wrapper (vs zlib/raw deflate). */
int gz_compress(const unsigned char* src, unsigned long srclen,
                unsigned char* dst, unsigned long* dstlen, int level) {
    z_stream s;
    memset(&s, 0, sizeof(s));
    if (deflateInit2(&s, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK)
        return 1;
    s.next_in = (unsigned char*)src;
    s.avail_in = (uInt)srclen;
    s.next_out = dst;
    s.avail_out = (uInt)(*dstlen);
    int r = deflate(&s, Z_FINISH);
    *dstlen = *dstlen - s.avail_out;
    deflateEnd(&s);
    return (r == Z_STREAM_END) ? 0 : 2;
}
