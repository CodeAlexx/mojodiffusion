/* alpn_shim.c — ALPN "h2" negotiation for the TLS+HTTP/2 server.
 *
 * OpenSSL picks the ALPN protocol via a C callback set on the SSL_CTX; Mojo
 * can't supply a C-ABI callback, so it lives here. nghttp2's helper selects "h2"
 * when the client offers it (else http/1.1). Build needs OpenSSL + nghttp2:
 *   cc -c -fPIC -O2 -I/usr/include -I<nghttp2>/include cshim/alpn_shim.c -o cshim/alpn_shim.o
 */
#include <openssl/ssl.h>
#include <nghttp2/nghttp2.h>

static int alpn_cb(SSL* ssl, const unsigned char** out, unsigned char* outlen,
                   const unsigned char* in, unsigned int inlen, void* arg) {
    (void)ssl;
    (void)arg;
    int r = nghttp2_select_next_protocol((unsigned char**)out, outlen, in, inlen);
    if (r < 0)
        return SSL_TLSEXT_ERR_NOACK; /* no common protocol */
    return SSL_TLSEXT_ERR_OK;
}

/* enable ALPN h2 selection on a server SSL_CTX (pointer passed as void*) */
void tls_set_alpn(void* ctx) {
    SSL_CTX_set_alpn_select_cb((SSL_CTX*)ctx, alpn_cb, NULL);
}

/* after handshake: 1 if the negotiated ALPN protocol is "h2", else 0 */
int tls_is_h2(void* ssl) {
    const unsigned char* p = 0;
    unsigned int len = 0;
    SSL_get0_alpn_selected((SSL*)ssl, &p, &len);
    return (len == 2 && p[0] == 'h' && p[1] == '2') ? 1 : 0;
}
