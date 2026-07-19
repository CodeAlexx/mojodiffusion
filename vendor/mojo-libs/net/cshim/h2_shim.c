/* h2_shim.c — HTTP/2 server plumbing via nghttp2, exposed to Mojo.
 *
 * Mojo can't register C-ABI callbacks, and nghttp2 is entirely callback-driven,
 * so ALL the protocol work (preface, HPACK, framing, stream state) lives here in
 * C. Mojo only shuttles bytes and pulls/pushes whole requests/responses:
 *
 *   h2_new()                          -> conn handle (queues initial SETTINGS)
 *   h2_recv(conn, buf, len)           feed bytes read from the socket
 *   h2_next_request(conn, &sid, ..., &blen)  pop one completed request
 *   h2_request_body(conn, sid, dst, cap)     copy that request's body bytes
 *   h2_respond(conn, sid, status, body, len, ctype)   queue a response
 *   h2_send(conn, out, cap)           pull bytes to write to the socket
 *   h2_free(conn)
 *
 * Scope: captures :method, :path and the request BODY (DATA frames are
 * accumulated per stream; nghttp2's default auto WINDOW_UPDATE lets bodies
 * larger than the initial flow-control window arrive across reads).
 * Build: cc -c -fPIC -O2 -I<nghttp2-include> cshim/h2_shim.c -o cshim/h2_shim.o
 */
#include <nghttp2/nghttp2.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAXQ 256

typedef struct {
    unsigned char* data;
    size_t len;
    size_t off;
} h2_body;

typedef struct {
    int32_t sid;
    char method[16];
    char path[1024];
    unsigned char* body; /* accumulated request body (DATA frames) */
    size_t body_len;
    size_t body_cap;
    h2_body* resp; /* response body source, freed on stream close */
} h2_req;

typedef struct {
    nghttp2_session* session;
    h2_req* ready[MAXQ]; /* ring of completed requests */
    int rhead;
    int rtail;
} h2_conn;

static nghttp2_nv mknv(const char* name, const char* value) {
    nghttp2_nv nv;
    nv.name = (uint8_t*)name;
    nv.value = (uint8_t*)value;
    nv.namelen = strlen(name);
    nv.valuelen = strlen(value);
    nv.flags = NGHTTP2_NV_FLAG_NONE;
    return nv;
}

/* ── nghttp2 callbacks (all C-side) ─────────────────────────────────────── */

static int on_begin_headers(nghttp2_session* session,
                            const nghttp2_frame* frame, void* user_data) {
    if (frame->hd.type != NGHTTP2_HEADERS ||
        frame->headers.cat != NGHTTP2_HCAT_REQUEST)
        return 0;
    h2_req* r = (h2_req*)calloc(1, sizeof(h2_req));
    r->sid = frame->hd.stream_id;
    nghttp2_session_set_stream_user_data(session, frame->hd.stream_id, r);
    return 0;
}

static int on_header(nghttp2_session* session, const nghttp2_frame* frame,
                     const uint8_t* name, size_t namelen, const uint8_t* value,
                     size_t valuelen, uint8_t flags, void* user_data) {
    (void)flags;
    h2_req* r = (h2_req*)nghttp2_session_get_stream_user_data(
        session, frame->hd.stream_id);
    if (!r)
        return 0;
    if (namelen == 7 && memcmp(name, ":method", 7) == 0) {
        size_t n = valuelen < 15 ? valuelen : 15;
        memcpy(r->method, value, n);
        r->method[n] = 0;
    } else if (namelen == 5 && memcmp(name, ":path", 5) == 0) {
        size_t n = valuelen < 1023 ? valuelen : 1023;
        memcpy(r->path, value, n);
        r->path[n] = 0;
    }
    return 0;
}

/* accumulate request-body DATA chunks onto the stream's h2_req. Called (possibly
 * several times) before on_frame_recv for the same DATA frame, so by the time
 * END_STREAM is seen in on_frame_recv the body is complete. */
static int on_data_chunk_recv(nghttp2_session* session, uint8_t flags,
                              int32_t stream_id, const uint8_t* data,
                              size_t len, void* user_data) {
    (void)flags;
    (void)user_data;
    h2_req* r = (h2_req*)nghttp2_session_get_stream_user_data(session, stream_id);
    if (!r)
        return 0;
    if (r->body_len + len > r->body_cap) {
        size_t nc = r->body_cap ? r->body_cap * 2 : 256;
        while (nc < r->body_len + len)
            nc *= 2;
        unsigned char* nb = (unsigned char*)realloc(r->body, nc);
        if (!nb)
            return NGHTTP2_ERR_CALLBACK_FAILURE;
        r->body = nb;
        r->body_cap = nc;
    }
    memcpy(r->body + r->body_len, data, len);
    r->body_len += len;
    return 0;
}

static int on_frame_recv(nghttp2_session* session, const nghttp2_frame* frame,
                         void* user_data) {
    h2_conn* c = (h2_conn*)user_data;
    /* request complete when the stream is closed by the client (END_STREAM) */
    if ((frame->hd.type == NGHTTP2_HEADERS || frame->hd.type == NGHTTP2_DATA) &&
        (frame->hd.flags & NGHTTP2_FLAG_END_STREAM)) {
        h2_req* r = (h2_req*)nghttp2_session_get_stream_user_data(
            session, frame->hd.stream_id);
        if (r) {
            c->ready[c->rtail] = r;
            c->rtail = (c->rtail + 1) % MAXQ;
        }
    }
    return 0;
}

static int on_stream_close(nghttp2_session* session, int32_t stream_id,
                           uint32_t error_code, void* user_data) {
    (void)error_code;
    (void)user_data;
    h2_req* r = (h2_req*)nghttp2_session_get_stream_user_data(session, stream_id);
    if (r) {
        if (r->resp) {
            free(r->resp->data);
            free(r->resp);
        }
        free(r->body);
        free(r);
        nghttp2_session_set_stream_user_data(session, stream_id, NULL);
    }
    return 0;
}

static ssize_t body_read_cb(nghttp2_session* session, int32_t stream_id,
                            uint8_t* buf, size_t length, uint32_t* data_flags,
                            nghttp2_data_source* source, void* user_data) {
    (void)session;
    (void)stream_id;
    (void)user_data;
    h2_body* b = (h2_body*)source->ptr;
    size_t remain = b->len - b->off;
    size_t n = remain < length ? remain : length;
    memcpy(buf, b->data + b->off, n);
    b->off += n;
    if (b->off >= b->len)
        *data_flags |= NGHTTP2_DATA_FLAG_EOF;
    return (ssize_t)n;
}

/* ── API exposed to Mojo ────────────────────────────────────────────────── */

void* h2_new(void) {
    h2_conn* c = (h2_conn*)calloc(1, sizeof(h2_conn));
    nghttp2_session_callbacks* cbs;
    nghttp2_session_callbacks_new(&cbs);
    nghttp2_session_callbacks_set_on_begin_headers_callback(cbs, on_begin_headers);
    nghttp2_session_callbacks_set_on_header_callback(cbs, on_header);
    nghttp2_session_callbacks_set_on_frame_recv_callback(cbs, on_frame_recv);
    nghttp2_session_callbacks_set_on_data_chunk_recv_callback(cbs, on_data_chunk_recv);
    nghttp2_session_callbacks_set_on_stream_close_callback(cbs, on_stream_close);
    nghttp2_session_server_new(&c->session, cbs, c);
    nghttp2_session_callbacks_del(cbs);
    /* send our SETTINGS (empty is valid) */
    nghttp2_submit_settings(c->session, NGHTTP2_FLAG_NONE, NULL, 0);
    return c;
}

/* feed bytes received from the socket; returns consumed count or <0 on error */
long h2_recv(void* conn, const unsigned char* data, size_t len) {
    h2_conn* c = (h2_conn*)conn;
    return (long)nghttp2_session_mem_recv(c->session, data, len);
}

/* pop one completed request; returns 1 if filled, 0 if none pending.
 * out_blen receives the body length; fetch the bytes with h2_request_body(). */
int h2_next_request(void* conn, int32_t* out_sid, char* method, size_t mcap,
                    char* path, size_t pcap, size_t* out_blen) {
    h2_conn* c = (h2_conn*)conn;
    if (c->rhead == c->rtail)
        return 0;
    h2_req* r = c->ready[c->rhead];
    c->rhead = (c->rhead + 1) % MAXQ;
    *out_sid = r->sid;
    strncpy(method, r->method, mcap - 1);
    method[mcap - 1] = 0;
    strncpy(path, r->path, pcap - 1);
    path[pcap - 1] = 0;
    *out_blen = r->body_len;
    return 1;
}

/* copy a popped request's body into dst[0:cap]; returns the ACTUAL body length
 * (may exceed cap, in which case dst holds a truncated prefix). The h2_req lives
 * until the stream closes (after we respond), so this is valid between
 * h2_next_request() and h2_respond(). */
long h2_request_body(void* conn, int32_t sid, unsigned char* dst, size_t cap) {
    h2_conn* c = (h2_conn*)conn;
    h2_req* r = (h2_req*)nghttp2_session_get_stream_user_data(c->session, sid);
    if (!r)
        return 0;
    size_t n = r->body_len < cap ? r->body_len : cap;
    if (n)
        memcpy(dst, r->body, n);
    return (long)r->body_len;
}

/* queue a response on a stream */
int h2_respond(void* conn, int32_t sid, int status, const unsigned char* body,
               size_t blen, const char* ctype) {
    h2_conn* c = (h2_conn*)conn;
    h2_body* b = (h2_body*)malloc(sizeof(h2_body));
    b->data = (unsigned char*)malloc(blen ? blen : 1);
    memcpy(b->data, body, blen);
    b->len = blen;
    b->off = 0;
    h2_req* r = (h2_req*)nghttp2_session_get_stream_user_data(c->session, sid);
    if (r)
        r->resp = b; /* freed on stream close */

    char statusstr[8];
    snprintf(statusstr, sizeof(statusstr), "%d", status);
    char clen[24];
    snprintf(clen, sizeof(clen), "%zu", blen);
    nghttp2_nv nva[4];
    nva[0] = mknv(":status", statusstr);
    nva[1] = mknv("content-type", ctype);
    nva[2] = mknv("content-length", clen);
    nva[3] = mknv("server", "MojoHttpServer/3.3");

    nghttp2_data_provider prd;
    prd.source.ptr = b;
    prd.read_callback = body_read_cb;
    return nghttp2_submit_response(c->session, sid, nva, 4, &prd);
}

/* pull the next chunk of pending output into out[0:cap]; returns bytes written
 * (0 = none, <0 = error). Emits ONE nghttp2_session_mem_send result per call and
 * copies it whole — the caller (flush) loops until 0. The previous code packed
 * several results into one buffer and broke mid-result when the last didn't fit
 * (`take < n`), but mem_send had already consumed that data, so the tail was lost
 * and any response larger than `cap` was silently truncated. cap (65536) exceeds
 * the max frame size, so a single result always fits and is never split. */
long h2_send(void* conn, unsigned char* out, size_t cap) {
    h2_conn* c = (h2_conn*)conn;
    const uint8_t* data = NULL;
    ssize_t n = nghttp2_session_mem_send(c->session, &data);
    if (n < 0)
        return -1;
    if (n == 0)
        return 0;
    size_t take = ((size_t)n <= cap) ? (size_t)n : cap;
    memcpy(out, data, take);
    return (long)take;
}

/* whether the session still wants to read or write (0 = can close) */
int h2_alive(void* conn) {
    h2_conn* c = (h2_conn*)conn;
    return (nghttp2_session_want_read(c->session) ||
            nghttp2_session_want_write(c->session))
               ? 1
               : 0;
}

void h2_free(void* conn) {
    h2_conn* c = (h2_conn*)conn;
    nghttp2_session_del(c->session);
    free(c);
}
