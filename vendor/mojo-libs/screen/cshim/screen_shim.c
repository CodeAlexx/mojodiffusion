/* screen_shim.c — minimal C floor for X11 screen capture.
 *
 * The repo's "C for the gaps, Mojo for the rest" pattern (cf. http/cshim,
 * net/cshim): Xlib's Display* is opaque and its struct/macros don't pass
 * cleanly across Mojo FFI, so the few Xlib calls live here. Mojo loads this
 * .so, asks for an RGB24 buffer, encodes the PNG, and frees the buffer.
 *
 * Build:
 *   gcc -shared -fPIC -O2 screen_shim.c -o screen_shim.so -lX11
 *
 * ABI (all C, thin):
 *   int  screen_x11_size(int* w, int* h);                 // 0 ok, -1 no display
 *   unsigned char* screen_x11_grab(int x,int y,int w,int h,int* ow,int* oh);
 *       // RGB24, row-major, caller frees with screen_free; NULL on failure.
 *       // w<=0 || h<=0  => full screen. Rect is clamped to the screen.
 *   void screen_free(unsigned char* p);
 */

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdlib.h>

int screen_x11_size(int *w, int *h) {
    Display *d = XOpenDisplay(NULL);
    if (!d) return -1;
    int s = DefaultScreen(d);
    *w = DisplayWidth(d, s);
    *h = DisplayHeight(d, s);
    XCloseDisplay(d);
    return 0;
}

static int mask_shift(unsigned long m) {
    int sh = 0;
    if (!m) return 0;
    while (!(m & 1UL)) { m >>= 1; sh++; }
    return sh;
}

unsigned char *screen_x11_grab(int x, int y, int w, int h, int *ow, int *oh) {
    Display *d = XOpenDisplay(NULL);
    if (!d) return NULL;
    int s = DefaultScreen(d);
    Window root = RootWindow(d, s);
    int sw = DisplayWidth(d, s), sh = DisplayHeight(d, s);

    if (w <= 0 || h <= 0) { x = 0; y = 0; w = sw; h = sh; }
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x + w > sw) w = sw - x;
    if (y + h > sh) h = sh - y;
    if (w <= 0 || h <= 0) { XCloseDisplay(d); return NULL; }

    XImage *img = XGetImage(d, root, x, y, w, h, AllPlanes, ZPixmap);
    if (!img) { XCloseDisplay(d); return NULL; }

    unsigned char *out = (unsigned char *)malloc((size_t)w * (size_t)h * 3);
    if (!out) { XDestroyImage(img); XCloseDisplay(d); return NULL; }

    unsigned long rm = img->red_mask, gm = img->green_mask, bm = img->blue_mask;
    int rsh = mask_shift(rm), gsh = mask_shift(gm), bsh = mask_shift(bm);

    for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
            unsigned long px = XGetPixel(img, i, j);
            size_t o = ((size_t)j * (size_t)w + (size_t)i) * 3;
            out[o + 0] = (unsigned char)((px & rm) >> rsh);
            out[o + 1] = (unsigned char)((px & gm) >> gsh);
            out[o + 2] = (unsigned char)((px & bm) >> bsh);
        }
    }

    *ow = w;
    *oh = h;
    XDestroyImage(img);
    XCloseDisplay(d);
    return out;
}

void screen_free(unsigned char *p) { free(p); }
