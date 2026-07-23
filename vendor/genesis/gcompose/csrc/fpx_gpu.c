// fpx_gpu.c — OpenCL compute shim for the MojoMedia editor's GPU pipeline.
//
// Replaces the Mojo/MAX (LayoutTensor + DeviceContext + enqueue_function) kernels so
// the editor compiles host-only (no MAX device-codegen → ~5min/60GB build collapses).
// The kernels are RUNTIME-compiled by the OpenCL driver (clBuildProgram at init), so
// there is ZERO GPU codegen in the Mojo/C build AND it runs on any OpenCL device
// (NVIDIA/AMD/Intel) — keeping the portability MAX gave us.
//
// Math is ported 1:1 from editor/mojo_max_reference/{main_editor_max,transitions_max}.mojo.
// Flat index matches Mojo's Layout.row_major(VH,VW,4):  i = (y*VW + x)*4 + c.
//
// Pipeline (one upload, chain on-device, one download — mirrors the Mojo flow):
//   upload base/over/trans (u8 -> f32 on GPU)
//   track1 = transition(base,trans,t,param)  OR  base           (fpx_gpu_track1)
//   in     = composite_pip(track1, over, op, px,py,pw,ph)        (fpx_gpu_pip)
//   mid    = brightness(in);  out = contrast(mid)                (fpx_gpu_grade)
//   look   = lut3d(out)|vhs(out)  (optional)                     (fpx_gpu_look)
//   download final (out|look) -> host f32 (encode) or u8 (draw)
//
// Build:  cc -O2 -fPIC -shared fpx_gpu.c -o libfpxgpu.so -I/usr/local/cuda/include -lOpenCL
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// fixed working resolution (matches the editor's VW/VH/vlayout)
#define GVW 1280
#define GVH 856
#define GPIX (GVW * GVH)
#define GN (GPIX * 4)
#define MAXLUTF (33 * 33 * 33 * 3)

enum { BASE = 0, OVER, TRANS, TRACK1, INB, MID, OUTB, LOOKB, NBUF };

static cl_context     g_ctx = NULL;
static cl_command_queue g_q = NULL;
static cl_program     g_prog = NULL;
static cl_device_id   g_dev = NULL;
static cl_mem g_buf[NBUF];        // VW*VH*4 floats each
static cl_mem g_lut = NULL;       // MAXLUTF floats
static cl_mem g_stage = NULL;     // VW*VH*4 bytes (u8 upload staging)
static cl_mem g_hist = NULL;      // 3*256 int histogram bins
static cl_mem g_grid = NULL;      // 256*256 int scope accumulator (waveform / vectorscope)
static cl_mem g_parade = NULL;    // 3*256*256 int scope accumulator (RGB parade: one panel per channel)
static cl_mem g_scope = NULL;     // 256*256*4 byte rendered scope image
static cl_mem g_tmp = NULL;       // VW*VH*4 float scratch (P2: transform source copy / blur ping-pong)
static cl_mem g_tmp2 = NULL;      // VW*VH*4 float scratch #2 (P9: glow bright-pass + blur partner)
static int g_ready = 0;

// every per-pixel kernel; VW/VH injected as -D build options.
static const char* KSRC =
"#define IDX(x,y) (((y)*VW+(x))*4)\n"
"float clamp01(float v){ return v<0.0f?0.0f:(v>1.0f?1.0f:v); }\n"
// P13 OLD-FILM/DISTORT determinism helper: a pure INTEGER hash of two coords (no time/frame seed,
// no real RNG) so the same input frame ALWAYS produces the same noise — identity/regression gates
// stay stable. 'kx'/'ky' select an independent stream (vary the salt for a 2nd/3rd value). Returns
// a float in [0,1). NB 'hsh' is NOT a reserved OpenCL word (avoid local/global/half/etc).
"float fpx_hash01(int xx,int yy,int kx,int ky){\n"
"  uint hsh=(uint)(xx+kx)*73856093u ^ (uint)(yy+ky)*19349663u; hsh^=hsh>>13; hsh*=2654435761u;\n"
"  return (float)(hsh>>8)*(1.0f/16777216.0f);\n"
"}\n"
// u8 -> f32 [0,1]
"__kernel void k_unpack(__global const uchar* s,__global float* d){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  d[i+0]=(float)s[i+0]/255.0f; d[i+1]=(float)s[i+1]/255.0f; d[i+2]=(float)s[i+2]/255.0f; d[i+3]=(float)s[i+3]/255.0f;\n"
"}\n"
// f32 -> u8 (round)
"__kernel void k_pack(__global const float* s,__global uchar* d){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  d[i+0]=(uchar)(clamp01(s[i+0])*255.0f+0.5f); d[i+1]=(uchar)(clamp01(s[i+1])*255.0f+0.5f);\n"
"  d[i+2]=(uchar)(clamp01(s[i+2])*255.0f+0.5f); d[i+3]=(uchar)(clamp01(s[i+3])*255.0f+0.5f);\n"
"}\n"
"__kernel void k_copy(__global const float* s,__global float* d){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  d[i+0]=s[i+0]; d[i+1]=s[i+1]; d[i+2]=s[i+2]; d[i+3]=s[i+3];\n"
"}\n"
// scope: RGB histogram — 3x256 int bins (R:0..255, G:256..511, B:512..767)
"__kernel void k_hist_clear(__global int* h){ int i=get_global_id(0); if(i<768) h[i]=0; }\n"
"__kernel void k_hist(__global const float* s,__global volatile int* h){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int r=(int)(clamp01(s[i+0])*255.0f+0.5f), g=(int)(clamp01(s[i+1])*255.0f+0.5f), b=(int)(clamp01(s[i+2])*255.0f+0.5f);\n"
"  atomic_inc(&h[r]); atomic_inc(&h[256+g]); atomic_inc(&h[512+b]);\n"
"}\n"
// scope: 256x256 int grid (waveform: col x value ; vectorscope: U x V) + RGBA8 renderers
"__kernel void k_grid_clear(__global int* g){ int i=get_global_id(0); if(i<65536) g[i]=0; }\n"
"__kernel void k_wave_acc(__global const float* s,__global volatile int* w){\n"   // w[col*256+val], luma waveform
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int col=x*256/VW; float luma=clamp01(s[i+0])*0.299f+clamp01(s[i+1])*0.587f+clamp01(s[i+2])*0.114f;\n"
"  int val=(int)(luma*255.0f+0.5f); if(val>255)val=255; atomic_inc(&w[col*256+val]);\n"
"}\n"
"__kernel void k_wave_img(__global const int* w,__global uchar* img,float gain){\n" // 256x256 greenish, y=value (top=bright)
"  int ix=get_global_id(0),iy=get_global_id(1); if(ix>=256||iy>=256) return; int o=(iy*256+ix)*4;\n"
"  int c=w[ix*256+(255-iy)]; float v=(float)c*gain; if(v>1.0f)v=1.0f; uchar g=(uchar)(v*255.0f);\n"
"  img[o+0]=(uchar)((float)g*0.55f); img[o+1]=g; img[o+2]=(uchar)((float)g*0.7f); img[o+3]=255;\n"
"}\n"
"__kernel void k_vec_acc(__global const float* s,__global volatile int* w){\n"  // U/V scatter (BT.601, 0..1 centred at .5)
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=clamp01(s[i+0]),g=clamp01(s[i+1]),b=clamp01(s[i+2]);\n"
"  float u=-0.169f*r-0.331f*g+0.5f*b+0.5f; float v=0.5f*r-0.419f*g-0.081f*b+0.5f;\n"
"  int ux=(int)(u*255.0f+0.5f), vy=(int)(v*255.0f+0.5f); if(ux<0)ux=0; if(ux>255)ux=255; if(vy<0)vy=0; if(vy>255)vy=255;\n"
"  atomic_inc(&w[vy*256+ux]);\n"
"}\n"
"__kernel void k_vec_img(__global const int* w,__global uchar* img,float gain){\n"
"  int ix=get_global_id(0),iy=get_global_id(1); if(ix>=256||iy>=256) return; int o=(iy*256+ix)*4;\n"
"  int c=w[(255-iy)*256+ix]; float v=(float)c*gain; if(v>1.0f)v=1.0f; uchar g=(uchar)(v*255.0f);\n"
"  img[o+0]=(uchar)((float)g*0.5f); img[o+1]=g; img[o+2]=(uchar)((float)g*0.55f); img[o+3]=255;\n"
"}\n"
// RGB PARADE (Triad-B P1): per-channel column waveform, 3 side-by-side panels (R|G|B). The grid is
// channel-major p[c*65536 + col*256 + val] — for each source pixel, accumulate (column, value) per
// channel (col = source x compressed to 0..255). The image renders R into x[0,85), G into [85,170),
// B into [170,256), with value on the y-axis (top = brightest, like Shotcut's 255-value layout).
"__kernel void k_parade_acc(__global const float* s,__global volatile int* p){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int col=x*256/VW;\n"
"  int rv=(int)(clamp01(s[i+0])*255.0f+0.5f); if(rv>255)rv=255;\n"
"  int gv=(int)(clamp01(s[i+1])*255.0f+0.5f); if(gv>255)gv=255;\n"
"  int bv=(int)(clamp01(s[i+2])*255.0f+0.5f); if(bv>255)bv=255;\n"
"  atomic_inc(&p[0*65536+col*256+rv]); atomic_inc(&p[1*65536+col*256+gv]); atomic_inc(&p[2*65536+col*256+bv]);\n"
"}\n"
// 256x256 image: three 85/85/86-wide panels. For output pixel (ix,iy): pick the panel/channel by ix,
// map ix within the panel back to a source column (0..255), read the grid at value (255-iy), scale by
// gain, and paint in that channel's color. A 1px gap between panels (dark) separates the channels.
"__kernel void k_parade_img(__global const int* p,__global uchar* img,float gain){\n"
"  int ix=get_global_id(0),iy=get_global_id(1); if(ix>=256||iy>=256) return; int o=(iy*256+ix)*4;\n"
"  int ch=-1; int loc=0; int pw=85;\n"
"  if(ix<85){ ch=0; loc=ix; pw=85; }\n"
"  else if(ix<170){ ch=1; loc=ix-85; pw=85; }\n"
"  else { ch=2; loc=ix-170; pw=86; }\n"
"  // 1px dark separator at the left edge of the G and B panels.\n"
"  if((ix==85)||(ix==170)){ img[o+0]=8; img[o+1]=10; img[o+2]=14; img[o+3]=255; return; }\n"
"  int col=loc*256/pw; if(col>255)col=255;\n"
"  int val=255-iy;\n"
"  int c=p[ch*65536+col*256+val]; float v=(float)c*gain; if(v>1.0f)v=1.0f; uchar g=(uchar)(v*255.0f);\n"
"  uchar r8=0,g8=0,b8=0;\n"
"  if(ch==0){ r8=g; g8=(uchar)((float)g*0.18f); b8=(uchar)((float)g*0.18f); }\n"
"  else if(ch==1){ r8=(uchar)((float)g*0.18f); g8=g; b8=(uchar)((float)g*0.18f); }\n"
"  else { r8=(uchar)((float)g*0.3f); g8=(uchar)((float)g*0.45f); b8=g; }\n"
"  img[o+0]=r8; img[o+1]=g8; img[o+2]=b8; img[o+3]=255;\n"
"}\n"
"__kernel void k_parade_clear(__global int* p){ int i=get_global_id(0); if(i<3*65536) p[i]=0; }\n"
// composite alpha-over: dst = over*aeff + base*(1-aeff), aeff=over.a*op ; alpha from base
"__kernel void k_composite(__global const float* base,__global const float* over,__global float* dst,float op){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float a=over[i+3]*op, inv=1.0f-a;\n"
"  dst[i+0]=over[i+0]*a+base[i+0]*inv; dst[i+1]=over[i+1]*a+base[i+1]*inv; dst[i+2]=over[i+2]*a+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"}\n"
// P31 BLEND MODES (V2 overlay compositing). Per-channel blend of base `b` and over `o` (both 0..1)
// selected by mode `m`. m==0 (Normal) returns `o` so the alpha-over in k_pip is BYTE-IDENTICAL to the
// pre-P31 plain composite — only modes 1..7 visibly combine base*over. Placed ABOVE k_pip so the
// kernel can call it. Modes mirror Shotcut's qtblend/cairoblend per-clip blend modes.
"float fpx_blend(float b,float o,int m){\n"
"  if(m==1) return b*o;                                   // Multiply\n"
"  if(m==2) return 1.0f-(1.0f-b)*(1.0f-o);                // Screen\n"
"  if(m==3) return b<0.5f ? 2.0f*b*o : 1.0f-2.0f*(1.0f-b)*(1.0f-o); // Overlay\n"
"  if(m==4){ float s=b+o; return s>1.0f?1.0f:s; }         // Add (clamp)\n"
"  if(m==5) return b<o?b:o;                               // Darken\n"
"  if(m==6) return b>o?b:o;                               // Lighten\n"
"  if(m==7){ float d=b-o; return d<0.0f?-d:d; }           // Difference\n"
"  return o;                                              // 0 = Normal\n"
"}\n"
// picture-in-picture: shrink whole over into normalized rect [px,py,pw,ph]. The composite weight is
// over.a * op, so a pixel whose OVER ALPHA was zeroed (e.g. by k_chroma keying out the green screen)
// contributes nothing and the base (track1) shows through — this is what makes the chroma key visible
// after compositing. (Identical to the pre-P4 behaviour when over.a==1 everywhere.)
// P31: the over RGB is first combined with the base through fpx_blend(base,over,blend) per channel,
// THEN the standard alpha-over runs on that blended colour. blend==0 (Normal) => fpx_blend returns the
// over colour unchanged => this is BYTE-IDENTICAL to the pre-P31 composite. Only blend 1..7 alter it.
"__kernel void k_pip(__global const float* base,__global const float* over,__global float* dst,float op,int blend,float px,float py,float pw,float ph){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float fx=(float)x/(float)VW, fy=(float)y/(float)VH;\n"
"  if(pw>0.0f && ph>0.0f && fx>=px && fx<px+pw && fy>=py && fy<py+ph){\n"
"    int sx=(int)((fx-px)/pw*(float)VW), sy=(int)((fy-py)/ph*(float)VH);\n"
"    if(sx<0)sx=0; if(sx>VW-1)sx=VW-1; if(sy<0)sy=0; if(sy>VH-1)sy=VH-1; int si=IDX(sx,sy);\n"
"    float a=over[si+3]*op, inv=1.0f-a;\n"
"    float cr=fpx_blend(base[i+0],over[si+0],blend), cg=fpx_blend(base[i+1],over[si+1],blend), cb=fpx_blend(base[i+2],over[si+2],blend);\n"
"    dst[i+0]=cr*a+base[i+0]*inv; dst[i+1]=cg*a+base[i+1]*inv; dst[i+2]=cb*a+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"  } else { dst[i+0]=base[i+0]; dst[i+1]=base[i+1]; dst[i+2]=base[i+2]; dst[i+3]=base[i+3]; }\n"
"}\n"
// P4 CHROMA KEY (green-screen). Runs IN PLACE on the OVER buffer BEFORE k_pip, when enabled. For each
// OVER pixel we measure a CHROMA-VECTOR distance to the key colour (kr,kg,kb): subtract each colour's
// luma (BT.601) to get its chroma vector (the colour minus its grey component), then take the
// euclidean distance between the pixel's chroma vector and the key's chroma vector. This removes the
// COMMON luma term so two colours that differ only by a flat brightness offset (same RGB ratios, e.g.
// a uniformly darkened patch) compare equal — but it is CHROMA-VECTOR PROXIMITY, not true hue: a
// DESATURATED or much darker shade of the key colour has a shorter chroma vector and so a larger
// distance, and is NOT reliably keyed (a follow-up pass could normalize the chroma vectors / compare
// their direction for hue-true keying on shaded real footage). It is gate-correct for the flat
// full-bright key-over-flat-base case and matches Shotcut's bluescreen0r frei0r 'Distance' threshold
// closely enough for that. dist is the euclidean length of the luma-removed difference, then we map it
// through ck_smoothstep(sim, sim+smth, dist): close to the key (dist<=sim) → 0 (fully keyed/
// transparent), far from the key (dist>=sim+smth) → 1 (opaque), with a soft edge band of width 'smth'.
// We ONLY scale over.a (rgb untouched), so k_pip's `over.a*op` weight then shows the base through the
// keyed pixels. NB var names avoid reserved OpenCL words (no local/global/half/etc).
"float ck_smoothstep(float e0,float e1,float v){\n"
"  float d=e1-e0; if(d<=0.0f){ return v<e0?0.0f:1.0f; }\n"
"  float tnorm=(v-e0)/d; if(tnorm<0.0f)tnorm=0.0f; if(tnorm>1.0f)tnorm=1.0f;\n"
"  return tnorm*tnorm*(3.0f-2.0f*tnorm);\n"
"}\n"
// NB param 'smth' (NOT 'smooth') deliberately avoids any chance of clashing with a reserved/qualifier
// word in a strict OpenCL-C compiler — a reserved var name = clBuildProgram FAIL = all rendering dead.
// P37: `spill` (>0) adds GREEN-SPILL SUPPRESSION after the alpha key — it pulls a kept pixel's GREEN
// channel down toward max(r,b) to remove the key-colour tint that bled onto the subject's edges
// (Shotcut spillsuppress / keyspillm0pup). It runs ONLY for a green-dominant key (kg>kr && kg>kb) and
// ONLY when over[i+1] exceeds max(pr,pb). spill==0 (or chroma disabled) => the if is skipped => the
// green channel is unchanged => byte-identical to pre-P37. spill rides the wire as the LAST f32 field.
"__kernel void k_chroma(__global float* over,float kr,float kg,float kb,float sim,float smth,float spill){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float pr=over[i+0], pg=over[i+1], pb=over[i+2];\n"
"  float plum=pr*0.299f+pg*0.587f+pb*0.114f;\n"          // pixel luma (BT.601)
"  float klum=kr*0.299f+kg*0.587f+kb*0.114f;\n"          // key   luma (BT.601)
"  float dr=(pr-plum)-(kr-klum), dg=(pg-plum)-(kg-klum), db=(pb-plum)-(kb-klum);\n" // chroma-vector diff (luma removed)
"  float dist=sqrt(dr*dr+dg*dg+db*db);\n"
"  float afac=ck_smoothstep(sim, sim+smth, dist);\n"     // 0 near key -> keyed, 1 far -> opaque
"  over[i+3]=over[i+3]*afac;\n"                          // scale alpha only; rgb untouched
"  if(spill>0.0f && kg>kr && kg>kb){\n"
"    float m=fmax(pr,pb);\n"
"    if(pg>m) over[i+1]=pg+(m-pg)*spill;\n"
"  }\n"
"}\n"
"__kernel void k_brightness(__global const float* s,__global float* d,float p){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  d[i+0]=clamp01(s[i+0]+p); d[i+1]=clamp01(s[i+1]+p); d[i+2]=clamp01(s[i+2]+p); d[i+3]=s[i+3];\n"
"}\n"
"__kernel void k_contrast(__global const float* s,__global float* d,float p){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  d[i+0]=clamp01((s[i+0]-0.5f)*p+0.5f); d[i+1]=clamp01((s[i+1]-0.5f)*p+0.5f); d[i+2]=clamp01((s[i+2]-0.5f)*p+0.5f); d[i+3]=s[i+3];\n"
"}\n"
// saturation about luma (BT.601), in-place per-pixel: s=0 -> grayscale, s=1 -> identity
"__kernel void k_saturation(__global float* d,float sat){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float luma=d[i+0]*0.299f+d[i+1]*0.587f+d[i+2]*0.114f;\n"
"  d[i+0]=clamp01(luma+(d[i+0]-luma)*sat); d[i+1]=clamp01(luma+(d[i+1]-luma)*sat); d[i+2]=clamp01(luma+(d[i+2]-luma)*sat);\n"
"}\n"
// 3D-LUT trilinear, RED-fastest: idx=((b*N+g)*N+r)*3 ; mixed by amt
"__kernel void k_lut3d(__global const float* s,__global float* d,__global const float* lut,int n,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=clamp01(s[i+0]), g=clamp01(s[i+1]), b=clamp01(s[i+2]); float fn1=(float)(n-1);\n"
"  float fr=r*fn1, fg=g*fn1, fb=b*fn1; int r0=(int)fr,g0=(int)fg,b0=(int)fb;\n"
"  int r1=r0+1,g1=g0+1,b1=b0+1; if(r1>n-1)r1=n-1; if(g1>n-1)g1=n-1; if(b1>n-1)b1=n-1;\n"
"  float dr=fr-(float)r0, dg=fg-(float)g0, db=fb-(float)b0;\n"
"  int i000=((b0*n+g0)*n+r0)*3,i100=((b0*n+g0)*n+r1)*3,i010=((b0*n+g1)*n+r0)*3,i110=((b0*n+g1)*n+r1)*3;\n"
"  int i001=((b1*n+g0)*n+r0)*3,i101=((b1*n+g0)*n+r1)*3,i011=((b1*n+g1)*n+r0)*3,i111=((b1*n+g1)*n+r1)*3;\n"
"  float w000=(1.0f-dr)*(1.0f-dg)*(1.0f-db),w100=dr*(1.0f-dg)*(1.0f-db),w010=(1.0f-dr)*dg*(1.0f-db),w110=dr*dg*(1.0f-db);\n"
"  float w001=(1.0f-dr)*(1.0f-dg)*db,w101=dr*(1.0f-dg)*db,w011=(1.0f-dr)*dg*db,w111=dr*dg*db;\n"
"  for(int c=0;c<3;c++){\n"
"    float v=lut[i000+c]*w000+lut[i100+c]*w100+lut[i010+c]*w010+lut[i110+c]*w110+lut[i001+c]*w001+lut[i101+c]*w101+lut[i011+c]*w011+lut[i111+c]*w111;\n"
"    float orig=s[i+c]; d[i+c]=clamp01(orig+(v-orig)*amt);\n"
"  }\n"
"  d[i+3]=s[i+3];\n"
"}\n"
// procedural VHS (sin-hash noise — approximate parity vs MAX, sin backend differs)
"__kernel void k_vhs(__global const float* s,__global float* d,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int sh=(int)(1.0f+amt*4.0f); int xr=x-sh; if(xr<0)xr=0; int xb=x+sh; if(xb>VW-1)xb=VW-1;\n"
"  float rr=s[IDX(xr,y)+0], gg=s[i+1], bb=s[IDX(xb,y)+2];\n"
"  float sl=1.0f; if((y%2)==0) sl=1.0f-0.20f*amt;\n"
"  float hsh=sin((float)x*12.9898f+(float)y*78.233f)*43758.5453f; float noise=(hsh-floor(hsh)-0.5f)*0.14f*amt;\n"
"  d[i+0]=clamp01(rr*sl+noise); d[i+1]=clamp01(gg*sl+noise); d[i+2]=clamp01(bb*sl+noise); d[i+3]=s[i+3];\n"
"}\n"
// ---- transitions (t in [0,1]; alpha from base) ----
"__kernel void k_crossfade(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y); float w=t,inv=1.0f-w;\n"
"  dst[i+0]=over[i+0]*w+base[i+0]*inv; dst[i+1]=over[i+1]*w+base[i+1]*inv; dst[i+2]=over[i+2]*w+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_wipe_lr(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float edge=t*(float)VW; float w=(edge-(float)x)/36.0f+0.5f; if(w<0.0f)w=0.0f; if(w>1.0f)w=1.0f; float inv=1.0f-w;\n"
"  dst[i+0]=over[i+0]*w+base[i+0]*inv; dst[i+1]=over[i+1]*w+base[i+1]*inv; dst[i+2]=over[i+2]*w+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_wipe_rl(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float edge=t*(float)VW; float xr=(float)(VW-1)-(float)x; float w=(edge-xr)/36.0f+0.5f; if(w<0.0f)w=0.0f; if(w>1.0f)w=1.0f; float inv=1.0f-w;\n"
"  dst[i+0]=over[i+0]*w+base[i+0]*inv; dst[i+1]=over[i+1]*w+base[i+1]*inv; dst[i+2]=over[i+2]*w+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_wipe_up(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float edge=t*(float)VH; float yr=(float)(VH-1)-(float)y; float w=(edge-yr)/36.0f+0.5f; if(w<0.0f)w=0.0f; if(w>1.0f)w=1.0f; float inv=1.0f-w;\n"
"  dst[i+0]=over[i+0]*w+base[i+0]*inv; dst[i+1]=over[i+1]*w+base[i+1]*inv; dst[i+2]=over[i+2]*w+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_wipe_down(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float edge=t*(float)VH; float w=(edge-(float)y)/36.0f+0.5f; if(w<0.0f)w=0.0f; if(w>1.0f)w=1.0f; float inv=1.0f-w;\n"
"  dst[i+0]=over[i+0]*w+base[i+0]*inv; dst[i+1]=over[i+1]*w+base[i+1]*inv; dst[i+2]=over[i+2]*w+base[i+2]*inv; dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_slide_lr(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float off=(1.0f-t)*(float)VW; int sx=(int)((float)x-off+0.5f);\n"
"  if(sx>=0&&sx<VW){ int si=IDX(sx,y); dst[i+0]=over[si+0]; dst[i+1]=over[si+1]; dst[i+2]=over[si+2]; }\n"
"  else { dst[i+0]=base[i+0]; dst[i+1]=base[i+1]; dst[i+2]=base[i+2]; }\n"
"  dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_zoom(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float cx=(float)VW*0.5f, cy=(float)VH*0.5f, scale=0.25f+0.75f*t;\n"
"  int sx=(int)(cx+((float)x-cx)/scale+0.5f), sy=(int)(cy+((float)y-cy)/scale+0.5f);\n"
"  if(sx>=0&&sx<VW&&sy>=0&&sy<VH){ int si=IDX(sx,sy); dst[i+0]=over[si+0]; dst[i+1]=over[si+1]; dst[i+2]=over[si+2]; }\n"
"  else { dst[i+0]=base[i+0]; dst[i+1]=base[i+1]; dst[i+2]=base[i+2]; }\n"
"  dst[i+3]=base[i+3];\n"
"}\n"
"__kernel void k_dissolve(__global const float* base,__global const float* over,__global float* dst,float t,float power){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float p=power; if(p<1.0f)p=1.0f; float br=base[i+0],bg=base[i+1],bb=base[i+2];\n"
"  float luma=br*0.299f+bg*0.587f+bb*0.114f; float w=(t*(1.0f+1.0f/p)-luma)*p; if(w<0.0f)w=0.0f; if(w>1.0f)w=1.0f; float inv=1.0f-w;\n"
"  dst[i+0]=over[i+0]*w+br*inv; dst[i+1]=over[i+1]*w+bg*inv; dst[i+2]=over[i+2]*w+bb*inv; dst[i+3]=base[i+3];\n"
"}\n"
// P36 luma-wipe transitions (kind 8/9/10). t in [0,1]; alpha from base. 'nx'/'ny'/'r'/'ang'/'a'/'in' not reserved.
// IRIS (8): circular reveal from the centre — inside a growing circle shows OVER, outside stays BASE.
"__kernel void k_iris(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float nx=((float)x/(float)VW-0.5f), ny=((float)y/(float)VH-0.5f);\n"
"  float r=sqrt(nx*nx+ny*ny); int in = r <= t*0.72f;\n"
"  dst[i+0]=in?over[i+0]:base[i+0]; dst[i+1]=in?over[i+1]:base[i+1]; dst[i+2]=in?over[i+2]:base[i+2]; dst[i+3]=base[i+3];\n"
"}\n"
// CLOCK (9): clockwise angular wipe from the top — the swept wedge (angle fraction <= t) shows OVER.
"__kernel void k_clock(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float nx=((float)x/(float)VW-0.5f), ny=((float)y/(float)VH-0.5f);\n"
"  float ang=atan2(nx,-ny); float a=(ang+M_PI_F)/(2.0f*M_PI_F); int in = a <= t;\n"
"  dst[i+0]=in?over[i+0]:base[i+0]; dst[i+1]=in?over[i+1]:base[i+1]; dst[i+2]=in?over[i+2]:base[i+2]; dst[i+3]=base[i+3];\n"
"}\n"
// BARN-DOOR (10): horizontal split opening from the centre — the centred band |x-0.5| <= t/2 shows OVER.
"__kernel void k_barndoor(__global const float* base,__global const float* over,__global float* dst,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float nx=fabs((float)x/(float)VW-0.5f); int in = nx <= t*0.5f;\n"
"  dst[i+0]=in?over[i+0]:base[i+0]; dst[i+1]=in?over[i+1]:base[i+1]; dst[i+2]=in?over[i+2]:base[i+2]; dst[i+3]=base[i+3];\n"
"}\n"
// ---- P2 color/transform effects (Shotcut-parity) ----
// 3-way color wheels: per channel out = clamp01( pow( clamp01(in*gain + lift), 1/gamma ) ). Identity
// at lift=0, gamma=1, gain=1 (Shotcut color filter lift_r..gain_b defaults). In-place on the OUT
// buffer; alpha untouched. White balance is FOLDED into gain by the UI (engine never sees temp/tint).
// NB: 'gar/gag/gab' = gamma per-channel, 'gnr/gng/gnb' = gain per-channel (NOT reserved words).
"__kernel void k_lgg(__global float* d,float lr,float lg,float lb,float gar,float gag,float gab,float gnr,float gng,float gnb){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float er=1.0f/(gar>1e-3f?gar:1e-3f), eg=1.0f/(gag>1e-3f?gag:1e-3f), eb=1.0f/(gab>1e-3f?gab:1e-3f);\n"
"  d[i+0]=clamp01(pow(clamp01(d[i+0]*gnr+lr),er));\n"
"  d[i+1]=clamp01(pow(clamp01(d[i+1]*gng+lg),eg));\n"
"  d[i+2]=clamp01(pow(clamp01(d[i+2]*gnb+lb),eb));\n"
"}\n"
// P5 CURVE: a 5-point master tone curve. 'ys' are the outputs at fixed inputs 0,.25,.5,.75,1; the
// input is mapped via piecewise-linear interpolation, applied to all 3 channels in place on OUTB.
// Identity (ys = 0,.25,.5,.75,1) is a no-op (caller skips). Runs AFTER blur, BEFORE look.
"__kernel void k_curve(__global float* d,float y0,float y1,float y2,float y3,float y4){\n"
"  int x=get_global_id(0),gy=get_global_id(1); if(x>=VW||gy>=VH) return; int i=IDX(x,gy);\n"
"  float ys[5]; ys[0]=y0; ys[1]=y1; ys[2]=y2; ys[3]=y3; ys[4]=y4;\n"
"  for(int c=0;c<3;c++){\n"
"    float v=clamp01(d[i+c])*4.0f; int seg=(int)v; if(seg>3)seg=3; float f=v-(float)seg;\n"
"    d[i+c]=clamp01(ys[seg]+(ys[seg+1]-ys[seg])*f);\n"
"  }\n"
"}\n"
// ---- P6 STYLIZE/UTILITY filters (Shotcut-parity). All run on the composited OUTB AFTER the curve,
// BEFORE the look, gated/skipped at their no-op defaults so an unfiltered clip is byte-identical.
// SIMPLE-FX (in place on OUTB): kind 1 invert (1-rgb), 2 sepia (BT-ish matrix), 3 grayscale (BT.601
// luma broadcast), 4 posterize (~6 quantization levels). kind 0 never reaches here (caller skips).
// 'kind' is an int, not a reserved word; alpha untouched throughout.
"__kernel void k_simplefx(__global float* d,int kind){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=clamp01(d[i+0]), g=clamp01(d[i+1]), b=clamp01(d[i+2]);\n"
"  if(kind==1){ d[i+0]=1.0f-r; d[i+1]=1.0f-g; d[i+2]=1.0f-b; }\n"
"  else if(kind==2){\n"  // sepia (standard sepia matrix)
"    float sr=r*0.393f+g*0.769f+b*0.189f; float sg=r*0.349f+g*0.686f+b*0.168f; float sb=r*0.272f+g*0.534f+b*0.131f;\n"
"    d[i+0]=clamp01(sr); d[i+1]=clamp01(sg); d[i+2]=clamp01(sb);\n"
"  }\n"
"  else if(kind==3){ float lum=r*0.299f+g*0.587f+b*0.114f; d[i+0]=lum; d[i+1]=lum; d[i+2]=lum; }\n"
"  else if(kind==4){\n"  // posterize to ~6 levels per channel
"    float lev=6.0f; float lm1=lev-1.0f;\n"
"    d[i+0]=clamp01(floor(r*lm1+0.5f)/lm1); d[i+1]=clamp01(floor(g*lm1+0.5f)/lm1); d[i+2]=clamp01(floor(b*lm1+0.5f)/lm1);\n"
"  }\n"
"}\n"
// VIGNETTE (in place on OUTB): radial edge darken. 'dist' = distance of the pixel from the image
// centre normalized so a corner is ~1.0; factor=1-amt*smoothstep(0.5,0.95,dist); rgb*=factor. amt<=0
// never reaches here (caller skips). Reuses ck_smoothstep (defined above) for the soft falloff.
"__kernel void k_vignette(__global float* d,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float cx=(float)VW*0.5f, cy=(float)VH*0.5f;\n"
"  float dx=((float)x+0.5f)-cx, dy=((float)y+0.5f)-cy;\n"
"  float maxr=sqrt(cx*cx+cy*cy);\n"
"  float dist=sqrt(dx*dx+dy*dy)/(maxr>1e-4f?maxr:1e-4f);\n"
"  float factor=1.0f-amt*ck_smoothstep(0.5f,0.95f,dist); if(factor<0.0f)factor=0.0f;\n"
"  d[i+0]=clamp01(d[i+0]*factor); d[i+1]=clamp01(d[i+1]*factor); d[i+2]=clamp01(d[i+2]*factor);\n"
"}\n"
// SHARPEN (unsharp): reads source 's' (a copy of OUTB in g_tmp), writes OUTB 'd'. Per channel
// out=center*(1+4a) - 4*neighbour_avg*a, i.e. center*(1+4a) - a*(left+right+up+down). Edge-clamped
// neighbour sampling. amt<=0 never reaches here (caller skips). 's' is the distinct source copy so
// reads see the pre-filter pixels. 'amt'/'cen' are not reserved words.
"__kernel void k_sharpen(__global const float* s,__global float* d,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int xl=x-1; if(xl<0)xl=0; int xr=x+1; if(xr>VW-1)xr=VW-1;\n"
"  int yu=y-1; if(yu<0)yu=0; int yd=y+1; if(yd>VH-1)yd=VH-1;\n"
"  int il=IDX(xl,y), ir=IDX(xr,y), iu=IDX(x,yu), id=IDX(x,yd);\n"
"  for(int c=0;c<3;c++){\n"
"    float cen=s[i+c]; float nb=s[il+c]+s[ir+c]+s[iu+c]+s[id+c];\n"
"    d[i+c]=clamp01(cen*(1.0f+4.0f*amt) - nb*amt);\n"
"  }\n"
"  d[i+3]=s[i+3];\n"
"}\n"
// FLIP (mirror): reads source 's' (a copy of OUTB in g_tmp), writes OUTB 'd' by sampling 's' at the
// flipped coordinate. mode 1 -> x mirrored (W-1-x), 2 -> y mirrored (H-1-y), 3 -> both. mode 0 never
// reaches here (caller skips). 'mode'/'sxf'/'syf' are not reserved words.
"__kernel void k_flip(__global const float* s,__global float* d,int mode){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int sxf=x, syf=y;\n"
"  if(mode==1){ sxf=VW-1-x; }\n"
"  else if(mode==2){ syf=VH-1-y; }\n"
"  else if(mode==3){ sxf=VW-1-x; syf=VH-1-y; }\n"
"  int si=IDX(sxf,syf);\n"
"  d[i+0]=s[si+0]; d[i+1]=s[si+1]; d[i+2]=s[si+2]; d[i+3]=s[si+3];\n"
"}\n"
// transform: rotate (degrees) + uniform scale about the image center, BILINEAR sampling of source 's'
// into 'd'. Inverse map: for each dst pixel, undo the scale (divide) and rotation (rotate by -rot),
// sample s. Out-of-bounds -> transparent black. Identity at rot=0, scale=1. Reads s, writes d (NEVER
// the same buffer in place — caller passes a distinct source copy). 'rot_deg'/'scl' are NOT reserved.
"__kernel void k_transform(__global const float* s,__global float* d,float rot_deg,float scl,int w,int h){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int di=IDX(x,y);\n"
"  float cx=(float)w*0.5f, cy=(float)h*0.5f;\n"
"  float sf=(scl>1e-4f?scl:1e-4f);\n"
"  float rad=-rot_deg*0.01745329252f; float cs=cos(rad), sn=sin(rad);\n"   // inverse rotation: -rot
"  float dx=((float)x+0.5f)-cx, dy=((float)y+0.5f)-cy;\n"
"  float rx=(dx*cs - dy*sn)/sf, ry=(dx*sn + dy*cs)/sf;\n"                   // inverse: rotate then /scale
"  float sx=rx+cx-0.5f, sy=ry+cy-0.5f;\n"
"  int x0=(int)floor(sx), y0=(int)floor(sy); int x1=x0+1, y1=y0+1;\n"
"  float fx=sx-(float)x0, fy=sy-(float)y0;\n"
"  if(x1<0||y1<0||x0>w-1||y0>h-1){ d[di+0]=0.0f; d[di+1]=0.0f; d[di+2]=0.0f; d[di+3]=0.0f; return; }\n"
"  int cx0=x0<0?0:(x0>w-1?w-1:x0), cx1=x1<0?0:(x1>w-1?w-1:x1);\n"
"  int cy0=y0<0?0:(y0>h-1?h-1:y0), cy1=y1<0?0:(y1>h-1?h-1:y1);\n"
"  int i00=IDX(cx0,cy0), i10=IDX(cx1,cy0), i01=IDX(cx0,cy1), i11=IDX(cx1,cy1);\n"
"  float w00=(1.0f-fx)*(1.0f-fy), w10=fx*(1.0f-fy), w01=(1.0f-fx)*fy, w11=fx*fy;\n"
"  for(int c=0;c<4;c++){ d[di+c]=s[i00+c]*w00+s[i10+c]*w10+s[i01+c]*w01+s[i11+c]*w11; }\n"
"}\n"
// separable gaussian blur, horizontal pass: read s, write d. 'sigma'<=0 => copy. radius=ceil(2*sigma)
// capped at 32; weights exp(-x^2/(2 sigma^2)) normalized. Edge clamps the sample coordinate. Vertical
// pass below mirrors it on the y-axis. 'sig'/'rad'/'wsum' are NOT reserved words.
"__kernel void k_blur_h(__global const float* s,__global float* d,float sigma){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  if(sigma<=0.0f){ d[i+0]=s[i+0]; d[i+1]=s[i+1]; d[i+2]=s[i+2]; d[i+3]=s[i+3]; return; }\n"
"  int rad=(int)ceil(2.0f*sigma); if(rad>32)rad=32; if(rad<1)rad=1;\n"
"  float inv2s2=1.0f/(2.0f*sigma*sigma);\n"
"  float acc0=0.0f,acc1=0.0f,acc2=0.0f,wsum=0.0f;\n"
"  for(int k=-rad;k<=rad;k++){\n"
"    int xx=x+k; if(xx<0)xx=0; if(xx>VW-1)xx=VW-1; int si=IDX(xx,y);\n"
"    float wgt=exp(-(float)(k*k)*inv2s2);\n"
"    acc0+=s[si+0]*wgt; acc1+=s[si+1]*wgt; acc2+=s[si+2]*wgt; wsum+=wgt;\n"
"  }\n"
"  float invw=1.0f/wsum; d[i+0]=acc0*invw; d[i+1]=acc1*invw; d[i+2]=acc2*invw; d[i+3]=s[i+3];\n"
"}\n"
"__kernel void k_blur_v(__global const float* s,__global float* d,float sigma){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  if(sigma<=0.0f){ d[i+0]=s[i+0]; d[i+1]=s[i+1]; d[i+2]=s[i+2]; d[i+3]=s[i+3]; return; }\n"
"  int rad=(int)ceil(2.0f*sigma); if(rad>32)rad=32; if(rad<1)rad=1;\n"
"  float inv2s2=1.0f/(2.0f*sigma*sigma);\n"
"  float acc0=0.0f,acc1=0.0f,acc2=0.0f,wsum=0.0f;\n"
"  for(int k=-rad;k<=rad;k++){\n"
"    int yy=y+k; if(yy<0)yy=0; if(yy>VH-1)yy=VH-1; int si=IDX(x,yy);\n"
"    float wgt=exp(-(float)(k*k)*inv2s2);\n"
"    acc0+=s[si+0]*wgt; acc1+=s[si+1]*wgt; acc2+=s[si+2]*wgt; wsum+=wgt;\n"
"  }\n"
"  float invw=1.0f/wsum; d[i+0]=acc0*invw; d[i+1]=acc1*invw; d[i+2]=acc2*invw; d[i+3]=s[i+3];\n"
"}\n"
// ---- P7 COLOR filters (Shotcut-parity). Both run on the composited OUTB AFTER the P6 filters (flip),
// BEFORE the look, in place per-pixel (own pixel only), in the pinned order HSL -> LEVELS. Each is a
// no-op at its identity default (HSL hue0/sat1/light0 ; LEVELS inb0/inw1/gam1) so the caller skips it
// and an unfiltered clip is byte-identical. NB: all var names avoid reserved OpenCL words (no
// local/global/half/double/kernel/constant/uniform/...); reserved-word var = clBuildProgram FAIL.
// HSL ADJUST: RGB->HSL (standard hexcone), then hue += hue_deg (wrapped mod 360), saturation *= sat,
// lightness += light, then HSL->RGB and clamp01. The hue/sat/lightness are the usual definitions:
//   maxc=max(r,g,b), minc=min(r,g,b), chr=maxc-minc ; lightness L=(maxc+minc)/2 ;
//   saturation S = chr / (1 - |2L-1|) (0 when chr==0) ; hue H from which channel is max (degrees).
// Reconstruction uses chr2 = (1-|2L'-1|)*S' and a per-channel hue ramp, matching the inverse.
"__kernel void k_hsl(__global float* d,float hue_deg,float sat,float light){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=clamp01(d[i+0]), g=clamp01(d[i+1]), b=clamp01(d[i+2]);\n"
"  float maxc=r; if(g>maxc)maxc=g; if(b>maxc)maxc=b;\n"
"  float minc=r; if(g<minc)minc=g; if(b<minc)minc=b;\n"
"  float chr=maxc-minc;\n"
"  float lgt=(maxc+minc)*0.5f;\n"
"  float hue=0.0f;\n"
"  if(chr>1e-6f){\n"
"    if(maxc==r){ hue=fmod((g-b)/chr,6.0f); }\n"
"    else if(maxc==g){ hue=(b-r)/chr+2.0f; }\n"
"    else { hue=(r-g)/chr+4.0f; }\n"
"    hue*=60.0f; if(hue<0.0f) hue+=360.0f;\n"
"  }\n"
"  float satr=0.0f;\n"
"  float denom=1.0f-fabs(2.0f*lgt-1.0f);\n"
"  if(denom>1e-6f) satr=chr/denom;\n"
"  hue+=hue_deg; hue=fmod(hue,360.0f); if(hue<0.0f) hue+=360.0f;\n"
"  satr*=sat; if(satr<0.0f) satr=0.0f; if(satr>1.0f) satr=1.0f;\n"
"  lgt+=light; if(lgt<0.0f) lgt=0.0f; if(lgt>1.0f) lgt=1.0f;\n"
"  float chr2=(1.0f-fabs(2.0f*lgt-1.0f))*satr;\n"
"  float hp=hue/60.0f;\n"
"  float xc=chr2*(1.0f-fabs(fmod(hp,2.0f)-1.0f));\n"
"  float mm=lgt-chr2*0.5f;\n"
"  float rr=0.0f,gg=0.0f,bb=0.0f;\n"
"  if(hp<1.0f){ rr=chr2; gg=xc; bb=0.0f; }\n"
"  else if(hp<2.0f){ rr=xc; gg=chr2; bb=0.0f; }\n"
"  else if(hp<3.0f){ rr=0.0f; gg=chr2; bb=xc; }\n"
"  else if(hp<4.0f){ rr=0.0f; gg=xc; bb=chr2; }\n"
"  else if(hp<5.0f){ rr=xc; gg=0.0f; bb=chr2; }\n"
"  else { rr=chr2; gg=0.0f; bb=xc; }\n"
"  d[i+0]=clamp01(rr+mm); d[i+1]=clamp01(gg+mm); d[i+2]=clamp01(bb+mm);\n"
"}\n"
// LEVELS: per channel out = clamp01( pow( clamp01((c-in_black)/max(in_white-in_black,1e-3)),
// 1/max(gamma,1e-3) ) ). Identity in_black=0,in_white=1,gamma=1 => caller skips. 'inb'/'inw'/'gam'
// are NOT reserved words.
"__kernel void k_levels(__global float* d,float in_black,float in_white,float gamma){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float spanv=in_white-in_black; if(spanv<1e-3f) spanv=1e-3f;\n"
"  float ginv=gamma; if(ginv<1e-3f) ginv=1e-3f; ginv=1.0f/ginv;\n"
"  for(int c=0;c<3;c++){\n"
"    float v=clamp01((d[i+c]-in_black)/spanv);\n"
"    d[i+c]=clamp01(pow(v,ginv));\n"
"  }\n"
"}\n"
// ---- P8 STYLIZE-2 filters (Shotcut-parity). Both run on the composited OUTB AFTER the P7 color
// filters (levels), BEFORE the look, in the pinned order MOSAIC -> GRADIENT-MAP. Each is a no-op
// at its default (mosaic block<=1 ; gmap amt<=0) so the caller skips it and an unfiltered clip is
// byte-identical. NB: all var names avoid reserved OpenCL words (no local/global/half/double/
// kernel/constant/uniform/...); 'block' is NOT a reserved word and matches the pinned API.
// MOSAIC (pixelate): reads source 's' (a copy of OUTB in g_tmp), writes OUTB 'd'. For each pixel,
// snap to the block top-left (bx=(x/block)*block, by=(y/block)*block) and copy that single source
// pixel across the whole block — so a block reads ONE source pixel (from the distinct g_tmp copy,
// avoiding the read/write race an in-place mosaic would hit). block<=1 never reaches here (caller
// skips). 'block'/'bx'/'by' are not reserved words.
"__kernel void k_mosaic(__global const float* s,__global float* d,int block){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int bx=(x/block)*block, by=(y/block)*block;\n"
"  if(bx>VW-1)bx=VW-1; if(by>VH-1)by=VH-1; int si=IDX(bx,by);\n"
"  d[i+0]=s[si+0]; d[i+1]=s[si+1]; d[i+2]=s[si+2]; d[i+3]=s[si+3];\n"
"}\n"
// GRADIENT MAP: luma -> colour ramp, IN PLACE on OUTB (per-own-pixel, no scratch). luma=dot(rgb,
// [.299,.587,.114]); mapped=mix(lo,hi,luma) per channel; rgb=mix(rgb,mapped,amt). amt<=0 never
// reaches here (caller skips). 'lr/lg/lb' = shadow colour, 'hr/hg/hb' = highlight colour, 'amt' =
// blend; 'mr/mg/mb' = mapped colour. None are reserved words. Alpha untouched.
"__kernel void k_gmap(__global float* d,float amt,float lr,float lg,float lb,float hr,float hg,float hb){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=clamp01(d[i+0]), g=clamp01(d[i+1]), b=clamp01(d[i+2]);\n"
"  float luma=r*0.299f+g*0.587f+b*0.114f;\n"
"  float mr=lr+(hr-lr)*luma, mg=lg+(hg-lg)*luma, mb=lb+(hb-lb)*luma;\n"
"  d[i+0]=clamp01(r+(mr-r)*amt); d[i+1]=clamp01(g+(mg-g)*amt); d[i+2]=clamp01(b+(mb-b)*amt);\n"
"}\n"
// ---- P9 FX filters (Shotcut-parity). All run on the composited OUTB AFTER the P8 gradient-map,
// BEFORE the look, in the pinned order DENOISE -> GLOW -> RGB-SHIFT. Each is a no-op at its
// default (denoise<=0 ; glow amt<=0 ; rgbshift off<=0) so the caller skips it and an unfiltered
// clip is byte-identical. NB: all var names avoid reserved OpenCL words (no local/global/half/
// double/kernel/constant/uniform/...); a reserved-word var = clBuildProgram FAIL = all rendering
// dead. The reserved-word avoidance below is deliberate (e.g. 'srng'/'spat'/'wgt'/'strength').
// DENOISE: edge-preserving 5x5 BILATERAL smooth. reads source 's' (a copy of OUTB in g_tmp),
// writes OUTB 'd'. For each output pixel, sum the 5x5 neighbourhood weighted by a SPATIAL gaussian
// (distance falloff) times a COLOUR-RANGE gaussian (penalize neighbours whose colour differs from
// the centre — this is what preserves edges). The weighted average is then blended back toward the
// centre by `strength` (0 = identity, 1 = full bilateral). strength<=0 never reaches here (caller
// skips). 'strength'/'srng'/'spat'/'wgt'/'cen*'/'acc*' are NOT reserved words.
"__kernel void k_denoise(__global const float* s,__global float* d,float strength){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float cenr=s[i+0], ceng=s[i+1], cenb=s[i+2];\n"
"  float spat2=2.0f*1.6f*1.6f;\n"        // spatial gaussian variance term (sigma~1.6 over a 5x5)
"  float srng2=2.0f*0.12f*0.12f;\n"      // colour-range gaussian variance term (sigma~0.12)
"  float accr=0.0f,accg=0.0f,accb=0.0f,wsum=0.0f;\n"
"  for(int ky=-2;ky<=2;ky++){\n"
"    int yy=y+ky; if(yy<0)yy=0; if(yy>VH-1)yy=VH-1;\n"
"    for(int kx=-2;kx<=2;kx++){\n"
"      int xx=x+kx; if(xx<0)xx=0; if(xx>VW-1)xx=VW-1; int si=IDX(xx,yy);\n"
"      float nr=s[si+0], ng=s[si+1], nb=s[si+2];\n"
"      float spat=exp(-(float)(kx*kx+ky*ky)/spat2);\n"
"      float dcr=nr-cenr, dcg=ng-ceng, dcb=nb-cenb;\n"
"      float srng=exp(-(dcr*dcr+dcg*dcg+dcb*dcb)/srng2);\n"
"      float wgt=spat*srng;\n"
"      accr+=nr*wgt; accg+=ng*wgt; accb+=nb*wgt; wsum+=wgt;\n"
"    }\n"
"  }\n"
"  float invw=(wsum>1e-6f)?1.0f/wsum:0.0f;\n"
"  float fr=accr*invw, fg=accg*invw, fb=accb*invw;\n"
"  float sclamp=strength; if(sclamp<0.0f)sclamp=0.0f; if(sclamp>1.0f)sclamp=1.0f;\n"
"  d[i+0]=clamp01(cenr+(fr-cenr)*sclamp);\n"
"  d[i+1]=clamp01(ceng+(fg-ceng)*sclamp);\n"
"  d[i+2]=clamp01(cenb+(fb-cenb)*sclamp);\n"
"  d[i+3]=s[i+3];\n"
"}\n"
// GLOW step 1 — BRIGHT-PASS extract: read source 's' (OUTB), write 'd' (g_tmp2). Where the pixel's
// luma exceeds `thr` keep its rgb, else 0. Alpha set to 1 (the extract buffer is blurred then added
// back, so its alpha is irrelevant). 'thr'/'lum' are NOT reserved words.
"__kernel void k_glow_extract(__global const float* s,__global float* d,float thr){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=clamp01(s[i+0]), g=clamp01(s[i+1]), b=clamp01(s[i+2]);\n"
"  float lum=r*0.299f+g*0.587f+b*0.114f;\n"
"  if(lum>thr){ d[i+0]=r; d[i+1]=g; d[i+2]=b; }\n"
"  else { d[i+0]=0.0f; d[i+1]=0.0f; d[i+2]=0.0f; }\n"
"  d[i+3]=1.0f;\n"
"}\n"
// GLOW step 3 — COMBINE: OUTB 'd' += amt * blurred-bright-pass 'g' (the buffer g_glow holds the
// blurred bright pass). Clamped to [0,1]. This brightens dark pixels AROUND a bright region (bloom
// halo). 'amt' is NOT a reserved word.
"__kernel void k_glow_combine(__global float* d,__global const float* g,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  d[i+0]=clamp01(d[i+0]+amt*g[i+0]); d[i+1]=clamp01(d[i+1]+amt*g[i+1]); d[i+2]=clamp01(d[i+2]+amt*g[i+2]);\n"
"}\n"
// RGB-SHIFT (chromatic aberration): reads source 's' (a copy of OUTB in g_tmp), writes OUTB 'd'.
// out.r samples s.r at (x+off,y), out.b samples s.b at (x-off,y), out.g/out.a sample s at (x,y).
// Sample x-coords are clamped to [0,VW-1]. off<=0 never reaches here (caller skips). 'off'/'xr'/'xb'
// are NOT reserved words.
"__kernel void k_rgbshift(__global const float* s,__global float* d,int off){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int xr=x+off; if(xr<0)xr=0; if(xr>VW-1)xr=VW-1;\n"
"  int xb=x-off; if(xb<0)xb=0; if(xb>VW-1)xb=VW-1;\n"
"  d[i+0]=s[IDX(xr,y)+0]; d[i+1]=s[i+1]; d[i+2]=s[IDX(xb,y)+2]; d[i+3]=s[i+3];\n"
"}\n"
// ---- P10 STYLIZE-4 filters (Shotcut-parity). All run on the composited OUTB AFTER the P9 fx
// filters (rgb-shift), BEFORE the look, in the pinned order HALFTONE -> EMBOSS -> EDGE. Each is a
// no-op at its default (halftone cell<=1 ; emboss amt<=0 ; edge mix<=0) so the caller skips it and
// an unfiltered clip is byte-identical. ALL THREE are spatial: the C wrapper copies OUTB->g_tmp
// via kCopy, then the kernel reads the g_tmp source copy ('s') and writes OUTB ('d'). NB: all var
// names avoid reserved OpenCL words (no local/global/half/double/kernel/constant/uniform/...); a
// reserved-word var = clBuildProgram FAIL = all rendering dead. 'cell'/'amt'/'mix' are NOT reserved.
// HALFTONE (dot screen): reads source 's' (a copy of OUTB in g_tmp), writes OUTB 'd'. For each
// pixel find its cell (cx=(x/cell)*cell+cell/2 clamped to [0,VW-1], cy likewise), sample the source
// LUMA at the cell CENTRE, dot radius = (1-luma)*0.5*cell (darker cell centre => bigger dot); the
// distance from the pixel to its cell centre decides black-dot vs white. out.rgb = dist<radius ?
// 0 : 1 (black dots on a white field); out.a passthrough. cell<=1 never reaches here (caller skips).
"__kernel void k_halftone(__global const float* s,__global float* d,int cell){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int hcell=cell/2;\n"  // NB 'half' is a RESERVED OpenCL type keyword; use 'hcell' to avoid clBuildProgram FAIL.
"  int cx=(x/cell)*cell+hcell; if(cx<0)cx=0; if(cx>VW-1)cx=VW-1;\n"
"  int cy=(y/cell)*cell+hcell; if(cy<0)cy=0; if(cy>VH-1)cy=VH-1;\n"
"  int ci=IDX(cx,cy);\n"
"  float cluma=s[ci+0]*0.299f+s[ci+1]*0.587f+s[ci+2]*0.114f;\n"
"  float radius=(1.0f-cluma)*0.5f*(float)cell;\n"
"  float dx=(float)x-(float)cx, dy=(float)y-(float)cy;\n"
"  float dist=sqrt(dx*dx+dy*dy);\n"
"  float v=(dist<radius)?0.0f:1.0f;\n"
"  d[i+0]=v; d[i+1]=v; d[i+2]=v; d[i+3]=s[i+3];\n"
"}\n"
// EMBOSS (directional relief): reads source 's' (a copy of OUTB in g_tmp), writes OUTB 'd'. Per
// channel out = clamp01(0.5 + amt*(centre - NW)), the NW directional difference (s[x,y]-s[x-1,y-1]).
// A flat region (centre==NW) yields ~0.5 (mid-gray relief); an edge yields a light/dark band along
// the NW->SE gradient. Neighbour coords clamped to [0,VW-1]x[0,VH-1]. out.a passthrough. amt<=0
// never reaches here (caller skips). 'amt' is NOT a reserved word.
"__kernel void k_emboss(__global const float* s,__global float* d,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int xn=x-1; if(xn<0)xn=0; int yn=y-1; if(yn<0)yn=0;\n"
"  int ni=IDX(xn,yn);\n"
"  for(int c=0;c<3;c++){\n"
"    d[i+c]=clamp01(0.5f+amt*(s[i+c]-s[ni+c]));\n"
"  }\n"
"  d[i+3]=s[i+3];\n"
"}\n"
// EDGE/SKETCH (Sobel edge-detect mixed back): reads source 's' (a copy of OUTB in g_tmp), writes
// OUTB 'd'. Compute the Sobel gradient on the LUMA of the 3x3 neighbourhood (gx/gy via the standard
// Sobel kernels), magnitude mag=clamp01(hypot(gx,gy)); out.rgb = mix*vec3(mag) + (1-mix)*orig (mix
// toward white-on-dark edges). Neighbour coords clamped to [0,VW-1]x[0,VH-1]. out.a passthrough.
// mix<=0 never reaches here (caller skips). 'mix'/'mag'/'gx'/'gy' are NOT reserved words.
"__kernel void k_edge(__global const float* s,__global float* d,float mix){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int xl=x-1; if(xl<0)xl=0; int xr=x+1; if(xr>VW-1)xr=VW-1;\n"
"  int yu=y-1; if(yu<0)yu=0; int yd=y+1; if(yd>VH-1)yd=VH-1;\n"
"  int i00=IDX(xl,yu), i10=IDX(x,yu), i20=IDX(xr,yu);\n"
"  int i01=IDX(xl,y),               i21=IDX(xr,y);\n"
"  int i02=IDX(xl,yd), i12=IDX(x,yd), i22=IDX(xr,yd);\n"
"  float l00=s[i00+0]*0.299f+s[i00+1]*0.587f+s[i00+2]*0.114f;\n"
"  float l10=s[i10+0]*0.299f+s[i10+1]*0.587f+s[i10+2]*0.114f;\n"
"  float l20=s[i20+0]*0.299f+s[i20+1]*0.587f+s[i20+2]*0.114f;\n"
"  float l01=s[i01+0]*0.299f+s[i01+1]*0.587f+s[i01+2]*0.114f;\n"
"  float l21=s[i21+0]*0.299f+s[i21+1]*0.587f+s[i21+2]*0.114f;\n"
"  float l02=s[i02+0]*0.299f+s[i02+1]*0.587f+s[i02+2]*0.114f;\n"
"  float l12=s[i12+0]*0.299f+s[i12+1]*0.587f+s[i12+2]*0.114f;\n"
"  float l22=s[i22+0]*0.299f+s[i22+1]*0.587f+s[i22+2]*0.114f;\n"
"  float gx=(l20+2.0f*l21+l22)-(l00+2.0f*l01+l02);\n"
"  float gy=(l02+2.0f*l12+l22)-(l00+2.0f*l10+l20);\n"
"  float mag=clamp01(sqrt(gx*gx+gy*gy));\n"
"  d[i+0]=clamp01(mix*mag+(1.0f-mix)*s[i+0]);\n"
"  d[i+1]=clamp01(mix*mag+(1.0f-mix)*s[i+1]);\n"
"  d[i+2]=clamp01(mix*mag+(1.0f-mix)*s[i+2]);\n"
"  d[i+3]=s[i+3];\n"
"}\n"
// ---- P13 OLD-FILM/DISTORT filters (Shotcut-parity). All run on the composited OUTB AFTER the P10
// stylize-4 filters (edge), BEFORE the look, in the pinned order GRAIN -> SCRATCHES -> DIFFUSION.
// Each is a no-op at its default (grain<=0 ; scratches<=0 ; diffusion radius<=0) so the caller skips
// it and an unfiltered clip is byte-identical. ALL THREE are spatial: the C wrapper copies OUTB->
// g_tmp via kCopy, then the kernel reads the g_tmp source copy ('s') and writes OUTB ('d'). The
// pseudo-randomness is a DETERMINISTIC integer hash of the pixel coords (fpx_hash01 above) — same
// input frame => same output, so the regression gates are stable. NB: all var names avoid reserved
// OpenCL words (no local/global/half/double/kernel/constant/uniform/...); a reserved-word var =
// clBuildProgram FAIL = all rendering dead. 'amt'/'radius'/'noise' are NOT reserved words.
// GRAIN (film noise): reads source 's' (a copy of OUTB in g_tmp), writes OUTB 'd'. A single LUMA
// noise value n=(hash(x,y)*2-1)*amt (same on all 3 channels => achromatic film grain) is ADDED to
// each rgb channel and clamped; alpha passthrough. amt<=0 never reaches here (caller skips). At
// amt>0 the per-pixel n spreads a flat frame's values, so the output std rises from ~0.
"__kernel void k_grain(__global const float* s,__global float* d,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float noise=(fpx_hash01(x,y,0,0)*2.0f-1.0f)*amt;\n"
"  d[i+0]=clamp01(s[i+0]+noise); d[i+1]=clamp01(s[i+1]+noise); d[i+2]=clamp01(s[i+2]+noise); d[i+3]=s[i+3];\n"
"}\n"
// SCRATCHES (old-film vertical lines): reads source 's' (a copy of OUTB in g_tmp), writes OUTB 'd'.
// A whole COLUMN is a scratch when hash(x,0) < amt*0.06 (density scales with amt). On a scratch
// column the column-wide signed offset (hash(x,7)-0.5)*0.9 is ADDED to every rgb (a bright or dark
// vertical line); off a scratch column the pixel passes through. alpha passthrough. amt<=0 never
// reaches here (caller skips). At amt=1 a few columns flip => the column-mean variance rises.
"__kernel void k_scratches(__global const float* s,__global float* d,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float ishit=fpx_hash01(x,0,0,0);\n"
"  if(ishit < amt*0.06f){\n"
"    float off=(fpx_hash01(x,7,0,0)-0.5f)*0.9f;\n"
"    d[i+0]=clamp01(s[i+0]+off); d[i+1]=clamp01(s[i+1]+off); d[i+2]=clamp01(s[i+2]+off);\n"
"  } else { d[i+0]=s[i+0]; d[i+1]=s[i+1]; d[i+2]=s[i+2]; }\n"
"  d[i+3]=s[i+3];\n"
"}\n"
// DIFFUSION (frosted-glass jitter): reads source 's' (a copy of OUTB in g_tmp), writes OUTB 'd'. For
// each pixel pick a deterministic neighbour offset within +/- r: dx=(hash(x,y)*2-1)*r, dy=(hash(x,
// y+101)*2-1)*r (the 2nd stream is salted by +101 so dx/dy are independent), then SAMPLE the source
// at the clamped neighbour (x+dx,y+dy). radius<=0 never reaches here (caller skips). At radius=8 a
// sharp edge gets sampled from jittered neighbours => the boundary transition widens/softens.
"__kernel void k_diffuse(__global const float* s,__global float* d,float radius){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int r=(int)(radius+0.5f);\n"
"  int dx=(int)((fpx_hash01(x,y,0,0)*2.0f-1.0f)*(float)r);\n"
"  int dy=(int)((fpx_hash01(x,y,0,101)*2.0f-1.0f)*(float)r);\n"
"  int sx=x+dx; if(sx<0)sx=0; if(sx>VW-1)sx=VW-1;\n"
"  int sy=y+dy; if(sy<0)sy=0; if(sy>VH-1)sy=VH-1;\n"
"  int si=IDX(sx,sy);\n"
"  d[i+0]=s[si+0]; d[i+1]=s[si+1]; d[i+2]=s[si+2]; d[i+3]=s[si+3];\n"
"}\n"
// ---- P16 DISTORT filters (Shotcut-parity). All run on the composited OUTB AFTER the P13 old-film
// filters (diffusion), BEFORE the look, in the pinned order WAVE -> SWIRL -> THRESHOLD. Each is a
// no-op at its default (wave amp<=0 ; swirl strength<=0 ; threshold level<=0) so the caller skips it
// and an unfiltered clip is byte-identical. WAVE and SWIRL are SPATIAL: the C wrapper copies OUTB->
// g_tmp via kCopy, then the kernel reads the g_tmp source copy ('s') and writes OUTB ('d').
// THRESHOLD is PER-PIXEL IN PLACE on OUTB (own pixel only, no scratch). NB: all var names avoid
// reserved OpenCL words (no local/global/half/double/kernel/constant/uniform/image2d_t/...); a
// reserved-word var = clBuildProgram FAIL = all rendering dead. 'amp'/'strength'/'level' are NOT
// reserved words.
// WAVE (horizontal sinusoidal row displacement): reads source 's' (a copy of OUTB in g_tmp), writes
// OUTB 'd'. Each row y is shifted horizontally by sin(y*0.05)*amp; the source x is the FLOATING shift
// position sampled with BILINEAR interpolation between the two bracketing columns (x clamped to
// [0,VW-1]), so a straight vertical edge becomes a smooth per-row wavy boundary. amp<=0 never reaches
// here (caller skips). 'amp'/'sxf'/'x0'/'x1'/'fx' are NOT reserved words.
"__kernel void k_wave(__global const float* s,__global float* d,float amp){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float sxf=(float)x + sin((float)y*0.05f)*amp;\n"
"  if(sxf<0.0f)sxf=0.0f; if(sxf>(float)(VW-1))sxf=(float)(VW-1);\n"
"  int x0=(int)sxf; int x1=x0+1; if(x1>VW-1)x1=VW-1; float fx=sxf-(float)x0;\n"
"  int i0=IDX(x0,y), i1=IDX(x1,y);\n"
"  d[i+0]=s[i0+0]*(1.0f-fx)+s[i1+0]*fx; d[i+1]=s[i0+1]*(1.0f-fx)+s[i1+1]*fx;\n"
"  d[i+2]=s[i0+2]*(1.0f-fx)+s[i1+2]*fx; d[i+3]=s[i0+3]*(1.0f-fx)+s[i1+3]*fx;\n"
"}\n"
// SWIRL (rotational distortion around the image centre): reads source 's' (a copy of OUTB in g_tmp),
// writes OUTB 'd'. For each pixel measure its offset (dx,dy) from the centre (cx=VW/2,cy=VH/2) and
// radius r=hypot(dx,dy); the rotation angle falls off linearly from the centre out to the corner:
// ang = strength*(1 - clamp(r/maxr,0,1)) where maxr=hypot(cx,cy). Rotate the offset by -ang to find
// the SOURCE coordinate (sx,sy), clamp to [0,VW-1]x[0,VH-1], nearest-sample g_tmp. So pixels near the
// centre are rotated most and the rim is untouched — a straight edge near the centre curves.
// strength<=0 never reaches here (caller skips). 'strength'/'ang'/'cs'/'sn'/'sx'/'sy' are NOT reserved.
"__kernel void k_swirl(__global const float* s,__global float* d,float strength){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float cx=(float)VW*0.5f, cy=(float)VH*0.5f;\n"
"  float dx=(float)x-cx, dy=(float)y-cy;\n"
"  float r=hypot(dx,dy); float maxr=hypot(cx,cy);\n"
"  float rn=r/(maxr>1e-4f?maxr:1e-4f); if(rn<0.0f)rn=0.0f; if(rn>1.0f)rn=1.0f;\n"
"  float ang=-strength*(1.0f-rn);\n"     // rotate the offset by -ang to find the source
"  float cs=cos(ang), sn=sin(ang);\n"
"  float rx=dx*cs - dy*sn, ry=dx*sn + dy*cs;\n"
"  int sx=(int)(rx+cx+0.5f); if(sx<0)sx=0; if(sx>VW-1)sx=VW-1;\n"
"  int sy=(int)(ry+cy+0.5f); if(sy<0)sy=0; if(sy>VH-1)sy=VH-1;\n"
"  int si=IDX(sx,sy);\n"
"  d[i+0]=s[si+0]; d[i+1]=s[si+1]; d[i+2]=s[si+2]; d[i+3]=s[si+3];\n"
"}\n"
// THRESHOLD (luma binarize): PER-PIXEL IN PLACE on OUTB (own pixel only, no g_tmp scratch). luma=
// dot(rgb,[.299,.587,.114]); v = luma>=level ? 1 : 0; out.rgb=(v,v,v); alpha passthrough. So a varied
// image collapses to pure black/white. level<=0 never reaches here (caller skips). 'level'/'lum'/'v'
// are NOT reserved words.
"__kernel void k_threshold(__global float* d,float level){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float lum=clamp01(d[i+0])*0.299f+clamp01(d[i+1])*0.587f+clamp01(d[i+2])*0.114f;\n"
"  float v=(lum>=level)?1.0f:0.0f;\n"
"  d[i+0]=v; d[i+1]=v; d[i+2]=v;\n"
"}\n"
// ---- P17 GEOMETRIC filters (Shotcut-parity). All run on the composited OUTB AFTER the P16 distort
// filters (threshold), BEFORE the look, in the pinned order LENS -> CROP -> GLITCH. Each is a no-op
// at its default (lens k==0 / crop margin<=0 / glitch maxpx<=0) so the caller skips it and an
// unfiltered clip is byte-identical. LENS and GLITCH are SPATIAL: the C wrapper copies OUTB->g_tmp
// via kCopy, then the kernel reads the g_tmp source copy ('s') and writes OUTB ('d'). CROP is
// PER-PIXEL IN PLACE on OUTB (own pixel only, no scratch). NB: all var names avoid reserved OpenCL
// words (no local/global/private/constant/kernel/uniform/half/double/image2d_t/sampler_t/...); a
// reserved-word var = clBuildProgram FAIL = all rendering dead. 'k'/'margin'/'maxpx' are NOT reserved.
// LENS (radial barrel/pincushion distortion): reads source 's' (a copy of OUTB in g_tmp), writes OUTB
// 'd'. For each OUTPUT pixel (x,y) the normalised radial offset from the centre (cx=VW/2,cy=VH/2) is
// nx=(x-cx)/cx, ny=(y-cy)/cy and r2=nx*nx+ny*ny; the radial scale f=1+k*r2 maps the output position
// back to a SOURCE position srcx=cx+(x-cx)*f, srcy=cy+(y-cy)*f, which is clamped to [0,VW-1]x[0,VH-1]
// and nearest-sampled from g_tmp. k>0 pushes the source outward (barrel: centre magnified), k<0 pulls
// it inward (pincushion). k==0 never reaches here (caller skips — note BOTH signs are active, only the
// exact 0 is the no-op). 'k'/'nx'/'ny'/'r2'/'sclf'/'srcx'/'srcy'/'sx'/'sy' are NOT reserved words.
"__kernel void k_lens(__global const float* s,__global float* d,float k){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float cx=(float)VW*0.5f, cy=(float)VH*0.5f;\n"
"  float nx=((float)x-cx)/cx, ny=((float)y-cy)/cy;\n"
"  float r2=nx*nx+ny*ny;\n"
"  float sclf=1.0f+k*r2;\n"
"  float srcx=cx+((float)x-cx)*sclf;\n"
"  float srcy=cy+((float)y-cy)*sclf;\n"
"  int sx=(int)(srcx+0.5f); if(sx<0)sx=0; if(sx>VW-1)sx=VW-1;\n"
"  int sy=(int)(srcy+0.5f); if(sy<0)sy=0; if(sy>VH-1)sy=VH-1;\n"
"  int si=IDX(sx,sy);\n"
"  d[i+0]=s[si+0]; d[i+1]=s[si+1]; d[i+2]=s[si+2]; d[i+3]=s[si+3];\n"
"}\n"
// CROP (margin to black): PER-PIXEL IN PLACE on OUTB (own pixel only, no g_tmp scratch). The centred
// keep-rect is [margin*VW, (1-margin)*VW) x [margin*VH, (1-margin)*VH); any pixel OUTSIDE it has its
// RGB zeroed to black (alpha left untouched). margin<=0 never reaches here (caller skips). 'margin'/
// 'mx0'/'mx1'/'my0'/'my1' are NOT reserved words.
"__kernel void k_crop(__global float* d,float margin){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float mx0=margin*(float)VW, mx1=(1.0f-margin)*(float)VW;\n"
"  float my0=margin*(float)VH, my1=(1.0f-margin)*(float)VH;\n"
"  if((float)x<mx0 || (float)x>=mx1 || (float)y<my0 || (float)y>=my1){ d[i+0]=0.0f; d[i+1]=0.0f; d[i+2]=0.0f; }\n"
"}\n"
// GLITCH (per-band horizontal channel shift): reads source 's' (a copy of OUTB in g_tmp), writes OUTB
// 'd'. The frame is split into 24px-high horizontal bands (band = y/24, integer); each band gets a
// DETERMINISTIC signed shift sh = (int)((fpx_hash01(band,0,..)*2-1)*maxpx) (an integer band hash, NO
// time/RNG seed, so the same input frame always gives the same output and the gates stay stable). The
// channels are split: out.r samples g_tmp at clamp(x+sh), out.b at clamp(x-sh), out.g/out.a at x — so
// a sharp vertical edge breaks into per-band horizontal displacements with R/B colour separation.
// maxpx<=0 never reaches here (caller skips). 'maxpx'/'band'/'sh'/'rx'/'bx' are NOT reserved words.
"__kernel void k_glitch(__global const float* s,__global float* d,float maxpx){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int band=y/24;\n"
"  int sh=(int)((fpx_hash01(band,0,0,0)*2.0f-1.0f)*maxpx);\n"
"  int rx=x+sh; if(rx<0)rx=0; if(rx>VW-1)rx=VW-1;\n"
"  int bx=x-sh; if(bx<0)bx=0; if(bx>VW-1)bx=VW-1;\n"
"  int ir=IDX(rx,y), ib=IDX(bx,y);\n"
"  d[i+0]=s[ir+0]; d[i+1]=s[i+1]; d[i+2]=s[ib+2]; d[i+3]=s[i+3];\n"
"}\n"
// ---- P23 360 REFRAME (equirectangular -> rectilinear). When enabled the source ('s', a copy of OUTB
// in g_tmp) is treated as a full 360x180 equirectangular panorama and reprojected to a flat rectilinear
// "360 viewer" pinhole view at (yaw,pitch) radians with horizontal half-FOV tangent 'htan'. Standard
// bigsh0t/Shotcut-style projection (NOT bit-exact). For each OUTPUT pixel we build a camera ray from
// the NDC, rotate it by pitch then yaw, convert to longitude/latitude, map to equirect UV (u wraps,
// v clamps), and nearest-sample g_tmp. yaw>0 rotates the view toward +longitude (samples u>0.5, the
// RIGHT half), yaw<0 toward the LEFT half (u<0.5). M_PI_F/floor/atan2/asin/cos/sin/sqrt are OpenCL
// built-ins; VW/VH/IDX are existing macros; none of the locals are reserved words.
"__kernel void k_eq2rect(__global const float* s,__global float* d,float yaw,float pitch,float htan){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float ndcx=2.0f*((float)x+0.5f)/(float)VW-1.0f;\n"
"  float ndcy=2.0f*((float)y+0.5f)/(float)VH-1.0f;\n"
"  float rx=ndcx*htan;\n"
"  float ry=ndcy*htan*((float)VH/(float)VW);\n"
"  float rz=1.0f;\n"
"  float cp=cos(pitch), sp=sin(pitch);\n"
"  float y1=ry*cp-rz*sp;\n"
"  float z1=ry*sp+rz*cp;\n"
"  float x1=rx;\n"
"  float cw=cos(yaw), sw=sin(yaw);\n"
"  float x2=x1*cw+z1*sw;\n"
"  float z2=-x1*sw+z1*cw;\n"
"  float y2=y1;\n"
"  float ln=sqrt(x2*x2+y2*y2+z2*z2); if(ln<1e-6f) ln=1e-6f;\n"
"  float lon=atan2(x2,z2);\n"
"  float lat=asin(y2/ln);\n"
"  float u=0.5f+lon/(2.0f*M_PI_F);\n"
"  float v=0.5f-lat/M_PI_F;\n"
"  u=u-floor(u);\n"
"  if(v<0.0f)v=0.0f; if(v>1.0f)v=1.0f;\n"
"  int sx=(int)(u*(float)VW); if(sx<0)sx=0; if(sx>VW-1)sx=VW-1;\n"
"  int sy=(int)(v*(float)VH); if(sy<0)sy=0; if(sy>VH-1)sy=VH-1;\n"
"  int si=IDX(sx,sy);\n"
"  d[i+0]=s[si+0]; d[i+1]=s[si+1]; d[i+2]=s[si+2]; d[i+3]=s[si+3];\n"
"}\n"
// ---- P34 SHAPE MASK (Shotcut-parity mask_shape). Zero (to black) the pixels OUTSIDE a centred
// rectangle (shape==1) or ellipse (shape==2), with a feathered edge and optional invert. Runs on the
// composited OUTB AFTER the P17 geometry (lens/crop/glitch) and the P23 360 reframe, BEFORE the look —
// the SAME slot the geometry filters use. IN-PLACE on OUTB: each pixel scales ONLY ITSELF (no g_tmp
// copy, like k_crop), so no read/write race. shape==0 never reaches here (the FFI wrapper skips), so
// mask_shape 0 leaves OUTB byte-identical. 'dist' is a normalized distance from the mask centre: for
// the rectangle it is the Chebyshev (max-of-axes) distance, for the ellipse the euclidean radius —
// both ==1 on the rect/ellipse boundary. The feather band of width 'fw' ramps the mask m from 1
// (inside) to 0 (outside): m=1 for dist<=1-fw, m=0 for dist>=1, linear in between; 'inv' flips it.
// RGB is scaled by m (alpha untouched). VW/VH/IDX are existing macros; fmax/fabs/sqrt are standard;
// none of the locals (nx/ny/dx/dy/dist/fw/m) are reserved OpenCL words.
"__kernel void k_mask(__global float* d,int shape,float cx,float cy,float rw,float rh,float feather,int inv){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float nx=((float)x+0.5f)/(float)VW, ny=((float)y+0.5f)/(float)VH;\n"
"  float dx=(nx-cx)/fmax(rw,1e-4f), dy=(ny-cy)/fmax(rh,1e-4f);\n"
"  float dist = (shape==2) ? sqrt(dx*dx+dy*dy) : fmax(fabs(dx),fabs(dy));\n"
"  float fw=fmax(feather,1e-4f);\n"
"  float m;\n"
"  if(dist<=1.0f-fw) m=1.0f; else if(dist>=1.0f) m=0.0f; else m=(1.0f-dist)/fw;\n"
"  if(inv) m=1.0f-m;\n"
"  d[i+0]*=m; d[i+1]*=m; d[i+2]*=m;\n"
"}\n"
// ---- P38 DISTORTION FILTER BATCH (Shotcut-parity distort family). Three per-clip filters run on the
// composited OUTB AFTER the P34 shape mask and BEFORE the look — the SAME slot the P17/P23/P34 OUTB
// filters use. Each is a no-op at its default: mirror_x 0 / kaleido <2 / dither 0 → byte-identical to
// pre-P38. mirror+kaleido SAMPLE other pixels (the C wrapper copies OUTB->g_tmp first, like k_lens/
// k_eq2rect, then reads g_tmp 's' and writes OUTB 'd'); dither is IN-PLACE on OUTB (like k_crop). The
// FFI wrappers skip seg<2 / amt<=0, so those kernels never reach the never-skipped branches. VW/VH/IDX/
// M_PI_F are existing macros; the locals (sx/sy/si/cx/cy/dx/dy/r/ang/segang/thr/levels/v/bayer) are
// not reserved OpenCL words.
// MIRROR: the RIGHT half mirrors the LEFT half (x>=VW/2 samples column VW-1-x).
"__kernel void k_mirror(__global const float* s,__global float* d){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  int sx = (x < VW/2) ? x : (VW-1-x); int si=IDX(sx,y);\n"
"  d[i+0]=s[si+0]; d[i+1]=s[si+1]; d[i+2]=s[si+2]; d[i+3]=s[si+3];\n"
"}\n"
// KALEIDOSCOPE: N-fold radial mirror. seg<2 never reaches here (caller skips).
"__kernel void k_kaleido(__global const float* s,__global float* d,int seg){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float cx=(float)VW*0.5f, cy=(float)VH*0.5f; float dx=(float)x-cx, dy=(float)y-cy;\n"
"  float r=sqrt(dx*dx+dy*dy); float ang=atan2(dy,dx);\n"
"  float segang=2.0f*M_PI_F/(float)seg; ang=fmod(fabs(ang),segang); if(ang>segang*0.5f) ang=segang-ang;\n"
"  int sx=(int)(cx+r*cos(ang)+0.5f), sy=(int)(cy+r*sin(ang)+0.5f);\n"
"  if(sx<0)sx=0; if(sx>VW-1)sx=VW-1; if(sy<0)sy=0; if(sy>VH-1)sy=VH-1; int si=IDX(sx,sy);\n"
"  d[i+0]=s[si+0]; d[i+1]=s[si+1]; d[i+2]=s[si+2]; d[i+3]=s[si+3];\n"
"}\n"
// DITHER: ordered 4x4 Bayer-dithered posterize (~8 levels). amt<=0 never reaches here (caller skips).
"__kernel void k_dither(__global float* d,float amt){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  const int bayer[16]={0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5};\n"
"  float thr=((float)bayer[(y&3)*4+(x&3)]/16.0f-0.5f)*amt*0.25f; float levels=8.0f;\n"
"  for(int c=0;c<3;c++){ float v=d[i+c]+thr; v=floor(v*levels+0.5f)/levels; if(v<0.0f)v=0.0f; if(v>1.0f)v=1.0f; d[i+c]=v; }\n"
"}\n"
// SELECTIVE COLOR (P39): adjust ONE hue band (hue rotation + saturation) on OUTB in place. band==0 never
// reaches here (caller skips) so band 0 is a no-op. Greyscale pixels and pixels outside the band untouched.
"__kernel void k_selcolor(__global float* d,int band,float hshift,float ssat){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=d[i+0],g=d[i+1],b=d[i+2];\n"
"  float mx=fmax(r,fmax(g,b)), mn=fmin(r,fmin(g,b)), dl=mx-mn;\n"
"  if(dl<=1e-5f) return;                              // greyscale pixel: no hue, nothing to select\n"
"  float h=0.0f;\n"
"  if(mx==r) h=fmod((g-b)/dl,6.0f); else if(mx==g) h=(b-r)/dl+2.0f; else h=(r-g)/dl+4.0f;\n"
"  h/=6.0f; if(h<0.0f)h+=1.0f;                        // hue 0..1\n"
"  float s=dl/mx, v=mx;\n"
"  float center=((float)(band-1))/6.0f;              // band centres at k/6\n"
"  float dh=fabs(h-center); if(dh>0.5f) dh=1.0f-dh;  // circular hue distance\n"
"  float band_w=1.0f/12.0f;                           // ~30 deg half-band\n"
"  if(dh>band_w) return;                              // outside the band: untouched\n"
"  float w=1.0f-dh/band_w;                            // feather to the band edge\n"
"  h=h+hshift*w; h=h-floor(h);                        // rotate hue\n"
"  s=s*(1.0f+(ssat-1.0f)*w); if(s<0.0f)s=0.0f; if(s>1.0f)s=1.0f;\n"
"  float hh=h*6.0f; int ii=(int)hh; float ff=hh-(float)ii;\n"
"  float p=v*(1.0f-s), q=v*(1.0f-s*ff), t=v*(1.0f-s*(1.0f-ff));\n"
"  float nr,ng,nb;\n"
"  if(ii==0){nr=v;ng=t;nb=p;} else if(ii==1){nr=q;ng=v;nb=p;} else if(ii==2){nr=p;ng=v;nb=t;}\n"
"  else if(ii==3){nr=p;ng=q;nb=v;} else if(ii==4){nr=t;ng=p;nb=v;} else {nr=v;ng=p;nb=q;}\n"
"  d[i+0]=nr; d[i+1]=ng; d[i+2]=nb;\n"
"}\n"
// SOLARIZE (P41): per-channel classic darkroom solarize — v>thr -> 1-v, in place on OUTB. thr<=0 never
// reaches here (caller skips) so thr 0 is a no-op.
"__kernel void k_solarize(__global float* d,float thr){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  for(int c=0;c<3;c++){ float v=d[i+c]; if(v>thr) v=1.0f-v; d[i+c]=v; }\n"
"}\n"
// COLOR TEMPERATURE (P41): warm (t>0) raises R and lowers B; cool (t<0) the reverse. Green unchanged.
// In place on OUTB. t==0 never reaches here (caller skips) so t 0 is a no-op.
"__kernel void k_temp(__global float* d,float t){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  float r=d[i+0]+t*0.30f, b=d[i+2]-t*0.30f;\n"
"  d[i+0]=clamp(r,0.0f,1.0f); d[i+2]=clamp(b,0.0f,1.0f);\n"
"}\n"
// FADE TO BLACK (P45): multiply RGB by the fade factor f (0=black, 1=full). f>=1 never reaches here
// (caller skips), so a non-faded frame is byte-identical. Alpha untouched.
"__kernel void k_fade(__global float* d,float f){\n"
"  int x=get_global_id(0),y=get_global_id(1); if(x>=VW||y>=VH) return; int i=IDX(x,y);\n"
"  d[i+0]*=f; d[i+1]*=f; d[i+2]*=f;\n"
"}\n";

// cached kernel objects (clCreateKernel once)
static cl_kernel K(const char* name){ cl_int e; cl_kernel k = clCreateKernel(g_prog, name, &e); return (e==CL_SUCCESS)?k:NULL; }
static cl_kernel kUnpack,kPack,kCopy,kComposite,kPip,kBright,kContrast,kSat,kLut,kVhs;
static cl_kernel kHistClear,kHist;
static cl_kernel kGridClear,kWaveAcc,kWaveImg,kVecAcc,kVecImg;
static cl_kernel kParadeClear,kParadeAcc,kParadeImg;
static cl_kernel kLgg,kTransform,kBlurH,kBlurV; // P2 color/transform effects
static cl_kernel kCurve; // P5 master tone curve
static cl_kernel kSimplefx,kVignette,kSharpen,kFlip; // P6 stylize/utility filters
static cl_kernel kHsl,kLevels; // P7 color filters (HSL adjust + levels)
static cl_kernel kMosaic,kGmap; // P8 stylize-2 filters (mosaic pixelate + gradient map)
static cl_kernel kDenoise,kGlowExtract,kGlowCombine,kRgbshift; // P9 fx filters (denoise/glow/rgb-shift)
static cl_kernel kHalftone,kEmboss,kEdge; // P10 stylize-4 filters (halftone/emboss/edge — all spatial via g_tmp)
static cl_kernel kGrain,kScratches,kDiffuse; // P13 old-film/distort filters (grain/scratches/diffusion — all spatial via g_tmp)
static cl_kernel kWave,kSwirl,kThreshold; // P16 distort filters (wave/swirl spatial via g_tmp; threshold in-place)
static cl_kernel kLens,kCrop,kGlitch; // P17 geometric filters (lens/glitch spatial via g_tmp; crop in-place)
static cl_kernel kEq2rect; // P23 360 reframe (equirectangular -> rectilinear, spatial via g_tmp)
static cl_kernel kMask; // P34 shape mask (centred rect/ellipse, feathered, optional invert; in-place on OUTB)
static cl_kernel kMirror,kKaleido,kDither; // P38 distortion batch (mirror/kaleido spatial via g_tmp; dither in-place)
static cl_kernel kSelcolor; // P39 selective color (one hue band rotate+saturate; in-place on OUTB)
static cl_kernel kSolarize; // P41 solarize (per-channel v>thr -> 1-v; in-place on OUTB)
static cl_kernel kTemp; // P41 colour temperature (warm/cool R/B shift, green unchanged; in-place on OUTB)
static cl_kernel kFade; // P45 video fade-to-black (multiply RGB by the per-frame fade factor; in-place on OUTB)
static cl_kernel kChroma; // P4 chroma key (green-screen) on the OVER buffer
static cl_kernel kTrans[11]; // 0..10 (P36 added 8=iris, 9=clock, 10=barndoor)

static int launch(cl_kernel k){
  size_t gl[2] = { GVW, GVH };
  size_t lo[2] = { 8, 8 };
  cl_int e = clEnqueueNDRangeKernel(g_q, k, 2, NULL, gl, lo, 0, NULL, NULL);
  return e==CL_SUCCESS ? 0 : -1;
}

int fpx_gpu_init(void){
  if (g_ready) return 0;
  cl_uint np=0; cl_platform_id plat[8];
  if (clGetPlatformIDs(8, plat, &np)!=CL_SUCCESS || np==0) return -1;
  cl_int e;
  // first platform with a GPU device
  for (cl_uint p=0;p<np && !g_dev;p++){
    cl_uint nd=0; if (clGetDeviceIDs(plat[p], CL_DEVICE_TYPE_GPU, 1, &g_dev, &nd)!=CL_SUCCESS || nd==0) g_dev=NULL;
  }
  if (!g_dev){ // fall back to any device
    for (cl_uint p=0;p<np && !g_dev;p++){ cl_uint nd=0; clGetDeviceIDs(plat[p], CL_DEVICE_TYPE_ALL, 1, &g_dev, &nd); }
  }
  if (!g_dev) return -2;
  g_ctx = clCreateContext(NULL, 1, &g_dev, NULL, NULL, &e); if (e!=CL_SUCCESS) return -3;
  g_q = clCreateCommandQueue(g_ctx, g_dev, 0, &e); if (e!=CL_SUCCESS) return -4;
  g_prog = clCreateProgramWithSource(g_ctx, 1, &KSRC, NULL, &e); if (e!=CL_SUCCESS) return -5;
  char opt[64]; snprintf(opt, sizeof opt, "-DVW=%d -DVH=%d", GVW, GVH);
  e = clBuildProgram(g_prog, 1, &g_dev, opt, NULL, NULL);
  if (e!=CL_SUCCESS){
    size_t ls=0; clGetProgramBuildInfo(g_prog, g_dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &ls);
    char* log=(char*)malloc(ls+1); clGetProgramBuildInfo(g_prog, g_dev, CL_PROGRAM_BUILD_LOG, ls, log, NULL); log[ls]=0;
    fprintf(stderr, "fpx_gpu clBuildProgram failed:\n%s\n", log); free(log); return -6;
  }
  for (int i=0;i<NBUF;i++){ g_buf[i]=clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, GN*sizeof(float), NULL, &e); if(e!=CL_SUCCESS) return -7; }
  g_lut   = clCreateBuffer(g_ctx, CL_MEM_READ_ONLY, MAXLUTF*sizeof(float), NULL, &e); if(e!=CL_SUCCESS) return -8;
  // READ_WRITE (not READ_ONLY): g_stage is used BOTH directions — host writes + k_unpack reads it on
  // upload, AND k_pack WRITES it on download_u8 (then host reads). A READ_ONLY buffer written by a
  // kernel is undefined behaviour, so the correct flag is READ_WRITE. (The primary cause of the
  // N-layer fold's black band was the non-blocking upload below; this is a related correctness fix.)
  g_stage = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, GN*sizeof(unsigned char), NULL, &e); if(e!=CL_SUCCESS) return -9;
  g_hist  = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, 768*sizeof(int), NULL, &e); if(e!=CL_SUCCESS) return -11;
  g_grid  = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, 65536*sizeof(int), NULL, &e); if(e!=CL_SUCCESS) return -12;
  g_scope = clCreateBuffer(g_ctx, CL_MEM_WRITE_ONLY, 256*256*4, NULL, &e); if(e!=CL_SUCCESS) return -13;
  g_parade= clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, 3*65536*sizeof(int), NULL, &e); if(e!=CL_SUCCESS) return -15;
  g_tmp   = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, GN*sizeof(float), NULL, &e); if(e!=CL_SUCCESS) return -17; // P2 scratch
  g_tmp2  = clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, GN*sizeof(float), NULL, &e); if(e!=CL_SUCCESS) return -30; // P9 scratch #2 (glow)
  kUnpack=K("k_unpack"); kPack=K("k_pack"); kCopy=K("k_copy"); kComposite=K("k_composite"); kPip=K("k_pip");
  kBright=K("k_brightness"); kContrast=K("k_contrast"); kSat=K("k_saturation"); kLut=K("k_lut3d"); kVhs=K("k_vhs");
  kHistClear=K("k_hist_clear"); kHist=K("k_hist");
  kGridClear=K("k_grid_clear"); kWaveAcc=K("k_wave_acc"); kWaveImg=K("k_wave_img"); kVecAcc=K("k_vec_acc"); kVecImg=K("k_vec_img");
  kParadeClear=K("k_parade_clear"); kParadeAcc=K("k_parade_acc"); kParadeImg=K("k_parade_img");
  kLgg=K("k_lgg"); kTransform=K("k_transform"); kBlurH=K("k_blur_h"); kBlurV=K("k_blur_v"); // P2
  kChroma=K("k_chroma"); // P4 chroma key
  kCurve=K("k_curve");   // P5 master tone curve
  kSimplefx=K("k_simplefx"); kVignette=K("k_vignette"); kSharpen=K("k_sharpen"); kFlip=K("k_flip"); // P6
  kHsl=K("k_hsl"); kLevels=K("k_levels"); // P7 color filters
  kMosaic=K("k_mosaic"); kGmap=K("k_gmap"); // P8 stylize-2 filters
  kDenoise=K("k_denoise"); kGlowExtract=K("k_glow_extract"); kGlowCombine=K("k_glow_combine"); kRgbshift=K("k_rgbshift"); // P9 fx
  kHalftone=K("k_halftone"); kEmboss=K("k_emboss"); kEdge=K("k_edge"); // P10 stylize-4
  kGrain=K("k_grain"); kScratches=K("k_scratches"); kDiffuse=K("k_diffuse"); // P13 old-film/distort
  kWave=K("k_wave"); kSwirl=K("k_swirl"); kThreshold=K("k_threshold"); // P16 distort
  kLens=K("k_lens"); kCrop=K("k_crop"); kGlitch=K("k_glitch"); // P17 geometric
  kEq2rect=K("k_eq2rect"); // P23 360 reframe
  kMask=K("k_mask"); // P34 shape mask
  kMirror=K("k_mirror"); kKaleido=K("k_kaleido"); kDither=K("k_dither"); // P38 distortion batch
  kSelcolor=K("k_selcolor"); // P39 selective color
  kSolarize=K("k_solarize"); // P41 solarize
  kTemp=K("k_temp"); // P41 colour temperature
  kFade=K("k_fade"); // P45 video fade-to-black
  kTrans[0]=K("k_crossfade"); kTrans[1]=K("k_wipe_lr"); kTrans[2]=K("k_wipe_rl"); kTrans[3]=K("k_wipe_up");
  kTrans[4]=K("k_wipe_down"); kTrans[5]=K("k_slide_lr"); kTrans[6]=K("k_zoom"); kTrans[7]=K("k_dissolve");
  kTrans[8]=K("k_iris"); kTrans[9]=K("k_clock"); kTrans[10]=K("k_barndoor"); // P36 luma wipes
  if(!kUnpack||!kPack||!kComposite||!kPip||!kBright||!kContrast||!kLut||!kVhs||!kTrans[7]||!kTrans[10]) return -10;
  // scope kernels must build too — else a SCOPE command later launches a NULL kernel (segfault).
  if(!kHistClear||!kHist||!kGridClear||!kWaveAcc||!kWaveImg||!kVecAcc||!kVecImg) return -14;
  // RGB parade kernels (Triad-B P1) — same NULL-kernel guard so a SCOPE 3 never segfaults.
  if(!kParadeClear||!kParadeAcc||!kParadeImg) return -16;
  // P2 color/transform kernels — same NULL-kernel guard so a compose that runs them never segfaults.
  if(!kLgg||!kTransform||!kBlurH||!kBlurV) return -18;
  // P4 chroma-key kernel — same NULL-kernel guard so a compose that runs it never segfaults.
  if(!kChroma) return -19;
  if(!kCurve) return -20;
  // P6 stylize/utility kernels — same NULL-kernel guard so a compose that runs them never segfaults.
  if(!kSimplefx) return -21;
  if(!kVignette) return -22;
  if(!kSharpen) return -23;
  if(!kFlip) return -24;
  // k_copy backs transform/blur (P2) AND now sharpen/flip (P6): if it failed to create, those
  // wrappers would launch a NULL kernel and segfault while fpx_gpu_init falsely reported ready.
  // (Pre-existing gap; folded in here since the P6 filters newly depend on it.)
  if(!kCopy) return -25;
  // P7 color kernels (HSL adjust + levels) — same NULL-kernel guard so a compose that runs them
  // (after the P6 flip, before the look) never launches a NULL kernel / segfaults.
  if(!kHsl) return -26;
  if(!kLevels) return -27;
  // P8 stylize-2 kernels (mosaic pixelate + gradient map) — same NULL-kernel guard so a compose that
  // runs them (after the P7 levels, before the look) never launches a NULL kernel / segfaults.
  if(!kMosaic) return -28;
  if(!kGmap) return -29;
  // P9 fx kernels (denoise bilateral / glow bright-pass+combine / rgb-shift) — same NULL-kernel guard
  // so a compose that runs them (after the P8 gradient-map, before the look) never launches a NULL
  // kernel / segfaults. g_tmp2 (the glow scratch #2) gets its own alloc guard (-30) above.
  if(!kDenoise) return -31;
  if(!kGlowExtract) return -32;
  if(!kGlowCombine) return -33;
  if(!kRgbshift) return -34;
  // P10 stylize-4 kernels (halftone dot-screen / emboss relief / edge Sobel) — same NULL-kernel
  // guard so a compose that runs them (after the P9 rgb-shift, before the look) never launches a
  // NULL kernel / segfaults. All three reuse g_tmp (no new buffer), so no new alloc guard is needed.
  if(!kHalftone) return -35;
  if(!kEmboss) return -36;
  if(!kEdge) return -37;
  // P13 old-film/distort kernels (grain luma-noise / scratches vertical-lines / diffusion neighbour-
  // jitter) — same NULL-kernel guard so a compose that runs them (after the P10 edge, before the
  // look) never launches a NULL kernel / segfaults. All three reuse g_tmp (no new buffer), so no new
  // alloc guard is needed.
  if(!kGrain) return -38;
  if(!kScratches) return -39;
  if(!kDiffuse) return -40;
  // P16 distort kernels (wave horizontal sinusoidal row displacement / swirl rotational distortion /
  // threshold luma binarize) — same NULL-kernel guard so a compose that runs them (after the P13
  // diffusion, before the look) never launches a NULL kernel / segfaults. wave/swirl reuse g_tmp (no
  // new buffer), threshold is in-place on OUTB — so no new alloc guard is needed.
  if(!kWave) return -41;
  if(!kSwirl) return -42;
  if(!kThreshold) return -43;
  // P17 geometric kernels (lens radial barrel/pincushion / crop margin-to-black / glitch per-band
  // channel shift) — same NULL-kernel guard so a compose that runs them (after the P16 threshold,
  // before the look) never launches a NULL kernel / segfaults. lens/glitch reuse g_tmp (no new
  // buffer), crop is in-place on OUTB — so no new alloc guard is needed.
  if(!kLens) return -44;
  if(!kCrop) return -45;
  if(!kGlitch) return -46;
  // P23 360 reframe kernel (equirectangular -> rectilinear) — same NULL-kernel guard so a compose that
  // runs it (after the P17 glitch, before the look) never launches a NULL kernel / segfaults. It reuses
  // g_tmp (no new buffer) — so no new alloc guard is needed.
  if(!kEq2rect) return -47;
  // P34 shape-mask kernel (centred rect/ellipse, feathered, optional invert) — same NULL-kernel guard
  // so a compose that runs it (after the P23 reframe, before the look) never launches a NULL kernel /
  // segfaults. It is in-place on OUTB (no new buffer) — so no new alloc guard is needed.
  if(!kMask) return -48;
  // P38 distortion kernels (mirror right-half-mirrors-left / kaleido N-fold radial / dither Bayer
  // posterize) — same NULL-kernel guard so a compose that runs them (after the P34 mask, before the
  // look) never launches a NULL kernel / segfaults. mirror/kaleido reuse g_tmp (no new buffer), dither
  // is in-place on OUTB — so no new alloc guard is needed.
  if(!kMirror) return -49;
  if(!kKaleido) return -50;
  if(!kDither) return -51;
  // P39 selective color (one hue band rotate+saturate, in-place on OUTB after the P38 distort filters,
  // before the look) — same NULL-kernel guard; band==0 caller-skips so no new buffer/alloc needed.
  if(!kSelcolor) return -52;
  // P41 solarize + colour temperature (both in-place on OUTB after the P39 selective color, before the
  // look) — same NULL-kernel guard; thr<=0 / t==0 caller-skips so no new buffer/alloc needed.
  if(!kSolarize) return -53;
  if(!kTemp) return -54;
  if(!kFade) return -55;
  g_ready=1; return 0;
}

// upload an RGBA8 frame to slot (0=base,1=over,2=trans): staging u8 -> unpack -> float buf.
// The write is BLOCKING (CL_TRUE): the host `rgba8` pointer is a transient caller buffer (a decoded
// Rust Vec that is dropped/reused right after this returns). A non-blocking (CL_FALSE) write lets the
// host->device DMA outlive the call and read freed memory → a corrupt, NON-DETERMINISTIC frame
// (manifested as a black band over the lower part of the image, worst on the cold first compose and
// in the back-to-back N-layer render fold). CL_TRUE guarantees the buffer is fully consumed first.
int fpx_gpu_upload_u8(int slot, const unsigned char* rgba8){
  if(!g_ready||slot<0||slot>2||!rgba8) return -1;
  if(clEnqueueWriteBuffer(g_q,g_stage,CL_TRUE,0,GN*sizeof(unsigned char),rgba8,0,NULL,NULL)!=CL_SUCCESS) return -2;
  clSetKernelArg(kUnpack,0,sizeof(cl_mem),&g_stage); clSetKernelArg(kUnpack,1,sizeof(cl_mem),&g_buf[slot]);
  return launch(kUnpack);
}
int fpx_gpu_upload_f32(int slot, const float* f32){
  if(!g_ready||slot<0||slot>2||!f32) return -1; // CL_TRUE: transient host buffer (see fpx_gpu_upload_u8).
  return clEnqueueWriteBuffer(g_q,g_buf[slot],CL_TRUE,0,GN*sizeof(float),f32,0,NULL,NULL)==CL_SUCCESS?0:-2;
}
int fpx_gpu_upload_lut(const float* lut, int nfloats){
  if(!g_ready||!lut||nfloats<=0||nfloats>MAXLUTF) return -1; // CL_TRUE: transient host buffer.
  return clEnqueueWriteBuffer(g_q,g_lut,CL_TRUE,0,nfloats*sizeof(float),lut,0,NULL,NULL)==CL_SUCCESS?0:-2;
}

// track1 = transition(base,trans,t,param)  OR (tt<0) composite(base,trans,0)=base
void fpx_gpu_track1(int tt, float t, float param){
  if(!g_ready) return;
  if(tt<0||tt>10){ clSetKernelArg(kComposite,0,sizeof(cl_mem),&g_buf[BASE]); clSetKernelArg(kComposite,1,sizeof(cl_mem),&g_buf[TRANS]);
    clSetKernelArg(kComposite,2,sizeof(cl_mem),&g_buf[TRACK1]); float z=0.0f; clSetKernelArg(kComposite,3,sizeof(float),&z); launch(kComposite); return; }
  cl_kernel k=kTrans[tt];
  clSetKernelArg(k,0,sizeof(cl_mem),&g_buf[BASE]); clSetKernelArg(k,1,sizeof(cl_mem),&g_buf[TRANS]); clSetKernelArg(k,2,sizeof(cl_mem),&g_buf[TRACK1]);
  clSetKernelArg(k,3,sizeof(float),&t);
  if(tt==7) clSetKernelArg(k,4,sizeof(float),&param);
  launch(k);
}
// P2 TRANSFORM: rotate (degrees) + uniform scale the BASE frame (TRACK1 buffer) about its center,
// bilinear. Runs RIGHT AFTER fpx_gpu_track1, BEFORE pip — so the PiP overlay composites onto the
// already-transformed base. Identity at rot=0,scale=1 (skipped → TRACK1 untouched, zero cost). Since
// the kernel cannot read+write the same buffer safely, copy TRACK1→g_tmp first, then transform the
// COPY back into TRACK1.
void fpx_gpu_transform(float rot_deg, float scale){
  if(!g_ready) return;
  int identity = (rot_deg>-0.001f && rot_deg<0.001f) && (scale>0.999f && scale<1.001f);
  if(identity) return; // identity transform: leave TRACK1 as-is.
  // TRACK1 -> g_tmp (source copy), then transform g_tmp -> TRACK1.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[TRACK1]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  int w=GVW, h=GVH;
  clSetKernelArg(kTransform,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kTransform,1,sizeof(cl_mem),&g_buf[TRACK1]);
  clSetKernelArg(kTransform,2,sizeof(float),&rot_deg); clSetKernelArg(kTransform,3,sizeof(float),&scale);
  clSetKernelArg(kTransform,4,sizeof(int),&w); clSetKernelArg(kTransform,5,sizeof(int),&h);
  launch(kTransform);
}
// P4 CHROMA KEY: zero/soften the OVER buffer's ALPHA where the pixel matches the key colour, IN PLACE
// on the OVER buffer. Runs AFTER the over upload (+ any over transform) and BEFORE fpx_gpu_pip, so the
// pip composite (`over.a*op`) shows the base through the keyed pixels. The caller only invokes this
// when the clip's chroma is ENABLED (ck_on==1); a disabled clip never calls it, so the OVER alpha is
// untouched and the composite is byte-identical to P3. RGB is never modified.
// P37: `spill` (>0) enables green-spill suppression in the SAME kernel, AFTER the alpha key (arg 6,
// the LAST kernel arg). spill==0 is a no-op (the kernel's if is skipped) → byte-identical to pre-P37.
void fpx_gpu_chroma(float kr, float kg, float kb, float sim, float smooth, float spill){
  if(!g_ready) return;
  clSetKernelArg(kChroma,0,sizeof(cl_mem),&g_buf[OVER]);
  clSetKernelArg(kChroma,1,sizeof(float),&kr); clSetKernelArg(kChroma,2,sizeof(float),&kg); clSetKernelArg(kChroma,3,sizeof(float),&kb);
  clSetKernelArg(kChroma,4,sizeof(float),&sim); clSetKernelArg(kChroma,5,sizeof(float),&smooth);
  clSetKernelArg(kChroma,6,sizeof(float),&spill);
  launch(kChroma);
}
// in = composite_pip(track1, over, op, blend, px,py,pw,ph)
// P31: `blend` (0=Normal..7=Difference) selects the per-channel blend of the over RGB with the base
// before the alpha-over (k_pip). blend==0 is byte-identical to the pre-P31 plain composite. The new
// int arg is index 4, so px/py/pw/ph shift to 5/6/7/8.
void fpx_gpu_pip(float op, int blend, float px, float py, float pw, float ph){
  if(!g_ready) return;
  clSetKernelArg(kPip,0,sizeof(cl_mem),&g_buf[TRACK1]); clSetKernelArg(kPip,1,sizeof(cl_mem),&g_buf[OVER]); clSetKernelArg(kPip,2,sizeof(cl_mem),&g_buf[INB]);
  clSetKernelArg(kPip,3,sizeof(float),&op); clSetKernelArg(kPip,4,sizeof(int),&blend); clSetKernelArg(kPip,5,sizeof(float),&px); clSetKernelArg(kPip,6,sizeof(float),&py);
  clSetKernelArg(kPip,7,sizeof(float),&pw); clSetKernelArg(kPip,8,sizeof(float),&ph); launch(kPip);
}
// mid = brightness(in); out = contrast(mid); out = saturation(out) in-place (skip if sat==1)
void fpx_gpu_grade(float bright, float contrast, float sat){
  if(!g_ready) return;
  clSetKernelArg(kBright,0,sizeof(cl_mem),&g_buf[INB]); clSetKernelArg(kBright,1,sizeof(cl_mem),&g_buf[MID]); clSetKernelArg(kBright,2,sizeof(float),&bright); launch(kBright);
  clSetKernelArg(kContrast,0,sizeof(cl_mem),&g_buf[MID]); clSetKernelArg(kContrast,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kContrast,2,sizeof(float),&contrast); launch(kContrast);
  if(sat<0.999f || sat>1.001f){ clSetKernelArg(kSat,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kSat,1,sizeof(float),&sat); launch(kSat); }
}
// PER-CLIP grade (Triad-B P1): grade the PiP-composite buffer (INB) IN PLACE, BEFORE the program
// grade. Same kernels/semantics as fpx_gpu_grade, but routes INB→MID(bright)→INB(contrast)→INB(sat)
// so the result lands back in INB — then the caller runs the program fpx_gpu_grade(INB→…→OUTB) on
// top, giving the documented "per-clip first, then program" additive order. A neutral grade
// (bright 0, contrast 1, sat 1) is a no-op (skipped) so a clip with the default grade costs nothing.
void fpx_gpu_grade_clip(float bright, float contrast, float sat){
  if(!g_ready) return;
  int neutral = (bright>-0.001f && bright<0.001f) && (contrast>0.999f && contrast<1.001f) && (sat>0.999f && sat<1.001f);
  if(neutral) return; // identity per-clip grade: leave INB untouched.
  clSetKernelArg(kBright,0,sizeof(cl_mem),&g_buf[INB]); clSetKernelArg(kBright,1,sizeof(cl_mem),&g_buf[MID]); clSetKernelArg(kBright,2,sizeof(float),&bright); launch(kBright);
  clSetKernelArg(kContrast,0,sizeof(cl_mem),&g_buf[MID]); clSetKernelArg(kContrast,1,sizeof(cl_mem),&g_buf[INB]); clSetKernelArg(kContrast,2,sizeof(float),&contrast); launch(kContrast);
  if(sat<0.999f || sat>1.001f){ clSetKernelArg(kSat,0,sizeof(cl_mem),&g_buf[INB]); clSetKernelArg(kSat,1,sizeof(float),&sat); launch(kSat); }
}
// P2 LGG (3-way color wheels): per-channel out=clamp01(pow(clamp01(in*gain+lift),1/gamma)) IN PLACE
// on the grade-result buffer (OUTB), BEFORE look. Identity (lift 0 / gamma 1 / gain 1 on all three
// channels) is skipped (zero cost). White balance is already folded into the gains by the UI.
void fpx_gpu_lgg(float lr,float lg,float lb,float gar,float gag,float gab,float gnr,float gng,float gnb){
  if(!g_ready) return;
  int identity =
    (lr>-0.001f&&lr<0.001f)&&(lg>-0.001f&&lg<0.001f)&&(lb>-0.001f&&lb<0.001f)&&
    (gar>0.999f&&gar<1.001f)&&(gag>0.999f&&gag<1.001f)&&(gab>0.999f&&gab<1.001f)&&
    (gnr>0.999f&&gnr<1.001f)&&(gng>0.999f&&gng<1.001f)&&(gnb>0.999f&&gnb<1.001f);
  if(identity) return;
  clSetKernelArg(kLgg,0,sizeof(cl_mem),&g_buf[OUTB]);
  clSetKernelArg(kLgg,1,sizeof(float),&lr); clSetKernelArg(kLgg,2,sizeof(float),&lg); clSetKernelArg(kLgg,3,sizeof(float),&lb);
  clSetKernelArg(kLgg,4,sizeof(float),&gar); clSetKernelArg(kLgg,5,sizeof(float),&gag); clSetKernelArg(kLgg,6,sizeof(float),&gab);
  clSetKernelArg(kLgg,7,sizeof(float),&gnr); clSetKernelArg(kLgg,8,sizeof(float),&gng); clSetKernelArg(kLgg,9,sizeof(float),&gnb);
  launch(kLgg);
}
// P2 BLUR: separable gaussian, 2 passes, IN PLACE on OUTB via g_tmp. sigma<=0 => no-op. Runs AFTER
// lgg, BEFORE look. Horizontal pass OUTB->g_tmp, vertical pass g_tmp->OUTB.
void fpx_gpu_blur(float sigma){
  if(!g_ready || sigma<=0.0f) return; // identity / no-op blur: leave OUTB untouched.
  clSetKernelArg(kBlurH,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kBlurH,1,sizeof(cl_mem),&g_tmp); clSetKernelArg(kBlurH,2,sizeof(float),&sigma); launch(kBlurH);
  clSetKernelArg(kBlurV,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kBlurV,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kBlurV,2,sizeof(float),&sigma); launch(kBlurV);
}
// P5 CURVE: a 5-point master tone curve, IN PLACE on OUTB. The identity (0,.25,.5,.75,1) is skipped
// so an un-curved clip is a true no-op. Runs AFTER blur, BEFORE look.
void fpx_gpu_curve(float y0,float y1,float y2,float y3,float y4){
  if(!g_ready) return;
  if(y0==0.0f && y1==0.25f && y2==0.5f && y3==0.75f && y4==1.0f) return; // identity
  clSetKernelArg(kCurve,0,sizeof(cl_mem),&g_buf[OUTB]);
  clSetKernelArg(kCurve,1,sizeof(float),&y0); clSetKernelArg(kCurve,2,sizeof(float),&y1);
  clSetKernelArg(kCurve,3,sizeof(float),&y2); clSetKernelArg(kCurve,4,sizeof(float),&y3);
  clSetKernelArg(kCurve,5,sizeof(float),&y4); launch(kCurve);
}
// P6 SIMPLE-FX: in place on OUTB. kind 0 = skip (no-op default); 1 invert, 2 sepia, 3 grayscale,
// 4 posterize. Runs AFTER curve, BEFORE the look (first of the four P6 filters, per pinned order).
void fpx_gpu_simplefx(int kind){
  if(!g_ready || kind==0) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kSimplefx,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kSimplefx,1,sizeof(int),&kind);
  launch(kSimplefx);
}
// P6 VIGNETTE: in place on OUTB, radial edge darken by `amt`. amt<=0 = skip (no-op default). Runs
// after simple-fx, before sharpen (pinned P6 order: simplefx -> vignette -> sharpen -> flip).
void fpx_gpu_vignette(float amt){
  if(!g_ready || amt<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kVignette,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kVignette,1,sizeof(float),&amt);
  launch(kVignette);
}
// P6 SHARPEN (unsharp): amt<=0 = skip. The kernel cannot read+write OUTB in place, so copy
// OUTB->g_tmp first (via k_copy, same convention as transform/blur), then sharpen g_tmp->OUTB.
void fpx_gpu_sharpen(float amt){
  if(!g_ready || amt<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kSharpen,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kSharpen,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kSharpen,2,sizeof(float),&amt);
  launch(kSharpen);
}
// P6 FLIP (mirror): mode 0 = skip; 1 H (x->W-1-x), 2 V (y->H-1-y), 3 both. Copy OUTB->g_tmp first,
// then sample g_tmp at the flipped coord into OUTB. Last of the four P6 filters (before the look).
void fpx_gpu_flip(int mode){
  if(!g_ready || mode==0) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kFlip,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kFlip,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kFlip,2,sizeof(int),&mode);
  launch(kFlip);
}
// P7 HSL ADJUST: in place on OUTB. RGB->HSL, hue += hue_deg (wrap 360), saturation *= sat, lightness
// += light, HSL->RGB, clamp01. Identity hue_deg=0,sat=1,light=0 = skip (no-op default). Runs AFTER
// the P6 flip, BEFORE the look (first of the two P7 color filters). Per-pixel in place — no scratch.
void fpx_gpu_hsl(float hue_deg, float sat, float light){
  if(!g_ready) return;
  int identity = (hue_deg>-0.001f && hue_deg<0.001f) && (sat>0.999f && sat<1.001f) && (light>-0.001f && light<0.001f);
  if(identity) return; // identity HSL: leave OUTB untouched.
  clSetKernelArg(kHsl,0,sizeof(cl_mem),&g_buf[OUTB]);
  clSetKernelArg(kHsl,1,sizeof(float),&hue_deg); clSetKernelArg(kHsl,2,sizeof(float),&sat); clSetKernelArg(kHsl,3,sizeof(float),&light);
  launch(kHsl);
}
// P7 LEVELS: in place on OUTB, per channel out=clamp01(pow(clamp01((c-in_black)/max(in_white-in_black,
// 1e-3)),1/max(gamma,1e-3))). Identity in_black=0,in_white=1,gamma=1 = skip (no-op default). Runs
// AFTER hsl, BEFORE the look (last of the two P7 color filters). Per-pixel in place — no scratch.
void fpx_gpu_levels(float in_black, float in_white, float gamma){
  if(!g_ready) return;
  int identity = (in_black>-0.001f && in_black<0.001f) && (in_white>0.999f && in_white<1.001f) && (gamma>0.999f && gamma<1.001f);
  if(identity) return; // identity levels: leave OUTB untouched.
  clSetKernelArg(kLevels,0,sizeof(cl_mem),&g_buf[OUTB]);
  clSetKernelArg(kLevels,1,sizeof(float),&in_black); clSetKernelArg(kLevels,2,sizeof(float),&in_white); clSetKernelArg(kLevels,3,sizeof(float),&gamma);
  launch(kLevels);
}
// P8 MOSAIC (pixelate): block-average via top-left sampling. block<=1 = skip (no-op default, and the
// div-by-zero guard: block 0 and 1 both skip). The kernel cannot read+write OUTB in place (one source
// pixel per block read by many threads = read/write race), so copy OUTB->g_tmp first (via k_copy, same
// convention as transform/blur/sharpen/flip), then sample g_tmp's block top-left into OUTB. Runs AFTER
// the P7 levels, BEFORE the look (first of the two P8 filters, per pinned order).
void fpx_gpu_mosaic(int block){
  if(!g_ready || block<=1) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kMosaic,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kMosaic,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kMosaic,2,sizeof(int),&block);
  launch(kMosaic);
}
// P8 GRADIENT MAP: luma -> colour ramp, IN PLACE on OUTB (per-own-pixel, no scratch — like hsl/levels).
// luma=dot(rgb,[.299,.587,.114]); mapped=mix(lo,hi,luma); rgb=mix(rgb,mapped,amt). amt<=0 = skip
// (no-op default). Runs AFTER mosaic, BEFORE the look (last of the two P8 filters).
void fpx_gpu_gmap(float amt, float lo_r, float lo_g, float lo_b, float hi_r, float hi_g, float hi_b){
  if(!g_ready || amt<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kGmap,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kGmap,1,sizeof(float),&amt);
  clSetKernelArg(kGmap,2,sizeof(float),&lo_r); clSetKernelArg(kGmap,3,sizeof(float),&lo_g); clSetKernelArg(kGmap,4,sizeof(float),&lo_b);
  clSetKernelArg(kGmap,5,sizeof(float),&hi_r); clSetKernelArg(kGmap,6,sizeof(float),&hi_g); clSetKernelArg(kGmap,7,sizeof(float),&hi_b);
  launch(kGmap);
}
// P9 DENOISE: edge-preserving 5x5 bilateral smooth blended by `strength`. strength<=0 = skip
// (no-op default). The kernel cannot read+write OUTB in place (a 5x5 neighbourhood read by many
// threads vs an in-place write = read/write race), so copy OUTB->g_tmp first (via k_copy, same
// convention as transform/blur/sharpen/flip/mosaic), then read g_tmp neighbours into OUTB. Runs
// AFTER the P8 gradient-map, BEFORE the look (first of the three P9 filters, per pinned order).
void fpx_gpu_denoise(float strength){
  if(!g_ready || strength<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kDenoise,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kDenoise,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kDenoise,2,sizeof(float),&strength);
  launch(kDenoise);
}
// P9 GLOW (bloom): bright-pass extract -> blur -> add back. amt<=0 = skip (no-op default). This runs
// as its OWN self-contained sequence using BOTH scratch buffers (g_tmp2 + g_tmp): extract the bright
// pass OUTB->g_tmp2, blur it at a FIXED sigma by REUSING the separable gaussian (kBlurH g_tmp2->g_tmp,
// kBlurV g_tmp->g_tmp2 — same arg order as fpx_gpu_blur), then combine OUTB = clamp01(OUTB +
// amt*g_tmp2). The blur ping-pongs g_tmp2->g_tmp->g_tmp2, so the final blurred bright pass lands back
// in g_tmp2 (what k_glow_combine reads). g_tmp is only borrowed mid-sequence and is not relied on
// after; OUTB is untouched until the combine. Runs after denoise, before rgb-shift.
void fpx_gpu_glow(float amt, float thr){
  if(!g_ready || amt<=0.0f) return; // no-op default: leave OUTB untouched.
  float sigma=8.0f; // fixed bloom blur radius (~ceil(2*8)=16px each side, capped at 32 by the kernel)
  // 1) bright pass: OUTB (luma>thr ? rgb : 0) -> g_tmp2.
  clSetKernelArg(kGlowExtract,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kGlowExtract,1,sizeof(cl_mem),&g_tmp2); clSetKernelArg(kGlowExtract,2,sizeof(float),&thr);
  launch(kGlowExtract);
  // 2) blur the bright pass: horiz g_tmp2->g_tmp, vert g_tmp->g_tmp2 (final blurred pass in g_tmp2).
  clSetKernelArg(kBlurH,0,sizeof(cl_mem),&g_tmp2); clSetKernelArg(kBlurH,1,sizeof(cl_mem),&g_tmp); clSetKernelArg(kBlurH,2,sizeof(float),&sigma); launch(kBlurH);
  clSetKernelArg(kBlurV,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kBlurV,1,sizeof(cl_mem),&g_tmp2); clSetKernelArg(kBlurV,2,sizeof(float),&sigma); launch(kBlurV);
  // 3) combine: OUTB = clamp01(OUTB + amt*g_tmp2).
  clSetKernelArg(kGlowCombine,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kGlowCombine,1,sizeof(cl_mem),&g_tmp2); clSetKernelArg(kGlowCombine,2,sizeof(float),&amt);
  launch(kGlowCombine);
}
// P9 RGB-SHIFT (chromatic aberration): channel-offset sample by `px` pixels. px<=0 = skip (no-op
// default). The kernel cannot read+write OUTB in place (it samples neighbouring x columns), so copy
// OUTB->g_tmp first, then sample g_tmp's r at (x+off,y), b at (x-off,y), g/a at (x,y) into OUTB. The
// offset is rounded to the nearest integer pixel. Runs after glow, before the look (last P9 filter).
void fpx_gpu_rgbshift(float px){
  if(!g_ready || px<=0.0f) return; // no-op default: leave OUTB untouched.
  int off=(int)(px+0.5f); if(off<1) off=1; // rounded, at least 1px (px>0 here)
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kRgbshift,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kRgbshift,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kRgbshift,2,sizeof(int),&off);
  launch(kRgbshift);
}
// P10 HALFTONE (luma-driven dot screen): cell<=1 = skip (no-op default; 0 and 1 both skip — and 1
// would also collapse the cell to a single pixel). The kernel cannot read+write OUTB in place (many
// threads in a cell read ONE cell-centre source pixel = read/write race), so copy OUTB->g_tmp first
// (via k_copy, same convention as the P6/P8/P9 spatial filters), then read g_tmp's cell-centre luma
// into OUTB (black dots vs white). Runs AFTER the P9 rgb-shift, BEFORE the look (first of the three
// P10 filters, per pinned order).
void fpx_gpu_halftone(int cell){
  if(!g_ready || cell<=1) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kHalftone,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kHalftone,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kHalftone,2,sizeof(int),&cell);
  launch(kHalftone);
}
// P10 EMBOSS (directional relief): amt<=0 = skip (no-op default). The kernel reads a NW neighbour, so
// it cannot read+write OUTB in place; copy OUTB->g_tmp first, then read g_tmp (centre - NW) into OUTB
// (out=clamp01(0.5+amt*diff) per channel). Runs after halftone, before edge.
void fpx_gpu_emboss(float amt){
  if(!g_ready || amt<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kEmboss,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kEmboss,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kEmboss,2,sizeof(float),&amt);
  launch(kEmboss);
}
// P10 EDGE/SKETCH (Sobel edge-detect mixed back): mix<=0 = skip (no-op default). The kernel reads a
// 3x3 neighbourhood, so it cannot read+write OUTB in place; copy OUTB->g_tmp first, then read g_tmp's
// 3x3 luma (Sobel gx/gy, mag=clamp01(hypot)) into OUTB (out=mix*vec3(mag)+(1-mix)*orig). Runs after
// emboss, before the look (last of the three P10 filters).
void fpx_gpu_edge(float mix){
  if(!g_ready || mix<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kEdge,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kEdge,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kEdge,2,sizeof(float),&mix);
  launch(kEdge);
}
// P13 GRAIN (film noise): amt<=0 = skip (no-op default). The kernel adds a per-pixel hashed luma
// noise (same on all 3 channels), so it could run in place, but for consistency with the other P13
// spatial filters (and so all three follow one g_tmp copy convention) copy OUTB->g_tmp first, then
// read g_tmp into OUTB. Runs AFTER the P10 edge, BEFORE the look (first of the three P13 filters,
// per pinned order).
void fpx_gpu_grain(float amt){
  if(!g_ready || amt<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kGrain,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kGrain,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kGrain,2,sizeof(float),&amt);
  launch(kGrain);
}
// P13 SCRATCHES (old-film vertical lines): amt<=0 = skip (no-op default). The per-column scratch
// decision/offset is the same for every pixel in a column, so the kernel could run in place, but we
// copy OUTB->g_tmp first (same convention as the other P13 filters), then read g_tmp into OUTB.
// Runs after grain, before diffusion.
void fpx_gpu_scratches(float amt){
  if(!g_ready || amt<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kScratches,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kScratches,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kScratches,2,sizeof(float),&amt);
  launch(kScratches);
}
// P13 DIFFUSION (frosted-glass jitter): radius<=0 = skip (no-op default). The kernel samples a
// jittered NEIGHBOUR, so it cannot read+write OUTB in place; copy OUTB->g_tmp first, then sample
// g_tmp's hashed neighbour (clamped) into OUTB. Runs after scratches, before the look (last of the
// three P13 filters).
void fpx_gpu_diffusion(float radius){
  if(!g_ready || radius<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kDiffuse,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kDiffuse,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kDiffuse,2,sizeof(float),&radius);
  launch(kDiffuse);
}
// P16 WAVE (horizontal sinusoidal row displacement): amp<=0 = skip (no-op default). The kernel samples
// a shifted source column (bilinear), so it cannot read+write OUTB in place; copy OUTB->g_tmp first
// (same convention as the P13 spatial filters), then read g_tmp's per-row shifted column into OUTB.
// Runs AFTER the P13 diffusion, BEFORE the look (first of the three P16 filters, per pinned order).
void fpx_gpu_wave(float amp){
  if(!g_ready || amp<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kWave,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kWave,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kWave,2,sizeof(float),&amp);
  launch(kWave);
}
// P16 SWIRL (rotational distortion around the centre): strength<=0 = skip (no-op default). The kernel
// samples a rotated source coordinate, so it cannot read+write OUTB in place; copy OUTB->g_tmp first,
// then sample g_tmp's rotated source (clamped) into OUTB. The rotation angle falls off from the centre
// to the rim (ang=strength*(1-r/maxr)). Runs after wave, before threshold.
void fpx_gpu_swirl(float strength){
  if(!g_ready || strength<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kSwirl,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kSwirl,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kSwirl,2,sizeof(float),&strength);
  launch(kSwirl);
}
// P16 THRESHOLD (luma binarize): level<=0 = skip (no-op default). PER-PIXEL IN PLACE on OUTB (own
// pixel only, no g_tmp copy — like hsl/levels/gmap): luma=dot(rgb,[.299,.587,.114]); out.rgb =
// luma>=level ? 1 : 0; alpha passthrough. Runs after swirl, before the look (last of the three P16
// filters).
void fpx_gpu_threshold(float level){
  if(!g_ready || level<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kThreshold,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kThreshold,1,sizeof(float),&level);
  launch(kThreshold);
}
// P17 LENS (radial barrel/pincushion distortion): k==0 = skip (no-op default; BOTH signs are active,
// only exact 0 is the no-op). The kernel samples a radially-scaled SOURCE coordinate, so it cannot
// read+write OUTB in place; copy OUTB->g_tmp first (same convention as the P13/P16 spatial filters),
// then sample g_tmp's radial source (clamped) into OUTB. Runs AFTER the P16 threshold, BEFORE the look
// (first of the three P17 filters, per pinned order). k>0 = barrel, k<0 = pincushion.
void fpx_gpu_lens(float k){
  if(!g_ready || k==0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kLens,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kLens,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kLens,2,sizeof(float),&k);
  launch(kLens);
}
// P17 CROP (margin to black): margin<=0 = skip (no-op default). PER-PIXEL IN PLACE on OUTB (own pixel
// only, no g_tmp copy — like threshold/hsl/levels): the centred keep-rect survives, everything OUTSIDE
// it has its RGB zeroed (alpha untouched). Runs after lens, before glitch.
void fpx_gpu_crop(float margin){
  if(!g_ready || margin<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCrop,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCrop,1,sizeof(float),&margin);
  launch(kCrop);
}
// P17 GLITCH (per-band horizontal channel shift): maxpx<=0 = skip (no-op default). The kernel samples
// a per-band horizontally-shifted source with R/B channel separation, so it cannot read+write OUTB in
// place; copy OUTB->g_tmp first, then sample g_tmp's shifted columns (clamped) into OUTB. The band
// shift is a DETERMINISTIC integer band hash (no time/RNG seed) so the gates stay stable. Runs after
// crop, before the look (last of the three P17 filters).
void fpx_gpu_glitch(float maxpx){
  if(!g_ready || maxpx<=0.0f) return; // no-op default: leave OUTB untouched.
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kGlitch,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kGlitch,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kGlitch,2,sizeof(float),&maxpx);
  launch(kGlitch);
}
// P23 360 REFRAME (equirectangular -> rectilinear): enable==0 (or fov out of (0,180)) = skip (no-op
// default — OUTB untouched → byte-identical to pre-P23). The kernel samples a reprojected SOURCE
// coordinate, so it cannot read+write OUTB in place; copy OUTB->g_tmp first (same convention as the
// P17 lens/glitch spatial filters), then sample g_tmp's reprojected source (nearest, u wraps / v clamps)
// into OUTB. yaw/pitch given in degrees → radians; htan = tan(fov/2). yaw>0 pans the view RIGHT
// (samples u>0.5), yaw<0 LEFT. Runs AFTER the P17 glitch, BEFORE the look.
void fpx_gpu_eq2rect(int enable, float yaw_deg, float pitch_deg, float fov_deg){
  if(!g_ready || !enable || fov_deg<=0.0f || fov_deg>=180.0f) return; // no-op default: leave OUTB untouched.
  float dr=(float)(M_PI/180.0);
  float yaw=yaw_deg*dr, pitch=pitch_deg*dr;
  float htan=tanf(fov_deg*0.5f*dr);
  clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy);
  clSetKernelArg(kEq2rect,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kEq2rect,1,sizeof(cl_mem),&g_buf[OUTB]);
  clSetKernelArg(kEq2rect,2,sizeof(float),&yaw); clSetKernelArg(kEq2rect,3,sizeof(float),&pitch); clSetKernelArg(kEq2rect,4,sizeof(float),&htan);
  launch(kEq2rect);
}
// P34 SHAPE MASK (centred rect/ellipse, feathered, optional invert): shape==0 = skip (no-op default →
// OUTB untouched → byte-identical to pre-P34). IN-PLACE on OUTB (each pixel scales only itself, like
// k_crop — NO g_tmp copy). cx/cy = mask centre (normalized 0..1), rw/rh = half-extents (normalized),
// feather = soft-edge band width (normalized), inv (1/0) flips inside<->outside. Runs AFTER the P23
// reframe, BEFORE the look. shape 1 = rectangle (Chebyshev distance), 2 = ellipse (euclidean radius).
void fpx_gpu_mask(int shape,float cx,float cy,float rw,float rh,float feather,int inv){
  if(!g_ready || shape==0) return;
  clSetKernelArg(kMask,0,sizeof(cl_mem),&g_buf[OUTB]);
  clSetKernelArg(kMask,1,sizeof(int),&shape);
  clSetKernelArg(kMask,2,sizeof(float),&cx); clSetKernelArg(kMask,3,sizeof(float),&cy);
  clSetKernelArg(kMask,4,sizeof(float),&rw); clSetKernelArg(kMask,5,sizeof(float),&rh);
  clSetKernelArg(kMask,6,sizeof(float),&feather); clSetKernelArg(kMask,7,sizeof(int),&inv);
  launch(kMask);
}
// P38 DISTORTION BATCH (mirror / kaleidoscope / dither): run on OUTB AFTER the P34 mask, BEFORE the
// look. Each is a no-op at its default (mirror_x 0 / kaleido <2 / dither 0): the wrapper returns early,
// leaving OUTB byte-identical to pre-P38. mirror/kaleido SAMPLE other pixels (copy OUTB->g_tmp via
// kCopy, then sample g_tmp into OUTB — mirror the fpx_gpu_lens copy pattern); dither runs in place.
void fpx_gpu_mirror(int on){ if(!g_ready || !on) return; clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy); clSetKernelArg(kMirror,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kMirror,1,sizeof(cl_mem),&g_buf[OUTB]); launch(kMirror); }
void fpx_gpu_kaleido(int seg){ if(!g_ready || seg<2) return; clSetKernelArg(kCopy,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kCopy,1,sizeof(cl_mem),&g_tmp); launch(kCopy); clSetKernelArg(kKaleido,0,sizeof(cl_mem),&g_tmp); clSetKernelArg(kKaleido,1,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kKaleido,2,sizeof(int),&seg); launch(kKaleido); }
void fpx_gpu_dither(float amt){ if(!g_ready || amt<=0.0f) return; clSetKernelArg(kDither,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kDither,1,sizeof(float),&amt); launch(kDither); }
// P39 selective color: rotate hue + scale saturation of ONE hue band on OUTB in place. band==0 -> no-op
// (skip) so the band-0 result is byte-identical to pre-P39. Runs after dither, before the look.
void fpx_gpu_selcolor(int band,float hshift,float ssat){
  if(!g_ready || band==0) return;
  clSetKernelArg(kSelcolor,0,sizeof(cl_mem),&g_buf[OUTB]);
  clSetKernelArg(kSelcolor,1,sizeof(int),&band); clSetKernelArg(kSelcolor,2,sizeof(float),&hshift); clSetKernelArg(kSelcolor,3,sizeof(float),&ssat);
  launch(kSelcolor);
}
// P41 solarize: per-channel v>thr -> 1-v on OUTB in place. thr<=0 -> no-op (skip) so the default result
// is byte-identical to pre-P41. Runs after selective color, before colour temperature/look.
void fpx_gpu_solarize(float thr){ if(thr<=0.0f) return; clSetKernelArg(kSolarize,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kSolarize,1,sizeof(float),&thr); launch(kSolarize); }
// P41 colour temperature: warm (t>0) raises R & lowers B; cool (t<0) the reverse; green unchanged. On OUTB
// in place. t==0 -> no-op (skip) so the default result is byte-identical to pre-P41. Runs after solarize.
void fpx_gpu_temp(float t){ if(t==0.0f) return; clSetKernelArg(kTemp,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kTemp,1,sizeof(float),&t); launch(kTemp); }
// P45 video fade-to-black: multiply OUTB rgb by f. f>=1 is a no-op (caller skips), so a non-faded frame is byte-identical.
void fpx_gpu_fade(float f){ if(f>=1.0f) return; if(f<0.0f) f=0.0f; clSetKernelArg(kFade,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kFade,1,sizeof(float),&f); launch(kFade); }
// look: 0=none (final=out), 1=vhs, 2=lut3d -> final=look ; returns 1 if final is LOOK else 0
int fpx_gpu_look(int kind, float amt, int lut_n){
  if(!g_ready) return 0;
  if(kind==1){ clSetKernelArg(kVhs,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kVhs,1,sizeof(cl_mem),&g_buf[LOOKB]); clSetKernelArg(kVhs,2,sizeof(float),&amt); launch(kVhs); return 1; }
  if(kind==2){ clSetKernelArg(kLut,0,sizeof(cl_mem),&g_buf[OUTB]); clSetKernelArg(kLut,1,sizeof(cl_mem),&g_buf[LOOKB]); clSetKernelArg(kLut,2,sizeof(cl_mem),&g_lut);
    clSetKernelArg(kLut,3,sizeof(int),&lut_n); clSetKernelArg(kLut,4,sizeof(float),&amt); launch(kLut); return 1; }
  return 0;
}
// download final buffer (0=out,1=look) -> host f32 (blocking)
void fpx_gpu_download_f32(int final_is_look, float* out){
  if(!g_ready||!out) return; int b = final_is_look?LOOKB:OUTB;
  clEnqueueReadBuffer(g_q,g_buf[b],CL_TRUE,0,GN*sizeof(float),out,0,NULL,NULL);
}
// download final -> host u8 (pack on GPU, then read)
void fpx_gpu_download_u8(int final_is_look, unsigned char* out){
  if(!g_ready||!out) return; int b = final_is_look?LOOKB:OUTB;
  clSetKernelArg(kPack,0,sizeof(cl_mem),&g_buf[b]); clSetKernelArg(kPack,1,sizeof(cl_mem),&g_stage); launch(kPack);
  clEnqueueReadBuffer(g_q,g_stage,CL_TRUE,0,GN*sizeof(unsigned char),out,0,NULL,NULL);
}
// scope: RGB histogram of the final buffer (0=out,1=look) -> out_hist[768] (R 0..255, G 256..511, B 512..767)
void fpx_gpu_histogram(int final_is_look, int* out_hist){
  if(!g_ready||!out_hist) return; int b = final_is_look?LOOKB:OUTB;
  clSetKernelArg(kHistClear,0,sizeof(cl_mem),&g_hist);
  size_t hg=768, hl=64; clEnqueueNDRangeKernel(g_q,kHistClear,1,NULL,&hg,&hl,0,NULL,NULL);
  clSetKernelArg(kHist,0,sizeof(cl_mem),&g_buf[b]); clSetKernelArg(kHist,1,sizeof(cl_mem),&g_hist); launch(kHist);
  clEnqueueReadBuffer(g_q,g_hist,CL_TRUE,0,768*sizeof(int),out_hist,0,NULL,NULL);
}
// scope helper: clear the 256x256 grid, accumulate (acc kernel over the frame), render to g_scope, read 256x256x4 u8
static void scope_render(int final_is_look, cl_kernel acc, cl_kernel img, float gain, unsigned char* out){
  int b = final_is_look?LOOKB:OUTB;
  clSetKernelArg(kGridClear,0,sizeof(cl_mem),&g_grid);
  size_t gg=65536, gl=64; clEnqueueNDRangeKernel(g_q,kGridClear,1,NULL,&gg,&gl,0,NULL,NULL);
  clSetKernelArg(acc,0,sizeof(cl_mem),&g_buf[b]); clSetKernelArg(acc,1,sizeof(cl_mem),&g_grid); launch(acc);
  clSetKernelArg(img,0,sizeof(cl_mem),&g_grid); clSetKernelArg(img,1,sizeof(cl_mem),&g_scope); clSetKernelArg(img,2,sizeof(float),&gain);
  size_t sg[2]={256,256}, sl[2]={8,8}; clEnqueueNDRangeKernel(g_q,img,2,NULL,sg,sl,0,NULL,NULL);
  clEnqueueReadBuffer(g_q,g_scope,CL_TRUE,0,256*256*4,out,0,NULL,NULL);
}
// luma waveform -> RGBA8 256x256 (out)
void fpx_gpu_waveform(int final_is_look, unsigned char* out){ if(g_ready&&out) scope_render(final_is_look,kWaveAcc,kWaveImg,0.06f,out); }
// vectorscope (U/V) -> RGBA8 256x256 (out)
void fpx_gpu_vectorscope(int final_is_look, unsigned char* out){ if(g_ready&&out) scope_render(final_is_look,kVecAcc,kVecImg,0.04f,out); }
// RGB parade (Triad-B P1): per-channel column waveform, 3 side-by-side panels (R|G|B) -> RGBA8
// 256x256 (out). Uses the dedicated 3*256*256 int parade grid: clear, accumulate all 3 channels over
// the frame, render the three panels into g_scope, read back. Same final-buffer selection as the
// other scopes (OUTB / LOOKB). Gain matches the waveform's so a busy frame fills similarly.
void fpx_gpu_parade(int final_is_look, unsigned char* out){
  if(!g_ready||!out) return;
  int b = final_is_look?LOOKB:OUTB;
  clSetKernelArg(kParadeClear,0,sizeof(cl_mem),&g_parade);
  size_t pg=3*65536, pl=64; clEnqueueNDRangeKernel(g_q,kParadeClear,1,NULL,&pg,&pl,0,NULL,NULL);
  clSetKernelArg(kParadeAcc,0,sizeof(cl_mem),&g_buf[b]); clSetKernelArg(kParadeAcc,1,sizeof(cl_mem),&g_parade); launch(kParadeAcc);
  float gain=0.06f;
  clSetKernelArg(kParadeImg,0,sizeof(cl_mem),&g_parade); clSetKernelArg(kParadeImg,1,sizeof(cl_mem),&g_scope); clSetKernelArg(kParadeImg,2,sizeof(float),&gain);
  size_t sg[2]={256,256}, sl[2]={8,8}; clEnqueueNDRangeKernel(g_q,kParadeImg,2,NULL,sg,sl,0,NULL,NULL);
  clEnqueueReadBuffer(g_q,g_scope,CL_TRUE,0,256*256*4,out,0,NULL,NULL);
}
void fpx_gpu_finish(void){ if(g_ready) clFinish(g_q); }
