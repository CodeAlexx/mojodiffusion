// bench_bf16_gemm.cpp — microbench: cublasGemmEx bf16 (Ampere-compat CUTLASS on
// Blackwell) vs cuBLASLt bf16 heuristic (native sm_120 kernel?) on Klein's real
// base-GEMM shapes. bf16 in/out, CUBLAS_COMPUTE_32F accumulate, tensor-op math.
//
// Reproduces the EXACT op/ld convention of the shim's bf16 path
// (serenity_cublas_gemm_bf16_rowmajor_nt): the row-major GEMM C[M,N] =
// A[M,K] @ B[N,K]ᵀ maps to cublas OP_T on B, OP_N on A, m=N n=M k=K,
// lda=ldb=K, ldc=N. Reports TOPS + µs/GEMM for GemmEx and Lt, the speedup,
// the Lt algoId(s) (and whether the heuristic returns >1 distinct algo), and
// f32-output parity (cos) between the two.
//
// Build (see build_bench_bf16.sh): links -lcublas -lcublasLt -lcudart.

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#define CK(x) do { auto _e=(x); if(_e!=cudaSuccess){ \
    fprintf(stderr,"cuda err %d at %s:%d\n",(int)_e,__FILE__,__LINE__); exit(1);} } while(0)
#define CB(x) do { auto _s=(x); if(_s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cublas err %d at %s:%d\n",(int)_s,__FILE__,__LINE__); exit(1);} } while(0)

static cublasHandle_t   g_h;
static cublasLtHandle_t g_lt;
static void*            g_ws = nullptr;
static size_t           g_ws_bytes = 128ull*1024*1024;   // generous for bf16 tiles

// ---- GemmEx bf16 NT (identical args to the shim's bf16 path) ---------------
static void gemmex_nt(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                      int M, int N, int K, cudaStream_t s) {
    const float alpha=1.f, beta=0.f;
    CB(cublasSetStream(g_h, s));
    CB(cublasGemmEx(g_h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                    &alpha, B, CUDA_R_16BF, K, A, CUDA_R_16BF, K,
                    &beta,  C, CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ---- Lt bf16 NT: build once, capture up to K heuristic algos ---------------
struct LtPlan {
    cublasLtMatmulDesc_t   op = nullptr;
    cublasLtMatrixLayout_t Ad=nullptr, Bd=nullptr, Cd=nullptr;
    cublasLtMatmulHeuristicResult_t heur[8];
    int nheur = 0;
    bool ok = false;
    std::string algo_info;
};

static LtPlan lt_plan_nt(int M, int N, int K) {
    LtPlan p;
    // mirror the shim: m=N, n=M, k=K, opA=T on B buf, opB=N on A buf
    int m=N, n=M, k=K;
    cublasOperation_t opA=CUBLAS_OP_T, opB=CUBLAS_OP_N;
    CB(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA,&opA,sizeof(opA));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB,&opB,sizeof(opB));
    uint64_t Ar=k, Ac=m;   // opA=T -> B buffer stored k×m, ld=K
    uint64_t Br=k, Bc=n;   // opB=N -> A buffer stored k×n, ld=K
    CB(cublasLtMatrixLayoutCreate(&p.Ad, CUDA_R_16BF, Ar, Ac, K));
    CB(cublasLtMatrixLayoutCreate(&p.Bd, CUDA_R_16BF, Br, Bc, K));
    CB(cublasLtMatrixLayoutCreate(&p.Cd, CUDA_R_32F, (uint64_t)m,(uint64_t)n, N));
    cublasLtMatmulPreference_t pref=nullptr;
    CB(cublasLtMatmulPreferenceCreate(&pref));
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&g_ws_bytes,sizeof(g_ws_bytes));
    int returned=0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
        g_lt, p.op, p.Ad, p.Bd, p.Cd, p.Cd, pref, 8, p.heur, &returned);
    p.ok = (hs==CUBLAS_STATUS_SUCCESS && returned>0);
    p.nheur = returned;
    if (p.ok) {
        char buf[256]; int off=0;
        off += snprintf(buf+off,sizeof(buf)-off,"n=%d ids=[",returned);
        for (int i=0;i<returned && i<8;i++){
            int algoId=-1;
            cublasLtMatmulAlgoConfigGetAttribute(&p.heur[i].algo,
                CUBLASLT_ALGO_CONFIG_ID,&algoId,sizeof(algoId),nullptr);
            off += snprintf(buf+off,sizeof(buf)-off,"%s%d",i?",":"",algoId);
        }
        off += snprintf(buf+off,sizeof(buf)-off,"] ws=%zu",p.heur[0].workspaceSize);
        p.algo_info = buf;
    }
    cublasLtMatmulPreferenceDestroy(pref);
    return p;
}

static void lt_run(const LtPlan& p, int which, const __nv_bfloat16* A,
                   const __nv_bfloat16* B, float* C, cudaStream_t s) {
    const float alpha=1.f, beta=0.f;
    // B buffer maps to layout Ad (the OP_T operand), A buffer maps to layout Bd.
    CB(cublasLtMatmul(g_lt, p.op, &alpha, B, p.Ad, A, p.Bd, &beta,
                      C, p.Cd, C, p.Cd, &p.heur[which].algo, g_ws, g_ws_bytes, s));
}

static double ms_event(cudaEvent_t a, cudaEvent_t b){ float ms; CK(cudaEventElapsedTime(&ms,a,b)); return ms; }

struct Shape { int M,N,K; const char* name; };

int main(){
    CB(cublasCreate(&g_h));
    CB(cublasLtCreate(&g_lt));
    CK(cudaMalloc(&g_ws, g_ws_bytes));

    std::vector<Shape> shapes = {
        {1536, 12288, 4096, "qkv/ffB single [M=1536 N=12288 K=4096]"},
        {1536,  4096, 4096, "proj single    [M=1536 N=4096  K=4096]"},
        {1536, 24576, 4096, "ff-in single   [M=1536 N=24576 K=4096]"},
        {3072, 12288, 4096, "qkv batch2     [M=3072 N=12288 K=4096]"},
        {4096,  4096, 4096, "square         [M=4096 N=4096  K=4096]"},
    };

    const int WARM=15, ITERS=60;
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    cudaStream_t s; CK(cudaStreamCreate(&s));

    printf("%-38s | %-26s | %-26s | speedup | parity | Lt-algo(s)\n",
           "shape","GemmEx bf16 (compat)","cuBLASLt bf16 (heuristic)");
    printf("%.*s\n", 150, "--------------------------------------------------------------------------------------------------------------------------------------------------------");

    for (auto& sh : shapes) {
        int M=sh.M, N=sh.N, K=sh.K;
        size_t aN=(size_t)M*K, bN=(size_t)N*K, cN=(size_t)M*N;
        __nv_bfloat16 *A,*B; float *Cg,*Cl;
        CK(cudaMalloc((void**)&A, aN*2)); CK(cudaMalloc((void**)&B, bN*2));
        CK(cudaMalloc((void**)&Cg, cN*4)); CK(cudaMalloc((void**)&Cl, cN*4));
        // deterministic non-degenerate fill in [-1,1], bf16-rounded
        std::vector<__nv_bfloat16> ha(aN), hb(bN);
        for (size_t i=0;i<aN;i++){ float v=((float)((i*131+7)%1000)/500.0f)-1.0f; ha[i]=__float2bfloat16(v); }
        for (size_t i=0;i<bN;i++){ float v=((float)((i*97+13)%1000)/500.0f)-1.0f; hb[i]=__float2bfloat16(v); }
        CK(cudaMemcpy(A,ha.data(),aN*2,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(B,hb.data(),bN*2,cudaMemcpyHostToDevice));

        double ops = 2.0*(double)M*(double)N*(double)K;

        // -- GemmEx timing --
        for(int i=0;i<WARM;i++) gemmex_nt(A,B,Cg,M,N,K,s);
        CK(cudaStreamSynchronize(s));
        CK(cudaEventRecord(e0,s));
        for(int i=0;i<ITERS;i++) gemmex_nt(A,B,Cg,M,N,K,s);
        CK(cudaEventRecord(e1,s)); CK(cudaEventSynchronize(e1));
        double g_us = ms_event(e0,e1)*1000.0/ITERS;
        double g_tops = ops/(g_us*1e-6)/1e12;

        // -- Lt timing (use best heuristic algo = index 0) --
        LtPlan p = lt_plan_nt(M,N,K);
        double l_us=0, l_tops=0; std::string lt_note;
        if (p.ok) {
            for(int i=0;i<WARM;i++) lt_run(p,0,A,B,Cl,s);
            CK(cudaStreamSynchronize(s));
            CK(cudaEventRecord(e0,s));
            for(int i=0;i<ITERS;i++) lt_run(p,0,A,B,Cl,s);
            CK(cudaEventRecord(e1,s)); CK(cudaEventSynchronize(e1));
            l_us = ms_event(e0,e1)*1000.0/ITERS;
            l_tops = ops/(l_us*1e-6)/1e12;
            lt_note = p.algo_info;
        } else {
            lt_note = "NO HEURISTIC";
        }

        // -- parity (cos of f32 outputs) --
        std::string par="n/a";
        if (p.ok) {
            gemmex_nt(A,B,Cg,M,N,K,s);
            lt_run(p,0,A,B,Cl,s);
            CK(cudaStreamSynchronize(s));
            std::vector<float> hg(cN), hl(cN);
            CK(cudaMemcpy(hg.data(),Cg,cN*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hl.data(),Cl,cN*4,cudaMemcpyDeviceToHost));
            double dot=0,na=0,nb=0;
            for(size_t i=0;i<cN;i++){ double x=hg[i],y=hl[i]; dot+=x*y; na+=x*x; nb+=y*y; }
            double cos = (na>0&&nb>0)? dot/(sqrt(na)*sqrt(nb)) : 0.0;
            char pb[32]; snprintf(pb,sizeof(pb),"cos=%.5f",cos); par=pb;
        }

        char gcol[64], lcol[64];
        snprintf(gcol,sizeof(gcol),"%8.1f us %6.1f TOPS", g_us, g_tops);
        if (p.ok) snprintf(lcol,sizeof(lcol),"%8.1f us %6.1f TOPS", l_us, l_tops);
        else      snprintf(lcol,sizeof(lcol),"%-26s","(none)");
        printf("%-38s | %-26s | %-26s | %6.2fx | %-10s | %s\n",
               sh.name, gcol, lcol, p.ok? g_us/l_us : 0.0, par.c_str(), lt_note.c_str());

        cudaFree(A); cudaFree(B); cudaFree(Cg); cudaFree(Cl);
    }
    printf("\nnote: TOPS = 2*M*N*K / time. RTX 5080 bf16 tensor-core peak ~ 225 TFLOP/s\n");
    printf("      (fp32-accumulate, no sparsity; ~450 TOPS marketing figure is fp16-acc/sparse).\n");
    return 0;
}
