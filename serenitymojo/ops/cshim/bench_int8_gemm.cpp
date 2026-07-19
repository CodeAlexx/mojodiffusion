// bench_int8_gemm.cpp — microbench: cublasGemmEx int8 (Ampere compat) vs
// cuBLASLt native-int8 heuristic on Klein's real GEMM shapes.
//
// Times BOTH paths for the NT GEMM  C[M,N] int32 = A[M,K] @ B[N,K]ᵀ (the fwd
// shape used by int8_linear), reproducing the EXACT op/ld convention of the
// shim (OP_T on B, OP_N on A, m=N n=M k=K, lda=ldb=K, ldc=N). Reports
// TOPS and µs/GEMM for both plus the speedup ratio, and whether the Lt
// heuristic returned an algo at all (native kernel present?). Also checks
// bit-parity of the two int32 outputs.
//
// Build (see build_bench.sh): links -lcublas -lcublasLt -lcudart.

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>

#define CK(x) do { auto _e=(x); if(_e!=cudaSuccess){ \
    fprintf(stderr,"cuda err %d at %s:%d\n",(int)_e,__FILE__,__LINE__); exit(1);} } while(0)
#define CB(x) do { auto _s=(x); if(_s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cublas err %d at %s:%d\n",(int)_s,__FILE__,__LINE__); exit(1);} } while(0)

static cublasHandle_t   g_h;
static cublasLtHandle_t g_lt;
static void*            g_ws = nullptr;
static size_t           g_ws_bytes = 32ull*1024*1024;

// ---- GemmEx int8 NT (identical args to the shim) --------------------------
static void gemmex_nt(const int8_t* A, const int8_t* B, int32_t* C,
                      int M, int N, int K, cudaStream_t s) {
    const int32_t alpha=1, beta=0;
    CB(cublasSetStream(g_h, s));
    CB(cublasGemmEx(g_h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                    &alpha, B, CUDA_R_8I, K, A, CUDA_R_8I, K,
                    &beta,  C, CUDA_R_32I, N,
                    CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ---- Lt int8 NT: build once, return whether heuristic found an algo -------
struct LtPlan {
    cublasLtMatmulDesc_t   op = nullptr;
    cublasLtMatrixLayout_t Ad=nullptr, Bd=nullptr, Cd=nullptr;
    cublasLtMatmulHeuristicResult_t heur;
    bool ok = false;
    std::string algo_info;
};

static LtPlan lt_plan_nt(int M, int N, int K) {
    LtPlan p;
    // mirror lt_gemm_s8s8s32(OP_T,OP_N, m=N,n=M,k=K, B,K, A,K, C,N)
    int m=N, n=M, k=K;
    cublasOperation_t opA=CUBLAS_OP_T, opB=CUBLAS_OP_N;
    CB(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA,&opA,sizeof(opA));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB,&opB,sizeof(opB));
    uint64_t Ar=k, Ac=m;   // opA=T -> stored k×m
    uint64_t Br=k, Bc=n;   // opB=N -> stored k×n
    CB(cublasLtMatrixLayoutCreate(&p.Ad, CUDA_R_8I,  Ar, Ac, K));  // ldb of B buf = K
    CB(cublasLtMatrixLayoutCreate(&p.Bd, CUDA_R_8I,  Br, Bc, K));  // lda of A buf = K
    CB(cublasLtMatrixLayoutCreate(&p.Cd, CUDA_R_32I, (uint64_t)m,(uint64_t)n, N));
    cublasLtMatmulPreference_t pref=nullptr;
    CB(cublasLtMatmulPreferenceCreate(&pref));
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&g_ws_bytes,sizeof(g_ws_bytes));
    int returned=0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
        g_lt, p.op, p.Ad, p.Bd, p.Cd, p.Cd, pref, 1, &p.heur, &returned);
    p.ok = (hs==CUBLAS_STATUS_SUCCESS && returned>0);
    if (p.ok) {
        int algoId=-1;
        cublasLtMatmulAlgoConfigGetAttribute(&p.heur.algo,
            CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr);
        char buf[128];
        snprintf(buf,sizeof(buf),"algoId=%d ws=%zu", algoId, p.heur.workspaceSize);
        p.algo_info = buf;
    }
    cublasLtMatmulPreferenceDestroy(pref);
    return p;
}

static void lt_run(const LtPlan& p, const int8_t* A, const int8_t* B, int32_t* C,
                   cudaStream_t s) {
    const int32_t alpha=1, beta=0;
    // A buffer maps to layout Bd, B buffer maps to layout Ad (see shim: B is
    // the first cublasLt matrix "A", A is the second "B").
    CB(cublasLtMatmul(g_lt, p.op, &alpha, B, p.Ad, A, p.Bd, &beta,
                      C, p.Cd, C, p.Cd, &p.heur.algo, g_ws, g_ws_bytes, s));
}

static double now_ms_event(cudaEvent_t a, cudaEvent_t b){ float ms; CK(cudaEventElapsedTime(&ms,a,b)); return ms; }

struct Shape { int M,N,K; const char* name; };

int main(){
    CB(cublasCreate(&g_h));
    CB(cublasLtCreate(&g_lt));
    CK(cudaMalloc(&g_ws, g_ws_bytes));

    std::vector<Shape> shapes = {
        {1024, 36864, 4096, "w1 single-block  [M=1024 N=36864 K=4096]"},
        {1024, 24576, 4096, "double mlp 2F    [M=1024 N=24576 K=4096]"},
        {4096,  4096,  4096, "square           [4096 4096 4096]"},
    };

    const int WARM=10, ITERS=50;
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    cudaStream_t s; CK(cudaStreamCreate(&s));

    printf("%-42s | %-28s | %-28s | speedup | Lt-algo\n",
           "shape","GemmEx (compat)","cuBLASLt (heuristic)");
    printf("%.*s\n", 130, "----------------------------------------------------------------------------------------------------------------------------------------");

    for (auto& sh : shapes) {
        int M=sh.M, N=sh.N, K=sh.K;
        size_t aN=(size_t)M*K, bN=(size_t)N*K, cN=(size_t)M*N;
        int8_t *A,*B; int32_t *Cg,*Cl;
        CK(cudaMalloc((void**)&A, aN)); CK(cudaMalloc((void**)&B, bN));
        CK(cudaMalloc((void**)&Cg, cN*4)); CK(cudaMalloc((void**)&Cl, cN*4));
        // deterministic-ish fill on host then copy
        std::vector<int8_t> ha(aN), hb(bN);
        for (size_t i=0;i<aN;i++) ha[i]=(int8_t)(((i*131+7)%251)-125);
        for (size_t i=0;i<bN;i++) hb[i]=(int8_t)(((i*97+13)%251)-125);
        CK(cudaMemcpy(A,ha.data(),aN,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(B,hb.data(),bN,cudaMemcpyHostToDevice));

        // flops = 2*M*N*K
        double ops = 2.0*(double)M*(double)N*(double)K;

        // -- GemmEx timing --
        for(int i=0;i<WARM;i++) gemmex_nt(A,B,Cg,M,N,K,s);
        CK(cudaStreamSynchronize(s));
        CK(cudaEventRecord(e0,s));
        for(int i=0;i<ITERS;i++) gemmex_nt(A,B,Cg,M,N,K,s);
        CK(cudaEventRecord(e1,s)); CK(cudaEventSynchronize(e1));
        double g_us = now_ms_event(e0,e1)*1000.0/ITERS;
        double g_tops = ops/(g_us*1e-6)/1e12;

        // -- Lt timing --
        LtPlan p = lt_plan_nt(M,N,K);
        double l_us=0, l_tops=0; std::string lt_note;
        if (p.ok) {
            for(int i=0;i<WARM;i++) lt_run(p,A,B,Cl,s);
            CK(cudaStreamSynchronize(s));
            CK(cudaEventRecord(e0,s));
            for(int i=0;i<ITERS;i++) lt_run(p,A,B,Cl,s);
            CK(cudaEventRecord(e1,s)); CK(cudaEventSynchronize(e1));
            l_us = now_ms_event(e0,e1)*1000.0/ITERS;
            l_tops = ops/(l_us*1e-6)/1e12;
            lt_note = p.algo_info;
        } else {
            lt_note = "NO HEURISTIC (fallback)";
        }

        // -- parity check (int32 bit-exact) --
        std::string par="n/a";
        if (p.ok) {
            gemmex_nt(A,B,Cg,M,N,K,s);
            lt_run(p,A,B,Cl,s);
            CK(cudaStreamSynchronize(s));
            std::vector<int32_t> hg(cN), hl(cN);
            CK(cudaMemcpy(hg.data(),Cg,cN*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hl.data(),Cl,cN*4,cudaMemcpyDeviceToHost));
            size_t mism=0; for(size_t i=0;i<cN;i++) if(hg[i]!=hl[i]) mism++;
            par = mism==0 ? "BIT-EXACT" : ("MISMATCH:"+std::to_string(mism));
        }

        char gcol[64], lcol[64];
        snprintf(gcol,sizeof(gcol),"%7.1f us  %6.1f TOPS", g_us, g_tops);
        if (p.ok) snprintf(lcol,sizeof(lcol),"%7.1f us  %6.1f TOPS", l_us, l_tops);
        else      snprintf(lcol,sizeof(lcol),"%-28s","(none)");
        printf("%-42s | %-28s | %-28s | %6.2fx | %s [%s]\n",
               sh.name, gcol, lcol, p.ok? g_us/l_us : 0.0, lt_note.c_str(), par.c_str());

        cudaFree(A); cudaFree(B); cudaFree(Cg); cudaFree(Cl);
    }
    return 0;
}
