// cudnn_conv3d.cpp — low-startup BF16 NDHWC Conv3d for the LTX2 video VAE.
//
// MAX's generic Conv3d helper runs cudnnFindConvolutionForwardAlgorithmEx for
// every unseen shape in every fresh process. A 121-frame LTX2 decode has many
// distinct shapes, so that exhaustive search costs substantially more than the
// steady-state decoder. This shim uses cuDNN's non-executing v7 heuristic,
// capped at 256 MiB workspace, while preserving the same NDHWC/FCQRS layouts.

#include <cudnn.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

static cudnnHandle_t g_conv3d_handle = nullptr;
static std::mutex g_conv3d_mutex;
static void* g_conv3d_workspace = nullptr;
static size_t g_conv3d_workspace_bytes = 0;

static int ensure_conv3d_handle() {
    if (g_conv3d_handle) return 0;
    std::lock_guard<std::mutex> lock(g_conv3d_mutex);
    if (g_conv3d_handle) return 0;
    cudnnStatus_t status = cudnnCreate(&g_conv3d_handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "[serenity_cudnn_conv3d] cudnnCreate failed: %d\n",
                     (int)status);
        return (int)status;
    }
    return 0;
}

extern "C" int serenity_cudnn_conv3d_bf16_ndhwc(
    const void* input,
    const void* filter,
    const void* bias,
    void* output,
    int n,
    int din,
    int hin,
    int win,
    int cin,
    int cout,
    int kd,
    int kh,
    int kw,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    void* stream
) {
    if (!input || !filter || !output || n <= 0 || din <= 0 || hin <= 0 ||
        win <= 0 || cin <= 0 || cout <= 0 || kd <= 0 || kh <= 0 || kw <= 0 ||
        stride_d <= 0 || stride_h <= 0 || stride_w <= 0 || pad_d < 0 ||
        pad_h < 0 || pad_w < 0) {
        return -1;
    }

    const int dout = (din + 2 * pad_d - kd) / stride_d + 1;
    const int hout = (hin + 2 * pad_h - kh) / stride_h + 1;
    const int wout = (win + 2 * pad_w - kw) / stride_w + 1;
    if (dout <= 0 || hout <= 0 || wout <= 0) return -1;

    int result = ensure_conv3d_handle();
    if (result != 0) return result;

    std::lock_guard<std::mutex> lock(g_conv3d_mutex);
    cudnnStatus_t status =
        cudnnSetStream(g_conv3d_handle, (cudaStream_t)stream);
    if (status != CUDNN_STATUS_SUCCESS) return (int)status;

    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnTensorDescriptor_t b_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    result = 0;
    const char* failed_stage = "none";

    do {
        failed_stage = "create_x_desc";
        if ((status = cudnnCreateTensorDescriptor(&x_desc)) !=
            CUDNN_STATUS_SUCCESS) break;
        failed_stage = "create_y_desc";
        if ((status = cudnnCreateTensorDescriptor(&y_desc)) !=
            CUDNN_STATUS_SUCCESS) break;
        failed_stage = "create_w_desc";
        if ((status = cudnnCreateFilterDescriptor(&w_desc)) !=
            CUDNN_STATUS_SUCCESS) break;
        failed_stage = "create_conv_desc";
        if ((status = cudnnCreateConvolutionDescriptor(&conv_desc)) !=
            CUDNN_STATUS_SUCCESS) break;

        // Logical descriptor dimensions are N,C,D,H,W. NHWC format makes the
        // physical activation layout N,D,H,W,C, matching the Mojo Tensor.
        int x_dims[5] = {n, cin, din, hin, win};
        failed_stage = "set_x_desc";
        status = cudnnSetTensorNdDescriptorEx(
            x_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_BFLOAT16, 5, x_dims);
        if (status != CUDNN_STATUS_SUCCESS) break;

        int w_dims[5] = {cout, cin, kd, kh, kw};
        failed_stage = "set_w_desc";
        status = cudnnSetFilterNdDescriptor(
            w_desc, CUDNN_DATA_BFLOAT16, CUDNN_TENSOR_NCHW, 5, w_dims);
        if (status != CUDNN_STATUS_SUCCESS) break;

        int pads[3] = {pad_d, pad_h, pad_w};
        int strides[3] = {stride_d, stride_h, stride_w};
        int dilations[3] = {1, 1, 1};
        failed_stage = "set_conv_desc";
        status = cudnnSetConvolutionNdDescriptor(
            conv_desc, 3, pads, strides, dilations, CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT);
        if (status != CUDNN_STATUS_SUCCESS) break;
        failed_stage = "set_math_type";
        status = cudnnSetConvolutionMathType(
            conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
        if (status != CUDNN_STATUS_SUCCESS) break;

        int y_dims[5] = {n, cout, dout, hout, wout};
        failed_stage = "set_y_desc";
        status = cudnnSetTensorNdDescriptorEx(
            y_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_BFLOAT16, 5, y_dims);
        if (status != CUDNN_STATUS_SUCCESS) break;

        int returned = 0;
        cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        failed_stage = "get_algo_v7";
        status = cudnnGetConvolutionForwardAlgorithm_v7(
            g_conv3d_handle, x_desc, w_desc, conv_desc, y_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned, perf);
        if (status != CUDNN_STATUS_SUCCESS || returned <= 0) break;

        constexpr size_t workspace_cap = 256ULL * 1024ULL * 1024ULL;
        bool found = false;
        cudnnConvolutionFwdAlgo_t algo =
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        for (int i = 0; i < returned; ++i) {
            if (perf[i].status == CUDNN_STATUS_SUCCESS &&
                perf[i].memory <= workspace_cap) {
                algo = perf[i].algo;
                workspace_bytes = perf[i].memory;
                found = true;
                break;
            }
        }
        if (!found) {
            failed_stage = "select_algo";
            status = CUDNN_STATUS_NOT_SUPPORTED;
            break;
        }

        if (std::getenv("SERENITY_CUDNN_CONV3D_TRACE")) {
            std::fprintf(
                stderr,
                "[serenity_cudnn_conv3d] in=%dx%dx%dx%dx%d out=%dx%dx%dx%dx%d "
                "kernel=%dx%dx%d stride=%dx%dx%d algo=%d workspace=%zu\n",
                n, din, hin, win, cin, n, dout, hout, wout, cout, kd, kh, kw,
                stride_d, stride_h, stride_w, (int)algo, workspace_bytes);
        }

        if (!g_conv3d_workspace) {
            failed_stage = "alloc_workspace";
            cudaError_t cuda_status =
                cudaMalloc(&g_conv3d_workspace, workspace_cap);
            if (cuda_status != cudaSuccess) {
                std::fprintf(
                    stderr,
                    "[serenity_cudnn_conv3d] cudaMalloc(%zu) failed: %d %s\n",
                    workspace_cap, (int)cuda_status,
                    cudaGetErrorString(cuda_status));
                result = 1000 + (int)cuda_status;
                break;
            }
            g_conv3d_workspace_bytes = workspace_cap;
        }
        workspace = g_conv3d_workspace;

        const float alpha = 1.0f;
        const float beta = 0.0f;
        failed_stage = "convolution_forward";
        status = cudnnConvolutionForward(
            g_conv3d_handle, &alpha, x_desc, input, w_desc, filter, conv_desc,
            algo, workspace, workspace_bytes, &beta, y_desc, output);
        if (status != CUDNN_STATUS_SUCCESS &&
            algo != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) {
            // The v7 heuristic can report PRECOMP_GEMM for large NDHWC 5-D
            // shapes that the legacy execution API then rejects. The
            // zero-workspace implicit algorithm is the reliable fallback.
            if (std::getenv("SERENITY_CUDNN_CONV3D_TRACE")) {
                std::fprintf(
                    stderr,
                    "[serenity_cudnn_conv3d] algo=%d rejected (%d); "
                    "retrying IMPLICIT_GEMM\n",
                    (int)algo, (int)status);
            }
            algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
            workspace_bytes = 0;
            status = cudnnConvolutionForward(
                g_conv3d_handle, &alpha, x_desc, input, w_desc, filter,
                conv_desc, algo, nullptr, 0, &beta, y_desc, output);
        }
        if (status != CUDNN_STATUS_SUCCESS) break;

        if (bias) {
            failed_stage = "create_bias_desc";
            if ((status = cudnnCreateTensorDescriptor(&b_desc)) !=
                CUDNN_STATUS_SUCCESS) break;
            int b_dims[5] = {1, cout, 1, 1, 1};
            failed_stage = "set_bias_desc";
            status = cudnnSetTensorNdDescriptorEx(
                b_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_BFLOAT16, 5, b_dims);
            if (status != CUDNN_STATUS_SUCCESS) break;
            const float one = 1.0f;
            failed_stage = "add_bias";
            status = cudnnAddTensor(
                g_conv3d_handle, &one, b_desc, bias, &one, y_desc, output);
            if (status != CUDNN_STATUS_SUCCESS) break;
        }
    } while (false);

    if (result == 0 && status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(
            stderr,
            "[serenity_cudnn_conv3d] %s failed: %d %s "
            "in=%dx%dx%dx%dx%d kernel=%dx%dx%d out=%dx%dx%dx%dx%d\n",
            failed_stage, (int)status, cudnnGetErrorString(status), n, din,
            hin, win, cin, kd, kh, kw, n, dout, hout, wout, cout);
        result = (int)status;
    }
    if (b_desc) cudnnDestroyTensorDescriptor(b_desc);
    if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
    if (w_desc) cudnnDestroyFilterDescriptor(w_desc);
    if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
    if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
    return result;
}
