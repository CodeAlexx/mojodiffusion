// cudnn_conv2d.cpp — native NCHW BF16 Conv2d for Creator LTX audio parity.
//
// LTX's audio VAE is defined with torch.nn.Conv2d over NCHW tensors.  Routing
// it through the shared singleton-axis Conv3d helper is numerically close, but
// the deep vocoder amplifies the remaining BF16 differences.  This shim keeps
// the checkpoint's native OIHW filter and NCHW activation layout and calls the
// classic cuDNN Conv2d API used by the Creator implementation.

#include <cudnn.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <mutex>

static cudnnHandle_t g_conv2d_handle = nullptr;
static std::mutex g_conv2d_mutex;

static int ensure_conv2d_handle() {
    if (g_conv2d_handle) return 0;
    std::lock_guard<std::mutex> lock(g_conv2d_mutex);
    if (g_conv2d_handle) return 0;
    cudnnStatus_t status = cudnnCreate(&g_conv2d_handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "[serenity_cudnn_conv2d] cudnnCreate failed: %d\n", (int)status);
        return (int)status;
    }
    return 0;
}

extern "C" int serenity_cudnn_conv2d_bf16_nchw(
    const void* input,
    const void* filter,
    const void* bias,
    void* output,
    int n,
    int cin,
    int hin,
    int win,
    int cout,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int algorithm,
    void* stream
) {
    if (!input || !filter || !output || n <= 0 || cin <= 0 || hin <= 0 ||
        win <= 0 || cout <= 0 || kh <= 0 || kw <= 0 || stride_h <= 0 ||
        stride_w <= 0 || pad_h < 0 || pad_w < 0) {
        return -1;
    }

    int rc = ensure_conv2d_handle();
    if (rc != 0) return rc;
    const size_t runtime_version = cudnnGetVersion();
    // cuDNN 9 keeps this classic descriptor API ABI-compatible across minor
    // releases. Creator's accepted numeric oracle was recorded on 9.10.2, but
    // refusing newer cuDNN 9 runtimes makes audio decode unusable as soon as
    // the project environment advances. Keep the major-version boundary and
    // the minimum tested API level strict; report (rather than reject) a newer
    // minor because its output still needs the normal end-to-end parity gate.
    if (runtime_version / 10000 != 9 || runtime_version < 91002) {
        std::fprintf(
            stderr,
            "[serenity_cudnn_conv2d] requires cuDNN 9.10.2 or newer cuDNN 9 "
            "runtime, loaded %zu\n",
            runtime_version);
        return -91002;
    }
    if (runtime_version != 91002) {
        static std::once_flag version_warning;
        std::call_once(version_warning, [runtime_version]() {
            std::fprintf(
                stderr,
                "[serenity_cudnn_conv2d] using cuDNN %zu; Creator numeric "
                "acceptance remains gated against the pinned 9.10.2 oracle\n",
                runtime_version);
        });
    }
    cudnnStatus_t status = cudnnSetStream(g_conv2d_handle, (cudaStream_t)stream);
    if (status != CUDNN_STATUS_SUCCESS) return (int)status;

    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnTensorDescriptor_t b_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    int result = 0;

    const int hout = (hin + 2 * pad_h - kh) / stride_h + 1;
    const int wout = (win + 2 * pad_w - kw) / stride_w + 1;
    if (hout <= 0 || wout <= 0) return -1;

    do {
        if ((status = cudnnCreateTensorDescriptor(&x_desc)) != CUDNN_STATUS_SUCCESS) break;
        if ((status = cudnnCreateTensorDescriptor(&y_desc)) != CUDNN_STATUS_SUCCESS) break;
        if ((status = cudnnCreateFilterDescriptor(&w_desc)) != CUDNN_STATUS_SUCCESS) break;
        if ((status = cudnnCreateConvolutionDescriptor(&conv_desc)) != CUDNN_STATUS_SUCCESS) break;

        status = cudnnSetTensor4dDescriptor(
            x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, n, cin, hin, win);
        if (status != CUDNN_STATUS_SUCCESS) break;
        status = cudnnSetFilter4dDescriptor(
            w_desc, CUDNN_DATA_BFLOAT16, CUDNN_TENSOR_NCHW, cout, cin, kh, kw);
        if (status != CUDNN_STATUS_SUCCESS) break;
        status = cudnnSetConvolution2dDescriptor(
            conv_desc, pad_h, pad_w, stride_h, stride_w, 1, 1,
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        if (status != CUDNN_STATUS_SUCCESS) break;
        status = cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);
        if (status != CUDNN_STATUS_SUCCESS) break;
        status = cudnnSetTensor4dDescriptor(
            y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, n, cout, hout, wout);
        if (status != CUDNN_STATUS_SUCCESS) break;

        cudnnConvolutionFwdAlgo_t algo;
        if (algorithm >= 0 && algorithm < CUDNN_CONVOLUTION_FWD_ALGO_COUNT) {
            algo = (cudnnConvolutionFwdAlgo_t)algorithm;
        } else {
            int returned = 0;
            cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
            status = cudnnGetConvolutionForwardAlgorithm_v7(
                g_conv2d_handle, x_desc, w_desc, conv_desc, y_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned, perf);
            if (status != CUDNN_STATUS_SUCCESS || returned <= 0) break;
            bool found = false;
            for (int i = 0; i < returned; ++i) {
                if (perf[i].status == CUDNN_STATUS_SUCCESS) {
                    algo = perf[i].algo;
                    found = true;
                    break;
                }
            }
            if (!found) {
                status = CUDNN_STATUS_NOT_SUPPORTED;
                break;
            }
        }
        if (std::getenv("SERENITY_CUDNN_CONV2D_TRACE")) {
            std::fprintf(
                stderr,
                "[serenity_cudnn_conv2d] n=%d cin=%d hin=%d win=%d "
                "cout=%d kh=%d kw=%d algo=%d cudnn=%zu\n",
                n, cin, hin, win, cout, kh, kw, (int)algo,
                runtime_version);
        }

        status = cudnnGetConvolutionForwardWorkspaceSize(
            g_conv2d_handle, x_desc, w_desc, conv_desc, y_desc, algo,
            &workspace_bytes);
        if (status != CUDNN_STATUS_SUCCESS) break;
        if (workspace_bytes > 0) {
            cudaError_t cuda_status = cudaMalloc(&workspace, workspace_bytes);
            if (cuda_status != cudaSuccess) {
                result = 1000 + (int)cuda_status;
                break;
            }
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        status = cudnnConvolutionForward(
            g_conv2d_handle, &alpha, x_desc, input, w_desc, filter, conv_desc,
            algo, workspace, workspace_bytes, &beta, y_desc, output);
        if (status != CUDNN_STATUS_SUCCESS) break;

        if (bias) {
            if ((status = cudnnCreateTensorDescriptor(&b_desc)) != CUDNN_STATUS_SUCCESS) break;
            status = cudnnSetTensor4dDescriptor(
                b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, 1, cout, 1, 1);
            if (status != CUDNN_STATUS_SUCCESS) break;
            const float one = 1.0f;
            status = cudnnAddTensor(
                g_conv2d_handle, &one, b_desc, bias, &one, y_desc, output);
            if (status != CUDNN_STATUS_SUCCESS) break;
        }
    } while (false);

    if (result == 0 && status != CUDNN_STATUS_SUCCESS) result = (int)status;
    if (workspace) cudaFree(workspace);
    if (b_desc) cudnnDestroyTensorDescriptor(b_desc);
    if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
    if (w_desc) cudnnDestroyFilterDescriptor(w_desc);
    if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
    if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
    return result;
}
