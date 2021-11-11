#include "Linear.h"
#include "utils.h"
#include <chrono>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <vector>

template <typename _Action> double wtime_cuda(_Action action, const int times) {
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times; ++i) {
        action();
    }
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - start_time)
                      .count() *
                  1.0;

    return time / times;
}

__inline__ int multiple_16(int x) { return ((x / 16) + ((x % 16) != 0)) * 16; }

int benchmark(half *weight, int M, int N, int K) {
    half *temp_in, *temp_out;
    cudaChk(cudaMalloc(&temp_in, sizeof(half) * M * K));
    cudaChk(cudaMalloc(&temp_out, sizeof(half) * M * N));
    int startAlgo = (int)CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    int endAlgo = (int)CUBLAS_GEMM_ALGO15_TENSOR_OP;
    half alpha = (half)1.0f;
    half beta = (half)0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);
    double best_time = 1e10;
    int best_algo = startAlgo;
#if NDEBUG
    for (int i = startAlgo; i <= endAlgo; i++) {
        cublasStatus_t status;
        auto temp_time = wtime_cuda(
            [&]() {
                status = cublasGemmEx(
                    handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, weight,
                    CUDA_R_16F, K, temp_in, CUDA_R_16F, K, &beta, temp_out,
                    CUDA_R_16F, N, CUDA_R_16F, (cublasGemmAlgo_t)i);
            },
            100);
        if (status == CUBLAS_STATUS_SUCCESS && temp_time < best_time) {
            best_time = temp_time;
            best_algo = i;
        }
    }
#endif
    cublasDestroy(handle);
    cudaChk(cudaFree(temp_in));
    cudaChk(cudaFree(temp_out));
    return best_algo;
}

Linear_row::Linear_row(int in_feat_, int out_feat_, int size_, half *weight_,
                       int *row_id_, int row_id_size_, half *bias_, int nhead_)
    : in_feat(in_feat_), out_feat(out_feat_), nhead(nhead_),
      row_id_size(row_id_size_), nrow(multiple_16(row_id_size_)),
      weight(in_feat_ * multiple_16(row_id_size_)),
      bias(multiple_16(row_id_size_)), row_id(row_id_, row_id_size_),
      size(size_), head_ptr(nhead_ + 1) {

    cudaChk(cudaMemcpy(weight.get(), weight_,
                       sizeof(half) * in_feat_ * row_id_size_,
                       cudaMemcpyHostToDevice));
    cudaChk(cudaMemcpy(bias.get(), bias_, sizeof(half) * row_id_size_,
                       cudaMemcpyHostToDevice));
    algo = (cublasGemmAlgo_t)benchmark(weight.get(), size, nrow, in_feat);
    std::vector<int> h_head_ptr(nhead + 1);
    h_head_ptr[0] = 0;
    int curr_head = 1;
    int row_counter = 0;
    int head_dim = out_feat / nhead;
    for (int i = 0; i < row_id_size; i++) {
        if (row_id_[i] < curr_head * head_dim) {
            row_counter++;
        } else {
            h_head_ptr[curr_head] = h_head_ptr[curr_head - 1] + row_counter;
            curr_head++;
            row_counter = 1;
        }
    }
    assert(curr_head == nhead);
    cudaChk(cudaMemcpy(head_ptr.get(), h_head_ptr.data(),
                       sizeof(int) * (nhead + 1), cudaMemcpyHostToDevice));
}