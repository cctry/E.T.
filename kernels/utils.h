#pragma once
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
template <typename T> __inline__ __device__ T warpReduceSum(T val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    return val;
}

template <typename T> __inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = warpReduceSum(val);
    return val;
}

#if CUDART_VERSION < 7000
#error "This code requires CUDA 7.0 or later"
#endif

namespace mma {
template <int M, int N, int K, typename T = half> struct mma_t {
    template <typename layout>
    using a_t = wmma::fragment<wmma::matrix_a, M, N, K, T, layout>;
    template <typename layout>
    using b_t = wmma::fragment<wmma::matrix_b, M, N, K, T, layout>;
    template <typename V>
    using c_t = wmma::fragment<wmma::accumulator, M, N, K, V>;
};
} // namespace mma
