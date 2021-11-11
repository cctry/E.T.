#include "kernels.h"
#include "utils.h"
#include <assert.h>
#include <cublas_v2.h>
__global__ void __kernel_fill_bias(half *out, const half *__restrict__ bias,
                                   int nrow, int ncol) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int nthd = blockDim.x * gridDim.x;
    int *out_ptr = (int *)out;
    int *bias_ptr = (int *)bias;
    int stride = ncol / 2;
    int nele = nrow * stride;
    for (int i = tid; i < nele; i += nthd) {
        out_ptr[i] = __ldg(&bias_ptr[i % stride]);
    }
}

void row_spmm_add_bias_kernelLauncher(half *out, const half *in,
                                      const half *weight, const half *bias,
                                      int M, int N, int K,
                                      cublasHandle_t handle,
                                      cublasGemmAlgo_t algo) {
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    int nblk, nthd;
    cudaOccupancyMaxPotentialBlockSize(&nblk, &nthd, __kernel_fill_bias);
    __kernel_fill_bias<<<nblk, nthd, 0, stream>>>(out, bias, M, N);
    const static half alpha = (half)1.0f;
    const static half beta = (half)1.0f;
    auto status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                               &alpha, weight, CUDA_R_16F, K, in, CUDA_R_16F, K,
                               &beta, out, CUDA_R_16F, N, CUDA_R_16F, algo);
    assert(status == CUBLAS_STATUS_SUCCESS);
}

__inline__ __device__ half2 gelu(half2 val) {
    half2 val_pow3 = __hmul2(val, __hmul2(val, val));
    float2 tmp_pow = __half22float2(val_pow3);
    float2 tmp = __half22float2(val);

    tmp.x =
        0.5f *
        (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
    tmp.y =
        0.5f *
        (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
    return __hmul2(val, __float22half2_rn(tmp));
}

// X * W.T = Y
// W_row_offset are the offsets of the row of each tile, so they are multiples
// of 16
static constexpr int FP16_skew = 16;
template <int num_thd>
__global__ void __launch_bounds__(128) __kernel_blk_mmul_blk_bias_smem_cta_skew(
    half *__restrict__ Y, const half *__restrict__ X,
    const int *__restrict__ W_row_ptr, const int *__restrict__ W_row_offset,
    const half *__restrict__ W_data, int M, int N,
    const half *__restrict__ bias) {
    static_assert(num_thd == 128);
    constexpr auto num_warp = num_thd >> 5;
    __shared__ half smem[16 * (num_warp * 16 + FP16_skew)];
    const auto warp_id = threadIdx.x >> 5;
    constexpr auto tile_ldm = num_warp * 16 + FP16_skew;
    const int out_row_16 = blockIdx.y * 16;
    const int out_col = blockIdx.x;
    // out_row and out_col are untransposed positions. Starting from
    // upper-left
    using frag_t = mma::mma_t<16, 16, 16>;
    frag_t::a_t<wmma::row_major> a;
    frag_t::b_t<wmma::col_major> b;
    frag_t::c_t<half> c;
    wmma::fill_fragment(c, 0);
    for (int i = W_row_ptr[out_col] + warp_id; i < W_row_ptr[out_col + 1];
         i += num_warp) {
        wmma::load_matrix_sync(b, &W_data[i << 8], 16);
        auto src = &X[out_row_16 * M + W_row_offset[i]];
        wmma::load_matrix_sync(a, src, M);
        wmma::mma_sync(c, a, b, c);
    }
    auto tile_temp = &smem[warp_id * 16];
    wmma::store_matrix_sync(tile_temp, c, tile_ldm, wmma::mem_row_major);
    __syncthreads();
    // reduce K across warp (2 per thread)
    int row = threadIdx.x / 8;
    int col = threadIdx.x % 8;
    // add bias
    auto bias2 = reinterpret_cast<const half2 *>(bias);
    half2 sum = bias2[out_col * 8 + col];
    constexpr int tile_ldm2 = tile_ldm / 2;
    half2 *smem2 = reinterpret_cast<half2 *>(smem);
    half2 *base = &smem2[row * tile_ldm2 + col];
#pragma unroll
    for (int i = 0; i < num_warp; i++) {
        sum += base[i * 8];
    }
    auto Y2 = reinterpret_cast<half2 *>(Y);
    int N2 = N / 2;
    half2 *dst = &Y2[out_row_16 * N2 + out_col * 8];
    dst[row * N2 + col] = sum;
}

template <int num_thd>
__global__ void __launch_bounds__(128)
    __kernel_blk_mmul_blk_bias_smem_cta_skew_gelu(
        half *__restrict__ Y, const half *__restrict__ X,
        const int *__restrict__ W_row_ptr, const int *__restrict__ W_row_offset,
        const half *__restrict__ W_data, int M, int N,
        const half *__restrict__ bias) {
    static_assert(num_thd == 128);
    constexpr auto num_warp = num_thd >> 5;
    __shared__ half smem[16 * (num_warp * 16 + FP16_skew)];
    const auto warp_id = threadIdx.x >> 5;
    constexpr auto tile_ldm = num_warp * 16 + FP16_skew;
    const int out_row_16 = blockIdx.y * 16;
    const int out_col = blockIdx.x;
    // out_row and out_col are untransposed positions. Starting from
    // upper-left
    using frag_t = mma::mma_t<16, 16, 16>;
    frag_t::a_t<wmma::row_major> a;
    frag_t::b_t<wmma::col_major> b;
    frag_t::c_t<half> c;
    wmma::fill_fragment(c, 0);
    for (int i = W_row_ptr[out_col] + warp_id; i < W_row_ptr[out_col + 1];
         i += num_warp) {
        wmma::load_matrix_sync(b, &W_data[i << 8], 16);
        auto src = &X[out_row_16 * M + W_row_offset[i]];
        wmma::load_matrix_sync(a, src, M);
        wmma::mma_sync(c, a, b, c);
    }
    auto tile_temp = &smem[warp_id * 16];
    wmma::store_matrix_sync(tile_temp, c, tile_ldm, wmma::mem_row_major);
    __syncthreads();
    // reduce K across warp (2 per thread)
    int row = threadIdx.x / 8;
    int col = threadIdx.x % 8;
    // add bias
    auto bias2 = reinterpret_cast<const half2 *>(bias);
    half2 sum = bias2[out_col * 8 + col];
    constexpr int tile_ldm2 = tile_ldm / 2;
    half2 *smem2 = reinterpret_cast<half2 *>(smem);
    half2 *base = &smem2[row * tile_ldm2 + col];
#pragma unroll
    for (int i = 0; i < num_warp; i++) {
        sum += base[i * 8];
    }
    auto Y2 = reinterpret_cast<half2 *>(Y);
    int N2 = N / 2;
    Y2[(out_row_16 + row) * N2 + out_col * 8 + col] = gelu(sum);
}

void tile_spmm_add_bias_kernelLauncher(half *Y, half *X, int *W_row_ptr,
                                       int *W_row_offset, half *W_data, int M,
                                       int N, half *bias, cudaStream_t stream) {
    __kernel_blk_mmul_blk_bias_smem_cta_skew<128>
        <<<dim3(N / 16, M / 16), 128, 0, stream>>>(
            Y, X, W_row_ptr, W_row_offset, W_data, M, N, bias);
}

void tile_spmm_add_bias_gelu_kernelLauncher(half *Y, half *X, int *W_row_ptr,
                                            int *W_row_offset, half *W_data,
                                            int M, int N, half *bias,
                                            cudaStream_t stream) {
    __kernel_blk_mmul_blk_bias_smem_cta_skew_gelu<128>
        <<<dim3(N / 16, M / 16), 128, 0, stream>>>(
            Y, X, W_row_ptr, W_row_offset, W_data, M, N, bias);
}
