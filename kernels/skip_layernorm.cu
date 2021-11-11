#include "utils.h"
#include <assert.h>
#include "kernels.h"
__global__ void skip_layernorm(half *in_out, const half *__restrict skip,
                               const half *__restrict gamma,
                               const half *__restrict beta, int len) {
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    float2 local_out_fp2;

    half2 *out_ptr = (half2 *)in_out;
    const half2 *skip_ptr = (const half2 *)skip;
    const half2 *gamma_ptr = (const half2 *)gamma;
    const half2 *beta_ptr = (const half2 *)beta;

    float local_out = 0.0f;
    int id = blockIdx.x * len / 2 + threadIdx.x;
    local_out_fp2 = __half22float2(__hadd2(out_ptr[id], __ldg(&skip_ptr[id])));
    local_out += local_out_fp2.x;
    local_out += local_out_fp2.y;

    mean = blockReduceSum<float>(local_out);
    if (threadIdx.x == 0)
        s_mean = mean / len;
    __syncthreads();

    variance = (local_out_fp2.x - s_mean) * (local_out_fp2.x - s_mean);
    variance += (local_out_fp2.y - s_mean) * (local_out_fp2.y - s_mean);
    variance = blockReduceSum<float>(variance);
    if (threadIdx.x == 0)
        s_variance = rsqrtf(variance / len + 1e-6f);
    __syncthreads();

    float2 gamma_val = __half22float2(__ldg(&gamma_ptr[threadIdx.x]));
    float2 beta_val = __half22float2(__ldg(&beta_ptr[threadIdx.x]));
    local_out_fp2.x =
        (local_out_fp2.x - s_mean) * s_variance * gamma_val.x + beta_val.x;
    local_out_fp2.y =
        (local_out_fp2.y - s_mean) * s_variance * gamma_val.y + beta_val.y;
    out_ptr[id] = __float22half2_rn(local_out_fp2);
}

__global__ void skip_layernorm_v2(half *in_out, const half *__restrict skip,
                                  const half *__restrict gamma,
                                  const half *__restrict beta, int n) {
    constexpr int ite = 4;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;
    half2 local_out_half2[ite];

    half2 *out_ptr = (half2 *)in_out;
    const half2 *skip_ptr = (const half2 *)skip;
    const half2 *gamma_ptr = (const half2 *)gamma;
    const half2 *beta_ptr = (const half2 *)beta;

    half2 sum = __float2half2_rn(0.0f);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id = bid * n / 2 + col_id;
        local_out_half2[i] = out_ptr[id] + __ldg(&skip_ptr[id]);
        sum += local_out_half2[i];
    }

    mean = blockReduceSum<float>((float)(sum.x + sum.y));
    if (threadIdx.x == 0)
        s_mean = mean / n;
    __syncthreads();

    float var = 0.0f;
    half2 s_mean_2 = __float2half2_rn(s_mean);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        local_out_half2[i] = local_out_half2[i] - s_mean_2;
        float v1 = (float)local_out_half2[i].x;
        float v2 = (float)local_out_half2[i].y;
        var += v1 * v1 + v2 * v2;
    }

    variance = blockReduceSum<float>(var);
    if (threadIdx.x == 0)
        s_variance = rsqrtf(variance / n + 1e-6f);
    __syncthreads();

    half2 s_var_2 = __float2half2_rn(s_variance);
#pragma unroll
    for (int i = 0; i < ite; i++) {
        int col_id = i * blockDim.x + tid;
        int id = bid * n / 2 + col_id;
        out_ptr[id] = local_out_half2[i] * s_var_2 * __ldg(&gamma_ptr[col_id]) +
                      __ldg(&beta_ptr[col_id]);
    }
}

void skip_layernorm_kernelLauncher(half *in_out, const half *skip,
                                   const half *gamma, const half *beta, int num,
                                   int size, cudaStream_t stream) {
    if (num >= 512 && (size == 768 || size == 1024))
        skip_layernorm_v2<<<num, size / 8, 0, stream>>>(in_out, skip, gamma,
                                                        beta, size);
    else {
        assert(size / 2 <= 1024);
        skip_layernorm<<<num, size / 2, 0, stream>>>(in_out, skip, gamma, beta,
                                                     size);
    }
}