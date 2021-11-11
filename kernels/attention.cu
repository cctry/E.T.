#include "utils.h"
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
#include <algorithm>
namespace cg = cooperative_groups;

template <typename _TyGroup>
__device__ void softmax_blk(const _TyGroup &cta, half *data, const int size,
                            const int num, const int ldm) {
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = warp.meta_group_rank();
    const auto lane_id = warp.thread_rank();
    const auto num_warp = warp.meta_group_size();
    const auto size2 = size >> 1;
    for (int row = warp_id; row < num; row += num_warp) {
        auto row_ptr = data + row * ldm;
        // find the max
        half val_max = (half)0.0f;
        for (auto i = lane_id; i < size; i += 32) {
            const auto temp = row_ptr[i];
            val_max = val_max > temp ? val_max : temp;
        }
        warp.sync();
        const auto max = cg::reduce(warp, val_max, cg::greater<half>());
        const half2 max2{max, max};
        // compute the sum of exp-ed and shifted array
        auto row_ptr2 = reinterpret_cast<half2 *>(row_ptr);
        half2 val_sum2{(half)0.0f, (half)0.0f};
        for (auto i = lane_id; i < size2; i += 32) {
            const auto temp = h2exp2(__hsub2(row_ptr2[i], max2));
            row_ptr2[i] = temp;
            val_sum2 += temp;
        }
        warp.sync();
        const auto sum2 = cg::reduce(warp, val_sum2, cg::plus<half2>());
        // update with softmax scaling
        const auto sum = sum2.x + sum2.y;
        const half2 one_over_sum =
            __h2div(half2{(half)1.0f, (half)1.0f}, half2{sum, sum});
        for (auto i = lane_id; i < size2; i += 32) {
            row_ptr2[i] = row_ptr2[i] * one_over_sum;
        }
    }
    cta.sync();
}

constexpr int FP16_skew = 16;
template <int num_thd>
__global__ void __launch_bounds__(num_thd)
    __kernel_multi_head_full_skew_warpSFM(
        const half *__restrict__ Q, const half *__restrict__ K,
        const half *__restrict__ V, half *__restrict__ Z, const int d_model,
        const int seq_len, const int head_dim, const half one_scale,
        const half *__restrict__ mask) {
    // blockIdx.x: block row id
    // blockIdx.y: head_id
    using frag_t = mma::mma_t<16, 16, 16>;
    extern __shared__ half smem[];
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    const int warp_id = warp.meta_group_rank();
    const int lane_id = warp.thread_rank();
    constexpr int num_warp = num_thd / 32;
    auto Q_ptr = &Q[16 * blockIdx.x * d_model + head_dim * blockIdx.y];
    auto temp_Q = smem;
    const auto ldQ = head_dim + FP16_skew;
    const auto ldRow = seq_len + FP16_skew;
    const half2 scale2{one_scale, one_scale};
#pragma unroll
    for (int r = warp_id; r < 16; r += num_warp) {
        auto dst = reinterpret_cast<half2 *>(&temp_Q[r * ldQ]);
        auto src = reinterpret_cast<const half2 *>(&Q_ptr[r * d_model]);
        for (int i = lane_id; i < (head_dim >> 1); i += 32)
            dst[i] = __hmul2(__ldg(&src[i]), scale2);
    }
    cta.sync();
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<half> c_frag;
    auto temp_row = &smem[16 * ldQ];
    for (int KR = warp_id * 16; KR < seq_len; KR += num_warp * 16) {
        auto K_ptr = &K[KR * d_model + head_dim * blockIdx.y];
        wmma::fill_fragment(c_frag, (half)0.0f);
        for (int i = 0; i < head_dim; i += 16) {
            wmma::load_matrix_sync(a_frag, smem + i, ldQ);
            wmma::load_matrix_sync(b_frag, K_ptr + i, d_model);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(temp_row + KR, c_frag, ldRow,
                                wmma::mem_row_major);
    }
    cta.sync();
    // mask
    auto mask_base = (const half2 *)(&mask[blockIdx.x * 16 * seq_len]);
    auto temp_row2 = (half2 *)(temp_row);
    const auto ldRow2 = ldRow >> 1;
    const auto seq_len2 = seq_len >> 1;
    for (int i = threadIdx.x; i < seq_len2 * 16; i += num_thd) {
        const auto r = i / seq_len2;
        const auto c = i % seq_len2;
        temp_row2[r * ldRow2 + c] += __ldg(&mask_base[i]);
    }
    cta.sync();
    softmax_blk(cta, temp_row, seq_len, 16, ldRow);
    for (int VC = warp_id * 16; VC < head_dim; VC += num_warp * 16) {
        frag_t::b_t<wmma::row_major> b_frag;
        wmma::fill_fragment(c_frag, (half)0.0f);
        for (int i = 0; i < seq_len; i += 16) {
            auto V_ptr = &V[d_model * i + blockIdx.y * head_dim + VC];
            wmma::load_matrix_sync(a_frag, temp_row + i, ldRow);
            wmma::load_matrix_sync(b_frag, V_ptr, d_model);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        auto res = &Z[blockIdx.x * 16 * d_model + blockIdx.y * head_dim + VC];
        wmma::store_matrix_sync(res, c_frag, d_model, wmma::mem_row_major);
    }
}

void OTF_attention_kernelLauncher(half *out, half *Q, half *K, half *V,
                                  half *mask, int seq_len, int d_model,
                                  int nhead, cudaStream_t stream = nullptr) {
    int head_dim = d_model / nhead;
    half scale = __float2half(sqrtf(1.0f / head_dim));
    constexpr int num_thd = 256;
    auto tempQ_size = 16 * ((d_model / nhead) + FP16_skew);
    auto temp_row_size = 16 * (seq_len + FP16_skew);
    auto smem_size = sizeof(half) * (tempQ_size + temp_row_size);
    __kernel_multi_head_full_skew_warpSFM<num_thd>
        <<<dim3(seq_len / 16, nhead), num_thd, smem_size, stream>>>(
            Q, K, V, out, d_model, seq_len, head_dim, scale, mask);
}

template <typename _Tg>
__device__ void clear_smem(const _Tg &group, void *smem, const int N) {
    auto ptr = reinterpret_cast<int *>(smem);
#pragma unroll
    for (int i = group.thread_rank(); i < N / sizeof(int); i += group.size()) {
        ptr[i] = 0;
    }
    group.sync();
}

/*
 * generate a dense matrix
 */
template <int num_thd>
__global__ void __launch_bounds__(num_thd) __kernel_multi_head_prune(
    const half *__restrict Q, const half *__restrict K,
    const half *__restrict V, half *__restrict Z, const int d_model,
    const int seq_len, const int head_dim, const half one_scale,
    const int num_head, const int *__restrict v_col_id,
    const int *__restrict v_head_ptr, const int nnz_col_v,
    const half *__restrict mask) {
    // blockIdx.x: block row id
    // blockIdx.y: head_id
    /*
     * smem partition:
     * First part: 16 * seq_len * sizeof(half)
     * temp_row (16 * seq_len * sizeof(half)): one block row of QK
     * Second part: max(temp_Q, temp_blk+temp_id)
     *  temp_Q (16 * head_dim * sizeof(half)): one block row of Q
     *  temp_blk (256 * num_warp * sizeof(half)): one compressed block of Z
     *  temp_id (16 * num_warp * sizeof(int))
     */
    using frag_t = mma::mma_t<16, 16, 16>;
    extern __shared__ half smem[];
    auto cta = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(cta);
    const int warp_id = warp.meta_group_rank();
    const int lane_id = warp.thread_rank();
    constexpr int num_warp = num_thd / 32;
    auto Q_ptr = &Q[16 * blockIdx.x * d_model + head_dim * blockIdx.y];
    half *temp_row = smem;
    const auto ldQ = head_dim + FP16_skew;
    const auto ldRow = seq_len + FP16_skew;
    auto temp_Q = &smem[16 * ldRow];
    const half2 scale2{one_scale, one_scale};
#pragma unroll
    for (int r = warp_id; r < 16; r += num_warp) {
        auto dst = (half2 *)(&temp_Q[r * ldQ]);
        auto src = (const half2 *)(&Q_ptr[r * d_model]);
        for (int i = lane_id; i < (head_dim >> 1); i += 32)
            dst[i] = __hmul2(__ldg(&src[i]), scale2);
    }
    cta.sync();
    frag_t::a_t<wmma::row_major> a_frag;
    frag_t::b_t<wmma::col_major> b_frag;
    frag_t::c_t<half> c_frag;
    for (int KR = warp_id * 16; KR < seq_len; KR += num_warp * 16) {
        auto K_ptr = &K[KR * d_model + head_dim * blockIdx.y];
        wmma::fill_fragment(c_frag, (half)0.0f);
        for (int i = 0; i < head_dim; i += 16) {
            wmma::load_matrix_sync(a_frag, temp_Q + i, ldQ);
            wmma::load_matrix_sync(b_frag, K_ptr + i, d_model);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(temp_row + KR, c_frag, ldRow,
                                wmma::mem_row_major);
    }
    cta.sync();
    // mask
    const auto mask_base =
        reinterpret_cast<const half2 *>(&mask[(blockIdx.x * 16) * seq_len]);
    auto temp_row_2 = reinterpret_cast<half2 *>(temp_row);
    for (int i = threadIdx.x; i < seq_len * 8; i += blockDim.x) {
        temp_row_2[i] += mask_base[i];
    }
    cta.sync();
    softmax_blk(cta, temp_row, seq_len, 16, ldRow);
    // compute Z
    constexpr int ld_tempblk = 16 * num_warp + FP16_skew;
    auto temp_blk = temp_Q + 16 * warp_id;
    auto temp_id =
        reinterpret_cast<int *>(&temp_Q[16 * ld_tempblk]) + (16 * warp_id);
    const auto VC_start = v_head_ptr[blockIdx.y];
    const auto VC_end = v_head_ptr[blockIdx.y + 1];
    for (int VC_blk = VC_start + warp_id * 16; VC_blk < VC_end;
         VC_blk += num_warp * 16) {
        frag_t::b_t<wmma::row_major> b_frag;
        wmma::fill_fragment(c_frag, (half)0.0f);
        const int num_valid_col = (VC_blk + 16) < VC_end ? 16 : VC_end - VC_blk;
        cg::memcpy_async(warp, temp_id, &v_col_id[VC_blk],
                         sizeof(int) * num_valid_col);
        auto two_thd = cg::tiled_partition<2>(warp);
        auto dst = &temp_blk[ld_tempblk * two_thd.meta_group_rank()];
        clear_smem(two_thd, dst, 16 * sizeof(half));
        for (int i = 0; i < seq_len; i += 16) {
            auto src = &V[nnz_col_v * (i + two_thd.meta_group_rank()) + VC_blk];
            for (int k = two_thd.thread_rank(); k < num_valid_col; k += 2)
                dst[k] = src[k];
            warp.sync();
            wmma::load_matrix_sync(b_frag, temp_blk, ld_tempblk);
            wmma::load_matrix_sync(a_frag, temp_row + i, ldRow);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(temp_blk, c_frag, ld_tempblk,
                                wmma::mem_row_major);
        cg::wait(warp);
        auto src = &temp_blk[16 * two_thd.meta_group_rank()];
        for (int k = two_thd.thread_rank(); k < num_valid_col; k += 2) {
            const auto col = temp_id[k];
            auto dst =
                &Z[(blockIdx.x * 16 + two_thd.meta_group_rank()) * d_model +
                   col];
        }
    }
}

void Prune_attention_kernelLauncher(half *out, half *Q, half *K, half *V,
                                    int *v_col_id, int *v_head_ptr,
                                    int nnz_col_v, half *mask, int seq_len,
                                    int d_model, int nhead,
                                    cudaStream_t stream = nullptr) {
    const int head_dim = d_model / nhead;
    auto smem_size = [seq_len, head_dim](int num_thd) {
        const auto temp_row = 16 * (seq_len + FP16_skew) * sizeof(half);
        const auto temp_Q = 16 * (head_dim + FP16_skew) * sizeof(half);
        const auto temp_blk = 16 * ((num_thd / 2) + FP16_skew) * sizeof(half);
        const auto temp_id = 16 * (num_thd / 32) * sizeof(int);
        return temp_row + std::max(temp_Q, temp_blk + temp_id);
    };
    half scale = __float2half(sqrtf(1.0f / head_dim));
    constexpr int num_thd = 512;
    __kernel_multi_head_prune<num_thd>
        <<<dim3(seq_len / 16, nhead), num_thd, smem_size(num_thd)>>>(
            Q, K, V, out, d_model, seq_len, head_dim, scale, nhead, v_col_id,
            v_head_ptr, nnz_col_v, mask);
}

template <int num_thd>
__global__ void __launch_bounds__(num_thd)
    __kernel_multi_head_sharedQK(const half *__restrict__ QK,
                                 const half *__restrict__ V,
                                 half *__restrict__ Z, const int d_model,
                                 const int seq_len, const int head_dim,
                                 const half *__restrict__ mask) {
    constexpr int FP16_skew = 16;
    using frag_t = mma::mma_t<16, 16, 16>;
    extern __shared__ half smem[]; // 16 * seq_len
    auto cta = cg::this_thread_block();
    const auto head_id = cta.group_index().y;
    const auto z_row = cta.group_index().x;
    auto warp = cg::tiled_partition<32>(cta);
    const auto warp_id = warp.meta_group_rank();
    constexpr auto num_warp = num_thd / 32;
    const auto ldm = seq_len + FP16_skew;
    const auto seq_len2 = seq_len >> 1;
    for (auto r = warp_id; r < 16; r += num_warp) {
        const auto row = 16 * z_row + r;
        auto dst = &smem[r * ldm];
        auto src = &QK[head_id * seq_len * seq_len + seq_len * row];
        cg::memcpy_async(warp, dst, src, sizeof(half) * seq_len);
        cg::wait(warp);
        // mask and find the max
        half val_max = (half)0.0f;
        for (auto i = warp.thread_rank(); i < seq_len2; i += 32) {
            const auto temp = dst[i] + mask[seq_len * row + i];
            dst[i] = temp;
            val_max = val_max > temp ? val_max : temp;
        }
        warp.sync();
        const auto max = cg::reduce(warp, val_max, cg::greater<half>());
        half2 max2{max, max};
        // compute the sum of exp-ed and shifted array
        half2 val_sum2{(half)0.0f, (half)0.0f};
        auto dst2 = reinterpret_cast<half2 *>(dst);
        for (auto i = warp.thread_rank(); i < seq_len2; i += 32) {
            const auto temp2 = h2exp2(dst2[i] - max2);
            val_sum2 += temp2;
            dst2[i] = temp2;
        }
        warp.sync();
        const auto sum2 = cg::reduce(warp, val_sum2, cg::plus<half2>());
        // update with softmax scaling
        const auto sum = sum2.x + sum2.y;
        half2 one_over_sum =
            __h2div(half2{(half)1.0f, (half)1.0f}, half2{sum, sum});
        for (auto i = warp.thread_rank(); i < seq_len2; i += 32) {
            dst2[i] = dst2[i] * one_over_sum;
        }
    }
    cta.sync();
    for (auto VC = warp_id; VC < head_dim / 16; VC += num_warp) {
        frag_t::a_t<wmma::row_major> a_frag;
        frag_t::b_t<wmma::row_major> b_frag;
        frag_t::c_t<half> c_frag;
        wmma::fill_fragment(c_frag, (half)0.0f);
        for (int i = 0; i < seq_len; i += 16) {
            auto V_ptr = &V[d_model * i + head_id * head_dim + VC * 16];
            wmma::load_matrix_sync(a_frag, smem + i, ldm);
            wmma::load_matrix_sync(b_frag, V_ptr, d_model);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        auto res = &Z[blockIdx.x * 16 * d_model + head_id * head_dim + VC * 16];
        wmma::store_matrix_sync(res, c_frag, d_model, wmma::mem_row_major);
    }
}

void sharedQK_attention_kernelLauncher(half *out, half *QK, half *V, half *mask,
                                       int seq_len, int d_model, int nhead,
                                       cudaStream_t stream = nullptr) {
    constexpr int num_thd = 256;
    const auto num_blk = dim3(seq_len / 16, nhead);
    __kernel_multi_head_sharedQK<num_thd>
        <<<num_blk, num_thd, sizeof(half) * 16 * (seq_len + FP16_skew)>>>(
            QK, V, out, d_model, seq_len, d_model / nhead, mask);
}