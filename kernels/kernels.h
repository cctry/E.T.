#pragma once
#include <cublas_v2.h>
void skip_layernorm_kernelLauncher(half *in_out, const half *skip,
                                   const half *gamma, const half *beta, int num,
                                   int size, cudaStream_t stream);

void row_spmm_add_bias_kernelLauncher(half *out, const half *in,
                                      const half *weight, const half *bias,
                                      int M, int N, int K,
                                      cublasHandle_t handle,
                                      cublasGemmAlgo_t algo);

void tile_spmm_add_bias_kernelLauncher(half *Y, half *X, int *W_row_ptr,
                                       int *W_row_offset, half *W_data, int M,
                                       int N, half *bias, cudaStream_t stream);

void tile_spmm_add_bias_gelu_kernelLauncher(half *Y, half *X, int *W_row_ptr,
                                            int *W_row_offset, half *W_data,
                                            int M, int N, half *bias,
                                            cudaStream_t stream);

void OTF_attention_kernelLauncher(half *out, half *Q, half *K, half *V,
                                  half *mask, int seq_len, int d_model,
                                  int nhead, cudaStream_t stream);

void Prune_attention_kernelLauncher(half *out, half *Q, half *K, half *V,
                                    int *v_col_id, int *v_head_ptr,
                                    int nnz_col_v, half *mask, int seq_len,
                                    int d_model, int nhead,
                                    cudaStream_t stream = nullptr);

void sharedQK_attention_kernelLauncher(half *out, half *QK, half *V, half *mask,
                                       int seq_len, int d_model, int nhead,
                                       cudaStream_t stream = nullptr)