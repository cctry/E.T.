#include "encoder.h"
#include "utils.h"

Encoder_tile::Encoder_tile(int d_model_, int seq_len_, int nheads_,
                           half *buffer_, Linear_tile &&Q_linear_,
                           Linear_tile &&K_linear_, Linear_tile &&V_linear_,
                           Linear_tile &&O_linear_, Linear_tile &&MLP1_linear_,
                           Linear_tile &&MLP2_linear_, half *ln_gamma_1,
                           half *ln_beta_1, half *ln_gamma_2, half *ln_beta_2,
                           half *mask_)
    : Encoder_base(d_model_, seq_len_, nheads_, buffer_, ln_gamma_1, ln_beta_1,
                   ln_gamma_2, ln_beta_2, mask_),
      Q_linear(std::move(Q_linear_)), K_linear(std::move(K_linear_)),
      V_linear(std::move(V_linear_)), O_linear(std::move(O_linear_)),
      MLP1_linear(std::move(MLP1_linear_)),
      MLP2_linear(std::move(MLP2_linear_)) {
    assert(MLP1_linear.is_gelu);
}

void Encoder_tile::run(half *out, half *in) {
    // Compute Q
    Q_linear.run(buffers[0], in, streams[0]);
    // Compute K
    K_linear.run(buffers[1], in, streams[1]);
    // Compute V
    V_linear.run(buffers[2], in, streams[2]);
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);
    // Compute self-attention
    OTF_attention_kernelLauncher(buffers[3], buffers[0], buffers[1], buffers[2],
                                 mask.get(), seq_len, d_model, nheads, NULL);
    // Compute output
    O_linear.run(buffers[0], buffers[3], nullptr);
    // skip LayerNorm 1
    skip_layernorm_kernelLauncher(buffers[0], in, ln_gamma[0].get(),
                                  ln_beta[0].get(), seq_len, d_model, NULL);
    // MLP 1
    MLP1_linear.run(buffers[1], buffers[0], NULL);
    // MLP 2
    MLP2_linear.run(out, buffers[1], NULL);
    // skip LayerNorm 2
    skip_layernorm_kernelLauncher(out, buffers[0], ln_gamma[1].get(),
                                  ln_beta[1].get(), seq_len, d_model, NULL);
}

Encoder_prune::Encoder_prune(int d_model_, int seq_len_, int nheads_,
                             half *buffer_, Linear_tile &&Q_linear_,
                             Linear_tile &&K_linear_, Linear_row &&V_linear_,
                             Linear_tile &&O_linear_,
                             Linear_tile &&MLP1_linear_,
                             Linear_tile &&MLP2_linear_, half *ln_gamma_1,
                             half *ln_beta_1, half *ln_gamma_2, half *ln_beta_2,
                             half *mask_)
    : Encoder_base(d_model_, seq_len_, nheads_, buffer_, ln_gamma_1, ln_beta_1,
                   ln_gamma_2, ln_beta_2, mask_),
      Q_linear(std::move(Q_linear_)), K_linear(std::move(K_linear_)),
      V_linear(std::move(V_linear_)), O_linear(std::move(O_linear_)),
      MLP1_linear(std::move(MLP1_linear_)),
      MLP2_linear(std::move(MLP2_linear_)) {
    assert(MLP1_linear.is_gelu);
    cublasSetStream(handle, streams[2]);
}

void Encoder_prune::run(half *out, half *in) {
    // Compute Q
    Q_linear.run(buffers[0], in, streams[0]);
    // Compute K
    K_linear.run(buffers[1], in, streams[1]);
    // Compute V
    V_linear.run(buffers[2], in, handle);
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);
    // Compute self-attention
    Prune_attention_kernelLauncher(
        buffers[3], buffers[0], buffers[1], buffers[2], V_linear.row_id.get(),
        V_linear.head_ptr.get(), V_linear.row_id_size, mask.get(), seq_len,
        d_model, nheads, NULL);
    // Compute output
    O_linear.run(buffers[0], buffers[3], nullptr);
    // skip LayerNorm 1
    skip_layernorm_kernelLauncher(buffers[0], in, ln_gamma[0].get(),
                                  ln_beta[0].get(), seq_len, d_model, NULL);
    // MLP 1
    MLP1_linear.run(buffers[1], buffers[0], NULL);
    // MLP 2
    MLP2_linear.run(out, buffers[1], NULL);
    // skip LayerNorm 2
    skip_layernorm_kernelLauncher(out, buffers[0], ln_gamma[1].get(),
                                  ln_beta[1].get(), seq_len, d_model, NULL);
}

Encoder_length::Encoder_length(int d_model_, int seq_len_, int nheads_,
                               half *buffer_, Linear_tile &&Q_linear_,
                               Linear_tile &&K_linear_, Linear_tile &&V_linear_,
                               Linear_tile &&O_linear_,
                               Linear_tile &&MLP1_linear_,
                               Linear_tile &&MLP2_linear_, half *ln_gamma_1,
                               half *ln_beta_1, half *ln_gamma_2,
                               half *ln_beta_2, half *mask_)
    : Encoder_base(d_model_, seq_len_, nheads_, buffer_, ln_gamma_1, ln_beta_1,
                   ln_gamma_2, ln_beta_2, mask_),
      Q_linear(std::move(Q_linear_)), K_linear(std::move(K_linear_)),
      V_linear(std::move(V_linear_)), O_linear(std::move(O_linear_)),
      MLP1_linear(std::move(MLP1_linear_)),
      MLP2_linear(std::move(MLP2_linear_)) {
    QK_buf = buffer + d_model * nheads * buffer_num;
    assert(MLP1_linear.is_gelu);
    cublasSetStream(handle, streams[0]);
    head_dim = d_model / nheads;
    scale = __float2half(1.0 / sqrtf((float)head_dim));
    batch_stride = head_dim * seq_len;
}

void Encoder_length::run(half *out, half *in) {
    // Compute Q
    Q_linear.run(buffers[0], in, streams[0]);
    // Compute K
    K_linear.run(buffers[1], in, streams[1]);
    // Compute V
    V_linear.run(buffers[2], in, streams[2]);
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    // Compute QK
    half beta = (half)0.0f;
    cublasHgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_len, seq_len, head_dim, &scale,
        buffers[1], head_dim, batch_stride, buffers[0], head_dim, batch_stride,
        &beta, QK_buf, seq_len, seq_len * seq_len, nheads);
    cudaStreamSynchronize(streams[2]);
    cudaStreamSynchronize(streams[1]);
    // Compute self-attention
    sharedQK_attention_kernelLauncher(buffers[3], QK_buf, buffers[2],
                                      mask.get(), seq_len, d_model, nheads,
                                      NULL);
    // Compute output
    O_linear.run(buffers[0], buffers[3], nullptr);
    // skip LayerNorm 1
    skip_layernorm_kernelLauncher(buffers[0], in, ln_gamma[0].get(),
                                  ln_beta[0].get(), seq_len, d_model, NULL);
    // MLP 1
    MLP1_linear.run(buffers[1], buffers[0], NULL);
    // MLP 2
    MLP2_linear.run(out, buffers[1], NULL);
    // skip LayerNorm 2
    skip_layernorm_kernelLauncher(out, buffers[0], ln_gamma[1].get(),
                                  ln_beta[1].get(), seq_len, d_model, NULL);
}