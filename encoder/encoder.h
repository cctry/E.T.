#pragma once
#include "Linear.h"
#include "utils.h"
#include <cublas_v2.h>
#include <kernels/kernels.h>

class Encoder_base {
  public:
    static constexpr int buffer_num = 5;

  protected:
    int d_model;
    int seq_len;
    int nheads;
    cublasHandle_t handle;
    // buffers
    half *buffer;
    half *buffers[buffer_num];
    // weights
    Weight<half> ln_gamma[2];
    Weight<half> ln_beta[2];
    // mask
    Weight<half> mask;

    // streams
    cudaStream_t streams[3];

  public:
    Encoder_base(int d_model_, int seq_len_, int nheads_, half *buffer_,
                 half *ln_gamma_1, half *ln_beta_1, half *ln_gamma_2,
                 half *ln_beta_2, half *mask_)
        : d_model(d_model_), seq_len(seq_len_), nheads(nheads_),
          buffer(buffer_), mask(mask_, seq_len_ * seq_len_),
          ln_gamma{Weight(ln_gamma_1, d_model_), Weight(ln_gamma_2, d_model_)},
          ln_beta{Weight(ln_beta_1, d_model_), Weight(ln_beta_2, d_model_)} {
        cublasCreate(&handle);
        for (int i = 0; i < 3; i++) {
            cudaStreamCreate(&streams[i]);
        }
        for (int i = 0; i < buffer_num; i++) {
            buffers[i] = buffer + i * seq_len * d_model;
        }
    }
    ~Encoder_base() {
        cublasDestroy(handle);
        for (int i = 0; i < 3; i++) {
            cudaStreamDestroy(streams[i]);
        }
    }
    virtual void run(half *out, half *in) = 0;
    static size_t buffer_size(int d_model_, int seq_len_, int nheads_) {
        return buffer_num * d_model_ * seq_len_ * sizeof(half);
    }
};

class Encoder_tile : public Encoder_base {
  private:
    // Q, K, V
    Linear_tile Q_linear;
    Linear_tile K_linear;
    Linear_tile V_linear;
    // O
    Linear_tile O_linear;
    // MLP1
    Linear_tile MLP1_linear;
    // MLP2
    Linear_tile MLP2_linear;

  public:
    Encoder_tile(int d_model_, int seq_len_, int nheads_, half *buffer_,
                 Linear_tile &&Q_linear_, Linear_tile &&K_linear_,
                 Linear_tile &&V_linear_, Linear_tile &&O_linear_,
                 Linear_tile &&MLP1_linear_, Linear_tile &&MLP2_linear_,
                 half *ln_gamma_1, half *ln_beta_1, half *ln_gamma_2,
                 half *ln_beta_2, half *mask_);
    void run(half *out, half *in);
};

class Encoder_prune : public Encoder_base {
  private:
    // Q, K, V
    Linear_tile Q_linear;
    Linear_tile K_linear;
    Linear_row V_linear;
    // O
    Linear_tile O_linear;
    // MLP1
    Linear_tile MLP1_linear;
    // MLP2
    Linear_tile MLP2_linear;

  public:
    Encoder_prune(int d_model_, int seq_len_, int nheads_, half *buffer_,
                  Linear_tile &&Q_linear_, Linear_tile &&K_linear_,
                  Linear_row &&V_linear_, Linear_tile &&O_linear_,
                  Linear_tile &&MLP1_linear_, Linear_tile &&MLP2_linear_,
                  half *ln_gamma_1, half *ln_beta_1, half *ln_gamma_2,
                  half *ln_beta_2, half *mask_);
    void run(half *out, half *in);
};

class Encoder_length : public Encoder_base {
  private:
    // Q, K, V
    Linear_tile Q_linear;
    Linear_tile K_linear;
    Linear_tile V_linear;
    // O
    Linear_tile O_linear;
    // MLP1
    Linear_tile MLP1_linear;
    // MLP2
    Linear_tile MLP2_linear;

    half *QK_buf;

    half scale;
    int head_dim;
    int batch_stride;

  public:
    Encoder_length(int d_model_, int seq_len_, int nheads_, half *buffer_,
                   Linear_tile &&Q_linear_, Linear_tile &&K_linear_,
                   Linear_tile &&V_linear_, Linear_tile &&O_linear_,
                   Linear_tile &&MLP1_linear_, Linear_tile &&MLP2_linear_,
                   half *ln_gamma_1, half *ln_beta_1, half *ln_gamma_2,
                   half *ln_beta_2, half *mask_);
    void run(half *out, half *in);
    static size_t buffer_size(int d_model_, int seq_len_, int nheads_) {
        return (buffer_num * d_model_ * seq_len_ +
                nheads_ * seq_len_ * seq_len_) *
               sizeof(half);
    }
};
