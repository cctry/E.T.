#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <encoder/encoder.h>
#include <encoder/utils.h>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 5) {
        puts("Usage: ./exec seq_len head_number d_model sparsity");
        return -1;
    }
    int seq_len = atoi(argv[1]);
    int nhead = atoi(argv[2]);
    int d_model = atoi(argv[3]);
    float sparsity = atof(argv[4]);

    auto Q_linear = gen_tile_linear(d_model, d_model, seq_len, sparsity);
    auto K_linear = gen_tile_linear(d_model, d_model, seq_len, sparsity);
    auto V_linear = gen_tile_linear(d_model, d_model, seq_len, sparsity);
    auto O_linear = gen_tile_linear(d_model, d_model, seq_len, sparsity);
    auto MLP1_linear =
        gen_tile_linear(d_model, d_model * 4, seq_len, sparsity, true);
    auto MLP2_linear =
        gen_tile_linear(d_model * 4, d_model, seq_len, sparsity);

    std::vector<half> mask(seq_len * seq_len, __float2half(1.0));
    std::vector<half> ln_gamma(seq_len, __float2half(0.1));
    std::vector<half> ln_beta(seq_len, __float2half(0.1));

    half *buffer, *in;
    cudaMalloc(&buffer,
               (Encoder_tile::buffer_num * d_model * seq_len) * sizeof(half));
    cudaMalloc(&in, (seq_len * d_model) * sizeof(half));
    Encoder_tile encoder(
        d_model, seq_len, nhead, buffer, std::move(Q_linear),
        std::move(K_linear), std::move(V_linear), std::move(O_linear),
        std::move(MLP1_linear), std::move(MLP2_linear), ln_gamma.data(),
        ln_beta.data(), ln_gamma.data(), ln_beta.data(), mask.data());
    encoder.run(in, in);
}