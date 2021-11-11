#pragma once
#include "utils.h"
#include <kernels/kernels.h>
class Linear_tile {
  private:
    int in_feat;
    int out_feat;
    int size;
    // weight
    Weight<half> data;
    Weight<int> row_ptr;
    Weight<int> row_offset;
    // bias
    Weight<half> bias;

  public:
    const bool is_gelu;
    Linear_tile(int in_feat_, int out_feat_, int size_, int ntile, half *data_,
                int *row_ptr_, int *row_offset_, half *bias_,
                bool is_gelu_ = false)
        : data(data_, ntile * 256), row_ptr(row_ptr_, (out_feat_ / 16) + 1),
          row_offset(row_offset_, ntile), bias(bias_, out_feat_),
          in_feat(in_feat_), out_feat(out_feat_), size(size_),
          is_gelu(is_gelu_) {}

    Linear_tile(const Linear_tile &other) = delete;
    Linear_tile &operator=(const Linear_tile &other) = delete;
    Linear_tile(Linear_tile &&other)
        : data(std::move(other.data)), row_ptr(std::move(other.row_ptr)),
          row_offset(std::move(other.row_offset)), bias(std::move(other.bias)),
          is_gelu(other.is_gelu) {
        in_feat = other.in_feat;
        out_feat = other.out_feat;
        size = other.size;
    }
    void run(half *out, half *in, cudaStream_t stream) {
        if (is_gelu)
            tile_spmm_add_bias_gelu_kernelLauncher(
                out, in, row_ptr.get(), row_offset.get(), data.get(), size,
                out_feat, bias.get(), stream);
        else
            tile_spmm_add_bias_kernelLauncher(
                out, in, row_ptr.get(), row_offset.get(), data.get(), size,
                out_feat, bias.get(), stream);
    }
};

class Linear_row {
  public:
    int in_feat;
    int out_feat;
    int size;
    int nhead;
    // weight
    Weight<half> weight;
    Weight<int> row_id; // sorted outside
    Weight<int> head_ptr;
    int nrow;
    int row_id_size;
    // bias
    Weight<half> bias;

    cublasGemmAlgo_t algo;

  public:
    Linear_row(int in_feat_, int out_feat_, int size_, half *weight_,
               int *row_id_, int nrow_, half *bias_, int nhead_);
    void run(half *out, half *in, cublasHandle_t handle) {
        row_spmm_add_bias_kernelLauncher(out, in, weight.get(), bias.get(),
                                         size, nrow, in_feat, handle, algo);
    }
};
