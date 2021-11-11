#include "utils.h"
#include <cstdio>
#include <encoder/Linear.h>
#include <encoder/utils.h>
Linear_tile gen_tile_linear(int in_feat, int out_feat, int size, float sparsity,
                            bool is_gelu) {
    std::vector<int> row_ptr, row_offset;
    std::vector<half> bias(out_feat, __float2half(0.1f));
    gen_sparse_tile_mat(out_feat, in_feat, sparsity, row_ptr, row_offset);
    std::vector<half> data(row_offset.size() * 256, __float2half(0.1f));
    Linear_tile tile(in_feat, out_feat, size, row_offset.size(), data.data(),
                     row_ptr.data(), row_offset.data(), bias.data(), is_gelu);
    return std::move(tile);
}

Linear_row gen_row_linear(int in_feat, int out_feat, int size, float sparsity,
                          int nhead) {
    std::vector<half> rows;
    std::vector<int> row_id;
    gen_sparse_row_mat(out_feat, in_feat, sparsity, rows, row_id);
    std::vector<half> bias(row_id.size(), __float2half(0.1f));
    Linear_row row(in_feat, out_feat, size, rows.data(), row_id.data(),
                   row_id.size(), bias.data(), nhead);
    return std::move(row);
}