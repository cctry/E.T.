#pragma once
#include <cuda_fp16.h>
#include <utility>
#include <vector>
struct Coord {
    int stride;
    std::pair<int, int> value;
    Coord operator++() {
        if (x() < stride - 1) {
            x()++;
        } else {
            x() = 0;
            y()++;
        }
        return *this;
    };
    int &x() { return value.first; };
    int &y() { return value.second; };
    const int &x() const { return value.first; };
    const int &y() const { return value.second; };
    Coord(int _stride) : stride(_stride), value{0, 0} {};
    Coord(int _stride, int _x, int _y) : stride(_stride), value{_x, _y} {};
};

void gen_sparse_tile_mat(int nrow, int ncol, float sparsity,
                         std::vector<int> &row_ptr,
                         std::vector<int> &row_offset);
void gen_sparse_row_mat(int nrow, int ncol, float sparsity,
                        std::vector<half> &rows, std::vector<int> &row_id);

#include <encoder/Linear.h>
#include <encoder/utils.h>

Linear_tile gen_tile_linear(int in_feat, int out_feat, int size, float sparsity,
                            bool is_gelu = false);

Linear_row gen_row_linear(int in_feat, int out_feat, int size, float sparsity,
                          int nhead);