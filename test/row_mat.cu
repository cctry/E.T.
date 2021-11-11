#include "utils.h"
#include <algorithm>
#include <assert.h>
#include <numeric>
void gen_sparse_row_mat(int nrow, int ncol, float sparsity,
                         std::vector<half> &rows, std::vector<int> &row_id) {
    int nn_row = nrow * (1.0 - sparsity);
    rows = std::vector<half>(ncol * nn_row, __float2half_rn(1.0));
    row_id = std::vector<int>(nrow);
    std::iota(row_id.begin(), row_id.end(), 0);
    std::random_shuffle(row_id.begin(), row_id.end());
    row_id.erase(row_id.begin() + nn_row, row_id.end());
    std::sort(row_id.begin(), row_id.end());
}