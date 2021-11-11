#include "utils.h"
#include <numeric>
#include <algorithm>
#include <assert.h>
void gen_sparse_tile_mat(int nrow, int ncol, float sparsity,
                         std::vector<int> &row_ptr,
                         std::vector<int> &row_offset) {
    const int blk_row_num = (nrow / 16);
    const int blk_col_num = (ncol / 16);
    const int total_tile = blk_col_num * blk_row_num;
    const int ntile = total_tile * (1.0 - sparsity);
    row_ptr.reserve(blk_row_num + 1);
    row_offset.reserve(ntile);
    std::vector<Coord> tile_id(total_tile, Coord(blk_col_num));
    std::iota(tile_id.begin(), tile_id.end(), Coord(blk_col_num));
    std::random_shuffle(tile_id.begin(), tile_id.end());
    tile_id.erase(tile_id.begin() + ntile, tile_id.end());
    int offset = 0;
    row_ptr.push_back(offset);
    for (int r = 0; r < blk_row_num; r++) {
        std::vector<Coord> tiles;
        std::copy_if(tile_id.begin(), tile_id.end(), std::back_inserter(tiles),
                     [r](const Coord &A) { return A.y() == r; });
        offset += tiles.size();
        row_ptr.push_back(offset);
        std::vector<int> col_id;
        std::transform(tiles.begin(), tiles.end(), std::back_inserter(col_id),
                       [](const Coord &A) { return A.x() * 16; });
        row_offset.insert(row_offset.end(), col_id.begin(), col_id.end());
    }
    assert(row_ptr.size() == blk_row_num + 1);
    assert(row_ptr.back() == ntile);
    assert(row_offset.size() == ntile);
}