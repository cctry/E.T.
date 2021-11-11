add_executable(encoder_tile_test
    tile_mat.cu
    row_mat.cu
    tile_test.cu
    gen_linear.cu
)

target_link_libraries(encoder_tile_test
encoder
kernels
)

add_executable(encoder_prune_test
    tile_mat.cu
    row_mat.cu
    prune_test.cu
    gen_linear.cu
)

target_link_libraries(encoder_prune_test
encoder
kernels
)

add_executable(encoder_length_test
    tile_mat.cu
    row_mat.cu
    length_test.cu
    gen_linear.cu
)

target_link_libraries(encoder_length_test
encoder
kernels
)