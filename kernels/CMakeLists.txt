add_library(kernels
skip_layernorm.cu
linear.cu
attention.cu
)

target_link_libraries(kernels
cublas
)