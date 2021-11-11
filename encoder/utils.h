#pragma once
#include <assert.h>
#include <cstdio>
#define cudaChk(stat) cudaErrCheck_((stat), __FILE__, __LINE__)

inline void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat),
                file, line);
        exit(1);
    }
}

template <typename T> class Weight {
    T *data;
    int size;

  public:
    Weight(int size_) : data(NULL), size(size_) {
        cudaChk(cudaMalloc(&data, size * sizeof(T)));
    }
    Weight(T *data_, int size_) : data(NULL), size(size_) {
        cudaChk(cudaMalloc(&data, size * sizeof(T)));
        cudaChk(cudaMemcpy(data, data_, size * sizeof(T), cudaMemcpyHostToDevice));
    }
    Weight() : data(NULL), size(0) {}
    ~Weight() {
        if (data)
            cudaChk(cudaFree(data));
    }
    Weight(const Weight &w) = delete;
    Weight &operator=(const Weight &w) = delete;
    Weight(Weight &&w) {
        data = w.data;
        size = w.size;
        w.data = NULL;
    }
    T *get() { return data; }
    void alloc(int size) { cudaChk(cudaMalloc(&data, size * sizeof(T))); }
    void set(T *src) {
        cudaChk(cudaMemcpy(data, src, size * sizeof(T), cudaMemcpyHostToDevice));
    }
};