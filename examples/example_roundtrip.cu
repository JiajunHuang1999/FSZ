#include "fsz/fsz.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    std::size_t n = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1u << 22);
    float    eb = (argc > 2) ? std::strtof(argv[2], nullptr) : 1e-3f;

    std::printf("FSZ %s  N=%zu  abs_eb=%g\n", fsz::version(), n, eb);

    std::vector<float> h_in(n);
    for (std::size_t i = 0; i < n; ++i)
        h_in[i] = std::sin(0.001f * (float)i) + 0.01f * std::cos(0.07f * (float)i);

    float* d_in   = nullptr;
    float* d_out  = nullptr;
    unsigned char* d_cmp = nullptr;
    check_cuda(cudaMalloc(&d_in,  n * sizeof(float)),                          "alloc d_in");
    check_cuda(cudaMalloc(&d_out, n * sizeof(float)),                          "alloc d_out");
    check_cuda(cudaMalloc(&d_cmp, fsz::max_compressed_bytes(n)),               "alloc d_cmp");
    check_cuda(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice),                             "memcpy h2d");

    fsz::Workspace ws(n);

    auto cresult = fsz::compress(d_in, d_cmp, n, eb, ws);
    fsz::decompress(d_out, d_cmp, n, eb, cresult, ws);
    check_cuda(cudaDeviceSynchronize(), "synchronize");

    std::vector<float> h_out(n);
    check_cuda(cudaMemcpy(h_out.data(), d_out, n * sizeof(float),
                          cudaMemcpyDeviceToHost),                             "memcpy d2h");

    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double e = std::fabs((double)h_in[i] - (double)h_out[i]);
        if (e > max_err) max_err = e;
    }

    double cr = (double)(n * sizeof(float)) / (double)cresult.cmp_size;
    bool ok   = (max_err <= (double)eb * 1.01);

    std::printf("  CR        = %.3f\n", cr);
    std::printf("  cmp_size  = %zu B\n", cresult.cmp_size);
    std::printf("  max_err   = %.3e (bound %.3e) -> %s\n",
                max_err, (double)eb, ok ? "PASS" : "FAIL");

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_cmp);
    return ok ? 0 : 1;
}
