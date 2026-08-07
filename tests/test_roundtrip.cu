#include "fsz/fsz.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int run_case(std::size_t n, float eb, const char* label) {
    std::vector<float> h_in(n);
    for (std::size_t i = 0; i < n; ++i)
        h_in[i] = std::sin(0.0017f * (float)i) + 0.05f * std::cos(0.13f * (float)i);

    float* d_in   = nullptr;
    float* d_out  = nullptr;
    unsigned char* d_cmp = nullptr;
    cudaMalloc(&d_in,  n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMalloc(&d_cmp, fsz::max_compressed_bytes(n));
    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    fsz::Workspace ws(n);
    auto r = fsz::compress(d_in, d_cmp, n, eb, ws);
    fsz::decompress(d_out, d_cmp, n, eb, r, ws);
    cudaDeviceSynchronize();

    std::vector<float> h_out(n);
    cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double e = std::fabs((double)h_in[i] - (double)h_out[i]);
        if (e > max_err) max_err = e;
    }
    double cr = (double)(n * sizeof(float)) / (double)r.cmp_size;
    bool ok   = (max_err <= (double)eb * 1.01);

    std::printf("[%-12s] N=%-9zu eb=%g  CR=%.3f  max_err=%.3e  %s\n",
                label, n, (double)eb, cr, max_err, ok ? "PASS" : "FAIL");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_cmp);
    return ok ? 0 : 1;
}

static int run_case_f64(std::size_t n, double eb, const char* label) {
    std::vector<double> h_in(n);
    for (std::size_t i = 0; i < n; ++i)
        h_in[i] = std::sin(0.0017 * (double)i) + 0.05 * std::cos(0.13 * (double)i);

    double* d_in  = nullptr;
    double* d_out = nullptr;
    unsigned char* d_cmp = nullptr;
    cudaMalloc(&d_in,  n * sizeof(double));
    cudaMalloc(&d_out, n * sizeof(double));
    cudaMalloc(&d_cmp, fsz::max_compressed_bytes(n));
    cudaMemcpy(d_in, h_in.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    fsz::Workspace ws(n);
    auto r = fsz::compress(d_in, d_cmp, n, eb, ws);
    fsz::decompress(d_out, d_cmp, n, eb, r, ws);
    cudaDeviceSynchronize();

    std::vector<double> h_out(n);
    cudaMemcpy(h_out.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double e = std::fabs(h_in[i] - h_out[i]);
        if (e > max_err) max_err = e;
    }
    double cr = (double)(n * sizeof(double)) / (double)r.cmp_size;
    bool ok   = (max_err <= eb * 1.01);

    std::printf("[%-12s] N=%-9zu eb=%g  CR=%.3f  max_err=%.3e  %s\n",
                label, n, eb, cr, max_err, ok ? "PASS" : "FAIL");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_cmp);
    return ok ? 0 : 1;
}

int main() {
    int fails = 0;
    fails += run_case(1u << 12, 1e-3f, "4K");
    fails += run_case(1u << 16, 1e-3f, "64K");
    fails += run_case(1u << 18, 1e-3f, "256K");
    fails += run_case(1u << 22, 1e-3f, "4M");
    fails += run_case(1u << 22, 1e-2f, "4M-eb1e-2");
    fails += run_case(1u << 22, 1e-4f, "4M-eb1e-4");
    fails += run_case(1u << 25, 1e-3f, "32M");
    fails += run_case(1000003,   1e-3f, "non-aligned");
    fails += run_case_f64(1u << 12, 1e-3, "4K-f64");
    fails += run_case_f64(1u << 22, 1e-3, "4M-f64");
    fails += run_case_f64(1u << 22, 1e-9, "4M-f64-tight");
    fails += run_case_f64(1u << 25, 1e-3, "32M-f64");
    fails += run_case_f64(1000003,  1e-3, "non-aligned-f64");
    if (fails) std::printf("FAILURES: %d\n", fails);
    else       std::printf("ALL PASS\n");
    return fails ? 1 : 0;
}
