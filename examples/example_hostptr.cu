
#include "fsz/fsz.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const std::size_t dim = 64;
    const std::size_t n   = dim * dim * dim;
    const float       eb  = 1e-3f;   // absolute error bound

    std::vector<float> in(n);
    for (std::size_t z = 0; z < dim; ++z)
        for (std::size_t y = 0; y < dim; ++y)
            for (std::size_t x = 0; x < dim; ++x)
                in[(z * dim + y) * dim + x] = std::sin(0.05f * (float)x)
                                            * std::cos(0.04f * (float)y)
                                            + 0.1f * std::sin(0.03f * (float)z);

    std::vector<unsigned char> cmp(fsz_max_compressed_bytes(n));
    std::size_t cmp_size = 0;
    fsz_status_t s = fsz_compress_hostptr(in.data(), cmp.data(), n, eb, &cmp_size);
    if (s != FSZ_STATUS_OK) {
        std::printf("compress failed: %s\n", fsz_status_string(s));
        return 1;
    }

    std::vector<float> out(n);
    s = fsz_decompress_hostptr(out.data(), cmp.data(), cmp_size, n, eb);
    if (s != FSZ_STATUS_OK) {
        std::printf("decompress failed: %s\n", fsz_status_string(s));
        return 1;
    }

    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double e = std::fabs((double)in[i] - (double)out[i]);
        if (e > max_err) max_err = e;
    }

    std::printf("FSZ %s  N=%zu  abs_eb=%g\n", fsz_version(), n, (double)eb);
    std::printf("  cmp_size  = %zu B (from %zu B)\n", cmp_size, n * sizeof(float));
    std::printf("  CR        = %.3f\n", (double)(n * sizeof(float)) / (double)cmp_size);
    std::printf("  max_err   = %.3e (bound %.3e)\n", max_err, (double)eb);
    return (max_err <= 1.01 * (double)eb) ? 0 : 1;
}
