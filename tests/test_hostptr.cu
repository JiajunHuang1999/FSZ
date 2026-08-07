#include "fsz/fsz.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const std::size_t NX = 97, NY = 113, NZ = 131;

static void fill_smooth(std::vector<float>& v) {
    for (std::size_t z = 0; z < NZ; ++z) {
        for (std::size_t y = 0; y < NY; ++y) {
            for (std::size_t x = 0; x < NX; ++x) {
                float fx = 0.031f * (float)x;
                float fy = 0.027f * (float)y;
                float fz = 0.019f * (float)z;
                v[(z * NY + y) * NX + x] = std::sin(fx) * std::cos(fy) * std::sin(fz)
                                         + 0.05f * std::cos(7.0f * fx + 5.0f * fy)
                                         + 0.02f * std::sin(11.0f * fz);
            }
        }
    }
}

static void fill_smooth_f64(std::vector<double>& v) {
    for (std::size_t z = 0; z < NZ; ++z) {
        for (std::size_t y = 0; y < NY; ++y) {
            for (std::size_t x = 0; x < NX; ++x) {
                double fx = 0.031 * (double)x;
                double fy = 0.027 * (double)y;
                double fz = 0.019 * (double)z;
                v[(z * NY + y) * NX + x] = std::sin(fx) * std::cos(fy) * std::sin(fz)
                                         + 0.05 * std::cos(7.0 * fx + 5.0 * fy)
                                         + 0.02 * std::sin(11.0 * fz);
            }
        }
    }
}

static void fill_near_constant(std::vector<float>& v, float base) {
    for (std::size_t i = 0; i < v.size(); ++i)
        v[i] = base + 1e-6f * std::sin(0.0001f * (float)i);
}

static bool compress_device(const std::vector<float>& h_in, float eb,
                            std::vector<unsigned char>& out_cmp,
                            fsz_compress_result_t* out_result) {
    std::size_t n = h_in.size();
    float* d_in = nullptr;
    unsigned char* d_cmp = nullptr;
    if (cudaMalloc(&d_in, n * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_cmp, fsz::max_compressed_bytes(n)) != cudaSuccess) {
        cudaFree(d_in);
        return false;
    }
    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    fsz::Workspace ws(n);
    *out_result = fsz::compress(d_in, d_cmp, n, eb, ws);

    out_cmp.resize(out_result->cmp_size);
    cudaMemcpy(out_cmp.data(), d_cmp, out_result->cmp_size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_cmp);
    return cudaDeviceSynchronize() == cudaSuccess;
}

static bool compress_device_f64(const std::vector<double>& h_in, double eb,
                                std::vector<unsigned char>& out_cmp,
                                fsz_compress_result_t* out_result) {
    std::size_t n = h_in.size();
    double* d_in = nullptr;
    unsigned char* d_cmp = nullptr;
    if (cudaMalloc(&d_in, n * sizeof(double)) != cudaSuccess) return false;
    if (cudaMalloc(&d_cmp, fsz::max_compressed_bytes(n)) != cudaSuccess) {
        cudaFree(d_in);
        return false;
    }
    cudaMemcpy(d_in, h_in.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    fsz::Workspace ws(n);
    *out_result = fsz::compress(d_in, d_cmp, n, eb, ws);

    out_cmp.resize(out_result->cmp_size);
    cudaMemcpy(out_cmp.data(), d_cmp, out_result->cmp_size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_cmp);
    return cudaDeviceSynchronize() == cudaSuccess;
}

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b,
                           std::size_t* out_nonfinite) {
    double max_err = 0.0;
    std::size_t bad = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(b[i])) { ++bad; continue; }
        double e = std::fabs((double)a[i] - (double)b[i]);
        if (e > max_err) max_err = e;
    }
    *out_nonfinite = bad;
    return max_err;
}

static double max_abs_diff_f64(const std::vector<double>& a, const std::vector<double>& b,
                               std::size_t* out_nonfinite) {
    double max_err = 0.0;
    std::size_t bad = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(b[i])) { ++bad; continue; }
        double e = std::fabs(a[i] - b[i]);
        if (e > max_err) max_err = e;
    }
    *out_nonfinite = bad;
    return max_err;
}

static bool same_result(const fsz_compress_result_t& a, const fsz_compress_result_t& b) {
    return a.cmp_size == b.cmp_size && a.num_tiles == b.num_tiles
        && a.data_offset == b.data_offset;
}

static int report(const char* name, bool ok, const char* detail) {
    std::printf("[%s] %-22s %s\n", ok ? "PASS" : "FAIL", name, detail);
    return ok ? 0 : 1;
}

int main() {
    int fails = 0;
    char detail[256];
    const float eb = 1e-3f;

    const std::size_t nA = NX * NY * NZ;
    std::vector<float> inA(nA);
    fill_smooth(inA);

    std::vector<unsigned char> cmpA_dev;
    fsz_compress_result_t rA;
    if (!compress_device(inA, eb, cmpA_dev, &rA)) {
        std::printf("[FAIL] setup                  device-pointer compress of field A failed\n");
        return 1;
    }

    std::vector<unsigned char> cmpA_host(fsz::max_compressed_bytes(nA));
    std::size_t cmpA_host_size = 0;
    fsz_status_t st = fsz_compress_hostptr(inA.data(), cmpA_host.data(), nA, eb,
                                           &cmpA_host_size);

    bool sizes_match = (st == FSZ_STATUS_OK) && (cmpA_host_size == rA.cmp_size);
    bool bytes_match = sizes_match
        && std::memcmp(cmpA_host.data(), cmpA_dev.data(), rA.cmp_size) == 0;
    std::snprintf(detail, sizeof(detail),
                  "n=%zu status=%s device=%zu B hostptr=%zu B memcmp=%s",
                  nA, fsz_status_string(st), rA.cmp_size, cmpA_host_size,
                  bytes_match ? "equal" : "differs");
    fails += report("parity", bytes_match, detail);

    std::vector<float> outA(nA, 0.0f);
    st = fsz_decompress_hostptr(outA.data(), cmpA_host.data(), cmpA_host_size, nA, eb);
    std::size_t badA = 0;
    double errA = max_abs_diff(inA, outA, &badA);
    bool okA = (st == FSZ_STATUS_OK) && (badA == 0) && (errA <= 1.01 * (double)eb);
    std::snprintf(detail, sizeof(detail),
                  "n=%zu status=%s max_err=%.3e bound=%.3e nonfinite=%zu",
                  nA, fsz_status_string(st), errA, 1.01 * (double)eb, badA);
    fails += report("roundtrip", okA, detail);

    const std::size_t nB = 3000000;
    std::vector<float> inB(nB);
    fill_near_constant(inB, 1e-4f);

    std::vector<unsigned char> cmpB_host(fsz::max_compressed_bytes(nB));
    std::size_t cmpB_host_size = 0;
    st = fsz_compress_hostptr(inB.data(), cmpB_host.data(), nB, eb, &cmpB_host_size);

    std::vector<float> outB(nB, 1.0f);
    fsz_status_t st_d = FSZ_STATUS_INVALID_ARGUMENT;
    if (st == FSZ_STATUS_OK)
        st_d = fsz_decompress_hostptr(outB.data(), cmpB_host.data(), cmpB_host_size,
                                      nB, eb);
    std::size_t badB = 0;
    double errB = max_abs_diff(inB, outB, &badB);
    double crB = cmpB_host_size ? (double)(nB * sizeof(float)) / (double)cmpB_host_size : 0.0;
    bool okB = (st == FSZ_STATUS_OK) && (st_d == FSZ_STATUS_OK) && (crB > 100.0)
            && (badB == 0) && (errB <= 1.01 * (double)eb);
    std::snprintf(detail, sizeof(detail),
                  "n=%zu CR=%.2f cmp=%zu B max_err=%.3e nonfinite=%zu",
                  nB, crB, cmpB_host_size, errB, badB);
    fails += report("high-ratio", okB, detail);

    const std::size_t nC = 4096;
    std::vector<float> inC(nC);
    for (std::size_t i = 0; i < nC; ++i)
        inC[i] = std::sin(0.05f * (float)i) + 0.2f * std::cos(0.31f * (float)i);
    std::vector<unsigned char> cmpC(fsz::max_compressed_bytes(nC));
    std::size_t cmpC_size = 0;
    st = fsz_compress_hostptr(inC.data(), cmpC.data(), nC, eb, &cmpC_size);
    std::vector<float> outC(nC, 0.0f);
    st_d = FSZ_STATUS_INVALID_ARGUMENT;
    if (st == FSZ_STATUS_OK)
        st_d = fsz_decompress_hostptr(outC.data(), cmpC.data(), cmpC_size, nC, eb);
    std::size_t badC = 0;
    double errC = max_abs_diff(inC, outC, &badC);
    std::size_t hdrC = fsz::num_tiles(nC) * 8;
    bool okC = (st == FSZ_STATUS_OK) && (st_d == FSZ_STATUS_OK) && (cmpC_size > hdrC)
            && (badC == 0) && (errC <= 1.01 * (double)eb);
    std::snprintf(detail, sizeof(detail),
                  "n=%zu cmp=%zu B (header %zu B) max_err=%.3e nonfinite=%zu",
                  nC, cmpC_size, hdrC, errC, badC);
    fails += report("small-array", okC, detail);

    std::vector<unsigned char> cmpB_dev;
    fsz_compress_result_t rB;
    if (!compress_device(inB, eb, cmpB_dev, &rB)) {
        std::printf("[FAIL] setup                  device-pointer compress of field B failed\n");
        return 1;
    }
    fsz_compress_result_t mA = fsz::make_result(nA, rA.cmp_size);
    fsz_compress_result_t mB = fsz::make_result(nB, rB.cmp_size);
    bool okM = same_result(mA, rA) && same_result(mB, rB);
    std::snprintf(detail, sizeof(detail),
                  "A{%zu,%zu,%zu} B{%zu,%zu,%zu}",
                  mA.cmp_size, mA.num_tiles, mA.data_offset,
                  mB.cmp_size, mB.num_tiles, mB.data_offset);
    fails += report("make-result", okM, detail);

    {
        float* d_in = nullptr;
        unsigned char* d_cmp = nullptr;
        float* d_out = nullptr;
        cudaMalloc(&d_in, nA * sizeof(float));
        cudaMalloc(&d_cmp, fsz::max_compressed_bytes(nA));
        cudaMalloc(&d_out, nA * sizeof(float));
        cudaMemcpy(d_in, inA.data(), nA * sizeof(float), cudaMemcpyHostToDevice);
        fsz_compress_result_t rN{};
        fsz_status_t stN = fsz_compress(d_in, d_cmp, nA, eb, nullptr, 0, &rN);
        std::vector<unsigned char> cmpN(rN.cmp_size ? rN.cmp_size : 1);
        if (stN == FSZ_STATUS_OK)
            cudaMemcpy(cmpN.data(), d_cmp, rN.cmp_size, cudaMemcpyDeviceToHost);
        bool stream_match = (stN == FSZ_STATUS_OK) && (rN.cmp_size == rA.cmp_size)
            && std::memcmp(cmpN.data(), cmpA_dev.data(), rA.cmp_size) == 0;
        fsz_status_t stD = fsz_decompress(d_out, d_cmp, nA, eb, &rN, nullptr, 0);
        std::vector<float> outN(nA, 0.0f);
        cudaMemcpy(outN.data(), d_out, nA * sizeof(float), cudaMemcpyDeviceToHost);
        std::size_t badN = 0;
        double errN = max_abs_diff(inA, outN, &badN);
        bool okN = stream_match && (stD == FSZ_STATUS_OK) && (badN == 0)
                && (errN <= 1.01 * (double)eb);
        std::snprintf(detail, sizeof(detail),
                      "cmp=%zu B memcmp=%s max_err=%.3e nonfinite=%zu",
                      (size_t)rN.cmp_size, stream_match ? "equal" : "differs",
                      errN, badN);
        fails += report("null-workspace", okN, detail);
        cudaFree(d_in);
        cudaFree(d_cmp);
        cudaFree(d_out);
    }

    {
        const std::size_t sizes[] = {1, 4, 8, 16, 31, 33, 127};
        bool okS = true;
        char worst[96] = "";
        for (std::size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); ++k) {
            const std::size_t ns = sizes[k];
            std::vector<float> inS(ns), outS(ns, 0.0f);
            for (std::size_t i = 0; i < ns; ++i)
                inS[i] = (i % 2) ? 4.0e9f : -4.0e9f;
            std::vector<unsigned char> cmpS(fsz::max_compressed_bytes(ns));
            std::size_t szS = 0;
            fsz_status_t s1 = fsz_compress_hostptr(inS.data(), cmpS.data(), ns, 1.0f, &szS);
            fsz_status_t s2 = (s1 == FSZ_STATUS_OK)
                ? fsz_decompress_hostptr(outS.data(), cmpS.data(), szS, ns, 1.0f)
                : s1;
            std::size_t badS = 0;
            double errS = max_abs_diff(inS, outS, &badS);
            if (!(s1 == FSZ_STATUS_OK && s2 == FSZ_STATUS_OK && szS <= cmpS.size()
                  && badS == 0 && errS <= 1.01)) {
                okS = false;
                std::snprintf(worst, sizeof(worst), "n=%zu status=%s err=%.2e",
                              ns, fsz_status_string(s1 != FSZ_STATUS_OK ? s1 : s2), errS);
            }
        }
        std::snprintf(detail, sizeof(detail), "7 sizes, wide range: %s",
                      okS ? "within bound and capacity" : worst);
        fails += report("tiny-arrays", okS, detail);
    }

    const double ebd = 1e-3;
    const std::size_t nD = NX * NY * NZ;
    std::vector<double> inD(nD);
    fill_smooth_f64(inD);

    std::vector<unsigned char> cmpD_dev;
    fsz_compress_result_t rD;
    if (!compress_device_f64(inD, ebd, cmpD_dev, &rD)) {
        std::printf("[FAIL] setup                  device-pointer compress of field D failed\n");
        return 1;
    }

    std::vector<unsigned char> cmpD_host(fsz::max_compressed_bytes(nD));
    std::size_t cmpD_host_size = 0;
    st = fsz_compress_hostptr_f64(inD.data(), cmpD_host.data(), nD, ebd,
                                  &cmpD_host_size);

    bool sizes_match_f64 = (st == FSZ_STATUS_OK) && (cmpD_host_size == rD.cmp_size);
    bool bytes_match_f64 = sizes_match_f64
        && std::memcmp(cmpD_host.data(), cmpD_dev.data(), rD.cmp_size) == 0;
    std::snprintf(detail, sizeof(detail),
                  "n=%zu status=%s device=%zu B hostptr=%zu B memcmp=%s",
                  nD, fsz_status_string(st), rD.cmp_size, cmpD_host_size,
                  bytes_match_f64 ? "equal" : "differs");
    fails += report("parity-f64", bytes_match_f64, detail);

    std::vector<double> outD(nD, 0.0);
    st = fsz_decompress_hostptr_f64(outD.data(), cmpD_host.data(), cmpD_host_size,
                                    nD, ebd);
    std::size_t badD = 0;
    double errD = max_abs_diff_f64(inD, outD, &badD);
    bool okD = (st == FSZ_STATUS_OK) && (badD == 0) && (errD <= 1.01 * ebd);
    std::snprintf(detail, sizeof(detail),
                  "n=%zu status=%s max_err=%.3e bound=%.3e nonfinite=%zu",
                  nD, fsz_status_string(st), errD, 1.01 * ebd, badD);
    fails += report("roundtrip-f64", okD, detail);

    const std::size_t nE = 4096;
    std::vector<double> inE(nE);
    for (std::size_t i = 0; i < nE; ++i)
        inE[i] = std::sin(0.05 * (double)i) + 0.2 * std::cos(0.31 * (double)i);
    std::vector<unsigned char> cmpE(fsz::max_compressed_bytes(nE));
    std::size_t cmpE_size = 0;
    st = fsz_compress_hostptr_f64(inE.data(), cmpE.data(), nE, ebd, &cmpE_size);
    std::vector<double> outE(nE, 0.0);
    st_d = FSZ_STATUS_INVALID_ARGUMENT;
    if (st == FSZ_STATUS_OK)
        st_d = fsz_decompress_hostptr_f64(outE.data(), cmpE.data(), cmpE_size, nE, ebd);
    std::size_t badE = 0;
    double errE = max_abs_diff_f64(inE, outE, &badE);
    std::size_t hdrE = fsz::num_tiles(nE) * 8;
    bool okE = (st == FSZ_STATUS_OK) && (st_d == FSZ_STATUS_OK) && (cmpE_size > hdrE)
            && (badE == 0) && (errE <= 1.01 * ebd);
    std::snprintf(detail, sizeof(detail),
                  "n=%zu cmp=%zu B (header %zu B) max_err=%.3e nonfinite=%zu",
                  nE, cmpE_size, hdrE, errE, badE);
    fails += report("small-array-f64", okE, detail);

    if (fails) std::printf("FAILURES: %d\n", fails);
    else       std::printf("ALL PASS\n");
    return fails ? 1 : 0;
}
