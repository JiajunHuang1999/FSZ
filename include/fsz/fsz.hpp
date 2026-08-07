#ifndef FSZ_HPP
#define FSZ_HPP

#include "fsz/fsz.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace fsz {

class Error : public std::runtime_error {
public:
    Error(fsz_status_t status, const std::string& where)
        : std::runtime_error(where + ": " + fsz_status_string(status)),
          status_(status) {}
    fsz_status_t status() const noexcept { return status_; }
private:
    fsz_status_t status_;
};

inline void check(fsz_status_t s, const char* where) {
    if (s != FSZ_STATUS_OK) throw Error(s, where);
}

class Workspace {
public:
    explicit Workspace(std::size_t max_n_elements) {
        check(fsz_workspace_create(&ws_, max_n_elements), "fsz_workspace_create");
    }
    ~Workspace() { if (ws_) fsz_workspace_destroy(ws_); }

    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace(Workspace&& other) noexcept : ws_(other.ws_) { other.ws_ = nullptr; }
    Workspace& operator=(Workspace&& other) noexcept {
        if (this != &other) {
            if (ws_) fsz_workspace_destroy(ws_);
            ws_ = other.ws_; other.ws_ = nullptr;
        }
        return *this;
    }

    fsz_workspace_t* handle() noexcept { return ws_; }
    std::size_t capacity() const noexcept { return fsz_workspace_capacity(ws_); }

private:
    fsz_workspace_t* ws_ = nullptr;
};

inline fsz_compress_result_t compress(
    const float* d_in, unsigned char* d_cmp,
    std::size_t n, float abs_error_bound,
    Workspace& ws, cudaStream_t stream = 0)
{
    fsz_compress_result_t r{};
    check(fsz_compress(d_in, d_cmp, n, abs_error_bound, ws.handle(), stream, &r),
          "fsz_compress");
    return r;
}

inline void decompress(
    float* d_out, const unsigned char* d_cmp,
    std::size_t n, float abs_error_bound,
    const fsz_compress_result_t& result,
    Workspace& ws, cudaStream_t stream = 0)
{
    check(fsz_decompress(d_out, d_cmp, n, abs_error_bound, &result, ws.handle(), stream),
          "fsz_decompress");
}

inline fsz_compress_result_t compress(
    const float* d_in, unsigned char* d_cmp,
    std::size_t n, float abs_error_bound, cudaStream_t stream = 0)
{
    fsz_compress_result_t r{};
    check(fsz_compress(d_in, d_cmp, n, abs_error_bound, nullptr, stream, &r),
          "fsz_compress");
    return r;
}

inline fsz_compress_result_t compress(
    const double* d_in, unsigned char* d_cmp,
    std::size_t n, double abs_error_bound, cudaStream_t stream = 0)
{
    fsz_compress_result_t r{};
    check(fsz_compress_f64(d_in, d_cmp, n, abs_error_bound, nullptr, stream, &r),
          "fsz_compress_f64");
    return r;
}

inline void decompress(
    float* d_out, const unsigned char* d_cmp,
    std::size_t n, float abs_error_bound,
    const fsz_compress_result_t& result, cudaStream_t stream = 0)
{
    check(fsz_decompress(d_out, d_cmp, n, abs_error_bound, &result, nullptr, stream),
          "fsz_decompress");
}

inline void decompress(
    double* d_out, const unsigned char* d_cmp,
    std::size_t n, double abs_error_bound,
    const fsz_compress_result_t& result, cudaStream_t stream = 0)
{
    check(fsz_decompress_f64(d_out, d_cmp, n, abs_error_bound, &result, nullptr, stream),
          "fsz_decompress_f64");
}

inline std::size_t compress_hostptr(
    const float* in, unsigned char* cmp,
    std::size_t n, float abs_error_bound)
{
    std::size_t cmp_size = 0;
    check(fsz_compress_hostptr(in, cmp, n, abs_error_bound, &cmp_size),
          "fsz_compress_hostptr");
    return cmp_size;
}

inline void decompress_hostptr(
    float* out, const unsigned char* cmp, std::size_t cmp_size,
    std::size_t n, float abs_error_bound)
{
    check(fsz_decompress_hostptr(out, cmp, cmp_size, n, abs_error_bound),
          "fsz_decompress_hostptr");
}

inline fsz_compress_result_t make_result(std::size_t n, std::size_t cmp_size) {
    return fsz_make_result(n, cmp_size);
}

inline fsz_compress_result_t compress(
    const double* d_in, unsigned char* d_cmp,
    std::size_t n, double abs_error_bound,
    Workspace& ws, cudaStream_t stream = 0)
{
    fsz_compress_result_t r{};
    check(fsz_compress_f64(d_in, d_cmp, n, abs_error_bound, ws.handle(), stream, &r),
          "fsz_compress_f64");
    return r;
}

inline void decompress(
    double* d_out, const unsigned char* d_cmp,
    std::size_t n, double abs_error_bound,
    const fsz_compress_result_t& result,
    Workspace& ws, cudaStream_t stream = 0)
{
    check(fsz_decompress_f64(d_out, d_cmp, n, abs_error_bound, &result, ws.handle(), stream),
          "fsz_decompress_f64");
}

inline std::size_t compress_hostptr(
    const double* in, unsigned char* cmp, std::size_t n, double abs_error_bound)
{
    std::size_t cmp_size = 0;
    check(fsz_compress_hostptr_f64(in, cmp, n, abs_error_bound, &cmp_size),
          "fsz_compress_hostptr_f64");
    return cmp_size;
}

inline void decompress_hostptr(
    double* out, const unsigned char* cmp, std::size_t cmp_size,
    std::size_t n, double abs_error_bound)
{
    check(fsz_decompress_hostptr_f64(out, cmp, cmp_size, n, abs_error_bound),
          "fsz_decompress_hostptr_f64");
}

inline std::size_t max_compressed_bytes(std::size_t n) { return fsz_max_compressed_bytes(n); }
inline std::size_t num_tiles(std::size_t n)            { return fsz_num_tiles(n); }
inline const char* version()                            { return fsz_version(); }

}

#endif
