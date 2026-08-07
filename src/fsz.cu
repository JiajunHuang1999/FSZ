#include "fsz/fsz.h"

#include "fsz_internal.cuh"
#include "fsz_kernels.cuh"

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

using fsz_detail::fsz_compress_kernel_T;
using fsz_detail::fsz_decompress_kernel_T;

struct fsz_workspace {
    unsigned int* d_workspace;
    unsigned int* cmpOffset;
    unsigned int* locOffset;
    int* flag;
    unsigned int* h_cmpSize;
    int gsize_max;
    int gsize_sm_max;
    int cmpOffSize;
    size_t max_n_elements;
};

static int fsz_grid_size(size_t numTiles) {
    return (int)((numTiles + FSZ_TBLOCK_SZ * FSZ_TILES_PER_THR - 1)
                 / (FSZ_TBLOCK_SZ * FSZ_TILES_PER_THR));
}

static int fsz_grid_size_sm(size_t numTiles) {
    return (int)((numTiles + FSZ_TBLOCK_SZ * FSZ_TILES_PER_THR_SM - 1)
                 / (FSZ_TBLOCK_SZ * FSZ_TILES_PER_THR_SM));
}

static void fsz_hostptr_release(void* d_a, void* d_b, fsz_workspace_t* ws) {
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (ws)  fsz_workspace_destroy(ws);
}

extern "C" const char* fsz_version(void) {
    return FSZ_VERSION_STRING;
}

extern "C" const char* fsz_status_string(fsz_status_t s) {
    switch (s) {
        case FSZ_STATUS_OK:                  return "OK";
        case FSZ_STATUS_INVALID_ARGUMENT:    return "invalid argument";
        case FSZ_STATUS_OUT_OF_MEMORY:       return "out of memory";
        case FSZ_STATUS_WORKSPACE_TOO_SMALL: return "workspace too small";
        case FSZ_STATUS_CUDA_ERROR:          return "CUDA error";
        case FSZ_STATUS_TOO_LARGE:           return "compressed stream too large";
    }
    return "unknown";
}

extern "C" size_t fsz_num_tiles(size_t n) {
    return (n + FSZ_TILE_SIZE - 1) / FSZ_TILE_SIZE;
}

extern "C" size_t fsz_max_compressed_bytes(size_t n) {
    return fsz_num_tiles(n) * FSZ_MAX_TILE_BYTES;
}

extern "C" fsz_compress_result_t fsz_make_result(size_t n, size_t cmp_size) {
    size_t numTiles = fsz_num_tiles(n);
    fsz_compress_result_t r;
    r.cmp_size    = cmp_size;
    r.num_tiles   = numTiles;
    r.data_offset = numTiles * FSZ_BLKS_PER_TILE;
    return r;
}

extern "C" fsz_status_t fsz_workspace_create(fsz_workspace_t** out_ws, size_t max_n) {
    if (!out_ws) return FSZ_STATUS_INVALID_ARGUMENT;
    *out_ws = nullptr;
    if (max_n == 0) return FSZ_STATUS_INVALID_ARGUMENT;

    fsz_workspace* w = new(std::nothrow) fsz_workspace();
    if (!w) return FSZ_STATUS_OUT_OF_MEMORY;

    size_t numTiles = fsz_num_tiles(max_n);
    w->max_n_elements = max_n;
    w->gsize_max    = fsz_grid_size(numTiles);
    w->gsize_sm_max = fsz_grid_size_sm(numTiles);

    int maxgs = w->gsize_max > w->gsize_sm_max ? w->gsize_max : w->gsize_sm_max;
    w->cmpOffSize = maxgs + 1;

    cudaError_t e = cudaMalloc(&w->d_workspace, sizeof(unsigned int) * w->cmpOffSize * 3);
    if (e != cudaSuccess) { delete w; return FSZ_STATUS_CUDA_ERROR; }

    w->cmpOffset = w->d_workspace;
    w->locOffset = w->d_workspace + w->cmpOffSize;
    w->flag      = (int*)(w->d_workspace + 2 * w->cmpOffSize);

    e = cudaHostAlloc(&w->h_cmpSize, sizeof(unsigned int), cudaHostAllocDefault);
    if (e != cudaSuccess) {
        cudaFree(w->d_workspace);
        delete w;
        return FSZ_STATUS_CUDA_ERROR;
    }

    *out_ws = w;
    return FSZ_STATUS_OK;
}

extern "C" void fsz_workspace_destroy(fsz_workspace_t* ws) {
    if (!ws) return;
    if (ws->d_workspace) cudaFree(ws->d_workspace);
    if (ws->h_cmpSize)   cudaFreeHost(ws->h_cmpSize);
    delete ws;
}

extern "C" size_t fsz_workspace_capacity(const fsz_workspace_t* ws) {
    return ws ? ws->max_n_elements : 0;
}

template <typename FpT>
static fsz_status_t fsz_compress_with_ws(
    const FpT* d_in,
    unsigned char* d_cmp,
    size_t n,
    FpT eb,
    fsz_workspace_t* ws,
    cudaStream_t stream,
    fsz_compress_result_t* out_result)
{
    if (!d_in || !d_cmp || !ws || !out_result) return FSZ_STATUS_INVALID_ARGUMENT;
    if (n == 0)        return FSZ_STATUS_INVALID_ARGUMENT;
    if (!(eb > (FpT)0.0f)) return FSZ_STATUS_INVALID_ARGUMENT;
    if (n > ws->max_n_elements) return FSZ_STATUS_WORKSPACE_TOO_SMALL;

    size_t numTiles  = fsz_num_tiles(n);
    size_t numBlocks = numTiles * FSZ_BLKS_PER_TILE;
    size_t dataOfs   = numBlocks;

    bool use_small = (n <= FSZ_SMALL_THRESHOLD);
    int gs = use_small ? fsz_grid_size_sm(numTiles) : fsz_grid_size(numTiles);

    cudaError_t e = cudaMemsetAsync(ws->flag, 0, sizeof(int) * (gs + 1), stream);
    if (e != cudaSuccess) return FSZ_STATUS_CUDA_ERROR;

    cudaGetLastError();

    if (use_small) {
        fsz_compress_kernel_T<FSZ_TILES_PER_THR_SM, FpT><<<gs, FSZ_TBLOCK_SZ, 8, stream>>>(
            d_in, d_cmp, ws->cmpOffset, ws->locOffset, ws->flag,
            eb, n, numTiles, dataOfs);
    } else {
        fsz_compress_kernel_T<FSZ_TILES_PER_THR, FpT><<<gs, FSZ_TBLOCK_SZ, 8, stream>>>(
            d_in, d_cmp, ws->cmpOffset, ws->locOffset, ws->flag,
            eb, n, numTiles, dataOfs);
    }
    if (cudaGetLastError() != cudaSuccess) return FSZ_STATUS_CUDA_ERROR;

    const unsigned int* d_total = (gs == 1) ? (ws->locOffset + 1)
                                            : (ws->cmpOffset + gs);
    e = cudaMemcpyAsync(ws->h_cmpSize, d_total,
                        sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return FSZ_STATUS_CUDA_ERROR;

    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) return FSZ_STATUS_CUDA_ERROR;

    if ((unsigned long long)numTiles * (FSZ_MAX_TILE_BYTES - FSZ_BLKS_PER_TILE)
            > 0xffffffffull) {
        std::vector<unsigned int> parts((size_t)gs);
        e = cudaMemcpy(parts.data(), ws->locOffset + 1,
                       sizeof(unsigned int) * (size_t)gs, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) return FSZ_STATUS_CUDA_ERROR;
        unsigned long long total = 0;
        for (int i = 0; i < gs; ++i) total += parts[(size_t)i];
        if (total > 0xffffffffull) return FSZ_STATUS_TOO_LARGE;
    }

    out_result->cmp_size    = dataOfs + *ws->h_cmpSize;
    out_result->num_tiles   = numTiles;
    out_result->data_offset = dataOfs;
    return FSZ_STATUS_OK;
}

template <typename FpT>
static fsz_status_t fsz_compress_core(
    const FpT* d_in,
    unsigned char* d_cmp,
    size_t n,
    FpT eb,
    fsz_workspace_t* ws,
    cudaStream_t stream,
    fsz_compress_result_t* out_result)
{
    if (ws) return fsz_compress_with_ws(d_in, d_cmp, n, eb, ws, stream, out_result);
    if (n == 0) return FSZ_STATUS_INVALID_ARGUMENT;

    fsz_workspace_t* tmp = nullptr;
    fsz_status_t s = fsz_workspace_create(&tmp, n);
    if (s != FSZ_STATUS_OK) return s;
    s = fsz_compress_with_ws(d_in, d_cmp, n, eb, tmp, stream, out_result);
    fsz_workspace_destroy(tmp);
    return s;
}

extern "C" fsz_status_t fsz_compress(
    const float* d_in, unsigned char* d_cmp, size_t n, float eb,
    fsz_workspace_t* ws, cudaStream_t stream, fsz_compress_result_t* out_result)
{
    return fsz_compress_core<float>(d_in, d_cmp, n, eb, ws, stream, out_result);
}

extern "C" fsz_status_t fsz_compress_f64(
    const double* d_in, unsigned char* d_cmp, size_t n, double eb,
    fsz_workspace_t* ws, cudaStream_t stream, fsz_compress_result_t* out_result)
{
    return fsz_compress_core<double>(d_in, d_cmp, n, eb, ws, stream, out_result);
}

template <typename FpT>
static fsz_status_t fsz_decompress_with_ws(
    FpT* d_out,
    const unsigned char* d_cmp,
    size_t n,
    FpT eb,
    const fsz_compress_result_t* result,
    fsz_workspace_t* ws,
    cudaStream_t stream)
{
    if (!d_out || !d_cmp || !ws || !result) return FSZ_STATUS_INVALID_ARGUMENT;
    if (n == 0)        return FSZ_STATUS_INVALID_ARGUMENT;
    if (!(eb > (FpT)0.0f)) return FSZ_STATUS_INVALID_ARGUMENT;
    if (n > ws->max_n_elements) return FSZ_STATUS_WORKSPACE_TOO_SMALL;

    const size_t expect_tiles = fsz_num_tiles(n);
    if (result->num_tiles != expect_tiles) return FSZ_STATUS_INVALID_ARGUMENT;
    if (result->data_offset != expect_tiles * FSZ_BLKS_PER_TILE)
        return FSZ_STATUS_INVALID_ARGUMENT;

    bool use_small = (n <= FSZ_SMALL_THRESHOLD);
    int gs = use_small ? fsz_grid_size_sm(result->num_tiles)
                       : fsz_grid_size(result->num_tiles);

    cudaError_t e = cudaMemsetAsync(ws->flag, 0, sizeof(int) * (gs + 1), stream);
    if (e != cudaSuccess) return FSZ_STATUS_CUDA_ERROR;

    size_t origSize = n * sizeof(FpT);
    bool high_cr = (result->cmp_size > 0) && (origSize / result->cmp_size > 100);
    if (high_cr) {
        e = cudaMemsetAsync(d_out, 0, origSize, stream);
        if (e != cudaSuccess) return FSZ_STATUS_CUDA_ERROR;
    }

    cudaGetLastError();

    if (use_small) {
        fsz_decompress_kernel_T<FSZ_TILES_PER_THR_SM, FpT><<<gs, FSZ_TBLOCK_SZ, 8, stream>>>(
            d_out, d_cmp, ws->cmpOffset, ws->locOffset, ws->flag,
            eb, n, result->num_tiles, result->data_offset, high_cr);
    } else {
        fsz_decompress_kernel_T<FSZ_TILES_PER_THR, FpT><<<gs, FSZ_TBLOCK_SZ, 8, stream>>>(
            d_out, d_cmp, ws->cmpOffset, ws->locOffset, ws->flag,
            eb, n, result->num_tiles, result->data_offset, high_cr);
    }
    if (cudaGetLastError() != cudaSuccess) return FSZ_STATUS_CUDA_ERROR;
    return FSZ_STATUS_OK;
}

template <typename FpT>
static fsz_status_t fsz_decompress_core(
    FpT* d_out,
    const unsigned char* d_cmp,
    size_t n,
    FpT eb,
    const fsz_compress_result_t* result,
    fsz_workspace_t* ws,
    cudaStream_t stream)
{
    if (ws) return fsz_decompress_with_ws(d_out, d_cmp, n, eb, result, ws, stream);
    if (n == 0) return FSZ_STATUS_INVALID_ARGUMENT;

    fsz_workspace_t* tmp = nullptr;
    fsz_status_t s = fsz_workspace_create(&tmp, n);
    if (s != FSZ_STATUS_OK) return s;
    s = fsz_decompress_with_ws(d_out, d_cmp, n, eb, result, tmp, stream);
    cudaError_t e = cudaStreamSynchronize(stream);
    fsz_workspace_destroy(tmp);
    if (s == FSZ_STATUS_OK && e != cudaSuccess) s = FSZ_STATUS_CUDA_ERROR;
    return s;
}

extern "C" fsz_status_t fsz_decompress(
    float* d_out, const unsigned char* d_cmp, size_t n, float eb,
    const fsz_compress_result_t* result, fsz_workspace_t* ws, cudaStream_t stream)
{
    return fsz_decompress_core<float>(d_out, d_cmp, n, eb, result, ws, stream);
}

extern "C" fsz_status_t fsz_decompress_f64(
    double* d_out, const unsigned char* d_cmp, size_t n, double eb,
    const fsz_compress_result_t* result, fsz_workspace_t* ws, cudaStream_t stream)
{
    return fsz_decompress_core<double>(d_out, d_cmp, n, eb, result, ws, stream);
}

template <typename FpT>
static fsz_status_t fsz_compress_hostptr_core(
    const FpT* in,
    unsigned char* cmp,
    size_t n,
    FpT eb,
    size_t* out_cmp_size)
{
    if (!in || !cmp || !out_cmp_size) return FSZ_STATUS_INVALID_ARGUMENT;
    if (n == 0)        return FSZ_STATUS_INVALID_ARGUMENT;
    if (!(eb > (FpT)0.0f)) return FSZ_STATUS_INVALID_ARGUMENT;

    *out_cmp_size = 0;

    FpT* d_in = nullptr;
    unsigned char* d_cmp = nullptr;
    fsz_workspace_t* ws = nullptr;

    cudaError_t e = cudaMalloc(&d_in, n * sizeof(FpT));
    if (e != cudaSuccess) return FSZ_STATUS_CUDA_ERROR;

    e = cudaMalloc(&d_cmp, fsz_max_compressed_bytes(n));
    if (e != cudaSuccess) {
        fsz_hostptr_release(d_in, nullptr, nullptr);
        return FSZ_STATUS_CUDA_ERROR;
    }

    fsz_status_t s = fsz_workspace_create(&ws, n);
    if (s != FSZ_STATUS_OK) {
        fsz_hostptr_release(d_in, d_cmp, nullptr);
        return s;
    }

    e = cudaMemcpy(d_in, in, n * sizeof(FpT), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        fsz_hostptr_release(d_in, d_cmp, ws);
        return FSZ_STATUS_CUDA_ERROR;
    }

    fsz_compress_result_t r;
    s = fsz_compress_core<FpT>(d_in, d_cmp, n, eb, ws, 0, &r);
    if (s != FSZ_STATUS_OK) {
        fsz_hostptr_release(d_in, d_cmp, ws);
        return s;
    }

    e = cudaMemcpy(cmp, d_cmp, r.cmp_size, cudaMemcpyDeviceToHost);
    fsz_hostptr_release(d_in, d_cmp, ws);
    if (e != cudaSuccess) return FSZ_STATUS_CUDA_ERROR;

    *out_cmp_size = r.cmp_size;
    return FSZ_STATUS_OK;
}

extern "C" fsz_status_t fsz_compress_hostptr(
    const float* in, unsigned char* cmp, size_t n, float eb, size_t* out_cmp_size)
{
    return fsz_compress_hostptr_core<float>(in, cmp, n, eb, out_cmp_size);
}

extern "C" fsz_status_t fsz_compress_hostptr_f64(
    const double* in, unsigned char* cmp, size_t n, double eb, size_t* out_cmp_size)
{
    return fsz_compress_hostptr_core<double>(in, cmp, n, eb, out_cmp_size);
}

template <typename FpT>
static fsz_status_t fsz_decompress_hostptr_core(
    FpT* out,
    const unsigned char* cmp,
    size_t cmp_size,
    size_t n,
    FpT eb)
{
    if (!out || !cmp)  return FSZ_STATUS_INVALID_ARGUMENT;
    if (n == 0)        return FSZ_STATUS_INVALID_ARGUMENT;
    if (!(eb > (FpT)0.0f)) return FSZ_STATUS_INVALID_ARGUMENT;

    size_t header = fsz_num_tiles(n) * FSZ_BLKS_PER_TILE;
    if (cmp_size == 0 || cmp_size < header) return FSZ_STATUS_INVALID_ARGUMENT;

    FpT* d_out = nullptr;
    unsigned char* d_cmp = nullptr;
    fsz_workspace_t* ws = nullptr;

    cudaError_t e = cudaMalloc(&d_out, n * sizeof(FpT));
    if (e != cudaSuccess) return FSZ_STATUS_CUDA_ERROR;

    e = cudaMalloc(&d_cmp, cmp_size);
    if (e != cudaSuccess) {
        fsz_hostptr_release(d_out, nullptr, nullptr);
        return FSZ_STATUS_CUDA_ERROR;
    }

    fsz_status_t s = fsz_workspace_create(&ws, n);
    if (s != FSZ_STATUS_OK) {
        fsz_hostptr_release(d_out, d_cmp, nullptr);
        return s;
    }

    e = cudaMemcpy(d_cmp, cmp, cmp_size, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        fsz_hostptr_release(d_out, d_cmp, ws);
        return FSZ_STATUS_CUDA_ERROR;
    }

    fsz_compress_result_t r = fsz_make_result(n, cmp_size);
    s = fsz_decompress_core<FpT>(d_out, d_cmp, n, eb, &r, ws, 0);
    if (s != FSZ_STATUS_OK) {
        fsz_hostptr_release(d_out, d_cmp, ws);
        return s;
    }

    e = cudaStreamSynchronize(0);
    if (e != cudaSuccess) {
        fsz_hostptr_release(d_out, d_cmp, ws);
        return FSZ_STATUS_CUDA_ERROR;
    }

    e = cudaMemcpy(out, d_out, n * sizeof(FpT), cudaMemcpyDeviceToHost);
    fsz_hostptr_release(d_out, d_cmp, ws);
    if (e != cudaSuccess) return FSZ_STATUS_CUDA_ERROR;

    return FSZ_STATUS_OK;
}

extern "C" fsz_status_t fsz_decompress_hostptr(
    float* out, const unsigned char* cmp, size_t cmp_size, size_t n, float eb)
{
    return fsz_decompress_hostptr_core<float>(out, cmp, cmp_size, n, eb);
}

extern "C" fsz_status_t fsz_decompress_hostptr_f64(
    double* out, const unsigned char* cmp, size_t cmp_size, size_t n, double eb)
{
    return fsz_decompress_hostptr_core<double>(out, cmp, cmp_size, n, eb);
}
