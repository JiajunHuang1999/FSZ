#ifndef FSZ_H
#define FSZ_H

#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FSZ_VERSION_MAJOR 1
#define FSZ_VERSION_MINOR 0
#define FSZ_VERSION_PATCH 0
#define FSZ_VERSION_STRING "1.0.0"

typedef enum {
    FSZ_STATUS_OK = 0,
    FSZ_STATUS_INVALID_ARGUMENT = 1,
    FSZ_STATUS_OUT_OF_MEMORY = 2,
    FSZ_STATUS_WORKSPACE_TOO_SMALL = 3,
    FSZ_STATUS_CUDA_ERROR = 4,
    FSZ_STATUS_TOO_LARGE = 5
} fsz_status_t;

typedef struct fsz_workspace fsz_workspace_t;

typedef struct {
    size_t cmp_size;
    size_t num_tiles;
    size_t data_offset;
} fsz_compress_result_t;

const char* fsz_version(void);
const char* fsz_status_string(fsz_status_t status);

size_t fsz_num_tiles(size_t n_elements);
size_t fsz_max_compressed_bytes(size_t n_elements);

fsz_status_t fsz_workspace_create(fsz_workspace_t** out_ws, size_t max_n_elements);
void fsz_workspace_destroy(fsz_workspace_t* ws);
size_t fsz_workspace_capacity(const fsz_workspace_t* ws);

fsz_status_t fsz_compress(
    const float* d_in,
    unsigned char* d_cmp,
    size_t n_elements,
    float abs_error_bound,
    fsz_workspace_t* ws,
    cudaStream_t stream,
    fsz_compress_result_t* out_result);

fsz_status_t fsz_decompress(
    float* d_out,
    const unsigned char* d_cmp,
    size_t n_elements,
    float abs_error_bound,
    const fsz_compress_result_t* result,
    fsz_workspace_t* ws,
    cudaStream_t stream);

fsz_compress_result_t fsz_make_result(size_t n_elements, size_t cmp_size);

fsz_status_t fsz_compress_hostptr(
    const float* in,
    unsigned char* cmp,
    size_t n_elements,
    float abs_error_bound,
    size_t* out_cmp_size);

fsz_status_t fsz_decompress_hostptr(
    float* out,
    const unsigned char* cmp,
    size_t cmp_size,
    size_t n_elements,
    float abs_error_bound);

fsz_status_t fsz_compress_f64(
    const double* d_in,
    unsigned char* d_cmp,
    size_t n_elements,
    double abs_error_bound,
    fsz_workspace_t* ws,
    cudaStream_t stream,
    fsz_compress_result_t* out_result);

fsz_status_t fsz_decompress_f64(
    double* d_out,
    const unsigned char* d_cmp,
    size_t n_elements,
    double abs_error_bound,
    const fsz_compress_result_t* result,
    fsz_workspace_t* ws,
    cudaStream_t stream);

fsz_status_t fsz_compress_hostptr_f64(
    const double* in,
    unsigned char* cmp,
    size_t n_elements,
    double abs_error_bound,
    size_t* out_cmp_size);

fsz_status_t fsz_decompress_hostptr_f64(
    double* out,
    const unsigned char* cmp,
    size_t cmp_size,
    size_t n_elements,
    double abs_error_bound);

#ifdef __cplusplus
}
#endif

#endif
