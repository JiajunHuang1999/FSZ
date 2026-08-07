#ifndef FSZ_INTERNAL_CUH
#define FSZ_INTERNAL_CUH

#include <cuda_runtime.h>
#include <cstring>

namespace fsz_detail {

#define FSZ_BLK_DIM         32
#define FSZ_TILE_SIZE       256
#define FSZ_BLKS_PER_TILE   8
#define FSZ_TILES_PER_THR   4
#define FSZ_TILES_PER_THR_SM 1
#define FSZ_THREAD_CHUNK    1024
#define FSZ_WARP_SZ         32
#define FSZ_TBLOCK_SZ       32
#define FSZ_RATE_MASK       0x1F
#define FSZ_LZ2_FLAG        0x80
#define FSZ_MEAN_FLAG       0x40

#define FSZ_MAX_TILE_BYTES  (FSZ_BLKS_PER_TILE                              \
                             + FSZ_BLKS_PER_TILE * (4 + FSZ_RATE_MASK * 4)  \
                             + 8)

#define FSZ_SMALL_THRESHOLD 20000000

__device__ __forceinline__ int fsz_quantize(float data, float recipPrec) {
    int result;
    asm("{\n\t"
        ".reg .f32 dr, t;\n\t"
        ".reg .s32 s;\n\t"
        ".reg .pred p;\n\t"
        "mul.f32 dr, %1, %2;\n\t"
        "setp.ge.f32 p, dr, -0.5;\n\t"
        "selp.s32 s, 0, 1, p;\n\t"
        "add.f32 t, dr, 0.5;\n\t"
        "cvt.rzi.s32.f32 %0, t;\n\t"
        "sub.s32 %0, %0, s;\n\t"
        "}": "=r"(result) : "f"(data), "f"(recipPrec));
    return result;
}

__device__ __forceinline__ int fsz_bitnum(unsigned int x) {
    int lz; asm("clz.b32 %0, %1;" : "=r"(lz) : "r"(x)); return 32 - lz;
}

__device__ __forceinline__ float4 fsz_load4(const float* __restrict__ data,
                                            size_t gi, size_t nbEle) {
    float4 v;
    if (gi + 3 < nbEle) {
        v = reinterpret_cast<const float4*>(data)[gi >> 2];
    } else {
        v.x = (gi     < nbEle) ? data[gi]     : 0.0f;
        v.y = (gi + 1 < nbEle) ? data[gi + 1] : 0.0f;
        v.z = (gi + 2 < nbEle) ? data[gi + 2] : 0.0f;
        v.w = (gi + 3 < nbEle) ? data[gi + 3] : 0.0f;
    }
    return v;
}

__device__ __forceinline__ uchar4 fsz_ldg_uchar4(const unsigned char* __restrict__ cmpData,
                                                 unsigned int byte_ofs) {
    return __ldg(reinterpret_cast<const uchar4*>(cmpData) + (byte_ofs >> 2));
}

__device__ __forceinline__ int fsz_quantize(double data, double recipPrec) {
    return __double2int_rn(data * recipPrec);
}

__device__ __forceinline__ double4 fsz_load4(const double* __restrict__ data,
                                             size_t gi, size_t nbEle) {
    double4 v;
    if (gi + 3 < nbEle) {
        v = reinterpret_cast<const double4*>(data)[gi >> 2];
    } else {
        v.x = (gi     < nbEle) ? data[gi]     : 0.0;
        v.y = (gi + 1 < nbEle) ? data[gi + 1] : 0.0;
        v.z = (gi + 2 < nbEle) ? data[gi + 2] : 0.0;
        v.w = (gi + 3 < nbEle) ? data[gi + 3] : 0.0;
    }
    return v;
}

__device__ __forceinline__ float fsz_mean_value(int mean_q, float eb) {
    return (float)((double)mean_q * (double)(2.0f * eb));
}
__device__ __forceinline__ double fsz_mean_value(int mean_q, double eb) {
    return (double)mean_q * (2.0 * eb);
}

__device__ __forceinline__ void fsz_write_mean(unsigned char* __restrict__ cmpData,
                                               unsigned int ofs, float mu) {
    unsigned int mu_bits;
    memcpy(&mu_bits, &mu, 4);
    uchar4 mc;
    mc.x = mu_bits & 0xff;
    mc.y = (mu_bits >> 8) & 0xff;
    mc.z = (mu_bits >> 16) & 0xff;
    mc.w = (mu_bits >> 24) & 0xff;
    reinterpret_cast<uchar4*>(cmpData)[ofs >> 2] = mc;
}
__device__ __forceinline__ void fsz_write_mean(unsigned char* __restrict__ cmpData,
                                               unsigned int ofs, double mu) {
    unsigned long long b;
    memcpy(&b, &mu, 8);
    uchar4 lo, hi;
    lo.x = (unsigned char)(b & 0xff);
    lo.y = (unsigned char)((b >> 8) & 0xff);
    lo.z = (unsigned char)((b >> 16) & 0xff);
    lo.w = (unsigned char)((b >> 24) & 0xff);
    hi.x = (unsigned char)((b >> 32) & 0xff);
    hi.y = (unsigned char)((b >> 40) & 0xff);
    hi.z = (unsigned char)((b >> 48) & 0xff);
    hi.w = (unsigned char)((b >> 56) & 0xff);
    reinterpret_cast<uchar4*>(cmpData)[ofs >> 2]       = lo;
    reinterpret_cast<uchar4*>(cmpData)[(ofs >> 2) + 1] = hi;
}

template <typename FpT>
__device__ __forceinline__ FpT fsz_read_mean(const unsigned char* __restrict__ cmpData,
                                             unsigned int ofs);
template <>
__device__ __forceinline__ float fsz_read_mean<float>(
    const unsigned char* __restrict__ cmpData, unsigned int ofs) {
    uchar4 mc = fsz_ldg_uchar4(cmpData, ofs);
    unsigned int mu_bits = (unsigned int)mc.x | ((unsigned int)mc.y << 8) |
                           ((unsigned int)mc.z << 16) | ((unsigned int)mc.w << 24);
    float mu;
    memcpy(&mu, &mu_bits, 4);
    return mu;
}
template <>
__device__ __forceinline__ double fsz_read_mean<double>(
    const unsigned char* __restrict__ cmpData, unsigned int ofs) {
    uchar4 lo = fsz_ldg_uchar4(cmpData, ofs);
    uchar4 hi = fsz_ldg_uchar4(cmpData, ofs + 4);
    unsigned long long b =
        (unsigned long long)lo.x         | ((unsigned long long)lo.y << 8)  |
        ((unsigned long long)lo.z << 16) | ((unsigned long long)lo.w << 24) |
        ((unsigned long long)hi.x << 32) | ((unsigned long long)hi.y << 40) |
        ((unsigned long long)hi.z << 48) | ((unsigned long long)hi.w << 56);
    double mu;
    memcpy(&mu, &b, 8);
    return mu;
}

__device__ __forceinline__ void fsz_store4(float* __restrict__ p, size_t gi,
                                           float a, float b, float c, float d) {
    float4 o;
    o.x = a; o.y = b; o.z = c; o.w = d;
    reinterpret_cast<float4*>(p)[gi >> 2] = o;
}
__device__ __forceinline__ void fsz_store4(double* __restrict__ p, size_t gi,
                                           double a, double b, double c, double d) {
    double4 o;
    o.x = a; o.y = b; o.z = c; o.w = d;
    reinterpret_cast<double4*>(p)[gi >> 2] = o;
}

}

#endif
