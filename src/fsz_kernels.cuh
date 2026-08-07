#ifndef FSZ_KERNELS_CUH
#define FSZ_KERNELS_CUH

#include "fsz_internal.cuh"

namespace fsz_detail {

template <int TILES_PER_THREAD, typename FpT>
__global__ void fsz_compress_kernel_T(
    const FpT* __restrict__ oriData,
    unsigned char* __restrict__ cmpData,
    volatile unsigned int* __restrict__ cmpOffset,
    volatile unsigned int* __restrict__ locOffset,
    volatile int* __restrict__ flag,
    const FpT eb, const size_t nbEle, const size_t numTiles,
    const size_t dataOfs)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * FSZ_TBLOCK_SZ + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    if (!tid) { excl_sum = 0; base_idx = 0; }
    __syncthreads();

    const FpT recipPrec = (FpT)0.5f / eb;
    const int TPT = TILES_PER_THREAD;
    const int BN = TPT * FSZ_BLKS_PER_TILE;
    const int baseTile = (warp * FSZ_WARP_SZ + lane) * TPT;

    int fixed_rate[BN];
    int tile_mean_q[TPT];
    bool tile_use_lz2[TPT];
    bool tile_use_mean[TPT];
    unsigned int thread_ofs = 0;

    for (int t = 0; t < TPT; t++) {
        int tileId = baseTile + t;
        size_t tBase = (size_t)tileId * FSZ_TILE_SIZE;

        if (tileId < (int)numTiles) {
            FpT fsum = (FpT)0.0f;

            int prev1_lz1 = 0;
            int prev1_lz2 = 0, prev2_lz2 = 0;
            int blk0_q0 = 0;
            int blk0_d1_nomean_lz2 = 0;
            int blk0_max_2_31_lz2 = 0;
            int blk0_max_1_31_lz1 = 0;
            int lz1_rate_t[FSZ_BLKS_PER_TILE], lz2_rate_t[FSZ_BLKS_PER_TILE];
            int lz1_pb1_t[FSZ_BLKS_PER_TILE];
            int lz2_pb1_t[FSZ_BLKS_PER_TILE], lz2_pb2_t[FSZ_BLKS_PER_TILE];

            for (int blk = 0; blk < FSZ_BLKS_PER_TILE; blk++) {
                size_t bBase = tBase + blk * FSZ_BLK_DIM;
                lz1_pb1_t[blk] = prev1_lz1;
                lz2_pb1_t[blk] = prev1_lz2;
                lz2_pb2_t[blk] = prev2_lz2;

                int lz1Max = 0, lz2Max = 0;
                for (int k = 0; k < FSZ_BLK_DIM; k += 4) {
                    auto v4 = fsz_load4(oriData, bBase + k, nbEle);
                    fsum += v4.x + v4.y + v4.z + v4.w;
                    FpT fv[4] = {v4.x, v4.y, v4.z, v4.w};
                    for (int kk = 0; kk < 4; kk++) {
                        int q = fsz_quantize(fv[kk], recipPrec);
                        int d1 = q - prev1_lz1;
                        prev1_lz1 = q;
                        int a1 = d1 < 0 ? -d1 : d1;
                        lz1Max = lz1Max > a1 ? lz1Max : a1;
                        int d2 = q - 2 * prev1_lz2 + prev2_lz2;
                        prev2_lz2 = prev1_lz2;
                        prev1_lz2 = q;
                        int a2 = d2 < 0 ? -d2 : d2;
                        lz2Max = lz2Max > a2 ? lz2Max : a2;
                        if (blk == 0) {
                            int elem = k + kk;
                            if (elem == 0) blk0_q0 = q;
                            else if (elem == 1) blk0_d1_nomean_lz2 = d2;
                            if (elem >= 2) blk0_max_2_31_lz2 = blk0_max_2_31_lz2 > a2 ? blk0_max_2_31_lz2 : a2;
                            if (elem >= 1) blk0_max_1_31_lz1 = blk0_max_1_31_lz1 > a1 ? blk0_max_1_31_lz1 : a1;
                        }
                    }
                }
                int lz1r = fsz_bitnum((unsigned int)lz1Max);
                if (lz1r > 31) lz1r = 31;
                lz1_rate_t[blk] = lz1r;
                int lz2r = fsz_bitnum((unsigned int)lz2Max);
                if (lz2r > 31) lz2r = 31;
                lz2_rate_t[blk] = lz2r;
            }

            int mean_q = fsz_quantize(fsum / (FpT)FSZ_TILE_SIZE, recipPrec);
            int lz1_d0_mean = blk0_q0 - mean_q;
            int lz1_ad0_mean = lz1_d0_mean < 0 ? -lz1_d0_mean : lz1_d0_mean;
            int lz1_max0_mean = lz1_ad0_mean > blk0_max_1_31_lz1 ? lz1_ad0_mean : blk0_max_1_31_lz1;
            int lz1_rate0_mean = fsz_bitnum((unsigned int)lz1_max0_mean);
            if (lz1_rate0_mean > 31) lz1_rate0_mean = 31;
            int lz1_rate0_nomean = lz1_rate_t[0];

            int lz2_d0_mean = blk0_q0 - mean_q;
            int lz2_ad0_mean = lz2_d0_mean < 0 ? -lz2_d0_mean : lz2_d0_mean;
            int lz2_d1_mean = blk0_d1_nomean_lz2 + mean_q;
            int lz2_ad1_mean = lz2_d1_mean < 0 ? -lz2_d1_mean : lz2_d1_mean;
            int lz2_max0_mean = lz2_ad0_mean;
            lz2_max0_mean = lz2_max0_mean > lz2_ad1_mean ? lz2_max0_mean : lz2_ad1_mean;
            lz2_max0_mean = lz2_max0_mean > blk0_max_2_31_lz2 ? lz2_max0_mean : blk0_max_2_31_lz2;
            int lz2_rate0_mean = fsz_bitnum((unsigned int)lz2_max0_mean);
            if (lz2_rate0_mean > 31) lz2_rate0_mean = 31;
            int lz2_rate0_nomean = lz2_rate_t[0];

            int lz1_blks17 = 0, lz2_blks17 = 0;
            for (int b = 1; b < FSZ_BLKS_PER_TILE; b++) {
                lz1_blks17 += lz1_rate_t[b] ? (4 + lz1_rate_t[b] * 4) : 0;
                lz2_blks17 += lz2_rate_t[b] ? (4 + lz2_rate_t[b] * 4) : 0;
            }

            int lz1_b0_mean_cost = lz1_rate0_mean ? (4 + lz1_rate0_mean * 4) : 0;
            int lz1_b0_nomean_cost = lz1_rate0_nomean ? (4 + lz1_rate0_nomean * 4) : 0;
            int lz2_b0_mean_cost = lz2_rate0_mean ? (4 + lz2_rate0_mean * 4) : 0;
            int lz2_b0_nomean_cost = lz2_rate0_nomean ? (4 + lz2_rate0_nomean * 4) : 0;

            int lz1_cost_mean   = lz1_blks17 + lz1_b0_mean_cost + (int)sizeof(FpT);
            int lz1_cost_nomean = lz1_blks17 + lz1_b0_nomean_cost;
            int lz2_cost_mean   = lz2_blks17 + lz2_b0_mean_cost + (int)sizeof(FpT);
            int lz2_cost_nomean = lz2_blks17 + lz2_b0_nomean_cost;

            bool lz1_use_mean = (lz1_cost_mean < lz1_cost_nomean);
            int lz1_best = lz1_use_mean ? lz1_cost_mean : lz1_cost_nomean;
            bool lz2_use_mean = (lz2_cost_mean < lz2_cost_nomean);
            int lz2_best = lz2_use_mean ? lz2_cost_mean : lz2_cost_nomean;
            bool use_lz2 = (lz2_best < lz1_best);
            bool use_mean = use_lz2 ? lz2_use_mean : lz1_use_mean;

            tile_use_lz2[t] = use_lz2;
            tile_use_mean[t] = use_mean;

            int chosen_rate0 = use_lz2 ? (use_mean ? lz2_rate0_mean : lz2_rate0_nomean)
                                       : (use_mean ? lz1_rate0_mean : lz1_rate0_nomean);
            int rate_byte_0 = chosen_rate0 | (use_lz2 ? FSZ_LZ2_FLAG : 0) | (use_mean ? FSZ_MEAN_FLAG : 0);
            int bIdx0 = t * FSZ_BLKS_PER_TILE;
            fixed_rate[bIdx0] = rate_byte_0;
            cmpData[tileId * FSZ_BLKS_PER_TILE] = (unsigned char)rate_byte_0;

            for (int b = 1; b < FSZ_BLKS_PER_TILE; b++) {
                int bIdx = t * FSZ_BLKS_PER_TILE + b;
                int rate = use_lz2 ? lz2_rate_t[b] : lz1_rate_t[b];
                fixed_rate[bIdx] = rate;
                cmpData[tileId * FSZ_BLKS_PER_TILE + b] = (unsigned char)rate;
            }

            tile_mean_q[t] = use_mean ? mean_q : 0;
            int tile_cost = use_lz2 ? lz2_best : lz1_best;
            thread_ofs += tile_cost;
        } else {
            tile_use_lz2[t] = false;
            tile_use_mean[t] = false;
            tile_mean_q[t] = 0;
            for (int blk = 0; blk < FSZ_BLKS_PER_TILE; blk++) {
                int bIdx = t * FSZ_BLKS_PER_TILE + blk;
                fixed_rate[bIdx] = 0;
            }
        }
    }

    #pragma unroll 5
    for (int i = 1; i < 32; i <<= 1) {
        unsigned int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if (lane >= i) thread_ofs += tmp;
    }
    __syncthreads();
    if (lane == 31) {
        locOffset[warp + 1] = thread_ofs; __threadfence();
        if (warp == 0) {
            cmpOffset[0] = 0;
            __threadfence();
            flag[0] = 2; __threadfence();
            flag[1] = 1; __threadfence();
        }
        else { flag[warp + 1] = 1; __threadfence(); }
    }
    __syncthreads();
    if (warp > 0) {
        if (!lane) {
            int lb = warp; unsigned int les = 0;
            while (lb > 0) {
                int st; do { st = flag[lb]; __threadfence(); } while (st == 0);
                if (st == 2) { les += cmpOffset[lb]; __threadfence(); break; }
                if (st == 1) les += locOffset[lb]; lb--; __threadfence();
            }
            excl_sum = les;
        }
        __syncthreads();
    }
    if (warp > 0) {
        if (!lane) {
            cmpOffset[warp] = excl_sum; __threadfence();
            if (warp == gridDim.x - 1) cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1];
            __threadfence(); flag[warp] = 2; __threadfence();
        }
    }
    __syncthreads();
    if (!lane) base_idx = excl_sum;
    __syncthreads();

    unsigned int base_cmp = base_idx + dataOfs;
    unsigned int cur_ofs = 0;
    uchar4 tmp_char;
    int enc_p1 = 0, enc_p2 = 0;
    int prev_tile = -1;

    for (int j = 0; j < BN; j++) {
        int t = j / FSZ_BLKS_PER_TILE;
        int blk = j % FSZ_BLKS_PER_TILE;
        bool use_lz2 = tile_use_lz2[t];
        bool mean_enabled = tile_use_mean[t];
        int mean_q = tile_mean_q[t];

        if (t != prev_tile) {
            enc_p1 = 0; enc_p2 = 0;
            prev_tile = t;
        }

        int rate = fixed_rate[j] & FSZ_RATE_MASK;
        unsigned int tmp_byte = rate ? (4 + rate * 4) : 0;
        if (blk == 0 && mean_enabled) tmp_byte += (unsigned int)sizeof(FpT);

        #pragma unroll 5
        for (int i = 1; i < 32; i <<= 1) {
            unsigned int t2 = __shfl_up_sync(0xffffffff, tmp_byte, i);
            if (lane >= i) tmp_byte += t2;
        }
        unsigned int prev_thr = __shfl_up_sync(0xffffffff, tmp_byte, 1);
        unsigned int cmp_ofs;
        if (!lane) cmp_ofs = base_cmp + cur_ofs;
        else cmp_ofs = base_cmp + cur_ofs + prev_thr;

        if (rate || (blk == 0 && mean_enabled)) {
            if (blk == 0 && mean_enabled) {
                FpT mu = fsz_mean_value(mean_q, eb);
                fsz_write_mean(cmpData, cmp_ofs, mu);
                cmp_ofs += (unsigned int)sizeof(FpT);
            }

            if (rate) {
                int absQ[FSZ_BLK_DIM];
                unsigned int sf = 0;

                size_t bBase = (size_t)(baseTile + t) * FSZ_TILE_SIZE + blk * FSZ_BLK_DIM;
                int p1 = enc_p1;
                int p2 = enc_p2;
                for (int k = 0; k < FSZ_BLK_DIM; k += 4) {
                    auto v4 = fsz_load4(oriData, bBase + k, nbEle);
                    FpT fv[4] = {v4.x, v4.y, v4.z, v4.w};
                    for (int kk = 0; kk < 4; kk++) {
                        int q = fsz_quantize(fv[kk], recipPrec) - mean_q;
                        int delta;
                        if (use_lz2) {
                            delta = q - 2 * p1 + p2;
                            p2 = p1; p1 = q;
                        } else {
                            delta = q - p1;
                            p1 = q;
                        }
                        sf |= (unsigned int)(delta < 0 ? 1 : 0) << (31 - (k + kk));
                        absQ[k + kk] = delta < 0 ? -delta : delta;
                    }
                }
                enc_p1 = p1;
                if (use_lz2) enc_p2 = p2;

                tmp_char.x = 0xff & (sf >> 24);
                tmp_char.y = 0xff & (sf >> 16);
                tmp_char.z = 0xff & (sf >> 8);
                tmp_char.w = 0xff & sf;
                reinterpret_cast<uchar4*>(cmpData)[cmp_ofs >> 2] = tmp_char;
                cmp_ofs += 4;
                int mask = 1;
                for (int b = 0; b < rate; b++) {
                    tmp_char.x = (unsigned char)(
                        (((absQ[0]&mask)>>b)<<7)|(((absQ[1]&mask)>>b)<<6)|
                        (((absQ[2]&mask)>>b)<<5)|(((absQ[3]&mask)>>b)<<4)|
                        (((absQ[4]&mask)>>b)<<3)|(((absQ[5]&mask)>>b)<<2)|
                        (((absQ[6]&mask)>>b)<<1)|(((absQ[7]&mask)>>b)));
                    tmp_char.y = (unsigned char)(
                        (((absQ[8]&mask)>>b)<<7)|(((absQ[9]&mask)>>b)<<6)|
                        (((absQ[10]&mask)>>b)<<5)|(((absQ[11]&mask)>>b)<<4)|
                        (((absQ[12]&mask)>>b)<<3)|(((absQ[13]&mask)>>b)<<2)|
                        (((absQ[14]&mask)>>b)<<1)|(((absQ[15]&mask)>>b)));
                    tmp_char.z = (unsigned char)(
                        (((absQ[16]&mask)>>b)<<7)|(((absQ[17]&mask)>>b)<<6)|
                        (((absQ[18]&mask)>>b)<<5)|(((absQ[19]&mask)>>b)<<4)|
                        (((absQ[20]&mask)>>b)<<3)|(((absQ[21]&mask)>>b)<<2)|
                        (((absQ[22]&mask)>>b)<<1)|(((absQ[23]&mask)>>b)));
                    tmp_char.w = (unsigned char)(
                        (((absQ[24]&mask)>>b)<<7)|(((absQ[25]&mask)>>b)<<6)|
                        (((absQ[26]&mask)>>b)<<5)|(((absQ[27]&mask)>>b)<<4)|
                        (((absQ[28]&mask)>>b)<<3)|(((absQ[29]&mask)>>b)<<2)|
                        (((absQ[30]&mask)>>b)<<1)|(((absQ[31]&mask)>>b)));
                    reinterpret_cast<uchar4*>(cmpData)[cmp_ofs >> 2] = tmp_char;
                    cmp_ofs += 4; mask <<= 1;
                }
            }
        }

        if (rate == 0) {
            if (use_lz2) {
                int stride = enc_p1 - enc_p2;
                int base_val = enc_p1 + stride;
                enc_p2 = base_val + 30 * stride;
                enc_p1 = base_val + 31 * stride;
            }
        }

        cur_ofs += __shfl_sync(0xffffffff, tmp_byte, 31);
    }
}

template <int TILES_PER_THREAD, typename FpT>
__global__ void fsz_decompress_kernel_T(
    FpT* __restrict__ decData,
    const unsigned char* __restrict__ cmpData,
    volatile unsigned int* __restrict__ cmpOffset,
    volatile unsigned int* __restrict__ locOffset,
    volatile int* __restrict__ flag,
    const FpT eb, const size_t nbEle, const size_t numTiles,
    const size_t dataOfs, const bool skipZeroWrites)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * FSZ_TBLOCK_SZ + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    if (!tid) { excl_sum = 0; base_idx = 0; }
    __syncthreads();

    const int TPT = TILES_PER_THREAD;
    const int BN = TPT * FSZ_BLKS_PER_TILE;
    const int baseTile = (warp * FSZ_WARP_SZ + lane) * TPT;

    int fixed_rate[BN];
    unsigned int thread_ofs = 0;
    for (int j = 0; j < BN; j++) {
        int t = j / FSZ_BLKS_PER_TILE, blk = j % FSZ_BLKS_PER_TILE;
        int tileId = baseTile + t;
        if (tileId < (int)numTiles) {
            int rate_byte = cmpData[tileId * FSZ_BLKS_PER_TILE + blk];
            fixed_rate[j] = rate_byte;
            int rate = rate_byte & FSZ_RATE_MASK;
            thread_ofs += rate ? (4 + rate * 4) : 0;
            if (blk == 0 && (rate_byte & FSZ_MEAN_FLAG)) thread_ofs += (unsigned int)sizeof(FpT);
        } else {
            fixed_rate[j] = 0;
        }
    }

    #pragma unroll 5
    for (int i = 1; i < 32; i <<= 1) {
        unsigned int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if (lane >= i) thread_ofs += tmp;
    }
    __syncthreads();
    if (lane == 31) {
        locOffset[warp + 1] = thread_ofs; __threadfence();
        if (warp == 0) {
            cmpOffset[0] = 0;
            __threadfence();
            flag[0] = 2; __threadfence();
            flag[1] = 1; __threadfence();
        }
        else { flag[warp + 1] = 1; __threadfence(); }
    }
    __syncthreads();
    if (warp > 0) {
        if (!lane) {
            int lb = warp; unsigned int les = 0;
            while (lb > 0) {
                int st; do { st = flag[lb]; __threadfence(); } while (st == 0);
                if (st == 2) { les += cmpOffset[lb]; __threadfence(); break; }
                if (st == 1) les += locOffset[lb]; lb--; __threadfence();
            }
            excl_sum = les;
        }
        __syncthreads();
    }
    if (warp > 0) {
        if (!lane) {
            cmpOffset[warp] = excl_sum; __threadfence();
            if (warp == gridDim.x - 1) cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1];
            __threadfence(); flag[warp] = 2; __threadfence();
        }
    }
    __syncthreads();
    if (!lane) base_idx = excl_sum;
    __syncthreads();

    const FpT step = (FpT)2.0f * eb;
    unsigned int base_cmp = base_idx + dataOfs;
    unsigned int cur_ofs = 0;
    int p1 = 0, p2 = 0;
    int cur_tile = -1;
    FpT mu = (FpT)0.0f;
    bool use_lz2 = false;
    bool mean_enabled = false;

    for (int j = 0; j < BN; j++) {
        int t = j / FSZ_BLKS_PER_TILE;
        int blk = j % FSZ_BLKS_PER_TILE;

        if (t != cur_tile) {
            cur_tile = t;
            p1 = 0; p2 = 0;
            mu = (FpT)0.0f;
            int rate_byte_0 = fixed_rate[t * FSZ_BLKS_PER_TILE];
            use_lz2 = (rate_byte_0 & FSZ_LZ2_FLAG) != 0;
            mean_enabled = (rate_byte_0 & FSZ_MEAN_FLAG) != 0;
        }

        int rate_byte = fixed_rate[j];
        int rate = rate_byte & FSZ_RATE_MASK;

        unsigned int tmp_byte = rate ? (4 + rate * 4) : 0;
        if (blk == 0 && mean_enabled) tmp_byte += (unsigned int)sizeof(FpT);

        #pragma unroll 5
        for (int i = 1; i < 32; i <<= 1) {
            unsigned int t2 = __shfl_up_sync(0xffffffff, tmp_byte, i);
            if (lane >= i) tmp_byte += t2;
        }
        unsigned int prev_thr = __shfl_up_sync(0xffffffff, tmp_byte, 1);
        unsigned int cmp_ofs;
        if (!lane) cmp_ofs = base_cmp + cur_ofs;
        else cmp_ofs = base_cmp + cur_ofs + prev_thr;

        if (blk == 0 && mean_enabled) {
            mu = fsz_read_mean<FpT>(cmpData, cmp_ofs);
            cmp_ofs += (unsigned int)sizeof(FpT);
        }

        int tileId = baseTile + t;

        if (rate == 0) {
            if (tileId < (int)numTiles) {
                size_t bBase = (size_t)tileId * FSZ_TILE_SIZE + blk * FSZ_BLK_DIM;
                if (use_lz2) {
                    FpT f_stride = (FpT)(p1 - p2) * step;
                    FpT f_base   = (FpT)(2*p1 - p2) * step + mu;
                    if (!(skipZeroWrites && f_base == (FpT)0.0f && f_stride == (FpT)0.0f)) {
                        size_t gi = bBase;
                        for (int k = 0; k < FSZ_BLK_DIM; k += 4) {
                            if (gi + 3 < nbEle) {
                                fsz_store4(decData, gi,
                                           f_base + (FpT)(k)   * f_stride,
                                           f_base + (FpT)(k+1) * f_stride,
                                           f_base + (FpT)(k+2) * f_stride,
                                           f_base + (FpT)(k+3) * f_stride);
                            } else {
                                for (int kk = 0; kk < 4 && gi+kk < nbEle; kk++)
                                    decData[gi+kk] = f_base + (FpT)(k+kk) * f_stride;
                            }
                            gi += 4;
                        }
                    }
                    int stride = p1 - p2;
                    int base_val = p1 + stride;
                    p2 = base_val + 30 * stride;
                    p1 = base_val + 31 * stride;
                } else {
                    FpT fill_val = (FpT)p1 * step + mu;
                    if (!(skipZeroWrites && fill_val == (FpT)0.0f)) {
                        size_t gi = bBase;
                        for (int k = 0; k < FSZ_BLK_DIM; k += 4) {
                            if (gi + 3 < nbEle) {
                                fsz_store4(decData, gi,
                                           fill_val, fill_val, fill_val, fill_val);
                            } else {
                                for (int kk = 0; kk < 4 && gi+kk < nbEle; kk++)
                                    decData[gi+kk] = fill_val;
                            }
                            gi += 4;
                        }
                    }
                }
            }
        } else {
            uchar4 sc = fsz_ldg_uchar4(cmpData, cmp_ofs);
            unsigned int signs = ((unsigned int)sc.x << 24) | ((unsigned int)sc.y << 16) |
                                 ((unsigned int)sc.z << 8) | (unsigned int)sc.w;
            unsigned int cd = cmp_ofs + 4;

            int mag[FSZ_BLK_DIM];
            #pragma unroll
            for (int k = 0; k < FSZ_BLK_DIM; k++) mag[k] = 0;

            for (int b = 0; b < rate; b++) {
                uchar4 bc = fsz_ldg_uchar4(cmpData, cd);
                cd += 4;
                #pragma unroll
                for (int k = 0; k < 8; k++) mag[k]    |= ((int)((bc.x >> (7-k)) & 1)) << b;
                #pragma unroll
                for (int k = 0; k < 8; k++) mag[8+k]  |= ((int)((bc.y >> (7-k)) & 1)) << b;
                #pragma unroll
                for (int k = 0; k < 8; k++) mag[16+k] |= ((int)((bc.z >> (7-k)) & 1)) << b;
                #pragma unroll
                for (int k = 0; k < 8; k++) mag[24+k] |= ((int)((bc.w >> (7-k)) & 1)) << b;
            }

            int vals[FSZ_BLK_DIM];
            for (int k = 0; k < FSZ_BLK_DIM; k++)
                vals[k] = (signs & (1u << (31-k))) ? -mag[k] : mag[k];

            if (use_lz2) {
                for (int k = 0; k < FSZ_BLK_DIM; k++) {
                    int q = 2*p1 - p2 + vals[k]; p2 = p1; p1 = q; vals[k] = q;
                }
            } else {
                for (int k = 0; k < FSZ_BLK_DIM; k++) {
                    int q = p1 + vals[k]; p1 = q; vals[k] = q;
                }
            }
            if (tileId < (int)numTiles) {
                size_t bBase = (size_t)tileId * FSZ_TILE_SIZE + blk * FSZ_BLK_DIM;
                for (int k = 0; k < FSZ_BLK_DIM; k += 4) {
                    size_t gi = bBase + k;
                    if (gi + 3 < nbEle) {
                        fsz_store4(decData, gi,
                                   (FpT)vals[k]   * step + mu,
                                   (FpT)vals[k+1] * step + mu,
                                   (FpT)vals[k+2] * step + mu,
                                   (FpT)vals[k+3] * step + mu);
                    } else {
                        for (int kk = 0; kk < 4 && gi+kk < nbEle; kk++)
                            decData[gi+kk] = (FpT)vals[k+kk] * step + mu;
                    }
                }
            }
        }

        cur_ofs += __shfl_sync(0xffffffff, tmp_byte, 31);
    }
}

}

#endif
