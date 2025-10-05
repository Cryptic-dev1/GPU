#include "CUDAMath.h"
#include "CUDAHash.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ uint32_t ror32(uint32_t x, int n) {
#if __CUDA_ARCH__ >= 350
    return __funnelshift_r(x, x, n);
#else
    return (x >> n) | (x << (32 - n));
#endif
}

__device__ __forceinline__ uint32_t rol32(uint32_t x, int n) { return ror32(x, 32 - n); }
__device__ __forceinline__ uint32_t bigS0(uint32_t x) { return ror32(x, 2) ^ ror32(x, 13) ^ ror32(x, 22); }
__device__ __forceinline__ uint32_t bigS1(uint32_t x) { return ror32(x, 6) ^ ror32(x, 11) ^ ror32(x, 25); }
__device__ __forceinline__ uint32_t smallS0(uint32_t x) { return ror32(x, 7) ^ ror32(x, 18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t smallS1(uint32_t x) { return ror32(x, 17) ^ ror32(x, 19) ^ (x >> 10); }
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (x & z) | (y & z); }

__device__ __constant__ uint32_t K[64] = {
    0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5,
    0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,
    0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
    0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967,
    0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,
    0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
    0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3,
    0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2
};

__device__ __constant__ uint32_t IV[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

__device__ __forceinline__ void SHA256Initialize(uint32_t s[8]) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) s[i] = IV[i];
}

__device__ void SHA256Transform(uint32_t state[8], const uint32_t W[16]) {
    extern __shared__ uint32_t sha_shared_mem[];
    uint32_t *s_state = sha_shared_mem;
    uint32_t *my_state = s_state + (threadIdx.x % WARP_SIZE) * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) my_state[i] = state[i];
    uint32_t a = my_state[0], b = my_state[1], c = my_state[2], d = my_state[3];
    uint32_t e = my_state[4], f = my_state[5], g = my_state[6], h = my_state[7];
    uint32_t W_exp[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) W_exp[i] = W[i];
    #pragma unroll
    for (int t = 16; t < 64; ++t) {
        W_exp[t] = smallS1(W_exp[t-2]) + W_exp[t-7] + smallS0(W_exp[t-15]) + W_exp[t-16];
    }
    #pragma unroll
    for (int t = 0; t < 64; ++t) {
        uint32_t T1 = h + bigS1(e) + Ch(e, f, g) + K[t] + W_exp[t];
        uint32_t T2 = bigS0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + T1; d = c; c = b; b = a; a = T1 + T2;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) my_state[i] += (i == 0 ? a : i == 1 ? b : i == 2 ? c : i == 3 ? d :
                                                 i == 4 ? e : i == 5 ? f : i == 6 ? g : h);
    #pragma unroll
    for (int i = 0; i < 8; ++i) state[i] = my_state[i];
    __syncthreads();
}

__device__ __forceinline__ uint32_t pack_be4(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    return ((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)c << 8) | d;
}

__device__ void SHA256_33_from_limbs(uint8_t prefix02_03, const uint64_t x_be_limbs[4], uint32_t out_state[8]) {
    uint32_t W[16];
    W[0] = pack_be4(prefix02_03, (uint8_t)(x_be_limbs[3]>>56), (uint8_t)(x_be_limbs[3]>>48), (uint8_t)(x_be_limbs[3]>>40));
    W[1] = pack_be4((uint8_t)(x_be_limbs[3]>>32), (uint8_t)(x_be_limbs[3]>>24), (uint8_t)(x_be_limbs[3]>>16), (uint8_t)(x_be_limbs[3]>>8));
    W[2] = pack_be4((uint8_t)(x_be_limbs[3]), (uint8_t)(x_be_limbs[2]>>56), (uint8_t)(x_be_limbs[2]>>48), (uint8_t)(x_be_limbs[2]>>40));
    W[3] = pack_be4((uint8_t)(x_be_limbs[2]>>32), (uint8_t)(x_be_limbs[2]>>24), (uint8_t)(x_be_limbs[2]>>16), (uint8_t)(x_be_limbs[2]>>8));
    W[4] = pack_be4((uint8_t)(x_be_limbs[2]), (uint8_t)(x_be_limbs[1]>>56), (uint8_t)(x_be_limbs[1]>>48), (uint8_t)(x_be_limbs[1]>>40));
    W[5] = pack_be4((uint8_t)(x_be_limbs[1]>>32), (uint8_t)(x_be_limbs[1]>>24), (uint8_t)(x_be_limbs[1]>>16), (uint8_t)(x_be_limbs[1]>>8));
    W[6] = pack_be4((uint8_t)(x_be_limbs[1]), (uint8_t)(x_be_limbs[0]>>56), (uint8_t)(x_be_limbs[0]>>48), (uint8_t)(x_be_limbs[0]>>40));
    W[7] = pack_be4((uint8_t)(x_be_limbs[0]>>32), (uint8_t)(x_be_limbs[0]>>24), (uint8_t)(x_be_limbs[0]>>16), (uint8_t)(x_be_limbs[0]>>8));
    W[8] = pack_be4((uint8_t)(x_be_limbs[0]), 0x80, 0, 0);
    #pragma unroll
    for (int i = 9; i < 15; ++i) W[i] = 0;
    W[15] = 33 * 8;
    SHA256Initialize(out_state);
    SHA256Transform(out_state, W);
}

__device__ void getSHA256_33bytes(const uint8_t* pubkey33, uint8_t sha[32]) {
    uint32_t W[16], state[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        W[i] = pack_be4(pubkey33[4*i], pubkey33[4*i+1], pubkey33[4*i+2], pubkey33[4*i+3]);
    }
    W[8] = pack_be4(pubkey33[32], 0x80, 0, 0);
    #pragma unroll
    for (int i = 9; i < 15; ++i) W[i] = 0;
    W[15] = 33 * 8;
    SHA256Initialize(state);
    SHA256Transform(state, W);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        sha[4*i] = (uint8_t)(state[i] >> 24);
        sha[4*i+1] = (uint8_t)(state[i] >> 16);
        sha[4*i+2] = (uint8_t)(state[i] >> 8);
        sha[4*i+3] = (uint8_t)(state[i]);
    }
}

__device__ __forceinline__ uint32_t f1(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
__device__ __forceinline__ uint32_t f2(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
__device__ __forceinline__ uint32_t f3(uint32_t x, uint32_t y, uint32_t z) { return (x | ~y) ^ z; }
__device__ __forceinline__ uint32_t f4(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
__device__ __forceinline__ uint32_t f5(uint32_t x, uint32_t y, uint32_t z) { return x ^ (y | ~z); }

__device__ __constant__ uint32_t RIPE_K[5] = {0, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
__device__ __constant__ uint32_t RIPE_KP[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0};
__device__ __constant__ int RIPE_S[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
__device__ __constant__ int RIPE_SP[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,12,8,9,11,7,7,12,7,6,15,13,11,9,7,15,12,8,9,11,7,7,12,7,6,15,13,11,9
};
__device__ __constant__ int RIPE_R[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
__device__ __constant__ int RIPE_RP[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};

__device__ void RIPEMD160Initialize(uint32_t s[5]) {
    s[0] = 0x67452301; s[1] = 0xEFCDAB89; s[2] = 0x98BADCFE; s[3] = 0x10325476; s[4] = 0xC3D2E1F0;
}

__device__ void RIPEMD160Transform(uint32_t state[5], const uint32_t W[16]) {
    extern __shared__ uint32_t ripe_shared_mem[];
    uint32_t *s_ripe_state = ripe_shared_mem;
    uint32_t *my_state = s_ripe_state + (threadIdx.x % WARP_SIZE) * 5;
    #pragma unroll
    for (int i = 0; i < 5; ++i) my_state[i] = state[i];
    uint32_t a = my_state[0], b = my_state[1], c = my_state[2], d = my_state[3], e = my_state[4];
    uint32_t ap = a, bp = b, cp = c, dp = d, ep = e;
    #pragma unroll
    for (int j = 0; j < 80; ++j) {
        uint32_t f, fp, k, kp;
        int r = RIPE_R[j], rp = RIPE_RP[j], s = RIPE_S[j], sp = RIPE_SP[j];
        if (j < 16) { f = f1(b, c, d); fp = f5(bp, cp, dp); k = RIPE_K[0]; kp = RIPE_KP[0]; }
        else if (j < 32) { f = f2(b, c, d); fp = f4(bp, cp, dp); k = RIPE_K[1]; kp = RIPE_KP[1]; }
        else if (j < 48) { f = f3(b, c, d); fp = f3(bp, cp, dp); k = RIPE_K[2]; kp = RIPE_KP[2]; }
        else if (j < 64) { f = f4(b, c, d); fp = f2(bp, cp, dp); k = RIPE_K[3]; kp = RIPE_KP[3]; }
        else { f = f5(b, c, d); fp = f1(bp, cp, dp); k = RIPE_K[4]; kp = RIPE_KP[4]; }
        uint32_t t = rol32(a + f + W[r] + k, s) + e;
        a = e; e = d; d = rol32(c, 10); c = b; b = t;
        t = rol32(ap + fp + W[rp] + kp, sp) + ep;
        ap = ep; ep = dp; dp = rol32(cp, 10); cp = bp; bp = t;
    }
    uint32_t t = my_state[1] + c + dp;
    my_state[1] = my_state[2] + d + ep;
    my_state[2] = my_state[3] + e + ap;
    my_state[3] = my_state[0] + a + bp;
    my_state[0] = my_state[4] + b + cp;
    my_state[4] = t;
    #pragma unroll
    for (int i = 0; i < 5; ++i) state[i] = my_state[i];
    __syncthreads();
}

__device__ void RIPEMD160_from_SHA256_state(const uint32_t sha_state_be[8], uint8_t ripemd20[20]) {
    uint32_t W[16], s[5];
    #pragma unroll
    for (int i = 0; i < 8; ++i) W[i] = bswap32(sha_state_be[i]);
    W[8] = 0x00000080; W[9] = W[10] = W[11] = W[12] = W[13] = 0; W[14] = 256; W[15] = 0;
    RIPEMD160Initialize(s);
    RIPEMD160Transform(s, W);
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        ripemd20[4*i+0] = (uint8_t)(s[i] >> 0);
        ripemd20[4*i+1] = (uint8_t)(s[i] >> 8);
        ripemd20[4*i+2] = (uint8_t)(s[i] >> 16);
        ripemd20[4*i+3] = (uint8_t)(s[i] >> 24);
    }
}

__device__ void getHash160_33_from_limbs(uint8_t prefix02_03, const uint64_t x_be_limbs[4], uint8_t out20[20]) {
    uint32_t sha_state[8];
    SHA256_33_from_limbs(prefix02_03, x_be_limbs, sha_state);
    RIPEMD160_from_SHA256_state(sha_state, out20);
}

__device__ void getHash160_33bytes(const uint8_t* pubkey33, uint8_t hash20[20]) {
    uint8_t sha[32];
    getSHA256_33bytes(pubkey33, sha);
    uint32_t W[16], s[5];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        W[i] = ((uint32_t)sha[4*i] << 24) | ((uint32_t)sha[4*i+1] << 16) |
               ((uint32_t)sha[4*i+2] << 8) | sha[4*i+3];
    }
    W[8] = 0x00000080; W[9] = W[10] = W[11] = W[12] = W[13] = 0; W[14] = 256; W[15] = 0;
    RIPEMD160Initialize(s);
    RIPEMD160Transform(s, W);
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        hash20[4*i+0] = (uint8_t)(s[i] >> 0);
        hash20[4*i+1] = (uint8_t)(s[i] >> 8);
        hash20[4*i+2] = (uint8_t)(s[i] >> 16);
        hash20[4*i+3] = (uint8_t)(s[i] >> 24);
    }
}

__device__ void addBigEndian32(uint8_t* data32, uint64_t offset) {
    uint64_t current = ((uint64_t)data32[28] << 24) | ((uint64_t)data32[29] << 16) |
                       ((uint64_t)data32[30] << 8) | data32[31];
    current += offset;
    data32[28] = (uint8_t)(current >> 24);
    data32[29] = (uint8_t)(current >> 16);
    data32[30] = (uint8_t)(current >> 8);
    data32[31] = (uint8_t)current;
}

__device__ void batch_getHash160_33bytes(const uint8_t* pubkeys, uint8_t* hashes, int n) {
    extern __shared__ uint32_t batch_shared_mem[];
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        const uint8_t* pubkey = pubkeys + idx * 33;
        uint8_t* hash = hashes + idx * 20;
        uint32_t W[16], state[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            W[i] = pack_be4(pubkey[4*i], pubkey[4*i+1], pubkey[4*i+2], pubkey[4*i+3]);
        }
        W[8] = pack_be4(pubkey[32], 0x80, 0, 0);
        #pragma unroll
        for (int i = 9; i < 15; ++i) W[i] = 0;
        W[15] = 33 * 8;
        SHA256Initialize(state);
        SHA256Transform(state, W);
        uint8_t sha[32];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sha[4*i] = (uint8_t)(state[i] >> 24);
            sha[4*i+1] = (uint8_t)(state[i] >> 16);
            sha[4*i+2] = (uint8_t)(state[i] >> 8);
            sha[4*i+3] = (uint8_t)(state[i]);
        }
        uint32_t W_ripe[16];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            W_ripe[i] = ((uint32_t)sha[4*i] << 24) | ((uint32_t)sha[4*i+1] << 16) |
                        ((uint32_t)sha[4*i+2] << 8) | sha[4*i+3];
        }
        W_ripe[8] = 0x00000080; W_ripe[9] = W_ripe[10] = W_ripe[11] = W_ripe[12] = W_ripe[13] = 0;
        W_ripe[14] = 256; W_ripe[15] = 0;
        uint32_t s[5];
        RIPEMD160Initialize(s);
        RIPEMD160Transform(s, W_ripe);
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            hash[4*i+0] = (uint8_t)(s[i] >> 0);
            hash[4*i+1] = (uint8_t)(s[i] >> 8);
            hash[4*i+2] = (uint8_t)(s[i] >> 16);
            hash[4*i+3] = (uint8_t)(s[i] >> 24);
        }
    }
    __syncthreads();
}