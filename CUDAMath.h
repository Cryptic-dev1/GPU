```cpp
#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CUDAStructures.h"

#define NBBLOCK 5
#define BIFULLSIZE 40
#define WARP_SIZE 32

// Verify unsigned long long size
static_assert(sizeof(unsigned long long) == 8, "unsigned long long must be 64 bits");

// PTX Assembly Macros
#define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory")
#define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory")
#define UADD(c, a, b) asm volatile ("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b))
#define UADDO1(c, a) asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory")
#define UADDC1(c, a) asm volatile ("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory")
#define UADD1(c, a) asm volatile ("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a))
#define USUBO(c, a, b) asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory")
#define USUBC(c, a, b) asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory")
#define USUB(c, a, b) asm volatile ("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b))
#define USUBO1(c, a) asm volatile ("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory")
#define USUBC1(c, a) asm volatile ("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory")
#define USUB1(c, a) asm volatile ("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a))
#define UMULLO(lo, a, b) asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b))
#define UMULHI(hi, a, b) asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b))
#define MADDO(r, a, b, c) asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory")
#define MADDC(r, a, b, c) asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory")
#define MADD(r, a, b, c) asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c))
#define MADDS(r, a, b, c) asm volatile ("madc.hi.s64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c))

__device__ __constant__ unsigned long long MM64 = 0xD838091DD2253531ULL;
__device__ __constant__ unsigned long long MSK62 = 0x3FFFFFFFFFFFFFFFULL;

// Host-side copy of c_p
static const unsigned long long host_c_p[4] = {0xfffffc2fULL, 0xffffffffULL, 0xffffffffULL, 0xffffffffULL};

// Utility function for zero check
__host__ __device__ __forceinline__ bool isZero256(const unsigned long long a[4]) {
    return (a[3] | a[2] | a[1] | a[0]) == 0ULL;
}

#define _IsPositive(x) (((long long)(x[3])) >= 0LL)
#define _IsNegative(x) (((long long)(x[3])) < 0LL)
#define _IsEqual(a, b) ((a[3] == b[3]) && (a[2] == b[2]) && (a[1] == b[1]) && (a[0] == b[0]))
#define _IsOne(a) ((a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) && (a[0] == 1ULL))

#define IDX threadIdx.x

#define bswap32(v) __byte_perm(v, 0, 0x0123)

#define __sright128(a, b, n) ((a) >> (n)) | ((b) << (64 - (n)))
#define __sleft128(a, b, n) ((b) << (n)) | ((a) >> (64 - (n)))

// Field Utility Functions
__host__ __device__ void fieldSetZero(unsigned long long a[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = 0ULL;
}

__host__ __device__ void fieldSetOne(unsigned long long a[4]) {
    a[0] = 1ULL;
    #pragma unroll
    for (int i = 1; i < 4; ++i) a[i] = 0ULL;
}

__host__ __device__ void fieldCopy(const unsigned long long a[4], unsigned long long b[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = a[i];
}

__device__ void lsl256(unsigned long long a[4], unsigned long long out[4], int n) {
    if (n >= 256) {
        fieldSetZero(out);
        return;
    }
    int limb_shift = n / 64;
    int bit_shift = n % 64;
    fieldSetZero(out);
    for (int i = 0; i < 4 - limb_shift; ++i) {
        out[i + limb_shift] = a[i] << bit_shift;
        if (bit_shift && i + limb_shift + 1 < 4) {
            out[i + limb_shift + 1] |= a[i] >> (64 - bit_shift);
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("lsl256: n=%d, out=%llx:%llx:%llx:%llx\n", n, out[0], out[1], out[2], out[3]);
    }
}

__device__ void lsr256(unsigned long long a[4], unsigned long long out[4], int n) {
    if (n >= 256) {
        fieldSetZero(out);
        return;
    }
    int limb_shift = n / 64;
    int bit_shift = n % 64;
    fieldSetZero(out);
    for (int i = limb_shift; i < 4; ++i) {
        out[i - limb_shift] = a[i] >> bit_shift;
        if (bit_shift && i - limb_shift - 1 >= 0) {
            out[i - limb_shift - 1] |= a[i] << (64 - bit_shift);
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("lsr256: n=%d, out=%llx:%llx:%llx:%llx\n", n, out[0], out[1], out[2], out[3]);
    }
}

__device__ void lsl512(const unsigned long long a[4], int n, unsigned long long out[8]) {
    fieldSetZero(out);
    int limb = n / 64;
    int bit_shift = n % 64;
    for (int i = 0; i < 4; ++i) {
        if (i + limb < 8) {
            out[i + limb] = a[i] << bit_shift;
            if (bit_shift && i + limb + 1 < 8) {
                out[i + limb + 1] |= a[i] >> (64 - bit_shift);
            }
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("lsl512: n=%d, out=%llx:%llx:%llx:%llx:%llx:%llx:%llx:%llx\n",
               n, out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    }
}

__device__ bool ge512(const unsigned long long a[8], const unsigned long long b[8]) {
    for (int i = 7; i >= 0; --i) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

__device__ void sub512(const unsigned long long a[8], const unsigned long long b[8], unsigned long long out[8]) {
    unsigned long long borrow = 0, temp;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        USUBO(temp, a[i], b[i]);
        USUB1(temp, borrow);
        out[i] = temp;
        borrow = (temp > a[i] || (temp == a[i] && b[i] != 0)) ? 1 : 0;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("sub512: out=%llx:%llx:%llx:%llx:%llx:%llx:%llx:%llx\n",
               out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    }
}

// Host versions of field operations
__host__ void fieldAdd_opt_host(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]) {
    unsigned long long carry = 0, temp;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        __uint128_t sum = (__uint128_t)a[i] + b[i] + carry;
        temp = (unsigned long long)sum;
        carry = (unsigned long long)(sum >> 64);
        out[i] = temp;
        carry = (temp < a[i] || (temp == a[i] && b[i] != 0)) ? 1 : 0;
    }
    if (carry || ge256(out, host_c_p)) {
        unsigned long long borrow = 0;
        for (int i = 0; i < 4; ++i) {
            __uint128_t diff = (__uint128_t)out[i] - host_c_p[i] - borrow;
            out[i] = (unsigned long long)diff;
            borrow = (diff >> 64) ? 1 : 0;
        }
    }
}

__host__ void fieldSub_opt_host(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]) {
    unsigned long long borrow = 0, temp;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        __uint128_t diff = (__uint128_t)a[i] - b[i] - borrow;
        temp = (unsigned long long)diff;
        borrow = (diff >> 64) ? 1 : 0;
        out[i] = temp;
        borrow = (temp > a[i] || (temp == a[i] && b[i] != 0)) ? 1 : 0;
    }
    if (borrow) {
        unsigned long long carry = 0;
        for (int i = 0; i < 4; ++i) {
            __uint128_t sum = (__uint128_t)out[i] + host_c_p[i] + carry;
            out[i] = (unsigned long long)sum;
            carry = (unsigned long long)(sum >> 64);
        }
    }
}

__host__ void mul256_host(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[8]) {
    unsigned long long temp_out[8] = {0};
    unsigned long long lo, hi, carry;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned long long ai = a[i];
            unsigned long long bj = b[j];
            __uint128_t prod = (__uint128_t)ai * bj;
            lo = (unsigned long long)prod;
            hi = (unsigned long long)(prod >> 64);
            unsigned long long sum = lo + carry;
            carry = (sum < lo) ? 1ULL : 0ULL;
            unsigned long long out_ij = *(temp_out + i + j);
            *(temp_out + i + j) = out_ij + sum;
            carry += (*(temp_out + i + j) < out_ij) ? 1ULL : 0ULL;
            unsigned long long out_ij1 = *(temp_out + i + j + 1);
            *(temp_out + i + j + 1) = out_ij1 + hi + carry;
            carry = (*(temp_out + i + j + 1) < hi) ? 1ULL : 0ULL;
        }
        if (i + 4 < 8) *(temp_out + i + 4) = carry;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = temp_out[i];
}

__host__ void mul_high_host(const unsigned long long a[4], const unsigned long long b[5], unsigned long long high[5]) {
    unsigned long long temp_prod[9] = {0};
    unsigned long long lo, hi, carry;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        carry = 0;
        #pragma unroll
        for (int j = 0; j < 5; ++j) {
            unsigned long long ai = a[i];
            unsigned long long bj = b[j];
            __uint128_t prod = (__uint128_t)ai * bj;
            lo = (unsigned long long)prod;
            hi = (unsigned long long)(prod >> 64);
            unsigned long long sum = lo + carry;
            carry = (sum < lo) ? 1ULL : 0ULL;
            unsigned long long prod_ij = *(temp_prod + i + j);
            *(temp_prod + i + j) = prod_ij + sum;
            carry += (*(temp_prod + i + j) < prod_ij) ? 1ULL : 0ULL;
            unsigned long long prod_ij1 = *(temp_prod + i + j + 1);
            *(temp_prod + i + j + 1) = prod_ij1 + hi + carry;
            carry = (*(temp_prod + i + j + 1) < hi) ? 1ULL : 0ULL;
        }
        if (i + 5 < 9) *(temp_prod + i + 5) = carry;
    }
    #pragma unroll
    for (int i = 0; i < 5; ++i) *(high + i) = *(temp_prod + i + 4);
}

__host__ void modred_barrett_opt_host(const unsigned long long input[8], unsigned long long out[4]) {
    unsigned long long q[5], tmp[8], r[4];
    mul_high_host(input + 4, c_mu, q);
    mul256_host(q, host_c_p, tmp);
    fieldSub_opt_host(input, tmp, r);
    if (ge256(r, host_c_p)) {
        fieldSub_opt_host(r, host_c_p, r);
    }
    if (_IsNegative(r)) {
        fieldAdd_opt_host(r, host_c_p, r);
    }
    fieldCopy(r, out);
}

__host__ void fieldMul_opt_host(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]) {
    unsigned long long prod[8];
    mul256_host(a, b, prod);
    modred_barrett_opt_host(prod, out);
}

__host__ void fieldSqr_opt_host(const unsigned long long a[4], unsigned long long out[4]) {
    fieldMul_opt_host(a, a, out);
}

__host__ void fieldInvFermat_host(const unsigned long long a[4], unsigned long long inv[4]) {
    if (isZero256(a)) {
        fieldSetZero(inv);
        return;
    }
    unsigned long long t[4], p_minus_2[4] = {0xfffffc2dULL, 0xffffffffULL, 0xffffffffULL, 0xffffffffULL};
    fieldCopy(a, t);
    for (int i = 255; i >= 1; --i) {
        fieldSqr_opt_host(t, t);
        if ((p_minus_2[i/64] >> (i%64)) & 1ULL) {
            fieldMul_opt_host(t, a, t);
        }
    }
    fieldCopy(t, inv);
}

// Device versions of field operations
__device__ void fieldAdd_opt_device(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]) {
    unsigned long long carry = 0, temp;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        UADDO(temp, a[i], b[i]);
        UADD1(temp, carry);
        out[i] = temp;
        carry = (temp < a[i] || (temp == a[i] && b[i] != 0)) ? 1 : 0;
    }
    if (carry || ge256(out, c_p)) {
        USUBO(out[0], out[0], c_p[0]);
        USUBC(out[1], out[1], c_p[1]);
        USUBC(out[2], out[2], c_p[2]);
        USUB(out[3], out[3], c_p[3]);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("fieldAdd_opt_device: a=%llx:%llx:%llx:%llx, b=%llx:%llx:%llx:%llx, out=%llx:%llx:%llx:%llx\n",
               a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], out[0], out[1], out[2], out[3]);
    }
}

__device__ void fieldSub_opt_device(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]) {
    unsigned long long borrow = 0, temp;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        USUBO(temp, a[i], b[i]);
        USUB1(temp, borrow);
        out[i] = temp;
        borrow = (temp > a[i] || (temp == a[i] && b[i] != 0)) ? 1 : 0;
    }
    if (borrow) {
        UADDO(out[0], out[0], c_p[0]);
        UADDC(out[1], out[1], c_p[1]);
        UADDC(out[2], out[2], c_p[2]);
        UADD(out[3], out[3], c_p[3]);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("fieldSub_opt_device: a=%llx:%llx:%llx:%llx, b=%llx:%llx:%llx:%llx, out=%llx:%llx:%llx:%llx\n",
               a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], out[0], out[1], out[2], out[3]);
    }
}

__device__ void mul256_device(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[8]) {
    unsigned long long temp_out[8] = {0};
    unsigned long long lo, hi, carry;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned long long ai = a[i];
            unsigned long long bj = b[j];
            UMULLO(lo, ai, bj);
            UMULHI(hi, ai, bj);
            unsigned long long sum = lo + carry;
            carry = (sum < lo) ? 1ULL : 0ULL;
            unsigned long long out_ij = *(temp_out + i + j);
            *(temp_out + i + j) = out_ij + sum;
            carry += (*(temp_out + i + j) < out_ij) ? 1ULL : 0ULL;
            unsigned long long out_ij1 = *(temp_out + i + j + 1);
            *(temp_out + i + j + 1) = out_ij1 + hi + carry;
            carry = (*(temp_out + i + j + 1) < hi) ? 1ULL : 0ULL;
        }
        if (i + 4 < 8) *(temp_out + i + 4) = carry;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = temp_out[i];
}

__device__ void mul_high_device(const unsigned long long a[4], const unsigned long long b[5], unsigned long long high[5]) {
    unsigned long long temp_prod[9] = {0};
    unsigned long long lo, hi, carry;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        carry = 0;
        #pragma unroll
        for (int j = 0; j < 5; ++j) {
            unsigned long long ai = a[i];
            unsigned long long bj = b[j];
            UMULLO(lo, ai, bj);
            UMULHI(hi, ai, bj);
            unsigned long long sum = lo + carry;
            carry = (sum < lo) ? 1ULL : 0ULL;
            unsigned long long prod_ij = *(temp_prod + i + j);
            *(temp_prod + i + j) = prod_ij + sum;
            carry += (*(temp_prod + i + j) < prod_ij) ? 1ULL : 0ULL;
            unsigned long long prod_ij1 = *(temp_prod + i + j + 1);
            *(temp_prod + i + j + 1) = prod_ij1 + hi + carry;
            carry = (*(temp_prod + i + j + 1) < hi) ? 1ULL : 0ULL;
        }
        if (i + 5 < 9) *(temp_prod + i + 5) = carry;
    }
    #pragma unroll
    for (int i = 0; i < 5; ++i) *(high + i) = *(temp_prod + i + 4);
}

__device__ void modred_barrett_opt_device(const unsigned long long input[8], unsigned long long out[4]) {
    unsigned long long q[5], tmp[8], r[4];
    mul_high_device(input + 4, c_mu, q);
    mul256_device(q, c_p, tmp);
    fieldSub_opt_device(input, tmp, r);
    if (ge256(r, c_p)) {
        fieldSub_opt_device(r, c_p, r);
    }
    if (_IsNegative(r)) {
        fieldAdd_opt_device(r, c_p, r);
    }
    fieldCopy(r, out);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("modred_barrett_opt_device: input=%llx:%llx:%llx:%llx:%llx:%llx:%llx:%llx, out=%llx:%llx:%llx:%llx\n",
               input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
               out[0], out[1], out[2], out[3]);
    }
}

__device__ void fieldMul_opt_device(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]) {
    unsigned long long prod[8];
    mul256_device(a, b, prod);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("fieldMul_opt_device: a=%llx:%llx:%llx:%llx, b=%llx:%llx:%llx:%llx, prod=%llx:%llx:%llx:%llx:%llx:%llx:%llx:%llx\n",
               a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
               prod[0], prod[1], prod[2], prod[3], prod[4], prod[5], prod[6], prod[7]);
    }
    modred_barrett_opt_device(prod, out);
}

__device__ void fieldSqr_opt_device(const unsigned long long a[4], unsigned long long out[4]) {
    fieldMul_opt_device(a, a, out);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("fieldSqr_opt_device: a=%llx:%llx:%llx:%llx, out=%llx:%llx:%llx:%llx\n",
               a[0], a[1], a[2], a[3], out[0], out[1], out[2], out[3]);
    }
}

__device__ void fieldInvFermat_device(const unsigned long long a[4], unsigned long long inv[4]) {
    if (isZero256(a)) {
        fieldSetZero(inv);
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("fieldInvFermat_device: input is zero, inv=0:0:0:0\n");
        }
        return;
    }
    unsigned long long t[4], p_minus_2[4] = {0xfffffc2dULL, 0xffffffffULL, 0xffffffffULL, 0xffffffffULL};
    fieldCopy(a, t);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("fieldInvFermat_device: input=%llx:%llx:%llx:%llx\n", a[0], a[1], a[2], a[3]);
    }
    for (int i = 255; i >= 1; --i) {
        fieldSqr_opt_device(t, t);
        if ((p_minus_2[i/64] >> (i%64)) & 1ULL) {
            fieldMul_opt_device(t, a, t);
        }
    }
    fieldCopy(t, inv);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("fieldInvFermat_device: inv=%llx:%llx:%llx:%llx\n", inv[0], inv[1], inv[2], inv[3]);
    }
}

__host__ __device__ void fieldNeg(const unsigned long long a[4], unsigned long long out[4]) {
    if (isZero256(a)) {
        fieldSetZero(out);
        return;
    }
#ifdef __CUDA_ARCH__
    fieldSub_opt_device(c_p, a, out);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("fieldNeg: a=%llx:%llx:%llx:%llx, out=%llx:%llx:%llx:%llx\n",
               a[0], a[1], a[2], a[3], out[0], out[1], out[2], out[3]);
    }
#else
    fieldSub_opt_host(host_c_p, a, out);
#endif
}

__device__ void batch_modinv_fermat(const unsigned long long* a, unsigned long long* inv, int n) {
    extern __shared__ unsigned long long shared_mem[];
    unsigned long long *prefix = shared_mem;
    unsigned long long prod[4], tmp[4];
    int tid = threadIdx.x % WARP_SIZE;
    if (tid == 0) {
        fieldSetOne(prefix);
        bool all_zero = true;
        for (int i = 0; i < n; ++i) {
            if (i + 1 <= n) { // Bounds check
                if (isZero256(a + i*4)) {
                    fieldSetZero(prefix + (i+1)*4);
                    if (threadIdx.x == 0 && blockIdx.x == 0) {
                        printf("batch_modinv_fermat: a[%d]=0:0:0:0\n", i);
                    }
                } else {
                    fieldMul_opt_device(prefix + i*4, a + i*4, prefix + (i+1)*4);
                    all_zero = false;
                    if (threadIdx.x == 0 && blockIdx.x == 0) {
                        printf("batch_modinv_fermat: a[%d]=%llx:%llx:%llx:%llx, prefix[%d]=%llx:%llx:%llx:%llx\n",
                               i, a[i*4], a[i*4+1], a[i*4+2], a[i*4+3],
                               i+1, prefix[(i+1)*4], prefix[(i+1)*4+1], prefix[(i+1)*4+2], prefix[(i+1)*4+3]);
                    }
                }
            }
        }
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("batch_modinv_fermat: prefix[%d]=%llx:%llx:%llx:%llx, all_zero=%d\n",
                   n, prefix[n*4], prefix[n*4+1], prefix[n*4+2], prefix[n*4+3], all_zero);
        }
        if (all_zero || isZero256(prefix + n*4)) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("batch_modinv_fermat: all inputs zero or prefix[%d] is zero, setting all inv=0\n", n);
            }
            for (int i = 0; i < n; ++i) {
                fieldSetZero(inv + i*4);
            }
            return;
        }
        fieldInvFermat_device(prefix + n*4, prod);
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("batch_modinv_fermat: prod=%llx:%llx:%llx:%llx\n", prod[0], prod[1], prod[2], prod[3]);
        }
        fieldCopy(prod, tmp);
        for (int i = n-1; i >= 0; --i) {
            if (isZero256(a + i*4)) {
                fieldSetZero(inv + i*4);
            } else {
                fieldMul_opt_device(tmp, prefix + i*4, inv + i*4);
                fieldMul_opt_device(tmp, a + i*4, tmp);
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                    printf("batch_modinv_fermat: inv[%d]=%llx:%llx:%llx:%llx, tmp=%llx:%llx:%llx:%llx\n",
                           i, inv[i*4], inv[i*4+1], inv[i*4+2], inv[i*4+3],
                           tmp[0], tmp[1], tmp[2], tmp[3]);
                }
            }
        }
    }
    __syncthreads();
}

__device__ void div512_256(const unsigned long long num[8], const unsigned long long den[4], unsigned long long quot[4], unsigned long long rem[4]) {
    unsigned long long dividend[8], shifted_den[8], q[4] = {0};
    fieldCopy(num, dividend);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("div512_256: num=%llx:%llx:%llx:%llx:%llx:%llx:%llx:%llx\n",
               num[0], num[1], num[2], num[3], num[4], num[5], num[6], num[7]);
    }
    // Check if num < den
    bool num_smaller = true;
    for (int i = 3; i >= 0; --i) {
        if (num[i+4] > 0) { num_smaller = false; break; }
        if (num[i] > den[i]) { num_smaller = false; break; }
        if (num[i] < den[i]) break;
    }
    if (num_smaller) {
        fieldCopy(num, rem);
        fieldSetZero(quot);
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("div512_256: num < den, q=0:0:0:0, rem=%llx:%llx:%llx:%llx\n",
                   rem[0], rem[1], rem[2], rem[3]);
        }
        return;
    }
    int msb_num = -1;
    for (int i = 7; i >= 0; --i) {
        if (num[i] != 0) { msb_num = i * 64 + 63 - __clzll(num[i]); break; }
    }
    int msb_den = -1;
    for (int i = 3; i >= 0; --i) {
        if (den[i] != 0) { msb_den = i * 64 + 63 - __clzll(den[i]); break; }
    }
    if (msb_num < 0 || msb_den < 0) {
        fieldSetZero(quot);
        fieldCopy(num, rem);
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("div512_256: msb_num=%d, msb_den=%d, q=0:0:0:0, rem=%llx:%llx:%llx:%llx\n",
                   msb_num, msb_den, rem[0], rem[1], rem[2], rem[3]);
        }
        return;
    }
    for (int bit = msb_num - msb_den; bit >= 0; --bit) {
        lsl512(den, bit, shifted_den);
        if (ge512(dividend, shifted_den)) {
            sub512(dividend, shifted_den, dividend);
            int limb = bit / 64;
            int bit_pos = bit % 64;
            if (limb < 4) {
                q[limb] |= (1ULL << bit_pos);
            }
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("div512_256: dividend=%llx:%llx:%llx:%llx:%llx:%llx:%llx:%llx, q=%llx:%llx:%llx:%llx\n",
               dividend[0], dividend[1], dividend[2], dividend[3], dividend[4], dividend[5], dividend[6], dividend[7],
               q[0], q[1], q[2], q[3]);
    }
    fieldCopy(q, quot);
    fieldCopy(dividend, rem);
}

// GLV Endomorphism
__device__ void split_glv(const unsigned long long scalar[4], unsigned long long k1[4], unsigned long long k2[4]) {
    if (isZero256(scalar)) {
        fieldSetZero(k1);
        fieldSetZero(k2);
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("split_glv: scalar=0:0:0:0, k1=0:0:0:0, k2=0:0:0:0\n");
        }
        return;
    }
    // Check if scalar < c_n
    bool scalar_small = true;
    for (int i = 3; i >= 0; --i) {
        if (scalar[i] > c_n[i]) { scalar_small = false; break; }
        if (scalar[i] < c_n[i]) break;
    }
    if (scalar_small) {
        fieldCopy(scalar, k1);
        fieldSetZero(k2);
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("split_glv: scalar_small, scalar=%llx:%llx:%llx:%llx, k1=%llx:%llx:%llx:%llx, k2=0:0:0:0\n",
                   scalar[0], scalar[1], scalar[2], scalar[3], k1[0], k1[1], k1[2], k1[3]);
        }
        return;
    }
    unsigned long long num[8], half_n[4], q1[4], q2[4], tmp1[4], tmp2[4], rem[4];
    fieldCopy(c_n, half_n);
    lsr256(half_n, half_n, 1);
    // q1 = round(b2 * scalar / n)
    fieldMul_opt_device(c_b2, scalar, num);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("split_glv: b2*scalar=%llx:%llx:%llx:%llx:%llx:%llx:%llx:%llx\n",
               num[0], num[1], num[2], num[3], num[4], num[5], num[6], num[7]);
    }
    fieldAdd_opt_device(num, half_n, num);
    div512_256(num, c_n, q1, rem);
    // q2 = round(b1 * scalar / n)
    fieldMul_opt_device(c_b1, scalar, num);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("split_glv: b1*scalar=%llx:%llx:%llx:%llx:%llx:%llx:%llx:%llx\n",
               num[0], num[1], num[2], num[3], num[4], num[5], num[6], num[7]);
    }
    fieldAdd_opt_device(num, half_n, num);
    div512_256(num, c_n, q2, rem);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("split_glv: q1=%llx:%llx:%llx:%llx, q2=%llx:%llx:%llx:%llx\n",
               q1[0], q1[1], q1[2], q1[3], q2[0], q2[1], q2[2], q2[3]);
    }
    // k1 = scalar - q1 * a1 - q2 * a2
    fieldMul_opt_device(q1, c_a1, tmp1);
    fieldMul_opt_device(q2, c_a2, tmp2);
    fieldAdd_opt_device(tmp1, tmp2, tmp1);
    fieldSub_opt_device(scalar, tmp1, k1);
    if (_IsNegative(k1)) {
        fieldAdd_opt_device(k1, c_n, k1);
    }
    // k2 = q1 * b1 - q2 * b2
    fieldMul_opt_device(q1, c_b1, tmp1);
    fieldMul_opt_device(q2, c_b2, tmp2);
    fieldSub_opt_device(tmp1, tmp2, k2);
    if (_IsNegative(k2)) {
        fieldAdd_opt_device(k2, c_n, k2);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("split_glv: k1=%llx:%llx:%llx:%llx, k2=%llx:%llx:%llx:%llx\n",
               k1[0], k1[1], k1[2], k1[3], k2[0], k2[1], k2[2], k2[3]);
    }
}

__host__ __device__ void fieldNeg(const unsigned long long a[4], unsigned long long out[4]) {
    if (isZero256(a)) {
        fieldSetZero(out);
        return;
    }
#ifdef __CUDA_ARCH__
    fieldSub_opt_device(c_p, a, out);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("fieldNeg: a=%llx:%llx:%llx:%llx, out=%llx:%llx:%llx:%llx\n",
               a[0], a[1], a[2], a[3], out[0], out[1], out[2], out[3]);
    }
#else
    fieldSub_opt_host(host_c_p, a, out);
#endif
}

__host__ __device__ void pointSetInfinity(JacobianPoint &P) {
    fieldSetZero(P.x);
    fieldSetZero(P.y);
    fieldSetZero(P.z);
    P.infinity = true;
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointSetInfinity: P.x=%llx:%llx:%llx:%llx, P.y=%llx:%llx:%llx:%llx, P.z=%llx:%llx:%llx:%llx, P.infinity=%d\n",
               P.x[0], P.x[1], P.x[2], P.x[3], P.y[0], P.y[1], P.y[2], P.y[3], P.z[0], P.z[1], P.z[2], P.z[3], P.infinity);
    }
#endif
}

__device__ void pointSetG(JacobianPoint &P) {
    fieldCopy(Gx_d, P.x);
    fieldCopy(Gy_d, P.y);
    fieldSetOne(P.z);
    P.infinity = false;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointSetG: P.x=%llx:%llx:%llx:%llx, P.y=%llx:%llx:%llx:%llx, P.z=%llx:%llx:%llx:%llx, P.infinity=%d\n",
               P.x[0], P.x[1], P.x[2], P.x[3], P.y[0], P.y[1], P.y[2], P.y[3], P.z[0], P.z[1], P.z[2], P.z[3], P.infinity);
    }
}

__device__ void pointToAffine(const JacobianPoint &P, unsigned long long outX[4], unsigned long long outY[4]) {
    if (P.infinity || isZero256(P.z)) {
        fieldSetZero(outX);
        fieldSetZero(outY);
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("pointToAffine: P.infinity=%d, P.z=%llx:%llx:%llx:%llx, outX=0:0:0:0, outY=0:0:0:0\n",
                   P.infinity, P.z[0], P.z[1], P.z[2], P.z[3]);
        }
        return;
    }
    unsigned long long zinv[4], zinv2[4];
    fieldInvFermat_device(P.z, zinv);
    fieldSqr_opt_device(zinv, zinv2);
    fieldMul_opt_device(P.x, zinv2, outX);
    fieldMul_opt_device(zinv, zinv2, zinv2);
    fieldMul_opt_device(P.y, zinv2, outY);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointToAffine: P.x=%llx:%llx:%llx:%llx, P.y=%llx:%llx:%llx:%llx, P.z=%llx:%llx:%llx:%llx, outX=%llx:%llx:%llx:%llx, outY=%llx:%llx:%llx:%llx\n",
               P.x[0], P.x[1], P.x[2], P.x[3], P.y[0], P.y[1], P.y[2], P.y[3], P.z[0], P.z[1], P.z[2], P.z[3],
               outX[0], outX[1], outX[2], outX[3], outY[0], outY[1], outY[2], outY[3]);
    }
}

__device__ void pointDoubleJacobian(const JacobianPoint &P, JacobianPoint &R) {
    if (P.infinity || isZero256(P.z)) {
        pointSetInfinity(R);
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("pointDoubleJacobian: P.infinity=%d, P.z=%llx:%llx:%llx:%llx, R set to infinity\n",
                   P.infinity, P.z[0], P.z[1], P.z[2], P.z[3]);
        }
        return;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointDoubleJacobian: input P.x=%llx:%llx:%llx:%llx, P.y=%llx:%llx:%llx:%llx, P.z=%llx:%llx:%llx:%llx\n",
               P.x[0], P.x[1], P.x[2], P.x[3], P.y[0], P.y[1], P.y[2], P.y[3], P.z[0], P.z[1], P.z[2], P.z[3]);
    }
    unsigned long long u[4], m[4], s[4], t[4], zz[4], tmp[4];
    fieldSqr_opt_device(P.y, u);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointDoubleJacobian: u=%llx:%llx:%llx:%llx\n", u[0], u[1], u[2], u[3]);
    }
    fieldSqr_opt_device(P.z, zz);
    fieldSqr_opt_device(u, t);
    fieldMul_opt_device(P.x, u, s);
    fieldAdd_opt_device(s, s, s);
    fieldSqr_opt_device(P.x, tmp);
    fieldAdd_opt_device(tmp, tmp, m);
    fieldAdd_opt_device(m, tmp, m);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointDoubleJacobian: m=%llx:%llx:%llx:%llx, s=%llx:%llx:%llx:%llx\n",
               m[0], m[1], m[2], m[3], s[0], s[1], s[2], s[3]);
    }
    fieldSqr_opt_device(m, R.x);
    fieldSub_opt_device(R.x, s, R.x);
    fieldSub_opt_device(R.x, s, R.x);
    fieldAdd_opt_device(P.y, P.z, R.z);
    fieldSqr_opt_device(R.z, R.z);
    fieldSub_opt_device(R.z, u, R.z);
    fieldSub_opt_device(R.z, zz, R.z);
    fieldSub_opt_device(s, R.x, tmp);
    fieldMul_opt_device(m, tmp, R.y);
    fieldAdd_opt_device(t, t, tmp);
    fieldAdd_opt_device(tmp, tmp, tmp);
    fieldSub_opt_device(R.y, tmp, R.y);
    R.infinity = false;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointDoubleJacobian: output R.x=%llx:%llx:%llx:%llx, R.y=%llx:%llx:%llx:%llx, R.z=%llx:%llx:%llx:%llx, R.infinity=%d\n",
               R.x[0], R.x[1], R.x[2], R.x[3], R.y[0], R.y[1], R.y[2], R.y[3], R.z[0], R.z[1], R.z[2], R.z[3], R.infinity);
    }
}

__device__ void pointAddJacobian(const JacobianPoint &P, const JacobianPoint &Q, JacobianPoint &R) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointAddJacobian: P.x=%llx:%llx:%llx:%llx, P.y=%llx:%llx:%llx:%llx, P.z=%llx:%llx:%llx:%llx, P.infinity=%d\n",
               P.x[0], P.x[1], P.x[2], P.x[3], P.y[0], P.y[1], P.y[2], P.y[3], P.z[0], P.z[1], P.z[2], P.z[3], P.infinity);
        printf("pointAddJacobian: Q.x=%llx:%llx:%llx:%llx, Q.y=%llx:%llx:%llx:%llx, Q.z=%llx:%llx:%llx:%llx, Q.infinity=%d\n",
               Q.x[0], Q.x[1], Q.x[2], Q.x[3], Q.y[0], Q.y[1], Q.y[2], Q.y[3], Q.z[0], Q.z[1], Q.z[2], Q.z[3], Q.infinity);
    }
    if (P.infinity || isZero256(P.z)) {
        R = Q;
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("pointAddJacobian: P is infinity, R=Q, R.x=%llx:%llx:%llx:%llx, R.infinity=%d\n",
                   R.x[0], R.x[1], R.x[2], R.x[3], R.infinity);
        }
        return;
    }
    if (Q.infinity || isZero256(Q.z)) {
        R = P;
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("pointAddJacobian: Q is infinity, R=P, R.x=%llx:%llx:%llx:%llx, R.infinity=%d\n",
                   R.x[0], R.x[1], R.x[2], R.x[3], R.infinity);
        }
        return;
    }
    unsigned long long z1z1[4], z2z2[4], u1[4], u2[4], s1[4], s2[4], h[4], i[4], j[4], r[4], v[4], tmp[4];
    fieldSqr_opt_device(P.z, z1z1);
    fieldSqr_opt_device(Q.z, z2z2);
    fieldMul_opt_device(P.x, z2z2, u1);
    fieldMul_opt_device(Q.x, z1z1, u2);
    fieldMul_opt_device(P.y, Q.z, s1);
    fieldMul_opt_device(s1, z2z2, s1);
    fieldMul_opt_device(Q.y, P.z, s2);
    fieldMul_opt_device(s2, z1z1, s2);
    if (_IsEqual(u1, u2)) {
        if (_IsEqual(s1, s2)) {
            pointDoubleJacobian(P, R);
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("pointAddJacobian: u1==u2, s1==s2, calling pointDoubleJacobian\n");
            }
        } else {
            pointSetInfinity(R);
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("pointAddJacobian: u1==u2, s1!=s2, R set to infinity\n");
            }
        }
        return;
    }
    fieldSub_opt_device(u2, u1, h);
    fieldAdd_opt_device(h, h, i);
    fieldSqr_opt_device(i, i);
    fieldMul_opt_device(i, h, j);
    fieldSub_opt_device(s2, s1, r);
    fieldAdd_opt_device(r, r, r);
    fieldMul_opt_device(u1, i, v);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointAddJacobian: h=%llx:%llx:%llx:%llx, i=%llx:%llx:%llx:%llx, j=%llx:%llx:%llx:%llx, r=%llx:%llx:%llx:%llx, v=%llx:%llx:%llx:%llx\n",
               h[0], h[1], h[2], h[3], i[0], i[1], i[2], i[3], j[0], j[1], j[2], j[3], r[0], r[1], r[2], r[3], v[0], v[1], v[2], v[3]);
    }
    fieldSqr_opt_device(r, R.x);
    fieldSub_opt_device(R.x, j, R.x);
    fieldSub_opt_device(R.x, v, R.x);
    fieldSub_opt_device(R.x, v, R.x);
    fieldSub_opt_device(v, R.x, tmp);
    fieldMul_opt_device(r, tmp, R.y);
    fieldMul_opt_device(s1, j, tmp);
    fieldAdd_opt_device(tmp, tmp, tmp);
    fieldSub_opt_device(R.y, tmp, R.y);
    fieldAdd_opt_device(P.z, Q.z, R.z);
    fieldSqr_opt_device(R.z, R.z);
    fieldSub_opt_device(R.z, z1z1, R.z);
    fieldSub_opt_device(R.z, z2z2, R.z);
    fieldMul_opt_device(R.z, h, R.z);
    R.infinity = false;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointAddJacobian: output R.x=%llx:%llx:%llx:%llx, R.y=%llx:%llx:%llx:%llx, R.z=%llx:%llx:%llx:%llx, R.infinity=%d\n",
               R.x[0], R.x[1], R.x[2], R.x[3], R.y[0], R.y[1], R.y[2], R.y[3], R.z[0], R.z[1], R.z[2], R.z[3], R.infinity);
    }
}

__device__ void pointAddMixed(const JacobianPoint &P, const unsigned long long Qx[4], const unsigned long long Qy[4], bool Qinf, JacobianPoint &R) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointAddMixed: P.x=%llx:%llx:%llx:%llx, P.y=%llx:%llx:%llx:%llx, P.z=%llx:%llx:%llx:%llx, P.infinity=%d\n",
               P.x[0], P.x[1], P.x[2], P.x[3], P.y[0], P.y[1], P.y[2], P.y[3], P.z[0], P.z[1], P.z[2], P.z[3], P.infinity);
        printf("pointAddMixed: Qx=%llx:%llx:%llx:%llx, Qy=%llx:%llx:%llx:%llx, Qinf=%d\n",
               Qx[0], Qx[1], Qx[2], Qx[3], Qy[0], Qy[1], Qy[2], Qy[3], Qinf);
    }
    if (P.infinity || isZero256(P.z)) {
        if (Qinf) {
            pointSetInfinity(R);
        } else {
            fieldCopy(Qx, R.x);
            fieldCopy(Qy, R.y);
            fieldSetOne(R.z);
            R.infinity = false;
        }
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("pointAddMixed: P is infinity, R.x=%llx:%llx:%llx:%llx, R.infinity=%d\n",
                   R.x[0], R.x[1], R.x[2], R.x[3], R.infinity);
        }
        return;
    }
    if (Qinf) {
        R = P;
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("pointAddMixed: Q is infinity, R=P, R.x=%llx:%llx:%llx:%llx, R.infinity=%d\n",
                   R.x[0], R.x[1], R.x[2], R.x[3], R.infinity);
        }
        return;
    }
    unsigned long long z1z1[4], u2[4], s2[4], h[4], i[4], j[4], r[4], v[4], tmp[4];
    fieldSqr_opt_device(P.z, z1z1);
    fieldMul_opt_device(Qx, z1z1, u2);
    fieldMul_opt_device(Qy, P.z, s2);
    fieldMul_opt_device(s2, z1z1, s2);
    fieldSub_opt_device(u2, P.x, h);
    fieldAdd_opt_device(h, h, i);
    fieldSqr_opt_device(i, i);
    fieldMul_opt_device(i, h, j);
    fieldSub_opt_device(s2, P.y, r);
    fieldAdd_opt_device(r, r, r);
    fieldMul_opt_device(P.x, i, v);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointAddMixed: h=%llx:%llx:%llx:%llx, i=%llx:%llx:%llx:%llx, j=%llx:%llx:%llx:%llx, r=%llx:%llx:%llx:%llx, v=%llx:%llx:%llx:%llx\n",
               h[0], h[1], h[2], h[3], i[0], i[1], i[2], i[3], j[0], j[1], j[2], j[3], r[0], r[1], r[2], r[3], v[0], v[1], v[2], v[3]);
    }
    fieldSqr_opt_device(r, R.x);
    fieldSub_opt_device(R.x, j, R.x);
    fieldSub_opt_device(R.x, v, R.x);
    fieldSub_opt_device(R.x, v, R.x);
    fieldSub_opt_device(v, R.x, tmp);
    fieldMul_opt_device(r, tmp, R.y);
    fieldMul_opt_device(P.y, j, tmp);
    fieldAdd_opt_device(tmp, tmp, tmp);
    fieldSub_opt_device(R.y, tmp, R.y);
    fieldMul_opt_device(P.z, h, R.z);
    R.infinity = false;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointAddMixed: output R.x=%llx:%llx:%llx:%llx, R.y=%llx:%llx:%llx:%llx, R.z=%llx:%llx:%llx:%llx, R.infinity=%d\n",
               R.x[0], R.x[1], R.x[2], R.x[3], R.y[0], R.y[1], R.y[2], R.y[3], R.z[0], R.z[1], R.z[2], R.z[3], R.infinity);
    }
}

__device__ int find_msb(const unsigned long long a[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] != 0) {
            int msb = i * 64 + 63 - __clzll(a[i]);
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("find_msb: a=%llx:%llx:%llx:%llx, msb=%d\n", a[0], a[1], a[2], a[3], msb);
            }
            return msb;
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("find_msb: a=0:0:0:0, msb=-1\n");
    }
    return -1;
}

__device__ uint32_t get_window(const unsigned long long a[4], int pos) {
    int limb = pos >> 6;
    int shift = pos & 63;
    if (limb >= 4) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("get_window: pos=%d, limb=%d, shift=%d, returning 0\n", pos, limb, shift);
        }
        return 0;
    }
    unsigned long long bits = a[limb] >> shift;
    if (shift > 64 - PRECOMPUTE_WINDOW && limb < 3) {
        bits |= a[limb+1] << (64 - shift);
    }
    uint32_t window = bits & ((1ULL << PRECOMPUTE_WINDOW) - 1);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("get_window: pos=%d, limb=%d, shift=%d, a=%llx:%llx:%llx:%llx, bits=%llx, window=%u\n",
               pos, limb, shift, a[0], a[1], a[2], a[3], bits, window);
    }
    return window;
}

__device__ void scalarMulBaseJacobian(const unsigned long long scalar_le[4], unsigned long long outX[4], unsigned long long outY[4], unsigned long long* d_pre_Gx, unsigned long long* d_pre_Gy, unsigned long long* d_pre_phiGx, unsigned long long* d_pre_phiGy) {
    unsigned long long k1[4], k2[4];
    split_glv(scalar_le, k1, k2);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("scalarMulBaseJacobian: scalar=%llx:%llx:%llx:%llx, k1=%llx:%llx:%llx:%llx, k2=%llx:%llx:%llx:%llx\n",
               scalar_le[0], scalar_le[1], scalar_le[2], scalar_le[3],
               k1[0], k1[1], k1[2], k1[3], k2[0], k2[1], k2[2], k2[3]);
    }
    JacobianPoint R1, R2, R;
    pointSetInfinity(R1);
    pointSetInfinity(R2);
    // Handle small k1 directly
    if (k1[3] == 0 && k1[2] == 0 && k1[1] == 0 && k1[0] <= 0xFFFFFFFF) {
        if (isZero256(k1)) {
            pointSetInfinity(R);
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("scalarMulBaseJacobian: k1=0, R set to infinity\n");
            }
        } else {
            pointSetG(R);
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("scalarMulBaseJacobian: small k1=%llx, starting R.x=%llx:%llx:%llx:%llx, R.y=%llx:%llx:%llx:%llx, R.z=%llx:%llx:%llx:%llx\n",
                       k1[0], R.x[0], R.x[1], R.x[2], R.x[3], R.y[0], R.y[1], R.y[2], R.y[3], R.z[0], R.z[1], R.z[2], R.z[3]);
            }
            for (uint64_t i = 0; i < k1[0]; ++i) {
                pointDoubleJacobian(R, R);
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                    printf("scalarMulBaseJacobian: after double %llu, R.x=%llx:%llx:%llx:%llx, R.y=%llx:%llx:%llx:%llx, R.z=%llx:%llx:%llx:%llx\n",
                           i+1, R.x[0], R.x[1], R.x[2], R.x[3], R.y[0], R.y[1], R.y[2], R.y[3], R.z[0], R.z[1], R.z[2], R.z[3]);
                }
            }
        }
        if (isZero256(k2)) {
            pointToAffine(R, outX, outY);
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("scalarMulBaseJacobian: small k1=%llx, k2=0, final R.x=%llx:%llx:%llx:%llx, R.y=%llx:%llx:%llx:%llx, R.infinity=%d\n",
                       k1[0], R.x[0], R.x[1], R.x[2], R.x[3], R.y[0], R.y[1], R.y[2], R.y[3], R.infinity);
            }
            return;
        }
    }
    int msb1 = find_msb(k1);
    int msb2 = find_msb(k2);
    int msb = max(msb1, msb2);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("scalarMulBaseJacobian: msb1=%d, msb2=%d, msb=%d\n", msb1, msb2, msb);
    }
    if (msb < 0) {
        pointSetInfinity(R);
        pointToAffine(R, outX, outY);
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("scalarMulBaseJacobian: msb<0, R set to infinity\n");
        }
        return;
    }
    for (int pos = msb - (msb % PRECOMPUTE_WINDOW); pos >= 0; pos -= PRECOMPUTE_WINDOW) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("scalarMulBaseJacobian: pos=%d, R1.x=%llx:%llx:%llx:%llx, R2.x=%llx:%llx:%llx:%llx\n",
                   pos, R1.x[0], R1.x[1], R1.x[2], R1.x[3], R2.x[0], R2.x[1], R2.x[2], R2.x[3]);
        }
        #pragma unroll
        for (int i = 0; i < PRECOMPUTE_WINDOW; ++i) {
            pointDoubleJacobian(R1, R1);
            pointDoubleJacobian(R2, R2);
        }
        uint32_t w1 = get_window(k1, pos);
        if (w1 && w1 < PRECOMPUTE_SIZE) {
            JacobianPoint P;
            fieldCopy(d_pre_Gx + w1 * 4, P.x);
            fieldCopy(d_pre_Gy + w1 * 4, P.y);
            fieldSetOne(P.z);
            P.infinity = false;
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("scalarMulBaseJacobian: w1=%u, P.x=%llx:%llx:%llx:%llx, P.y=%llx:%llx:%llx:%llx\n",
                       w1, P.x[0], P.x[1], P.x[2], P.x[3], P.y[0], P.y[1], P.y[2], P.y[3]);
            }
            pointAddMixed(R1, P.x, P.y, P.infinity, R1);
        }
        uint32_t w2 = get_window(k2, pos);
        if (w2 && w2 < PRECOMPUTE_SIZE) {
            JacobianPoint P;
            fieldCopy(d_pre_phiGx + w2 * 4, P.x);
            fieldCopy(d_pre_phiGy + w2 * 4, P.y);
            fieldSetOne(P.z);
            P.infinity = false;
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("scalarMulBaseJacobian: w2=%u, P.x=%llx:%llx:%llx:%llx, P.y=%llx:%llx:%llx:%llx\n",
                       w2, P.x[0], P.x[1], P.x[2], P.x[3], P.y[0], P.y[1], P.y[2], P.y[3]);
            }
            pointAddMixed(R2, P.x, P.y, P.infinity, R2);
        }
    }
    fieldMul_opt_device(R2.x, c_beta, R2.x);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("scalarMulBaseJacobian: after beta mul, R2.x=%llx:%llx:%llx:%llx, R2.infinity=%d\n",
               R2.x[0], R2.x[1], R2.x[2], R2.x[3], R2.infinity);
    }
    pointAddJacobian(R1, R2, R);
    pointToAffine(R, outX, outY);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("scalarMulBaseJacobian: final R.x=%llx:%llx:%llx:%llx, R.y=%llx:%llx:%llx:%llx, R.infinity=%d\n",
               R.x[0], R.x[1], R.x[2], R.x[3], R.y[0], R.y[1], R.y[2], R.y[3], R.infinity);
    }
}

__host__ __device__ bool ge256(const unsigned long long a[4], const unsigned long long b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

__host__ __device__ bool ge256_u64(const unsigned long long a[4], unsigned long long b) {
    if (a[3] != 0 || a[2] != 0 || a[1] != 0) return true;
    return a[0] >= b;
}

__global__ void scalarMulKernelBase(const unsigned long long* scalars_in, unsigned long long* outX, unsigned long long* outY, int N, unsigned long long* d_pre_Gx, unsigned long long* d_pre_Gy, unsigned long long* d_pre_phiGx, unsigned long long* d_pre_phiGy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("scalarMulKernelBase: idx=%d, scalar=%llx:%llx:%llx:%llx\n",
               idx, scalars_in[idx*4], scalars_in[idx*4+1], scalars_in[idx*4+2], scalars_in[idx*4+3]);
    }
    scalarMulBaseJacobian(scalars_in + idx*4, outX + idx*4, outY + idx*4, d_pre_Gx, d_pre_Gy, d_pre_phiGx, d_pre_phiGy);
}

__global__ void precompute_table_kernel(JacobianPoint base, unsigned long long* pre_x, unsigned long long* pre_y, unsigned long long size) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    JacobianPoint P = base;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("precompute_table_kernel: base.x=%llx:%llx:%llx:%llx, base.y=%llx:%llx:%llx:%llx\n",
               base.x[0], base.x[1], base.x[2], base.x[3], base.y[0], base.y[1], base.y[2], base.y[3]);
    }
    for (unsigned long long bit = 0; bit < idx; ++bit) {
        if (bit % 2 == 0) {
            pointDoubleJacobian(P, P);
        } else {
            pointAddJacobian(P, base, P);
        }
    }
    fieldCopy(P.x, pre_x + idx * 4);
    fieldCopy(P.y, pre_y + idx * 4);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("precompute_table_kernel: idx=%llu, P.x=%llx:%llx:%llx:%llx, P.y=%llx:%llx:%llx:%llx\n",
               idx, P.x[0], P.x[1], P.x[2], P.x[3], P.y[0], P.y[1], P.y[2], P.y[3]);
    }
}

#endif // CUDA_MATH_H