#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CUDAStructures.h"
#include "CUDAUtils.h"

#define NBBLOCK 5
#define BIFULLSIZE 40
#define WARP_SIZE 32

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

#define HSIZE (GRP_SIZE / 2 - 1)

__device__ __constant__ uint64_t MM64 = 0xD838091DD2253531ULL;
__device__ __constant__ uint64_t MSK62 = 0x3FFFFFFFFFFFFFFFULL;

#define _IsPositive(x) (((int64_t)(x[3])) >= 0LL)
#define _IsNegative(x) (((int64_t)(x[3])) < 0LL)
#define _IsEqual(a, b) ((a[3] == b[3]) && (a[2] == b[2]) && (a[1] == b[1]) && (a[0] == b[0]))
#define _IsZero(a) ((a[3] | a[2] | a[1] | a[0]) == 0ULL)
#define _IsOne(a) ((a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) && (a[0] == 1ULL))

#define IDX threadIdx.x

#define bswap32(v) __byte_perm(v, 0, 0x0123)

#define __sright128(a, b, n) ((a) >> (n)) | ((b) << (64 - (n)))
#define __sleft128(a, b, n) ((b) << (n)) | ((a) >> (64 - (n)))

#define AddP(r) { \
    UADDO1(r[0], c_p[0]); \
    UADDC1(r[1], c_p[1]); \
    UADDC1(r[2], c_p[2]); \
    UADD1(r[3], c_p[3]); \
}

// Field Utility Functions
__device__ void fieldSetZero(uint64_t a[4]);
__device__ void fieldSetOne(uint64_t a[4]);
__device__ void fieldCopy(const uint64_t a[4], uint64_t b[4]);
__device__ void lsl256(uint64_t a[4], uint64_t out[4], int n);
__device__ void lsr256(uint64_t a[4], uint64_t out[4], int n);
__device__ void lsl512(const uint64_t a[4], int n, uint64_t out[8]);
__device__ bool ge512(const uint64_t a[8], const uint64_t b[8]);
__device__ void sub512(const uint64_t a[8], const uint64_t b[8], uint64_t out[8]);

// Optimized Field Operations
__device__ void fieldAdd_opt(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    uint64_t carry = 0, temp;
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
}

__device__ void fieldSub_opt(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    uint64_t borrow = 0, temp;
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
}

__device__ void mul256(const uint64_t a[4], const uint64_t b[4], uint64_t out[8]) {
    fieldSetZero(out);
    uint64_t lo, hi, carry;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            UMULLO(lo, a[i], b[j]);
            UMULHI(hi, a[i], b[j]);
            UADDO(lo, lo, carry);
            UADD1(hi, 0);
            UADDO(out[i+j], out[i+j], lo);
            UADDC(out[i+j+1], out[i+j+1], hi);
            carry = (out[i+j+1] < hi) ? 1 : 0;
        }
        if (i + 4 < 8) out[i+4] += carry;
    }
}

__device__ void mul_high(const uint64_t a[4], const uint64_t b[5], uint64_t high[5]) {
    uint64_t prod[9] = {0};
    uint64_t carry, lo, hi;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        carry = 0;
        #pragma unroll
        for (int j = 0; j < 5; ++j) {
            UMULLO(lo, a[i], b[j]);
            UMULHI(hi, a[i], b[j]);
            UADDO(lo, lo, carry);
            UADD1(hi, 0);
            UADDO(prod[i+j], prod[i+j], lo);
            UADDC(prod[i+j+1], prod[i+j+1], hi);
            carry = (prod[i+j+1] < hi) ? 1 : 0;
        }
        if (i + 5 < 9) prod[i+5] += carry;
    }
    #pragma unroll
    for (int i = 0; i < 5; ++i) high[i] = prod[i+4];
}

__device__ void modred_barrett_opt(const uint64_t input[8], uint64_t out[4]) {
    uint64_t q[5], tmp[8], r[4];
    mul_high(input+4, c_mu, q);
    mul256(q, c_p, tmp);
    fieldSub_opt(input, tmp, r);
    if (ge256(r, c_p)) {
        fieldSub_opt(r, c_p, r);
    }
    if (_IsNegative(r)) {
        fieldAdd_opt(r, c_p, r);
    }
    fieldCopy(r, out);
}

__device__ void fieldMul_opt(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    uint64_t prod[8];
    mul256(a, b, prod);
    modred_barrett_opt(prod, out);
}

__device__ void fieldSqr_opt(const uint64_t a[4], uint64_t out[4]) {
    fieldMul_opt(a, a, out);
}

__device__ void fieldNeg(const uint64_t a[4], uint64_t out[4]) {
    if (_IsZero(a)) {
        fieldSetZero(out);
        return;
    }
    fieldSub_opt(c_p, a, out);
}

__device__ void fieldInvFermat(const uint64_t a[4], uint64_t inv[4]) {
    if (_IsZero(a)) {
        fieldSetZero(inv);
        return;
    }
    uint64_t t[4], p_minus_2[4] = {0xfffffc2d, 0xffffffff, 0xffffffff, 0xffffffff};
    fieldCopy(a, t);
    for (int i = 255; i >= 1; --i) {
        fieldSqr_opt(t, t);
        if ((p_minus_2[i/64] >> (i%64)) & 1ULL) {
            fieldMul_opt(t, a, t);
        }
    }
    fieldCopy(t, inv);
}

__device__ void batch_modinv_fermat(const uint64_t* a, uint64_t* inv, int n) {
    extern __shared__ uint64_t shared[];
    uint64_t *prefix = shared;
    uint64_t prod[4], tmp[4];
    int tid = threadIdx.x % WARP_SIZE;
    if (tid == 0) {
        fieldSetOne(prefix);
        for (int i = 0; i < n; ++i) {
            fieldMul_opt(prefix + i*4, a + i*4, prefix + (i+1)*4);
        }
        fieldInvFermat(prefix + n*4, prod);
    }
    __syncthreads();
    if (tid == 0) {
        fieldCopy(prod, tmp);
        for (int i = n-1; i >= 0; --i) {
            fieldMul_opt(tmp, prefix + i*4, inv + i*4);
            fieldMul_opt(tmp, a + i*4, tmp);
        }
    }
    __syncthreads();
}

// Division for GLV
__device__ void div512_256(const uint64_t num[8], const uint64_t den[4], uint64_t quot[4], uint64_t rem[4]) {
    uint64_t dividend[8], shifted_den[8], q[4] = {0};
    fieldCopy(num, dividend);
    for (int bit = 255; bit >= 0; --bit) {
        lsl256(q, q, 1);
        lsl512(den, bit, shifted_den);
        if (ge512(dividend, shifted_den)) {
            sub512(dividend, shifted_den, dividend);
            q[0] |= 1ULL;
        }
    }
    fieldCopy(q, quot);
    fieldCopy(dividend, rem);
}

// GLV Endomorphism
__device__ void split_glv(const uint64_t scalar[4], uint64_t k1[4], uint64_t k2[4]) {
    uint64_t num[8], half_n[4], q1[4], q2[4], tmp1[4], tmp2[4], rem[4];
    fieldCopy(c_n, half_n);
    lsr256(half_n, half_n, 1);
    // q1 = round(b2 * scalar / n)
    fieldMul_opt(c_b2, scalar, num);
    fieldAdd_opt(num, half_n, num);
    div512_256(num, c_n, q1, rem);
    // q2 = round(b1 * scalar / n)
    fieldMul_opt(c_b1, scalar, num);
    fieldAdd_opt(num, half_n, num);
    div512_256(num, c_n, q2, rem);
    // k1 = scalar - q1 * a1 - q2 * a2
    fieldMul_opt(q1, c_a1, tmp1);
    fieldMul_opt(q2, c_a2, tmp2);
    fieldAdd_opt(tmp1, tmp2, tmp1);
    fieldSub_opt(scalar, tmp1, k1);
    if (_IsNegative(k1)) {
        fieldAdd_opt(k1, c_n, k1);
    }
    // k2 = q1 * b1 - q2 * b2
    fieldMul_opt(q1, c_b1, tmp1);
    fieldMul_opt(q2, c_b2, tmp2);
    fieldSub_opt(tmp1, tmp2, k2);
    if (_IsNegative(k2)) {
        fieldAdd_opt(k2, c_n, k2);
    }
}

// Jacobian Point Operations
__device__ void pointSetInfinity(JacobianPoint &P);
__device__ void pointSetG(JacobianPoint &P);
__device__ void pointToAffine(const JacobianPoint &P, uint64_t outX[4], uint64_t outY[4]);
__device__ void pointDoubleJacobian(const JacobianPoint &P, JacobianPoint &R);
__device__ void pointAddJacobian(const JacobianPoint &P, const JacobianPoint &Q, JacobianPoint &R);
__device__ void pointAddMixed(const JacobianPoint &P, const uint64_t Qx[4], const uint64_t Qy[4], bool Qinf, JacobianPoint &R);
__device__ int find_msb(const uint64_t a[4]);
__device__ uint32_t get_window(const uint64_t a[4], int pos);
__device__ void scalarMulBaseJacobian(const uint64_t scalar_le[4], uint64_t outX[4], uint64_t outY[4], uint64_t* d_pre_Gx, uint64_t* d_pre_Gy, uint64_t* d_pre_phiGx, uint64_t* d_pre_phiGy);

#endif // CUDA_MATH_H