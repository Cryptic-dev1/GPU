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

__device__ __constant__ unsigned long long MM64 = 0xD838091DD2253531ULL;
__device__ __constant__ unsigned long long MSK62 = 0x3FFFFFFFFFFFFFFFULL;

// Utility function for zero check
__device__ __host__ __forceinline__ bool isZero256(const unsigned long long a[4]) {
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

#define AddP(r) { \
    UADDO1(r[0], c_p[0]); \
    UADDC1(r[1], c_p[1]); \
    UADDC1(r[2], c_p[2]); \
    UADD1(r[3], c_p[3]); \
}

// Field Utility Functions
__device__ void fieldSetZero(unsigned long long a[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = 0ULL;
}

__device__ void fieldSetOne(unsigned long long a[4]) {
    a[0] = 1ULL;
    #pragma unroll
    for (int i = 1; i < 4; ++i) a[i] = 0ULL;
}

__device__ void fieldCopy(const unsigned long long a[4], unsigned long long b[4]) {
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
}

// Optimized Field Operations
__device__ void fieldAdd_opt(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]) {
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
}

__device__ void fieldSub_opt(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]) {
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
}

__device__ void mul256(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[8]) {
    fieldSetZero(out);
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
            carry = (sum < lo) ? 1 : 0;
            unsigned long long out_ij = out[i+j];
            out[i+j] = out_ij + sum;
            carry += (out[i+j] < out_ij) ? 1 : 0;
            unsigned long long out_ij1 = out[i+j+1];
            out[i+j+1] = out_ij1 + hi + carry;
            carry = (out[i+j+1] < hi) ? 1 : 0;
        }
        if (i + 4 < 8) out[i+4] += carry;
    }
}

__device__ void mul_high(const unsigned long long a[4], const unsigned long long b[5], unsigned long long high[5]) {
    unsigned long long prod[9] = {0};
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
            carry = (sum < lo) ? 1 : 0;
            unsigned long long prod_ij = prod[i+j];
            prod[i+j] = prod_ij + sum;
            carry += (prod[i+j] < prod_ij) ? 1 : 0;
            unsigned long long prod_ij1 = prod[i+j+1];
            prod[i+j+1] = prod_ij1 + hi + carry;
            carry = (prod[i+j+1] < hi) ? 1 : 0;
        }
        if (i + 5 < 9) prod[i+5] += carry;
    }
    #pragma unroll
    for (int i = 0; i < 5; ++i) high[i] = prod[i+4];
}

__device__ void modred_barrett_opt(const unsigned long long input[8], unsigned long long out[4]) {
    unsigned long long q[5], tmp[8], r[4];
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

__device__ void fieldMul_opt(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]) {
    unsigned long long prod[8];
    mul256(a, b, prod);
    modred_barrett_opt(prod, out);
}

__device__ void fieldSqr_opt(const unsigned long long a[4], unsigned long long out[4]) {
    fieldMul_opt(a, a, out);
}

__device__ void fieldNeg(const unsigned long long a[4], unsigned long long out[4]) {
    if (isZero256(a)) {
        fieldSetZero(out);
        return;
    }
    fieldSub_opt(c_p, a, out);
}

__device__ void fieldInvFermat(const unsigned long long a[4], unsigned long long inv[4]) {
    if (isZero256(a)) {
        fieldSetZero(inv);
        return;
    }
    unsigned long long t[4], p_minus_2[4] = {0xfffffc2dULL, 0xffffffffULL, 0xffffffffULL, 0xffffffffULL};
    fieldCopy(a, t);
    for (int i = 255; i >= 1; --i) {
        fieldSqr_opt(t, t);
        if ((p_minus_2[i/64] >> (i%64)) & 1ULL) {
            fieldMul_opt(t, a, t);
        }
    }
    fieldCopy(t, inv);
}

__device__ void batch_modinv_fermat(const unsigned long long* a, unsigned long long* inv, int n) {
    extern __shared__ unsigned long long shared[];
    unsigned long long *prefix = shared;
    unsigned long long prod[4], tmp[4];
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
__device__ void div512_256(const unsigned long long num[8], const unsigned long long den[4], unsigned long long quot[4], unsigned long long rem[4]) {
    unsigned long long dividend[8], shifted_den[8], q[4] = {0};
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
__device__ void split_glv(const unsigned long long scalar[4], unsigned long long k1[4], unsigned long long k2[4]) {
    unsigned long long num[8], half_n[4], q1[4], q2[4], tmp1[4], tmp2[4], rem[4];
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
__device__ void pointSetInfinity(JacobianPoint &P) {
    fieldSetZero(P.x);
    fieldSetZero(P.y);
    fieldSetZero(P.z);
    P.infinity = true;
}

__device__ void pointSetG(JacobianPoint &P) {
    fieldCopy(Gx_d, P.x);
    fieldCopy(Gy_d, P.y);
    fieldSetOne(P.z);
    P.infinity = false;
}

__device__ void pointToAffine(const JacobianPoint &P, unsigned long long outX[4], unsigned long long outY[4]) {
    if (P.infinity || isZero256(P.z)) {
        fieldSetZero(outX);
        fieldSetZero(outY);
        return;
    }
    unsigned long long zinv[4], zinv2[4];
    fieldInvFermat(P.z, zinv);
    fieldSqr_opt(zinv, zinv2);
    fieldMul_opt(P.x, zinv2, outX);
    fieldMul_opt(zinv, zinv2, zinv2);
    fieldMul_opt(P.y, zinv2, outY);
}

__device__ void pointDoubleJacobian(const JacobianPoint &P, JacobianPoint &R) {
    if (P.infinity || isZero256(P.z)) {
        pointSetInfinity(R);
        return;
    }
    unsigned long long u[4], m[4], s[4], t[4], zz[4], tmp[4];
    fieldSqr_opt(P.y, u);
    fieldSqr_opt(P.z, zz);
    fieldSqr_opt(u, t);
    fieldMul_opt(P.x, u, s);
    fieldAdd_opt(s, s, s);
    fieldSqr_opt(P.x, tmp);
    fieldAdd_opt(tmp, tmp, m);
    fieldAdd_opt(m, tmp, m);
    fieldSqr_opt(m, R.x);
    fieldSub_opt(R.x, s, R.x);
    fieldSub_opt(R.x, s, R.x);
    fieldAdd_opt(P.y, P.z, R.z);
    fieldSqr_opt(R.z, R.z);
    fieldSub_opt(R.z, u, R.z);
    fieldSub_opt(R.z, zz, R.z);
    fieldSub_opt(s, R.x, tmp);
    fieldMul_opt(m, tmp, R.y);
    fieldAdd_opt(t, t, tmp);
    fieldAdd_opt(tmp, tmp, tmp);
    fieldSub_opt(R.y, tmp, R.y);
    R.infinity = false;
}

__device__ void pointAddJacobian(const JacobianPoint &P, const JacobianPoint &Q, JacobianPoint &R) {
    if (P.infinity || isZero256(P.z)) {
        R = Q;
        return;
    }
    if (Q.infinity || isZero256(Q.z)) {
        R = P;
        return;
    }
    unsigned long long z1z1[4], z2z2[4], u1[4], u2[4], s1[4], s2[4], h[4], i[4], j[4], r[4], v[4], tmp[4];
    fieldSqr_opt(P.z, z1z1);
    fieldSqr_opt(Q.z, z2z2);
    fieldMul_opt(P.x, z2z2, u1);
    fieldMul_opt(Q.x, z1z1, u2);
    fieldMul_opt(P.y, Q.z, s1);
    fieldMul_opt(s1, z2z2, s1);
    fieldMul_opt(Q.y, P.z, s2);
    fieldMul_opt(s2, z1z1, s2);
    if (_IsEqual(u1, u2)) {
        if (_IsEqual(s1, s2)) {
            pointDoubleJacobian(P, R);
        } else {
            pointSetInfinity(R);
        }
        return;
    }
    fieldSub_opt(u2, u1, h);
    fieldAdd_opt(h, h, i);
    fieldSqr_opt(i, i);
    fieldMul_opt(i, h, j);
    fieldSub_opt(s2, s1, r);
    fieldAdd_opt(r, r, r);
    fieldMul_opt(u1, i, v);
    fieldSqr_opt(r, R.x);
    fieldSub_opt(R.x, j, R.x);
    fieldSub_opt(R.x, v, R.x);
    fieldSub_opt(R.x, v, R.x);
    fieldSub_opt(v, R.x, tmp);
    fieldMul_opt(r, tmp, R.y);
    fieldMul_opt(s1, j, tmp);
    fieldAdd_opt(tmp, tmp, tmp);
    fieldSub_opt(R.y, tmp, R.y);
    fieldAdd_opt(P.z, Q.z, R.z);
    fieldSqr_opt(R.z, R.z);
    fieldSub_opt(R.z, z1z1, R.z);
    fieldSub_opt(R.z, z2z2, R.z);
    fieldMul_opt(R.z, h, R.z);
    R.infinity = false;
}

__device__ void pointAddMixed(const JacobianPoint &P, const unsigned long long Qx[4], const unsigned long long Qy[4], bool Qinf, JacobianPoint &R) {
    if (P.infinity || isZero256(P.z)) {
        if (Qinf) {
            pointSetInfinity(R);
        } else {
            fieldCopy(Qx, R.x);
            fieldCopy(Qy, R.y);
            fieldSetOne(R.z);
            R.infinity = false;
        }
        return;
    }
    if (Qinf) {
        R = P;
        return;
    }
    unsigned long long z1z1[4], u2[4], s2[4], h[4], i[4], j[4], r[4], v[4], tmp[4];
    fieldSqr_opt(P.z, z1z1);
    fieldMul_opt(Qx, z1z1, u2);
    fieldMul_opt(Qy, P.z, s2);
    fieldMul_opt(s2, z1z1, s2);
    fieldSub_opt(u2, P.x, h);
    fieldAdd_opt(h, h, i);
    fieldSqr_opt(i, i);
    fieldMul_opt(i, h, j);
    fieldSub_opt(s2, P.y, r);
    fieldAdd_opt(r, r, r);
    fieldMul_opt(P.x, i, v);
    fieldSqr_opt(r, R.x);
    fieldSub_opt(R.x, j, R.x);
    fieldSub_opt(R.x, v, R.x);
    fieldSub_opt(R.x, v, R.x);
    fieldSub_opt(v, R.x, tmp);
    fieldMul_opt(r, tmp, R.y);
    fieldMul_opt(P.y, j, tmp);
    fieldAdd_opt(tmp, tmp, tmp);
    fieldSub_opt(R.y, tmp, R.y);
    fieldMul_opt(P.z, h, R.z);
    R.infinity = false;
}

__device__ int find_msb(const unsigned long long a[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] != 0) return i * 64 + 63 - __clzll(a[i]);
    }
    return -1;
}

__device__ uint32_t get_window(const unsigned long long a[4], int pos) {
    int limb = pos >> 6;
    int shift = pos & 63;
    if (limb >= 4) return 0;
    unsigned long long bits = a[limb] >> shift;
    if (shift > 64 - PRECOMPUTE_WINDOW && limb < 3) {
        bits |= a[limb+1] << (64 - shift);
    }
    return bits & ((1ULL << PRECOMPUTE_WINDOW) - 1);
}

__device__ void scalarMulBaseJacobian(const unsigned long long scalar_le[4], unsigned long long outX[4], unsigned long long outY[4], unsigned long long* d_pre_Gx, unsigned long long* d_pre_Gy, unsigned long long* d_pre_phiGx, unsigned long long* d_pre_phiGy) {
    unsigned long long k1[4], k2[4];
    split_glv(scalar_le, k1, k2);
    JacobianPoint R1, R2, R;
    pointSetInfinity(R1);
    pointSetInfinity(R2);
    int msb1 = find_msb(k1);
    int msb2 = find_msb(k2);
    int msb = max(msb1, msb2);
    for (int pos = msb - (msb % PRECOMPUTE_WINDOW); pos >= 0; pos -= PRECOMPUTE_WINDOW) {
        #pragma unroll
        for (int i = 0; i < PRECOMPUTE_WINDOW; ++i) {
            pointDoubleJacobian(R1, R1);
            pointDoubleJacobian(R2, R2);
        }
        uint32_t w1 = get_window(k1, pos);
        if (w1) {
            JacobianPoint P;
            fieldCopy(d_pre_Gx + w1 * 4, P.x);
            fieldCopy(d_pre_Gy + w1 * 4, P.y);
            fieldSetOne(P.z);
            P.infinity = false;
            pointAddMixed(R1, P.x, P.y, P.infinity, R1);
        }
        uint32_t w2 = get_window(k2, pos);
        if (w2) {
            JacobianPoint P;
            fieldCopy(d_pre_phiGx + w2 * 4, P.x);
            fieldCopy(d_pre_phiGy + w2 * 4, P.y);
            fieldSetOne(P.z);
            P.infinity = false;
            pointAddMixed(R2, P.x, P.y, P.infinity, R2);
        }
    }
    fieldMul_opt(R2.x, c_beta, R2.x);
    pointAddJacobian(R1, R2, R);
    pointToAffine(R, outX, outY);
}

__global__ void scalarMulKernelBase(const unsigned long long* scalars_in, unsigned long long* outX, unsigned long long* outY, int N, unsigned long long* d_pre_Gx, unsigned long long* d_pre_Gy, unsigned long long* d_pre_phiGx, unsigned long long* d_pre_phiGy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    scalarMulBaseJacobian(scalars_in + idx*4, outX + idx*4, outY + idx*4, d_pre_Gx, d_pre_Gy, d_pre_phiGx, d_pre_phiGy);
}

__global__ void precompute_table_kernel(JacobianPoint base, unsigned long long* pre_x, unsigned long long* pre_y, unsigned long long size) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    JacobianPoint P = base;
    for (unsigned long long bit = 0; bit < idx; ++bit) {
        if (bit % 2 == 0) {
            pointDoubleJacobian(P, P);
        } else {
            pointAddJacobian(P, base, P);
        }
    }
    fieldCopy(P.x, pre_x + idx * 4);
    fieldCopy(P.y, pre_y + idx * 4);
}

#endif // CUDA_MATH_H