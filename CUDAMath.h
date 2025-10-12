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

// PTX Assembly Macros (kept for other functions)
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

// Constants
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

__host__ __device__ void fieldCopy(const unsigned long long a[4], unsigned long long b[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = a[i];
}

__device__ void fieldAdd_opt_device(const unsigned long long a[4], const unsigned long long b[4], unsigned long long c[4]) {
    unsigned long long carry = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned long long sum = a[i] + b[i] + carry;
        c[i] = sum;
        carry = (sum < a[i] || (sum == a[i] && carry)) ? 1ULL : 0ULL;
    }
    if (carry || ge256(c, c_p)) {
        unsigned long long temp[4];
        fieldCopy(c, temp);
        unsigned long long borrow = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned long long diff = temp[i] - c_p[i] - borrow;
            c[i] = diff;
            borrow = (diff > temp[i] || (diff == temp[i] && borrow)) ? 1ULL : 0ULL;
        }
    }
}

__device__ void fieldSub_opt_device(const unsigned long long a[4], const unsigned long long b[4], unsigned long long c[4]) {
    unsigned long long borrow = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned long long diff = a[i] - b[i] - borrow;
        c[i] = diff;
        borrow = (diff > a[i] || (diff == a[i] && borrow)) ? 1ULL : 0ULL;
    }
    if (borrow) {
        unsigned long long temp[4];
        fieldCopy(c, temp);
        unsigned long long carry = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned long long sum = temp[i] + c_p[i] + carry;
            c[i] = sum;
            carry = (sum < temp[i] || (sum == temp[i] && carry)) ? 1ULL : 0ULL;
        }
    }
}

__device__ void fieldMul_opt_device(const unsigned long long a[4], const unsigned long long b[4], unsigned long long c[8]) {
    unsigned long long temp[8] = {0};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned long long carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            __uint128_t prod = (__uint128_t)a[i] * b[j] + temp[i + j] + carry;
            temp[i + j] = (unsigned long long)prod;
            carry = (unsigned long long)(prod >> 64);
        }
        temp[i + 4] += carry;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) c[i] = temp[i];
}

__device__ void modred_barrett_opt_device(const unsigned long long in[8], unsigned long long out[4]) {
    unsigned long long q[5], tmp[8];
    fieldMul_opt_device(in + 4, c_mu, q);
    fieldMul_opt_device(q, c_p, tmp);
    fieldSub_opt_device(in, tmp, out);
    if (ge256(out, c_p)) {
        fieldSub_opt_device(out, c_p, out);
    }
}

__device__ void fieldSqr_opt_device(const unsigned long long a[4], unsigned long long c[8]) {
    fieldMul_opt_device(a, a, c);
}

__device__ void fieldInv_opt_device(const unsigned long long a[4], unsigned long long out[4]) {
    unsigned long long t[4], r[4];
    fieldCopy(a, t);
    fieldSetZero(r);
    r[0] = 1ULL;
    unsigned long long exp[4] = {0xfffffffc2dULL, 0xffffffffULL, 0xffffffffULL, 0xffffffffULL};
    for (int i = 255; i >= 0; --i) {
        fieldSqr_opt_device(r, t);
        if ((exp[i/64] >> (i % 64)) & 1ULL) {
            unsigned long long temp[8];
            fieldMul_opt_device(t, a, temp);
            modred_barrett_opt_device(temp, r);
        }
        fieldCopy(t, r);
    }
    fieldCopy(t, out);
}

__device__ void pointDoubleJacobian(JacobianPoint &P, JacobianPoint &R) {
    if (P.infinity || isZero256(P.y)) {
        R.infinity = true;
        return;
    }
    unsigned long long t1[8], t3[4], t4[4], t5[4];
    fieldSqr_opt_device(P.z, t1);
    modred_barrett_opt_device(t1, t3); // t3 = z^2
    fieldMul_opt_device(P.x, t3, t1);
    modred_barrett_opt_device(t1, t4); // t4 = x*z^2
    fieldSqr_opt_device(t3, t1);
    modred_barrett_opt_device(t1, t5); // t5 = z^4
    fieldAdd_opt_device(t4, t4, t4); // t4 = 2*x*z^2
    fieldAdd_opt_device(t4, P.x, t3); // t3 = 3*x*z^2
    fieldSqr_opt_device(t3, t1);
    modred_barrett_opt_device(t1, R.x); // R.x = (3*x*z^2)^2
    fieldMul_opt_device(t4, R.x, t1);
    modred_barrett_opt_device(t1, t3); // t3 = 2*x*z^2*(3*x*z^2)^2
    fieldSub_opt_device(R.x, t3, R.x); // R.x = (3*x*z^2)^2 - 2*x*z^2*(3*x*z^2)^2
    fieldAdd_opt_device(P.y, P.y, t3); // t3 = 2*y
    fieldMul_opt_device(t3, P.z, t1);
    modred_barrett_opt_device(t1, R.z); // R.z = 2*y*z
    fieldMul_opt_device(t3, t4, t1);
    modred_barrett_opt_device(t1, t3); // t3 = 2*y*2*x*z^2
    fieldSqr_opt_device(P.y, t1);
    modred_barrett_opt_device(t1, t4); // t4 = y^2
    fieldMul_opt_device(t4, t4, t1);
    modred_barrett_opt_device(t1, t5); // t5 = y^4
    fieldMul_opt_device(t4, P.x, t1);
    modred_barrett_opt_device(t1, t4); // t4 = x*y^2
    fieldAdd_opt_device(t4, t4, t4); // t4 = 2*x*y^2
    fieldSub_opt_device(t3, t4, R.y); // R.y = 2*y*2*x*z^2 - 2*x*y^2
    R.infinity = false;
}

__device__ void pointAddJacobian(const JacobianPoint &P, const JacobianPoint &Q, JacobianPoint &R) {
    if (P.infinity) {
        fieldCopy(Q.x, R.x);
        fieldCopy(Q.y, R.y);
        fieldCopy(Q.z, R.z);
        R.infinity = Q.infinity;
        return;
    }
    if (Q.infinity) {
        fieldCopy(P.x, R.x);
        fieldCopy(P.y, R.y);
        fieldCopy(P.z, R.z);
        R.infinity = P.infinity;
        return;
    }
    unsigned long long t1[8], t3[4], t4[4], t5[4];
    fieldSqr_opt_device(Q.z, t1);
    modred_barrett_opt_device(t1, t3); // t3 = Q.z^2
    fieldMul_opt_device(P.x, t3, t1);
    modred_barrett_opt_device(t1, t4); // t4 = P.x*Q.z^2
    fieldSqr_opt_device(P.z, t1);
    modred_barrett_opt_device(t1, t5); // t5 = P.z^2
    fieldMul_opt_device(Q.x, t5, t1);
    modred_barrett_opt_device(t1, t3); // t3 = Q.x*P.z^2
    fieldSub_opt_device(t4, t3, t4); // t4 = P.x*Q.z^2 - Q.x*P.z^2
    fieldMul_opt_device(Q.z, t4, t1);
    modred_barrett_opt_device(t1, R.z); // R.z = Q.z*(P.x*Q.z^2 - Q.x*P.z^2)
    fieldMul_opt_device(P.z, t4, t1);
    modred_barrett_opt_device(t1, t5); // t5 = P.z*(P.x*Q.z^2 - Q.x*P.z^2)
    fieldSqr_opt_device(t4, t1);
    modred_barrett_opt_device(t1, t3); // t3 = (P.x*Q.z^2 - Q.x*P.z^2)^2
    fieldMul_opt_device(t3, Q.x, t1);
    modred_barrett_opt_device(t1, R.x); // R.x = Q.x*(P.x*Q.z^2 - Q.x*P.z^2)^2
    fieldMul_opt_device(t5, P.y, t1);
    modred_barrett_opt_device(t1, t5); // t5 = P.y*P.z*(P.x*Q.z^2 - Q.x*P.z^2)
    fieldMul_opt_device(Q.y, t3, t1);
    modred_barrett_opt_device(t1, t3); // t3 = Q.y*(P.x*Q.z^2 - Q.x*P.z^2)^2
    fieldSub_opt_device(t5, t3, R.y); // R.y = P.y*P.z*(P.x*Q.z^2 - Q.x*P.z^2) - Q.y*(P.x*Q.z^2 - Q.x*P.z^2)^2
    fieldMul_opt_device(R.x, t4, t1);
    modred_barrett_opt_device(t1, t3); // t3 = (P.x*Q.z^2 - Q.x*P.z^2)^3
    fieldSub_opt_device(R.x, t3, R.x); // R.x = Q.x*(P.x*Q.z^2 - Q.x*P.z^2)^2 - (P.x*Q.z^2 - Q.x*P.z^2)^3
    R.infinity = false;
}

__device__ void scalarMulBaseJacobian(const unsigned long long scalar[4], unsigned long long outX[4], unsigned long long outY[4], const unsigned long long* pre_Gx, const unsigned long long* pre_Gy, const unsigned long long* pre_phiGx, const unsigned long long* pre_phiGy) {
    JacobianPoint R;
    fieldSetZero(R.x);
    fieldSetZero(R.y);
    fieldSetZero(R.z);
    R.infinity = true;
    for (int i = 255; i >= 0; --i) {
        pointDoubleJacobian(R, R);
        if ((scalar[i/64] >> (i % 64)) & 1ULL) {
            JacobianPoint P;
            fieldCopy(pre_Gx + (i * 4), P.x);
            fieldCopy(pre_Gy + (i * 4), P.y);
            fieldSetZero(P.z);
            P.z[0] = 1ULL;
            P.infinity = false;
            pointAddJacobian(R, P, R);
        }
    }
    if (!R.infinity) {
        unsigned long long z_inv[4], t[8], t3[4];
        fieldInv_opt_device(R.z, z_inv);
        fieldSqr_opt_device(z_inv, t);
        modred_barrett_opt_device(t, t3);
        fieldMul_opt_device(R.x, t3, t);
        modred_barrett_opt_device(t, outX);
        fieldMul_opt_device(z_inv, t3, t);
        modred_barrett_opt_device(t, t3);
        fieldMul_opt_device(R.y, t3, t);
        modred_barrett_opt_device(t, outY);
    } else {
        fieldSetZero(outX);
        fieldSetZero(outY);
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