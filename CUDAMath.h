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

__host__ __device__ void fieldCopy(unsigned long long a[4], const unsigned long long b[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = b[i];
}

__host__ __device__ void fieldAdd_opt_device(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4], bool* carry) {
    unsigned long long temp = 0ULL;
    *carry = false;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned long long sum = a[i] + b[i] + temp;
        out[i] = sum;
        temp = (sum < a[i] || sum < b[i]) ? 1ULL : 0ULL;
    }
    *carry = (temp != 0ULL);
}

__host__ __device__ void fieldSub_opt_device(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4], bool* borrow) {
    unsigned long long temp = 0ULL;
    *borrow = false;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned long long diff = a[i] - b[i] - temp;
        out[i] = diff;
        temp = (diff > a[i]) ? 1ULL : 0ULL;
    }
    *borrow = (temp != 0ULL);
}

__device__ void mul256_device(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[8]) {
    unsigned long long lo, hi;
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = 0ULL;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            UMULLO(lo, a[i], b[j]);
            UMULHI(hi, a[i], b[j]);
            unsigned long long sum = out[i + j] + lo;
            out[i + j] = sum;
            unsigned long long carry = (sum < lo) ? 1ULL : 0ULL;
            out[i + j + 1] += hi + carry;
        }
    }
}

__device__ void mul_high_device(const unsigned long long a[4], const unsigned long long b[5], unsigned long long out[5]) {
    unsigned long long lo, hi;
    #pragma unroll
    for (int i = 0; i < 5; ++i) out[i] = 0ULL;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 5; ++j) {
            UMULLO(lo, a[i], b[j]);
            UMULHI(hi, a[i], b[j]);
            unsigned long long sum = out[i + j] + lo;
            out[i + j] = sum;
            if (i + j + 1 < 5) out[i + j + 1] += hi + (sum < lo ? 1ULL : 0ULL);
        }
    }
}

__device__ void modred_barrett_opt_device(const unsigned long long input[8], unsigned long long out[4]) {
    unsigned long long q[5], tmp[8];
    bool carry;
    mul_high_device(input + 4, c_mu, q);
    mul256_device(q, c_p, tmp);
    fieldSub_opt_device(input, tmp, out, &carry);
    if (_IsNegative(out)) {
        fieldAdd_opt_device(out, c_p, out, &carry);
    }
}

__device__ void fieldMul_opt_device(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]) {
    unsigned long long tmp[8];
    mul256_device(a, b, tmp);
    modred_barrett_opt_device(tmp, out);
}

__device__ void fieldSqr_opt_device(const unsigned long long a[4], unsigned long long out[4]) {
    fieldMul_opt_device(a, a, out);
}

__device__ void fieldInvFermat_device(const unsigned long long a[4], unsigned long long out[4]) {
    unsigned long long t[4], t2[4];
    fieldSqr_opt_device(a, t);
    fieldMul_opt_device(t, a, t2);
    fieldSqr_opt_device(t2, t);
    fieldSqr_opt_device(t, t);
    fieldMul_opt_device(t, a, t2);
    fieldSqr_opt_device(t2, t);
    fieldSqr_opt_device(t, t);
    fieldSqr_opt_device(t, t);
    fieldSqr_opt_device(t, t);
    fieldMul_opt_device(t, t2, t);
    fieldSqr_opt_device(t, t);
    for (int i = 0; i < 5; ++i) fieldSqr_opt_device(t, t);
    fieldMul_opt_device(t, t2, t2);
    fieldSqr_opt_device(t2, t);
    for (int i = 0; i < 7; ++i) fieldSqr_opt_device(t, t);
    fieldMul_opt_device(t, t2, t);
    fieldSqr_opt_device(t, t);
    for (int i = 0; i < 14; ++i) fieldSqr_opt_device(t, t);
    fieldMul_opt_device(t, t2, t);
    fieldSqr_opt_device(t, t);
    for (int i = 0; i < 29; ++i) fieldSqr_opt_device(t, t);
    fieldMul_opt_device(t, t2, t);
    fieldSqr_opt_device(t, t);
    for (int i = 0; i < 59; ++i) fieldSqr_opt_device(t, t);
    fieldMul_opt_device(t, t2, t);
    fieldSqr_opt_device(t, t);
    for (int i = 0; i < 119; ++i) fieldSqr_opt_device(t, t);
    fieldMul_opt_device(t, t2, t);
    fieldSqr_opt_device(t, t);
    fieldSqr_opt_device(t, t);
    fieldMul_opt_device(t, a, out);
}

__device__ void batch_modinv_fermat(unsigned long long* a, int n, unsigned long long* prefix) {
    int lane = threadIdx.x % WARP_SIZE;
    extern __shared__ unsigned long long smem[];
    unsigned long long* products = smem;
    unsigned long long* results = smem + n * 4;

    // Debug z_values before inversion
    if (lane < n) {
        printf("batch_modinv_fermat: lane=%d, a[%d]=%llx:%llx:%llx:%llx\n",
               lane, lane, a[lane*4], a[lane*4+1], a[lane*4+2], a[lane*4+3]);
    }
    __syncwarp();

    // Compute prefix products
    unsigned long long my_a[4];
    fieldCopy(my_a, a + lane * 4);
    fieldCopy(products + lane * 4, my_a);
    prefix[0] = 1ULL;
    for (int i = 0; i < n; ++i) {
        if (lane == i) {
            printf("batch_modinv_fermat: a[%d]=%llx:%llx:%llx:%llx, prefix[%d]=%llx:%llx:%llx:%llx\n",
                   i, my_a[0], my_a[1], my_a[2], my_a[3], i + 1, prefix[(i + 1) * 4], prefix[(i + 1) * 4 + 1], prefix[(i + 1) * 4 + 2], prefix[(i + 1) * 4 + 3]);
        }
        __syncwarp();
    }

    // Compute inverse of product
    unsigned long long prod_inv[4];
    fieldInvFermat_device(products + (n - 1) * 4, prod_inv);
    if (lane == 0) {
        printf("batch_modinv_fermat: prefix[%d]=%llx:%llx:%llx:%llx, all_zero=%d\n",
               n, prefix[n * 4], prefix[n * 4 + 1], prefix[n * 4 + 2], prefix[n * 4 + 3], isZero256(prod_inv));
    }
    __syncwarp();

    // Compute individual inverses
    if (lane < n) {
        unsigned long long temp[4];
        if (lane == 0) {
            fieldCopy(prod_inv, results);
        } else {
            fieldMul_opt_device(prod_inv, a + (lane - 1) * 4, temp);
            fieldCopy(temp, results + lane * 4);
        }
    }
    __syncwarp();

    if (lane < n) {
        fieldCopy(results + lane * 4, prefix + (lane + 1) * 4);
    }
}

__device__ void pointDoubleJacobian(JacobianPoint* P, JacobianPoint* R) {
    if (P->infinity) {
        R->infinity = true;
        return;
    }
    unsigned long long z2[4], z4[4], t1[4], t2[4], t3[4], t4[4];
    bool borrow;
    fieldSqr_opt_device(P->z, z2);
    fieldSqr_opt_device(z2, z4);
    fieldMul_opt_device(P->x, z2, t1);
    fieldAdd_opt_device(t1, t1, t2, &borrow);
    fieldMul_opt_device(t2, t2, t3);
    fieldSub_opt_device(P->x, z4, t4, &borrow);
    fieldAdd_opt_device(P->x, z4, t1, &borrow);
    fieldMul_opt_device(t4, t1, t2);
    fieldMul_opt_device(t2, t3, t1);
    fieldSqr_opt_device(t3, t2);
    fieldSub_opt_device(t2, t1, R->x, &borrow);
    fieldSub_opt_device(t1, R->x, t2, &borrow);
    fieldMul_opt_device(t2, t3, t1);
    fieldMul_opt_device(P->y, z2, t2);
    fieldMul_opt_device(t2, P->z, R->z);
    fieldMul_opt_device(P->y, P->y, t3);
    fieldMul_opt_device(t3, z4, t4);
    fieldSub_opt_device(t1, t4, R->y, &borrow);
    R->infinity = false;
}

__device__ void pointAddJacobian(const JacobianPoint* P, const JacobianPoint* Q, JacobianPoint* R) {
    if (P->infinity) {
        fieldCopy(Q->x, R->x);
        fieldCopy(Q->y, R->y);
        fieldCopy(Q->z, R->z);
        R->infinity = Q->infinity;
        return;
    }
    if (Q->infinity) {
        fieldCopy(P->x, R->x);
        fieldCopy(P->y, R->y);
        fieldCopy(P->z, R->z);
        R->infinity = P->infinity;
        return;
    }
    unsigned long long z1z1[4], z2z2[4], u1[4], u2[4], s1[4], s2[4], h[4], i[4], j[4], r[4], v[4];
    bool borrow;
    fieldSqr_opt_device(P->z, z1z1);
    fieldSqr_opt_device(Q->z, z2z2);
    // Copy const inputs to non-const arrays to allow modification
    unsigned long long Px[4], Py[4], Qx[4], Qy[4];
    fieldCopy(Px, P->x);
    fieldCopy(Py, P->y);
    fieldCopy(Qx, Q->x);
    fieldCopy(Qy, Q->y);
    fieldMul_opt_device(Px, z2z2, u1);
    fieldMul_opt_device(Qx, z1z1, u2);
    fieldMul_opt_device(Py, z2z2, s1);
    fieldMul_opt_device(Qy, z1z1, s2);
    fieldSub_opt_device(u2, u1, h, &borrow);
    fieldSqr_opt_device(h, i);
    fieldMul_opt_device(h, i, j);
    fieldSub_opt_device(s2, s1, r, &borrow);
    fieldMul_opt_device(r, r, r);
    fieldMul_opt_device(u1, i, v);
    fieldSqr_opt_device(r, R->x);
    fieldSub_opt_device(R->x, j, R->x, &borrow);
    fieldSub_opt_device(R->x, v, R->x, &borrow);
    fieldSub_opt_device(v, R->x, v, &borrow);
    fieldMul_opt_device(s1, j, s1);
    fieldMul_opt_device(v, r, v);
    fieldSub_opt_device(v, s1, R->y, &borrow);
    fieldMul_opt_device(P->z, Q->z, R->z);
    fieldMul_opt_device(R->z, h, R->z);
    R->infinity = false;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointAddJacobian: output R.x=%llx:%llx:%llx:%llx, R.y=%llx:%llx:%llx:%llx, R.z=%llx:%llx:%llx:%llx, R.infinity=%d\n",
               R->x[0], R->x[1], R->x[2], R->x[3], R->y[0], R->y[1], R->y[2], R->y[3], R->z[0], R->z[1], R->z[2], R->z[3], R->infinity);
    }
}

__device__ void pointAddMixed(const JacobianPoint* P, const unsigned long long Qx[4], const unsigned long long Qy[4], JacobianPoint* R) {
    if (P->infinity) {
        fieldCopy(Qx, R->x);
        fieldCopy(Qy, R->y);
        fieldSetZero(R->z);
        R->z[0] = 1ULL;
        R->infinity = false;
        return;
    }
    unsigned long long z1z1[4], u2[4], s2[4], h[4], i[4], j[4], r[4], v[4];
    bool borrow;
    fieldSqr_opt_device(P->z, z1z1);
    // Copy const inputs to non-const arrays to allow modification
    unsigned long long Qx_copy[4], Qy_copy[4], Px[4], Py[4];
    fieldCopy(Qx_copy, Qx);
    fieldCopy(Qy_copy, Qy);
    fieldCopy(Px, P->x);
    fieldCopy(Py, P->y);
    fieldMul_opt_device(Qx_copy, z1z1, u2);
    fieldMul_opt_device(Qy_copy, z1z1, s2);
    fieldSub_opt_device(u2, Px, h, &borrow);
    fieldSqr_opt_device(h, i);
    fieldMul_opt_device(h, i, j);
    fieldSub_opt_device(s2, Py, r, &borrow);
    fieldMul_opt_device(r, r, r);
    fieldMul_opt_device(Px, i, v);
    fieldSqr_opt_device(r, R->x);
    fieldSub_opt_device(R->x, j, R->x, &borrow);
    fieldSub_opt_device(R->x, v, R->x, &borrow);
    fieldSub_opt_device(v, R->x, v, &borrow);
    fieldMul_opt_device(Py, j, s2);
    fieldMul_opt_device(v, r, v);
    fieldSub_opt_device(v, s2, R->y, &borrow);
    fieldMul_opt_device(P->z, h, R->z);
    R->infinity = false;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("pointAddMixed: z1z1=%llx:%llx:%llx:%llx, u2=%llx:%llx:%llx:%llx, s2=%llx:%llx:%llx:%llx, h=%llx:%llx:%llx:%llx, i=%llx:%llx:%llx:%llx, j=%llx:%llx:%llx:%llx, r=%llx:%llx:%llx:%llx, v=%llx:%llx:%llx:%llx\n",
               z1z1[0], z1z1[1], z1z1[2], z1z1[3], u2[0], u2[1], u2[2], u2[3], s2[0], s2[1], s2[2], s2[3],
               h[0], h[1], h[2], h[3], i[0], i[1], i[2], i[3], j[0], j[1], j[2], j[3], r[0], r[1], r[2], r[3], v[0], v[1], v[2], v[3]);
        printf("pointAddMixed: output R.x=%llx:%llx:%llx:%llx, R.y=%llx:%llx:%llx:%llx, R.z=%llx:%llx:%llx:%llx, R.infinity=%d\n",
               R->x[0], R->x[1], R->x[2], R->x[3], R->y[0], R->y[1], R->y[2], R->y[3], R->z[0], R->z[1], R->z[2], R->z[3], R->infinity);
    }
}

__device__ void scalarMulBaseJacobian(const unsigned long long scalar[4], unsigned long long outX[4], unsigned long long outY[4], const unsigned long long* pre_Gx, const unsigned long long* pre_Gy, const unsigned long long* pre_phiGx, const unsigned long long* pre_phiGy) {
    JacobianPoint R;
    R.infinity = true;
    fieldSetZero(R.x); fieldSetZero(R.y); fieldSetZero(R.z);

    for (int i = 0; i < 256; ++i) {
        if (!R.infinity) {
            pointDoubleJacobian(&R, &R);
        }
        unsigned long long bit = (scalar[i / 64] >> (i % 64)) & 1ULL;
        if (bit) {
            pointAddMixed(&R, pre_Gx + (i % PRECOMPUTE_SIZE) * 4, pre_Gy + (i % PRECOMPUTE_SIZE) * 4, &R);
        }
    }

    fieldCopy(R.x, outX);
    fieldCopy(R.y, outY);

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("scalarMulBaseJacobian: R.x=%llx:%llx:%llx:%llx, R.y=%llx:%llx:%llx:%llx, R.z=%llx:%llx:%llx:%llx, R.infinity=%d, outX=%llx:%llx:%llx:%llx, outY=%llx:%llx:%llx:%llx\n",
               R.x[0], R.x[1], R.x[2], R.x[3], R.y[0], R.y[1], R.y[2], R.y[3], R.z[0], R.z[1], R.z[2], R.z[3], R.infinity,
               outX[0], outX[1], outX[2], outX[3], outY[0], outY[1], outY[2], outY[3]);
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
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("scalarMulKernelBase: idx=%d, outX=%llx:%llx:%llx:%llx, outY=%llx:%llx:%llx:%llx\n",
               idx, outX[idx*4], outX[idx*4+1], outX[idx*4+2], outX[idx*4+3],
               outY[idx*4], outY[idx*4+1], outY[idx*4+2], outY[idx*4+3]);
    }
}

__global__ void precompute_table_kernel(JacobianPoint base, unsigned long long* pre_x, unsigned long long* pre_y, unsigned long long size) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    JacobianPoint P = base;
    if (threadIdx.x == 0 && blockIdx.x == 0 && idx < 4) {
        printf("precompute_table_kernel: base.x=%llx:%llx:%llx:%llx, base.y=%llx:%llx:%llx:%llx, base.z=%llx:%llx:%llx:%llx, base.infinity=%d\n",
               base.x[0], base.x[1], base.x[2], base.x[3], base.y[0], base.y[1], base.y[2], base.y[3],
               base.z[0], base.z[1], base.z[2], base.z[3], base.infinity);
    }
    for (unsigned long long bit = 0; bit < idx; ++bit) {
        if (bit % 2 == 0) {
            pointDoubleJacobian(&P, &P);
        } else {
            pointAddJacobian(&P, &base, &P);
        }
        if (threadIdx.x == 0 && blockIdx.x == 0 && idx < 4) {
            printf("precompute_table_kernel: idx=%llu, bit=%llu, P.x=%llx:%llx:%llx:%llx, P.y=%llx:%llx:%llx:%llx\n",
                   idx, bit, P.x[0], P.x[1], P.x[2], P.x[3], P.y[0], P.y[1], P.y[2], P.y[3]);
        }
    }
    fieldCopy(P.x, pre_x + idx * 4);
    fieldCopy(P.y, pre_y + idx * 4);
    if (threadIdx.x == 0 && blockIdx.x == 0 && idx < 4) {
        printf("precompute_table_kernel: idx=%llu, pre_x=%llx:%llx:%llx:%llx, pre_y=%llx:%llx:%llx:%llx\n",
               idx, pre_x[idx*4], pre_x[idx*4+1], pre_x[idx*4+2], pre_x[idx*4+3],
               pre_y[idx*4], pre_y[idx*4+1], pre_y[idx*4+2], pre_y[idx*4+3]);
    }
}

#endif // CUDA_MATH_H