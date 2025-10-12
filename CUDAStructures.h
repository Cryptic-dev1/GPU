#ifndef CUDA_STRUCTURES_H
#define CUDA_STRUCTURES_H

#define WARP_SIZE 32
#define FOUND_NONE  0
#define FOUND_LOCK  1
#define FOUND_READY 2
#define MAX_BATCH_SIZE 512 // Reduced from 1024

struct JacobianPoint {
    unsigned long long x[4];
    unsigned long long y[4];
    unsigned long long z[4];
    bool infinity;
};

// Constants (little-endian)
__device__ __constant__ uint8_t  c_target_hash160[20];
__device__ __constant__ uint32_t c_target_prefix;
__device__ __constant__ unsigned long long Gx_d[4] = {
    0x9f2815b16f81798ULL, 0x29bfcdb2dce28d95ULL, 0x55a06295ce870b07ULL, 0x79be667ef9dcbbacULL
};
__device__ __constant__ unsigned long long Gy_d[4] = {
    0x9c47d08ffb10d4b8ULL, 0xfd17b448a6855419ULL, 0x5da4fbfc0e1108a8ULL, 0x483ada7726a3c465ULL
};

// Precomputed tables (2^8 = 256 points, ~2 MB total)
#define PRECOMPUTE_WINDOW 8 // Reduced from 16
#define PRECOMPUTE_SIZE (1LL << PRECOMPUTE_WINDOW)

// Batch point tables for fused_ec_hash
__device__ __constant__ unsigned long long c_Gx[(MAX_BATCH_SIZE/2) * 4];
__device__ __constant__ unsigned long long c_Gy[(MAX_BATCH_SIZE/2) * 4];

// secp256k1 constants (little-endian)
__device__ __constant__ unsigned long long c_p[4] = {0xfffffc2fULL, 0xffffffffULL, 0xffffffffULL, 0xffffffffULL};
__device__ __constant__ unsigned long long c_n[4] = {0xd0364141ULL, 0xbaaedce6ULL, 0xfffffffeULL, 0xffffffffULL};
__device__ __constant__ unsigned long long c_beta[4] = {0x6b3c4f7eULL, 0x8de6997dULL, 0x7cf27b18ULL, 0x00000000ULL};
__device__ __constant__ unsigned long long c_mu[5] = {0x1000003d1ULL, 0ULL, 0ULL, 0ULL, 1ULL};

// GLV coefficients (precomputed, little-endian)
__device__ __constant__ unsigned long long c_a1[4] = {0xe86c90e4ULL, 0x3086d221ULL, 0ULL, 0ULL};
__device__ __constant__ unsigned long long c_a2[4] = {0x657c1108ULL, 0x114ca50fULL, 0ULL, 0ULL};
__device__ __constant__ unsigned long long c_b1[4] = {0x90ab805fULL, 0x1bbaf310ULL, 0xaef9b7d6ULL, 0ULL};
__device__ __constant__ unsigned long long c_b2[4] = {0xe86c90e4ULL, 0x3086d221ULL, 0ULL, 0ULL};

__device__ int found_flag = 0;

#define CUDA_CHECK(ans) do { cudaError_t err = ans; if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE); } } while(0)

// Utility functions
__host__ __device__ bool ge256(const unsigned long long a[4], const unsigned long long b[4]);
__host__ __device__ bool ge256_u64(const unsigned long long a[4], unsigned long long b);

__global__ void scalarMulKernelBase(const unsigned long long* scalars_in, unsigned long long* outX, unsigned long long* outY, int N, unsigned long long* d_pre_Gx, unsigned long long* d_pre_Gy, unsigned long long* d_pre_phiGx, unsigned long long* d_pre_phiGy);
__global__ void precompute_table_kernel(JacobianPoint base, unsigned long long* pre_x, unsigned long long* pre_y, unsigned long long size);

#endif // CUDA_STRUCTURES_H