#ifndef CUDA_STRUCTURES_H
#define CUDA_STRUCTURES_H

#define WARP_SIZE 32
#define FOUND_NONE  0
#define FOUND_LOCK  1
#define FOUND_READY 2

struct FoundResult {
    int      threadId;
    int      iter;
    uint64_t scalar[4];
    uint64_t Rx[4];
    uint64_t Ry[4];
};

struct JacobianPoint {
    uint64_t x[4];
    uint64_t y[4];
    uint64_t z[4];
    bool infinity;
};

// Constants (little-endian)
__device__ __constant__ uint8_t  c_target_hash160[20];
__device__ __constant__ uint32_t c_target_prefix;
__device__ __constant__ uint64_t Gx_d[4] = {0x59f2815b, 0x0ea3fe7f, 0x2e6ff0b0, 0x79e81dc6};
__device__ __constant__ uint64_t Gy_d[4] = {0x4fe342e2, 0xe0fa9e5b, 0x7c0cad3c, 0x9f07d8fb};

// Precomputed tables (2^24 = 16,777,216 points, ~1GB per table)
#define PRECOMPUTE_WINDOW 24
#define PRECOMPUTE_SIZE (1LL << PRECOMPUTE_WINDOW)
__global__ uint64_t* d_pre_Gx;    // Dynamically allocated on device
__global__ uint64_t* d_pre_Gy;
__global__ uint64_t* d_pre_phiGx;
__global__ uint64_t* d_pre_phiGy;

// secp256k1 constants (little-endian)
__device__ __constant__ uint64_t c_p[4] = {0xfffffc2f, 0xffffffff, 0xffffffff, 0xffffffff};
__device__ __constant__ uint64_t c_n[4] = {0xd0364141, 0xbaaedce6, 0xfffffffe, 0xffffffff};
__device__ __constant__ uint64_t c_lambda[4] = {0x1b23bd72, 0x20816678, 0x8812645a, 0x0c05c30e};
__device__ __constant__ uint64_t c_beta[4] = {0xc2a38c8f, 0x488e4478, 0xcdb4986e, 0x7c07107a};

// Barrett reduction constant mu = floor(2^512 / p)
__device__ __constant__ uint64_t c_mu[5] = {0x1000003d1, 0, 0, 0, 1};

// GLV coefficients (precomputed, little-endian)
__device__ __constant__ uint64_t c_a1[4] = {0xe86c90e4, 0x3086d221, 0, 0};
__device__ __constant__ uint64_t c_a2[4] = {0x657c1108, 0x114ca50f, 0, 0};
__device__ __constant__ uint64_t c_b1[4] = {0x06f547fa, 0xe4437ed6, 0, 0};
__device__ __constant__ uint64_t c_b2[4] = {0xe86c90e4, 0x3086d221, 0, 0};

__device__ FoundResult found_result;
__device__ int found_flag = 0;

#define CUDA_CHECK(ans) do { cudaError_t err = ans; if (err != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE); } } while(0)

__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);
__global__ void fused_ec_hash(
    JacobianPoint* P, JacobianPoint* R, uint64_t* start_scalars, uint64_t* counts256,
    uint64_t threadsTotal, uint32_t batch_size, uint32_t max_batches_per_launch,
    int* d_found_flag, FoundResult* d_found_result, unsigned long long* hashes_accum,
    unsigned int* d_any_left
);
__global__ void precompute_table_kernel(JacobianPoint base, uint64_t* pre_x, uint64_t* pre_y, uint64_t size);

#endif // CUDA_STRUCTURES_H