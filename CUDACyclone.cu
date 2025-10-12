#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>

#include "CUDAMath.h"
#include "CUDAStructures.h"
#include "CUDAUtils.h"

// Verify unsigned long long size
static_assert(sizeof(unsigned long long) == 8, "unsigned long long must be 64 bits");

// Local FoundResult struct for both host and device
struct FoundResult {
    int threadId;
    int iter;
    unsigned long long scalar_val[4];
    unsigned long long Rx_val[4];
    unsigned long long Ry_val[4];
};

// Declarations for functions defined in CUDAUtils.cu
__device__ unsigned long long warp_reduce_add_ull(unsigned long long val);
__device__ bool hash160_prefix_equals(const uint8_t h20[20], uint32_t target_prefix);
__device__ bool hash160_matches_prefix_then_full(const uint8_t h20[20], const uint8_t target[20], uint32_t target_prefix);
__device__ void sub256_u64_inplace(unsigned long long a[4], unsigned long long b);
__device__ void inc256_device(unsigned long long a[4], unsigned long long b);
__host__ bool hexToLE64(const std::string& hex, unsigned long long out[4]);
__host__ void sub256(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]);
__host__ void add256_u64(const unsigned long long a[4], unsigned long long b, unsigned long long out[4]);
__host__ bool decode_p2pkh_address(const std::string& address, uint8_t hash160[20]);
__host__ long double ld_from_u256(const unsigned long long a[4]);

// Declaration for getHash160_33_from_limbs (assumed defined in CUDAHash.cu)
__device__ void getHash160_33_from_limbs(uint8_t prefix, const unsigned long long x[4], uint8_t h20[20]);

// Namespace for utility functions
namespace CryptoUtils {
    std::string formatHex256(const unsigned long long* limbs) {
        std::ostringstream oss;
        oss << std::hex << std::uppercase << std::setfill('0');
        for (int i = 3; i >= 0; --i) {
            oss << std::setw(16) << limbs[i];
        }
        return oss.str();
    }

    std::string formatCompressedPubHex(const unsigned long long* Rx, const unsigned long long* Ry) {
        uint8_t out[33];
        out[0] = (Ry[0] & 1ULL) ? 0x03 : 0x02;
        int off = 1;
        for (int limb = 3; limb >= 0; --limb) {
            unsigned long long v = Rx[limb];
            out[off+0] = (uint8_t)(v >> 56); out[off+1] = (uint8_t)(v >> 48);
            out[off+2] = (uint8_t)(v >> 40); out[off+3] = (uint8_t)(v >> 32);
            out[off+4] = (uint8_t)(v >> 24); out[off+5] = (uint8_t)(v >> 16);
            out[off+6] = (uint8_t)(v >> 8);  out[off+7] = (uint8_t)(v >> 0);
            off += 8;
        }
        static const char* hexd = "0123456789ABCDEF";
        std::string s;
        s.resize(66);
        for (int i = 0; i < 33; ++i) {
            s[2*i] = hexd[(out[i] >> 4) & 0xF];
            s[2*i+1] = hexd[out[i] & 0xF];
        }
        return s;
    }
}

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

__device__ __forceinline__ int load_found_flag_relaxed(const int* p) {
    return *((const volatile int*)p);
}

__device__ __forceinline__ bool warp_found_ready(const int* __restrict__ d_found_flag, unsigned full_mask, unsigned lane) {
    int f = 0;
    if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
    f = __shfl_sync(full_mask, f, 0);
    return f == FOUND_READY;
}

// Test kernel to verify c_Gx and c_Gy
__global__ void test_constant_memory(unsigned long long* out, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size / 2) return;
    for (int i = 0; i < 4; ++i) {
        out[idx * 8 + i] = c_Gx[idx * 4 + i];
        out[idx * 8 + i + 4] = c_Gy[idx * 4 + i];
    }
    if (idx == 0 && threadIdx.x == 0 && blockIdx.x == 0) {
        printf("test_constant_memory: c_Gx[0]=%llx:%llx:%llx:%llx, c_Gy[0]=%llx:%llx:%llx:%llx\n",
               c_Gx[0], c_Gx[1], c_Gx[2], c_Gx[3], c_Gy[0], c_Gy[1], c_Gy[2], c_Gy[3]);
    }
}

__global__ void compute_phi_base_kernel(JacobianPoint base, JacobianPoint &phi_base) {
    unsigned long long tmp[8];
    unsigned long long tmp2[8];
    fieldMul_opt_device(c_beta, base.x, tmp);
    modred_barrett_opt_device(tmp, tmp2);
    fieldCopy(tmp2, phi_base.x);
    fieldCopy(base.y, phi_base.y);
    phi_base.infinity = base.infinity;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("compute_phi_base_kernel completed\n");
    }
}

void precompute_g_table_gpu(JacobianPoint base, unsigned long long* pre_x, unsigned long long* pre_y, int size, cudaStream_t stream, int blocks, int threads) {
    precompute_table_kernel<<<blocks, threads, 0, stream>>>(base, pre_x, pre_y, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
}

__global__ void fused_ec_hash(
    const unsigned long long* scalars,
    unsigned long long* P,
    unsigned long long* R,
    int batch_size,
    const unsigned long long* pre_Gx,
    const unsigned long long* pre_Gy,
    const unsigned long long* pre_phiGx,
    const unsigned long long* pre_phiGy,
    unsigned long long* hashes_accum,
    unsigned int* any_left,
    int* found_flag,
    FoundResult* found_result
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    unsigned full_mask = 0xFFFFFFFF;

    if (idx >= batch_size) return;

    unsigned long long my_scalar[4];
    fieldCopy(my_scalar, scalars + idx * 4);

    unsigned long long Rx[4], Ry[4];
    scalarMulBaseJacobian(my_scalar, Rx, Ry, pre_Gx, pre_Gy, pre_phiGx, pre_phiGy);

    unsigned long long z_values[batch_size * 4];
    unsigned long long x_affine[batch_size * 4];
    unsigned long long y_affine[batch_size * 4];
    unsigned long long prefix[batch_size * 4 + 4];

    if (lane < batch_size / 2) {
        int batch_idx = idx / (batch_size / 2);
        int local_idx = idx % (batch_size / 2);
        fieldCopy(Rx, P + idx * 8);
        fieldCopy(Ry, P + idx * 8 + 4);
        fieldCopy(Ry, z_values + (batch_idx * (batch_size / 2) + local_idx) * 4); // Using Ry as z for testing
        if (lane == 0) {
            printf("fused_ec_hash: Block %d, lane %d, writing z_values[%d]=%llx:%llx:%llx:%llx\n",
                   blockIdx.x, lane, batch_idx * (batch_size / 2) + local_idx,
                   z_values[(batch_idx * (batch_size / 2) + local_idx) * 4],
                   z_values[(batch_idx * (batch_size / 2) + local_idx) * 4 + 1],
                   z_values[(batch_idx * (batch_size / 2) + local_idx) * 4 + 2],
                   z_values[(batch_idx * (batch_size / 2) + local_idx) * 4 + 3]);
        }
    }
    __syncthreads();

    batch_modinv_fermat(z_values, batch_size / 2, prefix);

    if (lane < batch_size / 2) {
        int batch_idx = idx / (batch_size / 2);
        int local_idx = idx % (batch_size / 2);
        unsigned long long z_inv[4];
        fieldCopy(z_inv, prefix + (local_idx + 1) * 4);
        fieldMul_opt_device(P + idx * 8, z_inv, x_affine + idx * 4);
        fieldMul_opt_device(P + idx * 8 + 4, z_inv, y_affine + idx * 4);
    }
    __syncthreads();

    uint8_t pubkeys[batch_size * 33];
    uint8_t hashes[batch_size * 20];
    if (lane < batch_size / 2) {
        int batch_idx = idx / (batch_size / 2);
        int local_idx = idx % (batch_size / 2);
        uint8_t prefix = (y_affine[idx * 4] & 1ULL) ? 0x03 : 0x02;
        getHash160_33_from_limbs(prefix, x_affine + idx * 4, hashes + (batch_idx * (batch_size / 2) + local_idx) * 20);
        if (hash160_matches_prefix_then_full(hashes + (batch_idx * (batch_size / 2) + local_idx) * 20, c_target_hash160, c_target_prefix)) {
            int old = atomicCAS(found_flag, FOUND_NONE, FOUND_LOCK);
            if (old == FOUND_NONE) {
                found_result->threadId = idx;
                found_result->iter = 0;
                fieldCopy(my_scalar, found_result->scalar_val);
                fieldCopy(Rx, found_result->Rx_val);
                fieldCopy(Ry, found_result->Ry_val);
                *found_flag = FOUND_READY;
            }
        }
    }
    __syncthreads();

    unsigned long long my_hashes = (lane < batch_size / 2) ? 1ULL : 0ULL;
    my_hashes = warp_reduce_add_ull(my_hashes);
    if (lane == 0) {
        atomicAdd(hashes_accum, my_hashes);
    }

    if (lane == 0) {
        unsigned long long my_scalar_max[4];
        fieldCopy(my_scalar_max, scalars + (batch_size - 1) * 4);
        *any_left = ge256(my_scalar_max, c_n) ? 0 : 1;
    }
}

int main(int argc, char* argv[]) {
    std::string range_arg, address_arg, grid_arg;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--range" && i + 1 < argc) range_arg = argv[++i];
        else if (arg == "--address" && i + 1 < argc) address_arg = argv[++i];
        else if (arg == "--grid" && i + 1 < argc) grid_arg = argv[++i];
        else if (arg == "--verbose") verbose = true;
    }

    if (range_arg.empty() || address_arg.empty()) {
        std::cerr << "Usage: " << argv[0] << " --range START:END --address P2PKH_ADDRESS [--grid BLOCKS,THREADS] [--verbose]\n";
        return EXIT_FAILURE;
    }

    unsigned long long range_start[4], range_end[4];
    if (!hexToLE64(range_arg.substr(0, range_arg.find(':')), range_start) ||
        !hexToLE64(range_arg.substr(range_arg.find(':') + 1), range_end)) {
        std::cerr << "Invalid range format. Use HEX_START:HEX_END\n";
        return EXIT_FAILURE;
    }

    uint8_t target_hash160[20];
    if (!decode_p2pkh_address(address_arg, target_hash160)) {
        std::cerr << "Invalid P2PKH address\n";
        return EXIT_FAILURE;
    }

    uint32_t target_prefix = ((uint32_t)target_hash160[0] << 0) | ((uint32_t)target_hash160[1] << 8) |
                             ((uint32_t)target_hash160[2] << 16) | ((uint32_t)target_hash160[3] << 24);

    CUDA_CHECK(cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_target_prefix, &target_prefix, sizeof(uint32_t)));

    if (verbose) {
        unsigned long long h_n[4], h_beta[4], h_b1[4], h_b2[4], h_a1[4], h_a2[4], h_p[4], h_mu[5];
        CUDA_CHECK(cudaMemcpyFromSymbol(h_n, c_n, 4 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpyFromSymbol(h_beta, c_beta, 4 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpyFromSymbol(h_b1, c_b1, 4 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpyFromSymbol(h_b2, c_b2, 4 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpyFromSymbol(h_a1, c_a1, 4 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpyFromSymbol(h_a2, c_a2, 4 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpyFromSymbol(h_p, c_p, 4 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpyFromSymbol(h_mu, c_mu, 5 * sizeof(unsigned long long)));
        std::cout << "c_n: " << std::hex << h_n[0] << ":" << h_n[1] << ":" << h_n[2] << ":" << h_n[3] << "\n";
        std::cout << "c_beta: " << h_beta[0] << ":" << h_beta[1] << ":" << h_beta[2] << ":" << h_beta[3] << "\n";
        std::cout << "c_b1: " << h_b1[0] << ":" << h_b1[1] << ":" << h_b1[2] << ":" << h_b1[3] << "\n";
        std::cout << "c_b2: " << h_b2[0] << ":" << h_b2[1] << ":" << h_b2[2] << ":" << h_b2[3] << "\n";
        std::cout << "c_a1: " << h_a1[0] << ":" << h_a1[1] << ":" << h_a1[2] << ":" << h_a1[3] << "\n";
        std::cout << "c_a2: " << h_a2[0] << ":" << h_a2[1] << ":" << h_a2[2] << ":" << h_a2[3] << "\n";
        std::cout << "c_p: " << h_p[0] << ":" << h_p[1] << ":" << h_p[2] << ":" << h_p[3] << "\n";
        std::cout << "c_mu: " << h_mu[0] << ":" << h_mu[1] << ":" << h_mu[2] << ":" << h_mu[3] << ":" << h_mu[4] << "\n";
    }

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "======== PrePhase: GPU Information ====================\n";
    std::cout << "Device               : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << "SM                   : " << prop.multiProcessorCount << "\n";

    int THREADS = 32;
    int BLOCKS = 64;
    if (!grid_arg.empty()) {
        std::sscanf(grid_arg.c_str(), "%d,%d", &BLOCKS, &THREADS);
    }
    std::cout << "Parsed grid: blocks=" << BLOCKS << ", threadsPerBlock=" << THREADS << "\n";

    int batch_size = 8;
    if (batch_size > MAX_BATCH_SIZE) batch_size = MAX_BATCH_SIZE;
    int batches_per_sm = (prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor) / (BLOCKS * THREADS);
    std::cout << "ThreadsPerBlock      : " << THREADS << "\n";
    std::cout << "Blocks               : " << BLOCKS << "\n";
    std::cout << "Points batch size    : " << batch_size << "\n";
    std::cout << "Batches/SM           : " << batches_per_sm << "\n";
    std::cout << "Precomputed tables    : 2^" << PRECOMPUTE_WINDOW << " points (~" << (PRECOMPUTE_SIZE * 8 * sizeof(unsigned long long) / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "Memory utilization   : " << std::fixed << std::setprecision(1)
              << (PRECOMPUTE_SIZE * 8 * sizeof(unsigned long long) / (double)prop.totalGlobalMem) * 100.0
              << "% (" << human_bytes(PRECOMPUTE_SIZE * 8 * sizeof(unsigned long long)) << " / "
              << human_bytes(prop.totalGlobalMem) << ")\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Total threads        : " << BLOCKS * THREADS << "\n";
    std::cout << "Batch size: " << batch_size << "\n";

    unsigned long long* d_start_scalars, *d_counts256, *d_P, *d_R, *d_hashes_accum, *d_pre_Gx_local, *d_pre_Gy_local, *d_pre_phiGx_local, *d_pre_phiGy_local;
    unsigned int* d_any_left;
    int* d_found_flag;
    FoundResult* d_found_result;
    uint8_t* d_hashes;
    CUDA_CHECK(cudaMalloc(&d_start_scalars, batch_size * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_counts256, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_P, batch_size * 4 * 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_R, batch_size * 4 * 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_hashes, batch_size * 20 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_any_left, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_found_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found_result, sizeof(FoundResult)));
    CUDA_CHECK(cudaMalloc(&d_pre_Gx_local, PRECOMPUTE_SIZE * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_pre_Gy_local, PRECOMPUTE_SIZE * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_pre_phiGx_local, PRECOMPUTE_SIZE * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_pre_phiGy_local, PRECOMPUTE_SIZE * 4 * sizeof(unsigned long long)));

    unsigned long long* h_start_scalars, *h_counts256;
    CUDA_CHECK(cudaMallocHost(&h_start_scalars, batch_size * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocHost(&h_counts256, 4 * sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemset(d_found_flag, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_any_left, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_hashes_accum, 0, sizeof(unsigned long long)));

    JacobianPoint G;
    G.infinity = false;
    fieldCopy(Gx_d, G.x);
    fieldCopy(Gy_d, G.y);
    G.z[0] = 1ULL; G.z[1] = 0ULL; G.z[2] = 0ULL; G.z[3] = 0ULL;

    cudaStream_t streamKernel;
    CUDA_CHECK(cudaStreamCreate(&streamKernel));

    JacobianPoint phi_base;
    compute_phi_base_kernel<<<1, 1, 0, streamKernel>>>(G, phi_base);
    CUDA_CHECK(cudaStreamSynchronize(streamKernel));
    CUDA_CHECK(cudaGetLastError());

    if (verbose) {
        unsigned long long h_Gx[4], h_Gy[4], h_phiGx[4], h_phiGy[4];
        CUDA_CHECK(cudaMemcpyFromSymbol(h_Gx, Gx_d, 4 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpyFromSymbol(h_Gy, Gy_d, 4 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpy(h_phiGx, phi_base.x, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_phiGy, phi_base.y, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        std::cout << "precompute_g_table_gpu: base.x=" << std::hex << h_Gx[0] << ":" << h_Gx[1] << ":" << h_Gx[2] << ":" << h_Gx[3]
                  << ", base.y=" << h_Gy[0] << ":" << h_Gy[1] << ":" << h_Gy[2] << ":" << h_Gy[3]
                  << ", base.infinity=" << G.infinity << "\n";
        std::cout << "precompute_g_table_gpu: phi_base.x=" << h_phiGx[0] << ":" << h_phiGx[1] << ":" << h_phiGx[2] << ":" << h_phiGx[3]
                  << ", phi_base.y=" << h_phiGy[0] << ":" << h_phiGy[1] << ":" << h_phiGy[2] << ":" << h_phiGy[3]
                  << ", phi_base.infinity=" << phi_base.infinity << "\n";
    }

    precompute_g_table_gpu(G, d_pre_Gx_local, d_pre_Gy_local, PRECOMPUTE_SIZE, streamKernel, BLOCKS, THREADS);
    std::cout << "precompute_table_kernel (Gx, Gy) completed\n";
    precompute_g_table_gpu(phi_base, d_pre_phiGx_local, d_pre_phiGy_local, PRECOMPUTE_SIZE, streamKernel, BLOCKS, THREADS);
    std::cout << "precompute_table_kernel (phiGx, phiGy) completed\n";

    unsigned long long range_len[4];
    sub256(range_end, range_start, range_len);
    inc256(range_len, 1ULL);

    unsigned long long delta[4];
    divmod_256_by_u64(range_len, batch_size, delta, h_counts256[0]);
    for (int i = 0; i < 4; ++i) h_counts256[i] = delta[i];
    CUDA_CHECK(cudaMemcpy(d_counts256, h_counts256, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    for (int i = 0; i < batch_size; ++i) {
        add256_u64(range_start, i, h_start_scalars + i * 4);
    }
    CUDA_CHECK(cudaMemcpy(d_start_scalars, h_start_scalars, batch_size * 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    unsigned long long h_hashes = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0;
    bool completed_all = false;
    bool stop_all = false;

    while (!completed_all && !stop_all && !g_sigint) {
        fused_ec_hash<<<BLOCKS, THREADS, batch_size * 20 * sizeof(uint32_t), streamKernel>>>(d_start_scalars, d_P, d_R, batch_size, d_pre_Gx_local, d_pre_Gy_local, d_pre_phiGx_local, d_pre_phiGy_local, d_hashes_accum, d_any_left, d_found_flag, d_found_result);
        CUDA_CHECK(cudaGetLastError());

        auto now = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(now - tLast).count();
        if (dt > 1000.0) {
            unsigned long long delta;
            CUDA_CHECK(cudaMemcpy(&delta, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            h_hashes += delta;
            double mkeys = delta / (dt * 1e6);
            double elapsed = std::chrono::duration<double>(now - t0).count();
            long double total_keys = ld_from_u256(range_len);
            long double prog = total_keys > 0.0L ? ((long double)h_hashes / total_keys) * 100.0L : 0.0L;
            if (prog > 100.0L) prog = 100.0L;
            std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                      << " s | Speed: " << std::fixed << std::setprecision(1) << mkeys
                      << " Mkeys/s | Count: " << h_hashes
                      << " | Progress: " << std::fixed << std::setprecision(2) << (double)prog << " %";
            std::cout.flush();
            lastHashes = h_hashes;
            tLast = now;
        }

        int host_found = 0;
        CUDA_CHECK(cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost));
        if (host_found == FOUND_READY) {
            stop_all = true;
            break;
        }

        cudaError_t qs = cudaStreamQuery(streamKernel);
        if (qs == cudaSuccess) {
            // Debug output for public keys and hashes
            unsigned long long h_P[batch_size * 4 * 2];
            uint8_t h_hashes[batch_size * 20];
            CUDA_CHECK(cudaMemcpy(h_P, d_P, batch_size * 4 * 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_hashes, d_hashes, batch_size * 20 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
            for (int i = 0; i < batch_size; ++i) {
                printf("Batch %d: x=%llx:%llx:%llx:%llx, y=%llx:%llx:%llx:%llx, hash160=",
                       i, h_P[i*8], h_P[i*8+1], h_P[i*8+2], h_P[i*8+3],
                       h_P[i*8+4], h_P[i*8+5], h_P[i*8+6], h_P[i*8+7]);
                for (int j = 0; j < 20; ++j) printf("%02x", h_hashes[i*20 + j]);
                printf("\n");
            }
            break;
        }
        if (qs != cudaErrorNotReady) {
            CUDA_CHECK(cudaGetLastError());
            stop_all = true;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    CUDA_CHECK(cudaStreamSynchronize(streamKernel));
    std::cout.flush();
    if (stop_all || g_sigint) {
        // Handle early termination
    }

    unsigned int h_any = 0;
    CUDA_CHECK(cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    std::swap(d_P, d_R);
    if (h_any == 0u) {
        completed_all = true;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "\n";

    int h_found_flag = 0;
    CUDA_CHECK(cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost));
    int exit_code = EXIT_SUCCESS;

    if (h_found_flag == FOUND_READY) {
        FoundResult host_result;
        CUDA_CHECK(cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost));
        std::cout << "\n======== FOUND MATCH! =================================\n";
        std::cout << "Private Key   : " << CryptoUtils::formatHex256(host_result.scalar_val) << "\n";
        std::cout << "Public Key    : " << CryptoUtils::formatCompressedPubHex(host_result.Rx_val, host_result.Ry_val) << "\n";
        if (verbose) {
            std::cout << "Thread ID     : " << host_result.threadId << "\n";
            std::cout << "Iteration     : " << host_result.iter << "\n";
        }
    } else {
        if (g_sigint) {
            std::cout << "======== INTERRUPTED (Ctrl+C) ==========================\n";
            std::cout << "Search was interrupted by user. Partial progress above.\n";
            exit_code = 130;
        } else if (completed_all) {
            std::cout << "======== KEY NOT FOUND (exhaustive) ===================\n";
            std::cout << "Target hash160 was not found within the specified range.\n";
        } else {
            std::cout << "======== TERMINATED ===================================\n";
            std::cout << "Search terminated due to an error or incomplete range.\n";
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_start_scalars));
    CUDA_CHECK(cudaFree(d_counts256));
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_hashes));
    CUDA_CHECK(cudaFree(d_hashes_accum));
    CUDA_CHECK(cudaFree(d_any_left));
    CUDA_CHECK(cudaFree(d_found_flag));
    CUDA_CHECK(cudaFree(d_found_result));
    CUDA_CHECK(cudaFree(d_pre_Gx_local));
    CUDA_CHECK(cudaFree(d_pre_Gy_local));
    CUDA_CHECK(cudaFree(d_pre_phiGx_local));
    CUDA_CHECK(cudaFree(d_pre_phiGy_local));
    if (h_start_scalars) CUDA_CHECK(cudaFreeHost(h_start_scalars));
    if (h_counts256) CUDA_CHECK(cudaFreeHost(h_counts256));
    CUDA_CHECK(cudaStreamDestroy(streamKernel));

    return exit_code;
}