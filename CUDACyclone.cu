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
#include "CUDAHash.cuh"
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

// Forward declaration for searchKernel
__global__ void searchKernel(
    const unsigned long long* scalars_in,
    unsigned long long* outX,
    unsigned long long* outY,
    int N,
    unsigned long long* counts,
    int* found_flag,
    FoundResult* found_result,
    unsigned long long* hashes_accum,
    unsigned int* any_left,
    const unsigned long long* pre_Gx,
    const unsigned long long* pre_Gy,
    const unsigned long long* pre_phiGx,
    const unsigned long long* pre_phiGy
);

// Debug kernel to test memory writes
__global__ void debug_test_write_kernel(unsigned long long* pre_x, unsigned long long* pre_y, unsigned long long size) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    pre_x[idx * 4] = 0ULL;
    pre_y[idx * 4] = 0ULL;
}

__global__ void searchKernel(
    const unsigned long long* scalars_in,
    unsigned long long* outX,
    unsigned long long* outY,
    int N,
    unsigned long long* counts,
    int* found_flag,
    FoundResult* found_result,
    unsigned long long* hashes_accum,
    unsigned int* any_left,
    const unsigned long long* pre_Gx,
    const unsigned long long* pre_Gy,
    const unsigned long long* pre_phiGx,
    const unsigned long long* pre_phiGy
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane = threadIdx.x % WARP_SIZE;
    unsigned full_mask = 0xFFFFFFFF;
    if (idx >= N || idx >= MAX_BATCH_SIZE) return;

    unsigned long long scalar[4];
    fieldCopy(scalars_in + idx*4, scalar);
    unsigned long long Rx[4], Ry[4];
    uint8_t hash[20];

    for (int iter = 0; iter < 1000; ++iter) {
        if (warp_found_ready(found_flag, full_mask, lane)) return;

        scalarMulBaseJacobian(scalar, Rx, Ry, pre_Gx, pre_Gy, pre_phiGx, pre_phiGy);
        getHash160_33_from_limbs((Ry[0] & 1ULL) ? 0x03 : 0x02, Rx, hash);

        if (hash160_matches_prefix_then_full(hash, c_target_hash160, c_target_prefix)) {
            int old = atomicCAS(found_flag, FOUND_NONE, FOUND_LOCK);
            if (old == FOUND_NONE) {
                FoundResult res;
                res.threadId = idx;
                res.iter = iter;
                fieldCopy(scalar, res.scalar_val);
                fieldCopy(Rx, res.Rx_val);
                fieldCopy(Ry, res.Ry_val);
                *found_result = res;
                *found_flag = FOUND_READY;
                return;
            }
        }

        inc256_device(scalar, 1ULL);
        atomicAdd(counts, 1ULL);
    }

    *any_left = 1;
    __syncthreads();
}

int main(int argc, char* argv[]) {
    std::string range_str = "1:FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141";
    std::string address = "";
    std::string grid_str = "64,32";
    bool verbose = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--range" && i + 1 < argc) {
            range_str = argv[++i];
        } else if (arg == "--address" && i + 1 < argc) {
            address = argv[++i];
        } else if (arg == "--grid" && i + 1 < argc) {
            grid_str = argv[++i];
        } else if (arg == "--verbose") {
            verbose = true;
        }
    }

    // Parse grid
    int blocks = 64, threadsPerBlock = 32;
    sscanf(grid_str.c_str(), "%d,%d", &blocks, &threadsPerBlock);
    if (verbose) {
        std::cout << "Parsed grid: blocks=" << blocks << ", threadsPerBlock=" << threadsPerBlock << "\n";
    }

    // Ensure blocks * threadsPerBlock does not exceed MAX_BATCH_SIZE
    if (blocks * threadsPerBlock > MAX_BATCH_SIZE) {
        blocks = (MAX_BATCH_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        if (verbose) {
            std::cout << "Adjusted blocks to " << blocks << " to match MAX_BATCH_SIZE=" << MAX_BATCH_SIZE << "\n";
        }
    }

    // Parse range
    unsigned long long start[4], end[4], range_len[4];
    std::string start_str = range_str.substr(0, range_str.find(':'));
    std::string end_str = range_str.substr(range_str.find(':') + 1);
    if (!hexToLE64(start_str, start) || !hexToLE64(end_str, end)) {
        std::cerr << "Invalid range format\n";
        return EXIT_FAILURE;
    }
    sub256(end, start, range_len);

    // Decode address
    uint8_t target_hash160[20];
    if (!decode_p2pkh_address(address, target_hash160)) {
        std::cerr << "Invalid P2PKH address\n";
        return EXIT_FAILURE;
    }
    uint32_t target_prefix = ((uint32_t)target_hash160[0]) | ((uint32_t)target_hash160[1] << 8) |
                             ((uint32_t)target_hash160[2] << 16) | ((uint32_t)target_hash160[3] << 24);

    // CUDA setup
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const unsigned long long PRECOMPUTE_SIZE_LOCAL = 1ULL << 10; // 1024 points
    if (verbose) {
        std::cout << "======== PrePhase: GPU Information ====================\n";
        std::cout << "Device               : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
        std::cout << "SM                   : " << prop.multiProcessorCount << "\n";
        std::cout << "ThreadsPerBlock      : " << threadsPerBlock << "\n";
        std::cout << "Blocks               : " << blocks << "\n";
        std::cout << "Points batch size    : " << MAX_BATCH_SIZE << "\n";
        std::cout << "Precomputed tables    : " << PRECOMPUTE_SIZE_LOCAL << " points (~" << (PRECOMPUTE_SIZE_LOCAL * 8 * 2 / 1024.0 / 1024.0) << " MB)\n";
        std::cout << "Memory utilization   : " << std::fixed << std::setprecision(1) << (PRECOMPUTE_SIZE_LOCAL * 8 * 2 * 100.0 / prop.totalGlobalMem) << "% ("
                  << human_bytes(PRECOMPUTE_SIZE_LOCAL * 8 * 2) << " / " << human_bytes(prop.totalGlobalMem) << ")\n";
        std::cout << "-------------------------------------------------------\n";
        std::cout << "Total threads        : " << (blocks * threadsPerBlock) << "\n";
    }

    // Validate constants
    unsigned long long h_Gx_d[4], h_Gy_d[4], h_c_beta[4], h_c_mu[5];
    CUDA_CHECK(cudaMemcpyFromSymbol(h_Gx_d, Gx_d, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_Gy_d, Gy_d, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_c_beta, c_beta, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_c_mu, c_mu, 5 * sizeof(unsigned long long)));
    if (verbose) {
        std::cout << "Gx_d: " << CryptoUtils::formatHex256(h_Gx_d) << "\n";
        std::cout << "Gy_d: " << CryptoUtils::formatHex256(h_Gy_d) << "\n";
        std::cout << "c_beta: " << CryptoUtils::formatHex256(h_c_beta) << "\n";
        std::cout << "c_mu: " << std::hex << std::uppercase << std::setfill('0')
                  << std::setw(16) << h_c_mu[4] << std::setw(16) << h_c_mu[3]
                  << std::setw(16) << h_c_mu[2] << std::setw(16) << h_c_mu[1]
                  << std::setw(16) << h_c_mu[0] << "\n";
    }
    if (isZero256(h_Gx_d) || isZero256(h_Gy_d) || isZero256(h_c_beta)) {
        std::cerr << "Error: Gx_d, Gy_d, or c_beta is zero\n";
        return EXIT_FAILURE;
    }
    // Validate c_mu
    unsigned long long expected_c_mu[5] = {0x1000003d1ULL, 0ULL, 0ULL, 0ULL, 1ULL};
    bool c_mu_valid = true;
    for (int i = 0; i < 5; ++i) {
        if (h_c_mu[i] != expected_c_mu[i]) {
            c_mu_valid = false;
            break;
        }
    }
    if (!c_mu_valid) {
        std::cerr << "Warning: c_mu in CUDAStructures.h is incorrect\n";
    }
    // Warn about incorrect c_beta
    unsigned long long expected_c_beta[4] = {0x719501eeULL, 0xc1396c28ULL, 0x12f58995ULL, 0x9cf04975ULL};
    if (!_IsEqual(h_c_beta, expected_c_beta)) {
        std::cerr << "Warning: c_beta in CUDAStructures.h is incorrect, using c_beta_fallback\n";
    }

    // Validate base point on host
    if (verbose) {
        std::cout << "Validating base point on host...\n";
    }
    if (!isPointOnCurve(h_Gx_d, h_Gy_d)) {
        std::cerr << "Error: base point (Gx_d, Gy_d) is not on the secp256k1 curve\n";
        return EXIT_FAILURE;
    }

    // Allocate memory
    unsigned long long *d_start_scalars, *d_counts256, *d_P, *d_R, *d_pre_Gx_local, *d_pre_Gy_local, *d_pre_phiGx_local, *d_pre_phiGy_local;
    int *d_found_flag, *d_valid;
    FoundResult *d_found_result;
    unsigned long long *d_hashes_accum;
    unsigned int *d_any_left;
    unsigned long long *d_debug_precompute, *d_debug_invalid_idx, *d_debug_invalid_point, *d_debug_field_result;
    CUDA_CHECK(cudaMalloc(&d_start_scalars, MAX_BATCH_SIZE * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_counts256, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_P, MAX_BATCH_SIZE * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_R, MAX_BATCH_SIZE * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_found_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found_result, sizeof(FoundResult)));
    CUDA_CHECK(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_any_left, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_pre_Gx_local, PRECOMPUTE_SIZE_LOCAL * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_pre_Gy_local, PRECOMPUTE_SIZE_LOCAL * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_pre_phiGx_local, PRECOMPUTE_SIZE_LOCAL * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_pre_phiGy_local, PRECOMPUTE_SIZE_LOCAL * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_debug_precompute, PRECOMPUTE_SIZE_LOCAL * 8 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_valid, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_debug_invalid_idx, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_debug_invalid_point, 8 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_debug_field_result, 4 * sizeof(unsigned long long)));

    // Initialize memory
    CUDA_CHECK(cudaMemset(d_counts256, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_found_flag, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hashes_accum, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_any_left, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_P, 0, MAX_BATCH_SIZE * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_R, 0, MAX_BATCH_SIZE * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_pre_Gx_local, 0, PRECOMPUTE_SIZE_LOCAL * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_pre_Gy_local, 0, PRECOMPUTE_SIZE_LOCAL * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_pre_phiGx_local, 0, PRECOMPUTE_SIZE_LOCAL * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_pre_phiGy_local, 0, PRECOMPUTE_SIZE_LOCAL * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_debug_precompute, 0, PRECOMPUTE_SIZE_LOCAL * 8 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_valid, 1, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_debug_invalid_idx, 0xFF, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_debug_invalid_point, 0, 8 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_debug_field_result, 0, 4 * sizeof(unsigned long long)));

    // Set constants
    CUDA_CHECK(cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_target_prefix, &target_prefix, sizeof(uint32_t)));

    // Validate field arithmetic for phi_base
    if (verbose) {
        std::cout << "Validating field multiplication for phi_base on device...\n";
    }
    debug_field_multiply_kernel<<<1, 1>>>(d_debug_field_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long h_field_result[4];
    CUDA_CHECK(cudaMemcpy(h_field_result, d_debug_field_result, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if (verbose) {
        std::cout << "phi_x (Gx_d * c_beta_fallback mod p): " << CryptoUtils::formatHex256(h_field_result) << "\n";
    }

    // Precompute G tables
    JacobianPoint base;
    fieldCopy(h_Gx_d, base.x);
    fieldCopy(h_Gy_d, base.y);
    fieldSetZero(base.z);
    base.z[0] = 1ULL;
    base.infinity = false;
    // Debug: Print base point
    if (verbose) {
        std::cout << "base.x: " << CryptoUtils::formatHex256(base.x) << "\n";
        std::cout << "base.y: " << CryptoUtils::formatHex256(base.y) << "\n";
    }
    // Validate base point
    if (isZero256(base.x) || isZero256(base.y)) {
        std::cerr << "Error: base point initialization failed\n";
        CUDA_CHECK(cudaFree(d_start_scalars));
        CUDA_CHECK(cudaFree(d_counts256));
        CUDA_CHECK(cudaFree(d_P));
        CUDA_CHECK(cudaFree(d_R));
        CUDA_CHECK(cudaFree(d_found_flag));
        CUDA_CHECK(cudaFree(d_found_result));
        CUDA_CHECK(cudaFree(d_hashes_accum));
        CUDA_CHECK(cudaFree(d_any_left));
        CUDA_CHECK(cudaFree(d_pre_Gx_local));
        CUDA_CHECK(cudaFree(d_pre_Gy_local));
        CUDA_CHECK(cudaFree(d_pre_phiGx_local));
        CUDA_CHECK(cudaFree(d_pre_phiGy_local));
        CUDA_CHECK(cudaFree(d_debug_precompute));
        CUDA_CHECK(cudaFree(d_valid));
        CUDA_CHECK(cudaFree(d_debug_invalid_idx));
        CUDA_CHECK(cudaFree(d_debug_invalid_point));
        CUDA_CHECK(cudaFree(d_debug_field_result));
        return EXIT_FAILURE;
    }
    int precompute_blocks = (PRECOMPUTE_SIZE_LOCAL + threadsPerBlock - 1) / threadsPerBlock;
    precompute_table_kernel<<<precompute_blocks, threadsPerBlock>>>(base, d_pre_Gx_local, d_pre_Gy_local, PRECOMPUTE_SIZE_LOCAL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Validate precomputed G tables
    if (verbose) {
        std::cout << "Validating precomputed G tables...\n";
    }
    CUDA_CHECK(cudaMemset(d_valid, 1, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_debug_invalid_idx, 0xFF, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_debug_invalid_point, 0, 8 * sizeof(unsigned long long)));
    validate_precompute_kernel<<<precompute_blocks, threadsPerBlock>>>(d_pre_Gx_local, d_pre_Gy_local, PRECOMPUTE_SIZE_LOCAL, d_valid, d_debug_invalid_idx, d_debug_invalid_point);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int h_valid;
    unsigned long long h_invalid_idx;
    unsigned long long h_invalid_point[8];
    CUDA_CHECK(cudaMemcpy(&h_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_invalid_idx, d_debug_invalid_idx, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_invalid_point, d_debug_invalid_point, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if (!h_valid) {
        std::cerr << "Error: precomputed G tables contain invalid points at index " << h_invalid_idx << "\n";
        std::cerr << "Invalid point x: " << CryptoUtils::formatHex256(h_invalid_point) << "\n";
        std::cerr << "Invalid point y: " << CryptoUtils::formatHex256(h_invalid_point + 4) << "\n";
        CUDA_CHECK(cudaFree(d_start_scalars));
        CUDA_CHECK(cudaFree(d_counts256));
        CUDA_CHECK(cudaFree(d_P));
        CUDA_CHECK(cudaFree(d_R));
        CUDA_CHECK(cudaFree(d_found_flag));
        CUDA_CHECK(cudaFree(d_found_result));
        CUDA_CHECK(cudaFree(d_hashes_accum));
        CUDA_CHECK(cudaFree(d_any_left));
        CUDA_CHECK(cudaFree(d_pre_Gx_local));
        CUDA_CHECK(cudaFree(d_pre_Gy_local));
        CUDA_CHECK(cudaFree(d_pre_phiGx_local));
        CUDA_CHECK(cudaFree(d_pre_phiGy_local));
        CUDA_CHECK(cudaFree(d_debug_precompute));
        CUDA_CHECK(cudaFree(d_valid));
        CUDA_CHECK(cudaFree(d_debug_invalid_idx));
        CUDA_CHECK(cudaFree(d_debug_invalid_point));
        CUDA_CHECK(cudaFree(d_debug_field_result));
        return EXIT_FAILURE;
    }

    // Debug: Check first few precomputed points
    unsigned long long h_pre_Gx[4], h_pre_Gy[4];
    CUDA_CHECK(cudaMemcpy(h_pre_Gx, d_pre_Gx_local, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pre_Gy, d_pre_Gy_local, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if (verbose) {
        std::cout << "First precomputed Gx: " << CryptoUtils::formatHex256(h_pre_Gx) << "\n";
        std::cout << "First precomputed Gy: " << CryptoUtils::formatHex256(h_pre_Gy) << "\n";
    }
    // Validate precomputed points
    if (isZero256(h_pre_Gx) || isZero256(h_pre_Gy)) {
        std::cerr << "Error: precomputed Gx or Gy is zero\n";
        CUDA_CHECK(cudaFree(d_start_scalars));
        CUDA_CHECK(cudaFree(d_counts256));
        CUDA_CHECK(cudaFree(d_P));
        CUDA_CHECK(cudaFree(d_R));
        CUDA_CHECK(cudaFree(d_found_flag));
        CUDA_CHECK(cudaFree(d_found_result));
        CUDA_CHECK(cudaFree(d_hashes_accum));
        CUDA_CHECK(cudaFree(d_any_left));
        CUDA_CHECK(cudaFree(d_pre_Gx_local));
        CUDA_CHECK(cudaFree(d_pre_Gy_local));
        CUDA_CHECK(cudaFree(d_pre_phiGx_local));
        CUDA_CHECK(cudaFree(d_pre_phiGy_local));
        CUDA_CHECK(cudaFree(d_debug_precompute));
        CUDA_CHECK(cudaFree(d_valid));
        CUDA_CHECK(cudaFree(d_debug_invalid_idx));
        CUDA_CHECK(cudaFree(d_debug_invalid_point));
        CUDA_CHECK(cudaFree(d_debug_field_result));
        return EXIT_FAILURE;
    }

    // Precompute phi(G)
    unsigned long long *d_phi_x, *d_phi_y;
    CUDA_CHECK(cudaMalloc(&d_phi_x, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_phi_y, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_phi_x, 0, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_phi_y, 0, 4 * sizeof(unsigned long long)));

    // Debug: Test simple write to d_phi_x and d_phi_y
    if (verbose) {
        std::cout << "Testing memory write to d_phi_x and d_phi_y...\n";
    }
    debug_test_write_kernel<<<1, threadsPerBlock>>>(d_phi_x, d_phi_y, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (verbose) {
        std::cout << "Debug write test passed\n";
    }

    // Run phi base computation
    if (verbose) {
        std::cout << "Computing phi(G)...\n";
    }
    compute_phi_base_kernel<<<1, 1>>>(d_phi_x, d_phi_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy phi(G) for debugging
    unsigned long long h_phi_x[4], h_phi_y[4];
    CUDA_CHECK(cudaMemcpy(h_phi_x, d_phi_x, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_phi_y, d_phi_y, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if (verbose) {
        std::cout << "phi_x: " << CryptoUtils::formatHex256(h_phi_x) << "\n";
        std::cout << "phi_y: " << CryptoUtils::formatHex256(h_phi_y) << "\n";
    }

    // Validate phi(G) on device
    if (verbose) {
        std::cout << "Validating phi(G) on device...\n";
    }
    CUDA_CHECK(cudaMemset(d_valid, 1, sizeof(int)));
    validate_point_kernel<<<1, 1>>>(d_phi_x, d_phi_y, d_valid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost));
    if (!h_valid) {
        std::cerr << "Error: phi(G) point is not on the secp256k1 curve (device validation)\n";
        CUDA_CHECK(cudaFree(d_phi_x));
        CUDA_CHECK(cudaFree(d_phi_y));
        CUDA_CHECK(cudaFree(d_start_scalars));
        CUDA_CHECK(cudaFree(d_counts256));
        CUDA_CHECK(cudaFree(d_P));
        CUDA_CHECK(cudaFree(d_R));
        CUDA_CHECK(cudaFree(d_found_flag));
        CUDA_CHECK(cudaFree(d_found_result));
        CUDA_CHECK(cudaFree(d_hashes_accum));
        CUDA_CHECK(cudaFree(d_any_left));
        CUDA_CHECK(cudaFree(d_pre_Gx_local));
        CUDA_CHECK(cudaFree(d_pre_Gy_local));
        CUDA_CHECK(cudaFree(d_pre_phiGx_local));
        CUDA_CHECK(cudaFree(d_pre_phiGy_local));
        CUDA_CHECK(cudaFree(d_debug_precompute));
        CUDA_CHECK(cudaFree(d_valid));
        CUDA_CHECK(cudaFree(d_debug_invalid_idx));
        CUDA_CHECK(cudaFree(d_debug_invalid_point));
        CUDA_CHECK(cudaFree(d_debug_field_result));
        return EXIT_FAILURE;
    }

    // Validate phi(G) on host
    if (verbose) {
        std::cout << "Validating phi(G) on host...\n";
    }
    if (!isPointOnCurve(h_phi_x, h_phi_y)) {
        std::cerr << "Error: phi_base point is not on the secp256k1 curve (host validation)\n";
        CUDA_CHECK(cudaFree(d_phi_x));
        CUDA_CHECK(cudaFree(d_phi_y));
        CUDA_CHECK(cudaFree(d_start_scalars));
        CUDA_CHECK(cudaFree(d_counts256));
        CUDA_CHECK(cudaFree(d_P));
        CUDA_CHECK(cudaFree(d_R));
        CUDA_CHECK(cudaFree(d_found_flag));
        CUDA_CHECK(cudaFree(d_found_result));
        CUDA_CHECK(cudaFree(d_hashes_accum));
        CUDA_CHECK(cudaFree(d_any_left));
        CUDA_CHECK(cudaFree(d_pre_Gx_local));
        CUDA_CHECK(cudaFree(d_pre_Gy_local));
        CUDA_CHECK(cudaFree(d_pre_phiGx_local));
        CUDA_CHECK(cudaFree(d_pre_phiGy_local));
        CUDA_CHECK(cudaFree(d_debug_precompute));
        CUDA_CHECK(cudaFree(d_valid));
        CUDA_CHECK(cudaFree(d_debug_invalid_idx));
        CUDA_CHECK(cudaFree(d_debug_invalid_point));
        CUDA_CHECK(cudaFree(d_debug_field_result));
        return EXIT_FAILURE;
    }

    // Precompute phi(G) tables
    JacobianPoint phi_base;
    fieldCopy(h_phi_x, phi_base.x);
    fieldCopy(h_phi_y, phi_base.y);
    fieldSetZero(phi_base.z);
    phi_base.z[0] = 1ULL;
    phi_base.infinity = false;
    // Debug: Print phi_base point
    if (verbose) {
        std::cout << "phi_base.x: " << CryptoUtils::formatHex256(phi_base.x) << "\n";
        std::cout << "phi_base.y: " << CryptoUtils::formatHex256(phi_base.y) << "\n";
    }
    // Verify phi_base
    if (isZero256(phi_base.x) || isZero256(phi_base.y)) {
        std::cerr << "Error: phi_base initialization failed\n";
        CUDA_CHECK(cudaFree(d_phi_x));
        CUDA_CHECK(cudaFree(d_phi_y));
        CUDA_CHECK(cudaFree(d_start_scalars));
        CUDA_CHECK(cudaFree(d_counts256));
        CUDA_CHECK(cudaFree(d_P));
        CUDA_CHECK(cudaFree(d_R));
        CUDA_CHECK(cudaFree(d_found_flag));
        CUDA_CHECK(cudaFree(d_found_result));
        CUDA_CHECK(cudaFree(d_hashes_accum));
        CUDA_CHECK(cudaFree(d_any_left));
        CUDA_CHECK(cudaFree(d_pre_Gx_local));
        CUDA_CHECK(cudaFree(d_pre_Gy_local));
        CUDA_CHECK(cudaFree(d_pre_phiGx_local));
        CUDA_CHECK(cudaFree(d_pre_phiGy_local));
        CUDA_CHECK(cudaFree(d_debug_precompute));
        CUDA_CHECK(cudaFree(d_valid));
        CUDA_CHECK(cudaFree(d_debug_invalid_idx));
        CUDA_CHECK(cudaFree(d_debug_invalid_point));
        CUDA_CHECK(cudaFree(d_debug_field_result));
        return EXIT_FAILURE;
    }
    // Debug: Test write to d_pre_phiGx_local and d_pre_phiGy_local
    if (verbose) {
        std::cout << "Testing memory write to d_pre_phiGx_local and d_pre_phiGy_local...\n";
    }
    debug_test_write_kernel<<<precompute_blocks, threadsPerBlock>>>(d_pre_phiGx_local, d_pre_phiGy_local, PRECOMPUTE_SIZE_LOCAL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (verbose) {
        std::cout << "Debug precompute write test passed\n";
    }
    precompute_table_kernel<<<precompute_blocks, threadsPerBlock>>>(phi_base, d_pre_phiGx_local, d_pre_phiGy_local, PRECOMPUTE_SIZE_LOCAL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Validate precomputed phi(G) tables
    if (verbose) {
        std::cout << "Validating precomputed phi(G) tables...\n";
    }
    CUDA_CHECK(cudaMemset(d_valid, 1, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_debug_invalid_idx, 0xFF, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_debug_invalid_point, 0, 8 * sizeof(unsigned long long)));
    validate_precompute_kernel<<<precompute_blocks, threadsPerBlock>>>(d_pre_phiGx_local, d_pre_phiGy_local, PRECOMPUTE_SIZE_LOCAL, d_valid, d_debug_invalid_idx, d_debug_invalid_point);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_invalid_idx, d_debug_invalid_idx, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_invalid_point, d_debug_invalid_point, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if (!h_valid) {
        std::cerr << "Error: precomputed phi(G) tables contain invalid points at index " << h_invalid_idx << "\n";
        std::cerr << "Invalid point x: " << CryptoUtils::formatHex256(h_invalid_point) << "\n";
        std::cerr << "Invalid point y: " << CryptoUtils::formatHex256(h_invalid_point + 4) << "\n";
        CUDA_CHECK(cudaFree(d_phi_x));
        CUDA_CHECK(cudaFree(d_phi_y));
        CUDA_CHECK(cudaFree(d_start_scalars));
        CUDA_CHECK(cudaFree(d_counts256));
        CUDA_CHECK(cudaFree(d_P));
        CUDA_CHECK(cudaFree(d_R));
        CUDA_CHECK(cudaFree(d_found_flag));
        CUDA_CHECK(cudaFree(d_found_result));
        CUDA_CHECK(cudaFree(d_hashes_accum));
        CUDA_CHECK(cudaFree(d_any_left));
        CUDA_CHECK(cudaFree(d_pre_Gx_local));
        CUDA_CHECK(cudaFree(d_pre_Gy_local));
        CUDA_CHECK(cudaFree(d_pre_phiGx_local));
        CUDA_CHECK(cudaFree(d_pre_phiGy_local));
        CUDA_CHECK(cudaFree(d_debug_precompute));
        CUDA_CHECK(cudaFree(d_valid));
        CUDA_CHECK(cudaFree(d_debug_invalid_idx));
        CUDA_CHECK(cudaFree(d_debug_invalid_point));
        CUDA_CHECK(cudaFree(d_debug_field_result));
        return EXIT_FAILURE;
    }

    // Debug: Verify precomputed phi tables
    if (verbose) {
        std::cout << "Verifying precomputed phi tables...\n";
    }
    debug_precompute_verify_kernel<<<precompute_blocks, threadsPerBlock>>>(d_pre_phiGx_local, d_pre_phiGy_local, d_debug_precompute, PRECOMPUTE_SIZE_LOCAL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long h_debug_precompute[8];
    CUDA_CHECK(cudaMemcpy(h_debug_precompute, d_debug_precompute, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if (verbose) {
        std::cout << "First precomputed phi_Gx: " << CryptoUtils::formatHex256(h_debug_precompute) << "\n";
        std::cout << "First precomputed phi_Gy: " << CryptoUtils::formatHex256(h_debug_precompute + 4) << "\n";
    }
    // Validate precomputed phi points
    if (isZero256(h_debug_precompute) || isZero256(h_debug_precompute + 4)) {
        std::cerr << "Error: precomputed phi_Gx or phi_Gy is zero\n";
        CUDA_CHECK(cudaFree(d_phi_x));
        CUDA_CHECK(cudaFree(d_phi_y));
        CUDA_CHECK(cudaFree(d_start_scalars));
        CUDA_CHECK(cudaFree(d_counts256));
        CUDA_CHECK(cudaFree(d_P));
        CUDA_CHECK(cudaFree(d_R));
        CUDA_CHECK(cudaFree(d_found_flag));
        CUDA_CHECK(cudaFree(d_found_result));
        CUDA_CHECK(cudaFree(d_hashes_accum));
        CUDA_CHECK(cudaFree(d_any_left));
        CUDA_CHECK(cudaFree(d_pre_Gx_local));
        CUDA_CHECK(cudaFree(d_pre_Gy_local));
        CUDA_CHECK(cudaFree(d_pre_phiGx_local));
        CUDA_CHECK(cudaFree(d_pre_phiGy_local));
        CUDA_CHECK(cudaFree(d_debug_precompute));
        CUDA_CHECK(cudaFree(d_valid));
        CUDA_CHECK(cudaFree(d_debug_invalid_idx));
        CUDA_CHECK(cudaFree(d_debug_invalid_point));
        CUDA_CHECK(cudaFree(d_debug_field_result));
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaFree(d_phi_x));
    CUDA_CHECK(cudaFree(d_phi_y));
    CUDA_CHECK(cudaFree(d_debug_precompute));
    CUDA_CHECK(cudaFree(d_valid));
    CUDA_CHECK(cudaFree(d_debug_invalid_idx));
    CUDA_CHECK(cudaFree(d_debug_invalid_point));
    CUDA_CHECK(cudaFree(d_debug_field_result));

    // Initialize scalars
    unsigned long long *h_start_scalars;
    CUDA_CHECK(cudaMallocHost(&h_start_scalars, MAX_BATCH_SIZE * 4 * sizeof(unsigned long long)));
    for (int i = 0; i < MAX_BATCH_SIZE; ++i) {
        unsigned long long offset[4], remainder;
        divmod_256_by_u64(start, MAX_BATCH_SIZE, offset, remainder);
        fieldCopy(offset, h_start_scalars + i*4);
        inc256(h_start_scalars + i*4, i + remainder);
    }
    CUDA_CHECK(cudaMemcpy(d_start_scalars, h_start_scalars, MAX_BATCH_SIZE * 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // Main search loop
    cudaStream_t streamKernel;
    CUDA_CHECK(cudaStreamCreate(&streamKernel));
    bool stop_all = false, completed_all = false;
    unsigned long long h_hashes = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto tLast = t0;

    signal(SIGINT, handle_sigint);
    while (!stop_all && !g_sigint) {
        CUDA_CHECK(cudaMemsetAsync(d_any_left, 0, sizeof(unsigned int), streamKernel));
        searchKernel<<<blocks, threadsPerBlock, 0, streamKernel>>>(d_start_scalars, d_P, d_R, MAX_BATCH_SIZE,
                                                                 d_counts256, d_found_flag, d_found_result, d_hashes_accum,
                                                                 d_any_left, d_pre_Gx_local, d_pre_Gy_local, d_pre_phiGx_local, d_pre_phiGy_local);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streamKernel));

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - tLast).count() >= 0.5) {
            unsigned long long delta;
            CUDA_CHECK(cudaMemcpy(&delta, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            h_hashes += delta;
            double dt = std::chrono::duration<double>(now - tLast).count();
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
            unsigned int h_any = 0;
            CUDA_CHECK(cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            if (h_any == 0) {
                completed_all = true;
                break;
            }
            for (int i = 0; i < MAX_BATCH_SIZE; ++i) {
                inc256(h_start_scalars + i*4, MAX_BATCH_SIZE);
            }
            CUDA_CHECK(cudaMemcpy(d_start_scalars, h_start_scalars, MAX_BATCH_SIZE * 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice));
            continue;
        }
        if (qs != cudaErrorNotReady) {
            CUDA_CHECK(qs);
            stop_all = true;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    CUDA_CHECK(cudaStreamSynchronize(streamKernel));
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
    CUDA_CHECK(cudaFree(d_found_flag));
    CUDA_CHECK(cudaFree(d_found_result));
    CUDA_CHECK(cudaFree(d_hashes_accum));
    CUDA_CHECK(cudaFree(d_any_left));
    CUDA_CHECK(cudaFree(d_pre_Gx_local));
    CUDA_CHECK(cudaFree(d_pre_Gy_local));
    CUDA_CHECK(cudaFree(d_pre_phiGx_local));
    CUDA_CHECK(cudaFree(d_pre_phiGy_local));
    CUDA_CHECK(cudaFreeHost(h_start_scalars));
    CUDA_CHECK(cudaStreamDestroy(streamKernel));

    return exit_code;
}