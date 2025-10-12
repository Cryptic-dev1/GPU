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

__global__ void compute_phi_base_kernel(unsigned long long* phi_x, unsigned long long* phi_y) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long temp[8];
        fieldMul_opt_device(Gx_d, c_beta, temp);
        modred_barrett_opt_device(temp, phi_x);
        fieldCopy(Gy_d, phi_y);
    }
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
    if (idx >= N) return;

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
    if (verbose) {
        std::cout << "======== PrePhase: GPU Information ====================\n";
        std::cout << "Device               : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
        std::cout << "SM                   : " << prop.multiProcessorCount << "\n";
        std::cout << "ThreadsPerBlock      : " << threadsPerBlock << "\n";
        std::cout << "Blocks               : " << blocks << "\n";
        std::cout << "Points batch size    : " << MAX_BATCH_SIZE << "\n";
        std::cout << "Precomputed tables    : 2^" << PRECOMPUTE_WINDOW << " points (~" << (PRECOMPUTE_SIZE * 8 * 2 / 1024.0 / 1024.0) << " MB)\n";
        std::cout << "Memory utilization   : " << std::fixed << std::setprecision(1) << (PRECOMPUTE_SIZE * 8 * 2 * 100.0 / prop.totalGlobalMem) << "% ("
                  << human_bytes(PRECOMPUTE_SIZE * 8 * 2) << " / " << human_bytes(prop.totalGlobalMem) << ")\n";
        std::cout << "-------------------------------------------------------\n";
        std::cout << "Total threads        : " << (blocks * threadsPerBlock) << "\n";
    }

    // Allocate memory
    unsigned long long *d_start_scalars, *d_counts256, *d_P, *d_R, *d_pre_Gx_local, *d_pre_Gy_local, *d_pre_phiGx_local, *d_pre_phiGy_local;
    int *d_found_flag;
    FoundResult *d_found_result;
    unsigned long long *d_hashes_accum;
    unsigned int *d_any_left;
    CUDA_CHECK(cudaMalloc(&d_start_scalars, blocks * threadsPerBlock * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_counts256, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_P, blocks * threadsPerBlock * 8 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_R, blocks * threadsPerBlock * 8 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_found_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found_result, sizeof(FoundResult)));
    CUDA_CHECK(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_any_left, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_pre_Gx_local, PRECOMPUTE_SIZE * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_pre_Gy_local, PRECOMPUTE_SIZE * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_pre_phiGx_local, PRECOMPUTE_SIZE * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_pre_phiGy_local, PRECOMPUTE_SIZE * 4 * sizeof(unsigned long long)));

    // Initialize memory
    CUDA_CHECK(cudaMemset(d_counts256, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_found_flag, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hashes_accum, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_any_left, 0, sizeof(unsigned int)));

    // Set constants
    CUDA_CHECK(cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_target_prefix, &target_prefix, sizeof(uint32_t)));

    // Precompute phi(G)
    unsigned long long *d_phi_x, *d_phi_y;
    CUDA_CHECK(cudaMalloc(&d_phi_x, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_phi_y, 4 * sizeof(unsigned long long)));
    compute_phi_base_kernel<<<1, 1>>>(d_phi_x, d_phi_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Precompute tables
    JacobianPoint base;
    fieldCopy(Gx_d, base.x);
    fieldCopy(Gy_d, base.y);
    fieldSetZero(base.z);
    base.z[0] = 1ULL;
    base.infinity = false;
    precompute_table_kernel<<<(PRECOMPUTE_SIZE + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(base, d_pre_Gx_local, d_pre_Gy_local, PRECOMPUTE_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    JacobianPoint phi_base;
    CUDA_CHECK(cudaMemcpy(phi_base.x, d_phi_x, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(phi_base.y, d_phi_y, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    fieldSetZero(phi_base.z);
    phi_base.z[0] = 1ULL;
    phi_base.infinity = false;
    precompute_table_kernel<<<(PRECOMPUTE_SIZE + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(phi_base, d_pre_phiGx_local, d_pre_phiGy_local, PRECOMPUTE_SIZE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_phi_x));
    CUDA_CHECK(cudaFree(d_phi_y));

    // Initialize scalars
    unsigned long long *h_start_scalars;
    CUDA_CHECK(cudaMallocHost(&h_start_scalars, blocks * threadsPerBlock * 4 * sizeof(unsigned long long)));
    for (int i = 0; i < blocks * threadsPerBlock; ++i) {
        unsigned long long offset[4], remainder;
        divmod_256_by_u64(start, blocks * threadsPerBlock, offset, remainder);
        fieldCopy(offset, h_start_scalars + i*4);
        inc256(h_start_scalars + i*4, i + remainder);
    }
    CUDA_CHECK(cudaMemcpy(d_start_scalars, h_start_scalars, blocks * threadsPerBlock * 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // Main search loop
    cudaStream_t streamKernel;
    CUDA_CHECK(cudaStreamCreate(&streamKernel));
    bool stop_all = false, completed_all = false;
    unsigned long long h_hashes = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0;

    signal(SIGINT, handle_sigint);
    while (!stop_all && !g_sigint) {
        CUDA_CHECK(cudaMemsetAsync(d_any_left, 0, sizeof(unsigned int), streamKernel));
        searchKernel<<<blocks, threadsPerBlock, 0, streamKernel>>>(d_start_scalars, d_P, d_R, blocks * threadsPerBlock,
                                                                 d_counts256, d_found_flag, d_found_result, d_hashes_accum,
                                                                 d_any_left, d_pre_Gx_local, d_pre_Gy_local, d_pre_phiGx_local, d_pre_phiGy_local);
        CUDA_CHECK(cudaGetLastError());

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
            unsigned int h_any = 0;
            CUDA_CHECK(cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost));
            if (h_any == 0) {
                completed_all = true;
                break;
            }
            CUDA_CHECK(cudaMemcpy(d_start_scalars, h_start_scalars, blocks * threadsPerBlock * 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice));
            for (int i = 0; i < blocks * threadsPerBlock; ++i) {
                inc256(h_start_scalars + i*4, blocks * threadsPerBlock);
            }
            continue;
        }
        if (qs != cudaErrorNotReady) {
            CUDA_CHECK(cudaGetLastError());
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