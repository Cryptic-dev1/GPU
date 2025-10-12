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

// Test kernel to verify precomputed tables
__global__ void test_precomputed_tables(unsigned long long* d_pre_Gx, unsigned long long* d_pre_Gy, unsigned long long* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    for (int i = 0; i < 4; ++i) {
        out[idx * 8 + i] = d_pre_Gx[idx * 4 + i];
        out[idx * 8 + i + 4] = d_pre_Gy[idx * 4 + i];
    }
    if (idx <= 1 && threadIdx.x == 0 && blockIdx.x == 0) {
        printf("test_precomputed_tables: idx=%d, pre_x=%llx:%llx:%llx:%llx, pre_y=%llx:%llx:%llx:%llx\n",
               idx, out[idx*8], out[idx*8+1], out[idx*8+2], out[idx*8+3],
               out[idx*8+4], out[idx*8+5], out[idx*8+6], out[idx*8+7]);
    }
}

// Modified scalarMulKernelBase for debugging
__global__ void scalarMulKernelBase(
    const unsigned long long* __restrict__ scalars,
    unsigned long long* __restrict__ outX,
    unsigned long long* __restrict__ outY,
    unsigned long long threadsTotal,
    const unsigned long long* __restrict__ d_pre_Gx,
    const unsigned long long* __restrict__ d_pre_Gy,
    const unsigned long long* __restrict__ d_pre_phiGx,
    const unsigned long long* __restrict__ d_pre_phiGy
) {
    unsigned long long gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    unsigned long long k1[4], k2[4], scalar[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        scalar[i] = scalars[gid * 4 + i];
    }
    split_glv(scalar, k1, k2);

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("scalarMulKernelBase: gid=%llu, scalar=%llx:%llx:%llx:%llx, k1=%llx:%llx:%llx:%llx, k2=%llx:%llx:%llx:%llx\n",
               gid, scalar[0], scalar[1], scalar[2], scalar[3],
               k1[0], k1[1], k1[2], k1[3], k2[0], k2[1], k2[2], k2[3]);
    }

    JacobianPoint R;
    if (k1[3] == 0 && k1[2] == 0 && k1[1] == 0 && k1[0] <= 0xFFFFFFFF) {
        pointSetG(R);
        for (uint64_t i = 0; i < k1[0]; ++i) {
            pointDoubleJacobian(R, R);
            if (threadIdx.x == 0 && blockIdx.x == 0 && i % 10 == 0) {
                printf("scalarMulKernelBase: doubling %llu, R.x=%llx:%llx:%llx:%llx, R.z=%llx:%llx:%llx:%llx, R.infinity=%d\n",
                       i, R.x[0], R.x[1], R.x[2], R.x[3], R.z[0], R.z[1], R.z[2], R.z[3], R.infinity);
            }
        }
        pointToAffine(R, outX + gid * 4, outY + gid * 4);
    } else {
        pointSetInfinity(R);
        int msb_k1 = find_msb(k1);
        int msb_k2 = find_msb(k2);
        int max_msb = (msb_k1 > msb_k2) ? msb_k1 : msb_k2;
        for (int i = max_msb; i >= 0; i -= PRECOMPUTE_WINDOW) {
            for (int j = 0; j < PRECOMPUTE_WINDOW && i - j >= 0; ++j) {
                pointDoubleJacobian(R, R);
            }
            int pos = i - PRECOMPUTE_WINDOW + 1;
            if (pos < 0) pos = 0;
            uint32_t window_k1 = get_window(k1, pos);
            if (window_k1) {
                unsigned long long Qx[4], Qy[4];
                fieldCopy(d_pre_Gx + (window_k1 - 1) * 4, Qx);
                fieldCopy(d_pre_Gy + (window_k1 - 1) * 4, Qy);
                pointAddMixed(R, Qx, Qy, false, R);
            }
            uint32_t window_k2 = get_window(k2, pos);
            if (window_k2) {
                unsigned long long Qx[4], Qy[4];
                fieldCopy(d_pre_phiGx + (window_k2 - 1) * 4, Qx);
                fieldCopy(d_pre_phiGy + (window_k2 - 1) * 4, Qy);
                pointAddMixed(R, Qx, Qy, false, R);
            }
        }
        pointToAffine(R, outX + gid * 4, outY + gid * 4);
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("scalarMulKernelBase: gid=%llu, outX=%llx:%llx:%llx:%llx, outY=%llx:%llx:%llx:%llx\n",
               gid, outX[gid*4], outX[gid*4+1], outX[gid*4+2], outX[gid*4+3],
               outY[gid*4], outY[gid*4+1], outY[gid*4+2], outY[gid*4+3]);
    }
}

__launch_bounds__(256, 2)
__global__ void fused_ec_hash(
    JacobianPoint* __restrict__ P,
    JacobianPoint* __restrict__ R,
    unsigned long long* __restrict__ start_scalars,
    unsigned long long* __restrict__ counts256,
    unsigned long long threadsTotal,
    uint32_t batch_size,
    uint32_t max_batches_per_launch,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    unsigned int* __restrict__ d_any_left
) {
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;

    const unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 65536u
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    JacobianPoint P_local = P[gid];
    if (lane == 0 && blockIdx.x == 0) {
        printf("fused_ec_hash: gid=%llu, P_local.x=%llx:%llx:%llx:%llx, P_local.y=%llx:%llx:%llx:%llx, P_local.z=%llx:%llx:%llx:%llx, P_local.infinity=%d\n",
               gid, P_local.x[0], P_local.x[1], P_local.x[2], P_local.x[3],
               P_local.y[0], P_local.y[1], P_local.y[2], P_local.y[3],
               P_local.z[0], P_local.z[1], P_local.z[2], P_local.z[3], P_local.infinity);
    }
    unsigned long long S[4], rem[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        S[i] = start_scalars[gid*4 + i];
        rem[i] = counts256[gid*4 + i];
    }

    if (isZero256(rem)) {
        R[gid] = P_local;
        WARP_FLUSH_HASHES();
        return;
    }

    uint32_t batches_done = 0;
    extern __shared__ unsigned long long shared_mem[];
    unsigned long long* z_values = shared_mem;

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (unsigned long long)B)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) {
            WARP_FLUSH_HASHES();
            return;
        }

        // Batch point additions
        for (int i = 0; i < half; ++i) {
            if (lane + i * WARP_SIZE >= batch_size / 2) continue;
            JacobianPoint Q;
            if ((lane + i * WARP_SIZE) * 4 < batch_size * 4) {
                fieldCopy(c_Gx + (lane + i * WARP_SIZE) * 4, Q.x);
                fieldCopy(c_Gy + (lane + i * WARP_SIZE) * 4, Q.y);
            } else {
                fieldSetZero(Q.x);
                fieldSetZero(Q.y);
            }
            fieldSetOne(Q.z);
            Q.infinity = false;
            pointAddMixed(P_local, Q.x, Q.y, Q.infinity, P_local);
            if (lane + half + i * WARP_SIZE >= batch_size / 2) continue;
            if ((lane + half + i * WARP_SIZE) * 4 < batch_size * 4) {
                fieldCopy(c_Gx + (lane + half + i * WARP_SIZE) * 4, Q.x);
                fieldCopy(c_Gy + (lane + half + i * WARP_SIZE) * 4, Q.y);
            } else {
                fieldSetZero(Q.x);
                fieldSetZero(Q.y);
            }
            pointAddMixed(P_local, Q.x, Q.y, Q.infinity, P_local);
            if (lane == 0 && blockIdx.x == 0) {
                printf("fused_ec_hash: Block %d, after pointAddMixed, P_local.x=%llx:%llx:%llx:%llx, P_local.z=%llx:%llx:%llx:%llx, P_local.infinity=%d\n",
                       blockIdx.x, P_local.x[0], P_local.x[1], P_local.x[2], P_local.x[3],
                       P_local.z[0], P_local.z[1], P_local.z[2], P_local.z[3], P_local.infinity);
            }
        }
        __syncthreads(); // Ensure all point additions are complete

        // Batch inversion
        bool skip_inversion = P_local.infinity || isZero256(P_local.z);
        int any_non_infinity = __syncthreads_count(!skip_inversion);
        if (lane < B && !skip_inversion) {
            fieldCopy(P_local.z, z_values + lane * 4);
            if (lane == 0 && blockIdx.x == 0) {
                printf("fused_ec_hash: Block %d, lane %d, writing z_values[0]=%llx:%llx:%llx:%llx\n",
                       blockIdx.x, lane, z_values[0], z_values[1], z_values[2], z_values[3]);
            }
        } else if (lane < B) {
            fieldSetZero(z_values + lane * 4);
            if (lane == 0 && blockIdx.x == 0) {
                printf("fused_ec_hash: Block %d, lane %d, setting z_values[0]=0:0:0:0 (skip_inversion=%d)\n",
                       blockIdx.x, lane, skip_inversion);
            }
        }
        __syncthreads();
        if (lane == 0 && any_non_infinity) {
            batch_modinv_fermat(z_values, z_values, B);
            printf("fused_ec_hash: Block %d, lane 0, after batch_modinv_fermat, z_values[0]=%llx:%llx:%llx:%llx\n",
                   blockIdx.x, z_values[0], z_values[1], z_values[2], z_values[3]);
        }
        __syncthreads();

        // Convert to affine and hash
        unsigned long long x_affine[4], y_affine[4];
        if (lane < B && !skip_inversion) {
            unsigned long long zinv[4], zinv2[4];
            fieldCopy(z_values + lane * 4, zinv);
            fieldSqr_opt_device(zinv, zinv2);
            fieldMul_opt_device(P_local.x, zinv2, x_affine);
            fieldMul_opt_device(zinv, zinv2, zinv2);
            fieldMul_opt_device(P_local.y, zinv2, y_affine);
            if (lane == 0 && blockIdx.x == 0) {
                printf("fused_ec_hash: Block %d, lane %d, x_affine=%llx:%llx:%llx:%llx, y_affine=%llx:%llx:%llx:%llx\n",
                       blockIdx.x, lane, x_affine[0], x_affine[1], x_affine[2], x_affine[3],
                       y_affine[0], y_affine[1], y_affine[2], y_affine[3]);
            }
        } else {
            fieldSetZero(x_affine);
            fieldSetZero(y_affine);
        }

        uint8_t h20[20];
        uint8_t prefix = (y_affine[0] & 1ULL) ? 0x03 : 0x02;
        getHash160_33_from_limbs(prefix, x_affine, h20);
        ++local_hashes;
        MAYBE_WARP_FLUSH();

        bool pref = hash160_prefix_equals(h20, c_target_prefix);
        if (__any_sync(full_mask, pref)) {
            if (pref && hash160_matches_prefix_then_full(h20, c_target_hash160, c_target_prefix)) {
                if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                    d_found_result->threadId = (int)gid;
                    d_found_result->iter = batches_done;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        d_found_result->scalar_val[i] = S[i];
                        d_found_result->Rx_val[i] = x_affine[i];
                        d_found_result->Ry_val[i] = y_affine[i];
                    }
                    atomicExch(d_found_flag, FOUND_READY);
                }
            }
        }

        sub256_u64_inplace(rem, (unsigned long long)B);
        inc256_device(S, (unsigned long long)B);
        batches_done++;
        if (isZero256(rem)) {
            atomicOr(d_any_left, 0u);
        } else {
            atomicOr(d_any_left, 1u);
        }
    }

    R[gid] = P_local;
    WARP_FLUSH_HASHES();
}

__global__ void precompute_batch_points_kernel(unsigned long long* d_Gx, unsigned long long* d_Gy, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size / 2) return;

    JacobianPoint G, tmp;
    fieldCopy(Gx_d, G.x);
    fieldCopy(Gy_d, G.y);
    fieldSetOne(G.z);
    G.infinity = false;

    // Compute 2^i * G for first half
    for (int i = 0; i < idx; ++i) {
        pointDoubleJacobian(G, tmp);
        G = tmp;
    }
    fieldCopy(G.x, d_Gx + idx * 4);
    fieldCopy(G.y, d_Gy + idx * 4);

    // Compute 2^(i + batch_size/2) * G for second half
    if (idx + batch_size / 2 < batch_size) {
        pointDoubleJacobian(G, tmp);
        G = tmp;
        fieldCopy(G.x, d_Gx + (idx + batch_size / 2) * 4);
        fieldCopy(G.y, d_Gy + (idx + batch_size / 2) * 4);
    }
}

__global__ void compute_phi_base_kernel(const unsigned long long* beta, const unsigned long long* Gx_d, unsigned long long* phi_base_x) {
    fieldMul_opt_device(beta, Gx_d, phi_base_x);
}

std::string human_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024 && unit_idx < 4) {
        size /= 1024;
        unit_idx++;
    }
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    return ss.str();
}

void precompute_g_table_gpu(JacobianPoint base, JacobianPoint phi_base, unsigned long long** d_pre_Gx, unsigned long long** d_pre_Gy, unsigned long long** d_pre_phiGx, unsigned long long** d_pre_phiGy) {
    size_t table_size = PRECOMPUTE_SIZE * 4 * sizeof(unsigned long long);
    size_t total_size = table_size * 4; // 4 tables: Gx, Gy, phiGx, phiGy
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    if (free_mem < total_size + 1e9) { // Reserve ~1GB for other allocations
        std::cerr << "Insufficient VRAM for 2^" << PRECOMPUTE_WINDOW << " precomputed tables (~" << human_bytes(total_size) << ")\n";
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMalloc(d_pre_Gx, table_size));
    CUDA_CHECK(cudaMalloc(d_pre_Gy, table_size));
    CUDA_CHECK(cudaMalloc(d_pre_phiGx, table_size));
    CUDA_CHECK(cudaMalloc(d_pre_phiGy, table_size));

    int threads = 256;
    int blocks = (PRECOMPUTE_SIZE + threads - 1) / threads;
    printf("precompute_g_table_gpu: base.x=%llx:%llx:%llx:%llx, base.y=%llx:%llx:%llx:%llx, base.infinity=%d\n",
           base.x[0], base.x[1], base.x[2], base.x[3], base.y[0], base.y[1], base.y[2], base.y[3], base.infinity);
    printf("precompute_g_table_gpu: phi_base.x=%llx:%llx:%llx:%llx, phi_base.y=%llx:%llx:%llx:%llx, phi_base.infinity=%d\n",
           phi_base.x[0], phi_base.x[1], phi_base.x[2], phi_base.x[3], phi_base.y[0], phi_base.y[1], phi_base.y[2], phi_base.y[3], phi_base.infinity);
    precompute_table_kernel<<<blocks, threads>>>(base, *d_pre_Gx, *d_pre_Gy, PRECOMPUTE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "precompute_table_kernel (Gx, Gy) completed" << std::endl;
    precompute_table_kernel<<<blocks, threads>>>(phi_base, *d_pre_phiGx, *d_pre_phiGy, PRECOMPUTE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "precompute_table_kernel (phiGx, phiGy) completed" << std::endl;
}

void print_gpu_info(const cudaDeviceProp& prop, int blocks, int threadsPerBlock, int batch_size, unsigned long long threadsTotal) {
    size_t table_size = PRECOMPUTE_SIZE * 4 * sizeof(unsigned long long) * 4; // 4 tables
    size_t mem_used = (threadsTotal * (4 * 3 + 4 + 4) * sizeof(unsigned long long)) + sizeof(FoundResult) +
                      sizeof(int) + sizeof(unsigned long long) + sizeof(unsigned int) + table_size;
    std::cout << "======== PrePhase: GPU Information ====================\n";
    std::cout << "Device               : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << "SM                   : " << prop.multiProcessorCount << "\n";
    std::cout << "ThreadsPerBlock      : " << threadsPerBlock << "\n";
    std::cout << "Blocks               : " << blocks << "\n";
    std::cout << "Points batch size    : " << batch_size << "\n";
    std::cout << "Batches/SM           : " << (batch_size / prop.multiProcessorCount) << "\n";
    std::cout << "Precomputed tables    : 2^" << PRECOMPUTE_WINDOW << " points (~" << human_bytes(table_size) << ")\n";
    std::cout << "Memory utilization   : " << std::fixed << std::setprecision(1)
              << (mem_used / (double)prop.totalGlobalMem) * 100.0 << "% ("
              << human_bytes(mem_used) << " / " << human_bytes(prop.totalGlobalMem) << ")\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Total threads        : " << threadsTotal << "\n";
}

int main(int argc, char* argv[]) {
    signal(SIGINT, handle_sigint);

    // Argument parsing
    unsigned long long range_start[4] = {0}, range_end[4] = {0}, range_len[4];
    uint8_t target_hash160[20] = {0};
    int blocks = 64, threadsPerBlock = 32; // Default to match --grid 64,32
    int batch_size = 8;
    uint32_t max_batches_per_launch = 64;
    std::string range_str, address_str, grid_str;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--range" && i + 1 < argc) {
            range_str = argv[++i];
        } else if (arg == "--address" && i + 1 < argc) {
            address_str = argv[++i];
        } else if (arg == "--grid" && i + 1 < argc) {
            grid_str = argv[++i];
        } else if (arg == "--slices" && i + 1 < argc) {
            max_batches_per_launch = std::atoi(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return EXIT_FAILURE;
        }
    }

    if (range_str.empty() || address_str.empty()) {
        std::cerr << "Usage: " << argv[0] << " --range START:END --address ADDRESS [--grid BLOCKS,THREADS] [--slices SLICES] [--verbose]\n";
        return EXIT_FAILURE;
    }

    // Parse range
    auto colon = range_str.find(':');
    if (colon == std::string::npos) {
        std::cerr << "Invalid range format. Use START:END\n";
        return EXIT_FAILURE;
    }
    std::string start_str = range_str.substr(0, colon);
    std::string end_str = range_str.substr(colon + 1);
    if (!hexToLE64(start_str, range_start) || !hexToLE64(end_str, range_end)) {
        std::cerr << "Invalid range hex values\n";
        return EXIT_FAILURE;
    }
    sub256(range_end, range_start, range_len);

    // Parse address
    if (!decode_p2pkh_address(address_str, target_hash160)) {
        std::cerr << "Invalid Bitcoin address\n";
        return EXIT_FAILURE;
    }

    // Parse grid and debug
    if (!grid_str.empty()) {
        auto comma = grid_str.find(',');
        if (comma != std::string::npos) {
            blocks = std::atoi(grid_str.substr(0, comma).c_str());
            threadsPerBlock = std::atoi(grid_str.substr(comma + 1).c_str());
            if (threadsPerBlock % WARP_SIZE != 0 || threadsPerBlock > 1024) {
                std::cerr << "Threads per block must be multiple of " << WARP_SIZE << " and <= 1024\n";
                return EXIT_FAILURE;
            }
        }
    }
    if (verbose) {
        std::cout << "Parsed grid: blocks=" << blocks << ", threadsPerBlock=" << threadsPerBlock << "\n";
    }

    // Validate batch size
    if (batch_size <= 0 || (batch_size & 1) || batch_size > MAX_BATCH_SIZE) {
        std::cerr << "Batch size must be even and <= " << MAX_BATCH_SIZE << "\n";
        return EXIT_FAILURE;
    }

    // Debug constants from CUDAStructures.h
    unsigned long long h_n[4], h_beta[4], h_b1[4], h_b2[4], h_a1[4], h_a2[4], h_p[4], h_mu[5];
    CUDA_CHECK(cudaMemcpyFromSymbol(h_n, c_n, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_beta, c_beta, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_b1, c_b1, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_b2, c_b2, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_a1, c_a1, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_a2, c_a2, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_p, c_p, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_mu, c_mu, 5 * sizeof(unsigned long long)));
    if (verbose) {
        std::cout << "c_n: " << std::hex << h_n[0] << ":" << h_n[1] << ":" << h_n[2] << ":" << h_n[3] << std::endl;
        std::cout << "c_beta: " << std::hex << h_beta[0] << ":" << h_beta[1] << ":" << h_beta[2] << ":" << h_beta[3] << std::endl;
        std::cout << "c_b1: " << std::hex << h_b1[0] << ":" << h_b1[1] << ":" << h_b1[2] << ":" << h_b1[3] << std::endl;
        std::cout << "c_b2: " << std::hex << h_b2[0] << ":" << h_b2[1] << ":" << h_b2[2] << ":" << h_b2[3] << std::endl;
        std::cout << "c_a1: " << std::hex << h_a1[0] << ":" << h_a1[1] << ":" << h_a1[2] << ":" << h_a1[3] << std::endl;
        std::cout << "c_a2: " << std::hex << h_a2[0] << ":" << h_a2[1] << ":" << h_a2[2] << ":" << h_a2[3] << std::endl;
        std::cout << "c_p: " << std::hex << h_p[0] << ":" << h_p[1] << ":" << h_p[2] << ":" << h_p[3] << std::endl;
        std::cout << "c_mu: " << std::hex << h_mu[0] << ":" << h_mu[1] << ":" << h_mu[2] << ":" << h_mu[3] << ":" << h_mu[4] << std::endl;
    }

    // Initialize Gx_d and Gy_d with correct secp256k1 generator point
    unsigned long long h_Gx_d[4] = {
        0x9f2815b16f81798ULL, 0x29bfcdb2dce28d95ULL, 0x55a06295ce870b07ULL, 0x79be667ef9dcbbacULL
    };
    unsigned long long h_Gy_d[4] = {
        0x9c47d08ffb10d4b8ULL, 0xfd17b448a6855419ULL, 0x5da4fbfc0e1108a8ULL, 0x483ada7726a3c465ULL
    };
    CUDA_CHECK(cudaMemcpyToSymbol(Gx_d, h_Gx_d, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyToSymbol(Gy_d, h_Gy_d, 4 * sizeof(unsigned long long)));

    // GPU setup
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    unsigned long long threadsTotal = (unsigned long long)blocks * threadsPerBlock;
    if (verbose) {
        print_gpu_info(prop, blocks, threadsPerBlock, batch_size, threadsTotal);
    }

    std::cout << "Batch size: " << batch_size << std::endl;

    // Precompute tables
    JacobianPoint h_base, h_phi_base;
    fieldCopy(h_Gx_d, h_base.x);
    fieldCopy(h_Gy_d, h_base.y);
    fieldSetOne(h_base.z);
    h_base.infinity = false;

    // Compute phi_base.x on GPU
    unsigned long long *d_beta, *d_Gx_d, *d_phi_base_x;
    CUDA_CHECK(cudaMalloc(&d_beta, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_Gx_d, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_phi_base_x, 4 * sizeof(unsigned long long)));
    unsigned long long h_beta_local[4] = {
        0x6B3C4F7EULL, 0x8DE6997DULL, 0x7CF27B18ULL, 0x00000000ULL
    };
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta_local, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Gx_d, h_Gx_d, 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    compute_phi_base_kernel<<<1, 1>>>(d_beta, d_Gx_d, d_phi_base_x);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "compute_phi_base_kernel completed" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_phi_base.x, d_phi_base_x, 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_Gx_d));
    CUDA_CHECK(cudaFree(d_phi_base_x));
    fieldCopy(h_Gy_d, h_phi_base.y);
    fieldSetOne(h_phi_base.z);
    h_phi_base.infinity = false;

    // Precompute tables
    unsigned long long *d_pre_Gx_local, *d_pre_Gy_local, *d_pre_phiGx_local, *d_pre_phiGy_local;
    precompute_g_table_gpu(h_base, h_phi_base, &d_pre_Gx_local, &d_pre_Gy_local, &d_pre_phiGx_local, &d_pre_phiGy_local);

    // Test precomputed tables
    unsigned long long *d_test_table_out;
    CUDA_CHECK(cudaMalloc(&d_test_table_out, PRECOMPUTE_SIZE * 8 * sizeof(unsigned long long)));
    test_precomputed_tables<<<(PRECOMPUTE_SIZE + 255) / 256, 256>>>(d_pre_Gx_local, d_pre_Gy_local, d_test_table_out, PRECOMPUTE_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_precomputed_tables completed" << std::endl;
    unsigned long long h_test_table_out[PRECOMPUTE_SIZE * 8];
    CUDA_CHECK(cudaMemcpy(h_test_table_out, d_test_table_out, PRECOMPUTE_SIZE * 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_test_table_out));
    if (verbose) {
        for (int i = 0; i < 2 && i < PRECOMPUTE_SIZE; ++i) {
            std::cout << "d_pre_Gx[" << i << "]: " << std::hex << h_test_table_out[i*8] << ":" << h_test_table_out[i*8+1] << ":"
                      << h_test_table_out[i*8+2] << ":" << h_test_table_out[i*8+3] << std::endl;
            std::cout << "d_pre_Gy[" << i << "]: " << std::hex << h_test_table_out[i*8+4] << ":" << h_test_table_out[i*8+5] << ":"
                      << h_test_table_out[i*8+6] << ":" << h_test_table_out[i*8+7] << std::endl;
        }
    }

    // Precompute batch points on GPU
    unsigned long long *d_Gx, *d_Gy;
    CUDA_CHECK(cudaMalloc(&d_Gx, batch_size * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_Gy, batch_size * 4 * sizeof(unsigned long long)));
    int threads = 256;
    int blocks_batch = (batch_size / 2 + threads - 1) / threads;
    precompute_batch_points_kernel<<<blocks_batch, threads>>>(d_Gx, d_Gy, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "precompute_batch_points_kernel completed" << std::endl;

    // Test c_Gx and c_Gy
    CUDA_CHECK(cudaMemcpyToSymbol(c_Gx, d_Gx, batch_size * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Gy, d_Gy, batch_size * 4 * sizeof(unsigned long long)));
    unsigned long long *d_test_out;
    CUDA_CHECK(cudaMalloc(&d_test_out, (batch_size / 2) * 8 * sizeof(unsigned long long)));
    test_constant_memory<<<1, threads>>>(d_test_out, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_constant_memory completed" << std::endl;
    unsigned long long h_test_out[(batch_size / 2) * 8];
    CUDA_CHECK(cudaMemcpy(h_test_out, d_test_out, (batch_size / 2) * 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_test_out));
    if (verbose) {
        std::cout << "c_Gx[0]: " << std::hex << h_test_out[0] << ":" << h_test_out[1] << ":"
                  << h_test_out[2] << ":" << h_test_out[3] << std::endl;
        std::cout << "c_Gy[0]: " << std::hex << h_test_out[4] << ":" << h_test_out[5] << ":"
                  << h_test_out[6] << ":" << h_test_out[7] << std::endl;
    }

    CUDA_CHECK(cudaFree(d_Gx));
    CUDA_CHECK(cudaFree(d_Gy));

    // Set target
    CUDA_CHECK(cudaMemcpyToSymbol(c_target_hash160, target_hash160, sizeof(target_hash160)));
    uint32_t target_prefix = *(uint32_t*)target_hash160;
    CUDA_CHECK(cudaMemcpyToSymbol(c_target_prefix, &target_prefix, sizeof(target_prefix)));

    // Allocate device memory
    unsigned long long *d_start_scalars, *d_counts256;
    unsigned long long *d_hashes_accum;
    int *d_found_flag;
    unsigned int *d_any_left;
    FoundResult *d_found_result;
    JacobianPoint *d_P, *d_R;
    CUDA_CHECK(cudaMalloc(&d_start_scalars, threadsTotal * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_counts256, threadsTotal * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_P, threadsTotal * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_R, threadsTotal * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_found_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found_result, sizeof(FoundResult)));
    CUDA_CHECK(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_any_left, sizeof(unsigned int)));

    // Initialize scalars and counts
    unsigned long long *h_start_scalars = nullptr, *h_counts256 = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_start_scalars, threadsTotal * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocHost(&h_counts256, threadsTotal * 4 * sizeof(unsigned long long)));
    for (unsigned long long i = 0; i < threadsTotal; ++i) {
        add256_u64(range_start, i, h_start_scalars + i * 4);
        unsigned long long end_plus_1[4];
        add256_u64(range_start, threadsTotal, end_plus_1);
        unsigned long long count[4];
        sub256(range_end, h_start_scalars + i * 4, count);
        if (ge256_u64(end_plus_1, range_end[0])) {
            unsigned long long remaining[4];
            sub256(end_plus_1, range_end, remaining);
            sub256(count, remaining, count);
        }
        fieldCopy(count, h_counts256 + i * 4);
    }
    if (verbose) {
        std::cout << "First scalar: " << std::hex << h_start_scalars[0] << ":" << h_start_scalars[1] << ":"
                  << h_start_scalars[2] << ":" << h_start_scalars[3] << std::endl;
        std::cout << "First count: " << std::hex << h_counts256[0] << ":" << h_counts256[1] << ":"
                  << h_counts256[2] << ":" << h_counts256[3] << std::endl;
    }
    CUDA_CHECK(cudaMemcpy(d_start_scalars, h_start_scalars, threadsTotal * 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_counts256, h_counts256, threadsTotal * 4 * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // Initialize points
    unsigned long long *d_outX, *d_outY;
    CUDA_CHECK(cudaMalloc(&d_outX, threadsTotal * 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_outY, threadsTotal * 4 * sizeof(unsigned long long)));
    scalarMulKernelBase<<<blocks, threadsPerBlock>>>(d_start_scalars, d_outX, d_outY, threadsTotal, d_pre_Gx_local, d_pre_Gy_local, d_pre_phiGx_local, d_pre_phiGy_local);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "scalarMulKernelBase completed" << std::endl;
    JacobianPoint *h_P = new JacobianPoint[threadsTotal];
    unsigned long long *h_outX = new unsigned long long[threadsTotal * 4], *h_outY = new unsigned long long[threadsTotal * 4];
    CUDA_CHECK(cudaMemcpy(h_outX, d_outX, threadsTotal * 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_outY, d_outY, threadsTotal * 4 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    for (unsigned long long i = 0; i < threadsTotal; ++i) {
        fieldCopy(h_outX + i * 4, h_P[i].x);
        fieldCopy(h_outY + i * 4, h_P[i].y);
        fieldSetOne(h_P[i].z);
        h_P[i].infinity = isZero256(h_outX + i * 4) && isZero256(h_outY + i * 4);
    }
    if (verbose) {
        std::cout << "d_P[0].x: " << std::hex << h_P[0].x[0] << ":" << h_P[0].x[1] << ":"
                  << h_P[0].x[2] << ":" << h_P[0].x[3] << std::endl;
        std::cout << "d_P[0].y: " << std::hex << h_P[0].y[0] << ":" << h_P[0].y[1] << ":"
                  << h_P[0].y[2] << ":" << h_P[0].y[3] << std::endl;
        std::cout << "d_P[0].infinity: " << h_P[0].infinity << std::endl;
    }
    CUDA_CHECK(cudaMemcpy(d_P, h_P, threadsTotal * sizeof(JacobianPoint), cudaMemcpyHostToDevice));
    delete[] h_P; delete[] h_outX; delete[] h_outY;
    CUDA_CHECK(cudaFree(d_outX));
    CUDA_CHECK(cudaFree(d_outY));

    // Initialize device memory
    CUDA_CHECK(cudaMemset(d_found_flag, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hashes_accum, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_any_left, 0, sizeof(unsigned int)));

    cudaStream_t streamKernel;
    CUDA_CHECK(cudaStreamCreate(&streamKernel));

    bool stop_all = false, completed_all = false;
    unsigned long long lastHashes = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;

    std::cout << "\n======== Phase-1: BruteForce (sliced) =================\n";

    while (!stop_all) {
        dim3 gridDim(blocks, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);
        size_t sharedMem = (batch_size + 1) * 4 * sizeof(unsigned long long); // Adjusted for n+1 elements
        fused_ec_hash<<<gridDim, blockDim, sharedMem, streamKernel>>>(
            d_P, d_R, d_start_scalars, d_counts256, threadsTotal, batch_size,
            max_batches_per_launch, d_found_flag, d_found_result, d_hashes_accum, d_any_left
        );
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            std::cerr << "\nKernel launch error: " << cudaGetErrorString(launchErr) << "\n";
            stop_all = true;
        }

        while (!stop_all) {
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - tLast).count();
            if (dt >= 1.0) {
                unsigned long long h_hashes = 0;
                CUDA_CHECK(cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
                double delta = (double)(h_hashes - lastHashes);
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
            if (qs == cudaSuccess) break;
            if (qs != cudaErrorNotReady) {
                CUDA_CHECK(cudaGetLastError());
                stop_all = true;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        CUDA_CHECK(cudaStreamSynchronize(streamKernel));
        std::cout.flush();
        if (stop_all || g_sigint) break;

        unsigned int h_any = 0;
        CUDA_CHECK(cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        std::swap(d_P, d_R);
        if (h_any == 0u) {
            completed_all = true;
            break;
        }
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
    CUDA_CHECK(cudaFree(d_found_flag));
    CUDA_CHECK(cudaFree(d_found_result));
    CUDA_CHECK(cudaFree(d_hashes_accum));
    CUDA_CHECK(cudaFree(d_any_left));
    CUDA_CHECK(cudaFree(d_pre_Gx_local));
    CUDA_CHECK(cudaFree(d_pre_Gy_local));
    CUDA_CHECK(cudaFree(d_pre_phiGx_local));
    CUDA_CHECK(cudaFree(d_pre_phiGy_local));
    if (h_start_scalars) CUDA_CHECK(cudaFreeHost(h_start_scalars));
    if (h_counts256) CUDA_CHECK(cudaFreeHost(h_counts256));
    CUDA_CHECK(cudaStreamDestroy(streamKernel));

    return exit_code;
}