#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstdint>
#include <string>
#include <sstream>

#define WARP_SIZE 32

__host__ void add256_u64(const unsigned long long a[4], unsigned long long b, unsigned long long out[4]);
__host__ void add256(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]);
__host__ void sub256(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]);
__host__ void inc256(unsigned long long a[4], unsigned long long inc);
__host__ void divmod_256_by_u64(const unsigned long long value[4], unsigned long long divisor, unsigned long long quotient[4], unsigned long long &remainder);
bool hexToLE64(const std::string& h_in, unsigned long long w[4]);
bool hexToHash160(const std::string& h, uint8_t hash160[20]);
bool decode_p2pkh_address(const std::string& addr, uint8_t out_hash160[20]);
__device__ void inc256_device(unsigned long long a[4], unsigned long long inc);
__device__ uint32_t load_u32_le(const uint8_t* p);
__device__ bool hash160_matches_prefix_then_full(const uint8_t* h, const uint8_t* target, uint32_t target_prefix_le);
__device__ bool eq256_u64(const unsigned long long a[4], unsigned long long b);
__device__ bool hash160_prefix_equals(const uint8_t* h, uint32_t target_prefix);
__device__ void sub256_u64_inplace(unsigned long long a[4], unsigned long long dec);
__device__ unsigned long long warp_reduce_add_ull(unsigned long long v);
std::string human_bytes(double bytes);
long double ld_from_u256(const unsigned long long v[4]);

#endif // CUDA_UTILS_H