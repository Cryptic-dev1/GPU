#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstdint>
#include <string>
#include <sstream>

__host__ void add256_u64(const uint64_t a[4], uint64_t b, uint64_t out[4]);
__host__ void add256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]);
__host__ void sub256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]);
__host__ void inc256(uint64_t a[4], uint64_t inc);
__host__ void divmod_256_by_u64(const uint64_t value[4], uint64_t divisor, uint64_t quotient[4], uint64_t &remainder);
bool hexToLE64(const std::string& h_in, uint64_t w[4]);
bool hexToHash160(const std::string& h, uint8_t hash160[20]);
__device__ void inc256_device(uint64_t a[4], uint64_t inc);
__device__ uint32_t load_u32_le(const uint8_t* p);
__device__ bool hash160_matches_prefix_then_full(const uint8_t* h, const uint8_t* target, uint32_t target_prefix_le);
__device__ bool eq256_u64(const uint64_t a[4], uint64_t b);
__device__ bool hash160_prefix_equals(const uint8_t* h, uint32_t target_prefix);
__device__ bool ge256_u64(const uint64_t a[4], uint64_t b);
__device__ void sub256_u64_inplace(uint64_t a[4], uint64_t dec);
__device__ unsigned long long warp_reduce_add_ull(unsigned long long v);
std::string human_bytes(double bytes);
long double ld_from_u256(const uint64_t v[4]);
bool decode_p2pkh_address(const std::string& addr, uint8_t hash160[20]);

#endif // CUDA_UTILS_H