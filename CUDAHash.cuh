#ifndef CUDA_HASH_CUH
#define CUDA_HASH_CUH

#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t ror32(uint32_t x, int n);
__device__ __forceinline__ uint32_t rol32(uint32_t x, int n);
__device__ __forceinline__ uint32_t bigS0(uint32_t x);
__device__ __forceinline__ uint32_t bigS1(uint32_t x);
__device__ __forceinline__ uint32_t smallS0(uint32_t x);
__device__ __forceinline__ uint32_t smallS1(uint32_t x);
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z);
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z);

__device__ __forceinline__ void SHA256Initialize(uint32_t s[8]);
__device__ void SHA256Transform(uint32_t state[8], const uint32_t W[16]);
__device__ __forceinline__ uint32_t pack_be4(uint8_t a, uint8_t b, uint8_t c, uint8_t d);
__device__ void SHA256_33_from_limbs(uint8_t prefix02_03, const uint64_t x_be_limbs[4], uint32_t out_state[8]);
__device__ void getSHA256_33bytes(const uint8_t* pubkey33, uint8_t sha[32]);

__device__ __forceinline__ uint32_t f1(uint32_t x, uint32_t y, uint32_t z);
__device__ __forceinline__ uint32_t f2(uint32_t x, uint32_t y, uint32_t z);
__device__ __forceinline__ uint32_t f3(uint32_t x, uint32_t y, uint32_t z);
__device__ __forceinline__ uint32_t f4(uint32_t x, uint32_t y, uint32_t z);
__device__ __forceinline__ uint32_t f5(uint32_t x, uint32_t y, uint32_t z);

__device__ void RIPEMD160Initialize(uint32_t s[5]);
__device__ void RIPEMD160Transform(uint32_t state[5], const uint32_t W[16]);
__device__ void RIPEMD160_from_SHA256_state(const uint32_t sha_state_be[8], uint8_t ripemd20[20]);
__device__ void getHash160_33_from_limbs(uint8_t prefix02_03, const uint64_t x_be_limbs[4], uint8_t out20[20]);
__device__ void getHash160_33bytes(const uint8_t* pubkey33, uint8_t hash20[20]);
__device__ void addBigEndian32(uint8_t* data32, uint64_t offset);
__device__ void batch_getHash160_33bytes(const uint8_t* pubkeys, uint8_t* hashes, int n);

// Additional declarations used in CUDACyclone.cu (may be in CUDAUtils.h)
__device__ bool hash160_prefix_equals(const uint8_t hash[20], uint32_t prefix);
__device__ bool hash160_matches_prefix_then_full(const uint8_t hash[20], const uint8_t target[20], uint32_t prefix);

#endif // CUDA_HASH_CUH