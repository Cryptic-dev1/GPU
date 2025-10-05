#ifndef CUDA_HASH_H
#define CUDA_HASH_H

#include <cstdint>

__device__ void getHash160_33_from_limbs(uint8_t prefix, const unsigned long long x[4], uint8_t out[20]);

#endif // CUDA_HASH_H