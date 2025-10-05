#include "CUDAUtils.h"
#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <stdexcept>

__host__ void add256_u64(const uint64_t a[4], uint64_t b, uint64_t out[4]) {
    __uint128_t sum = (__uint128_t)a[0] + b;
    out[0] = (uint64_t)sum;
    uint64_t carry = (uint64_t)(sum >> 64);
    for (int i = 1; i < 4; ++i) {
        sum = (__uint128_t)a[i] + carry;
        out[i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }
}

__host__ void add256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    __uint128_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t s = (__uint128_t)a[i] + b[i] + carry;
        out[i] = (uint64_t)s;
        carry = s >> 64;
    }
}

__host__ void sub256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t bi = b[i] + borrow;
        if (a[i] < bi) {
            out[i] = (uint64_t)(((__uint128_t(1) << 64) + a[i]) - bi);
            borrow = 1;
        } else {
            out[i] = a[i] - bi;
            borrow = 0;
        }
    }
}

__host__ void inc256(uint64_t a[4], uint64_t inc) {
    __uint128_t cur = (__uint128_t)a[0] + inc;
    a[0] = (uint64_t)cur;
    uint64_t carry = (uint64_t)(cur >> 64);
    for (int i = 1; i < 4 && carry; ++i) {
        cur = (__uint128_t)a[i] + carry;
        a[i] = (uint64_t)cur;
        carry = (uint64_t)(cur >> 64);
    }
}

__host__ void divmod_256_by_u64(const uint64_t value[4], uint64_t divisor, uint64_t quotient[4], uint64_t &remainder) {
    remainder = 0;
    for (int i = 3; i >= 0; --i) {
        __uint128_t cur = (__uint128_t(remainder) << 64) | value[i];
        quotient[i] = (uint64_t)(cur / divisor);
        remainder = (uint64_t)(cur % divisor);
    }
}

bool hexToLE64(const std::string& h_in, uint64_t w[4]) {
    std::string h = h_in;
    if (h.size() >= 2 && (h[0] == '0') && (h[1] == 'x' || h[1] == 'X')) h = h.substr(2);
    if (h.size() > 64) return false;
    if (h.size() < 64) h = std::string(64 - h.size(), '0') + h;
    if (h.size() != 64) return false;
    for (int i = 0; i < 4; ++i) {
        std::string part = h.substr(i * 16, 16);
        w[3 - i] = std::stoull(part, nullptr, 16);
    }
    return true;
}

bool hexToHash160(const std::string& h, uint8_t hash160[20]) {
    if (h.size() != 40) return false;
    for (int i = 0; i < 20; ++i) {
        std::string byteStr = h.substr(i * 2, 2);
        hash160[i] = (uint8_t)std::stoul(byteStr, nullptr, 16);
    }
    return true;
}

__device__ void inc256_device(uint64_t a[4], uint64_t inc) {
    unsigned __int128 cur = (unsigned __int128)a[0] + inc;
    a[0] = (uint64_t)cur;
    uint64_t carry = (uint64_t)(cur >> 64);
    for (int i = 1; i < 4 && carry; ++i) {
        cur = (unsigned __int128)a[i] + carry;
        a[i] = (uint64_t)cur;
        carry = (uint64_t)(cur >> 64);
    }
}

__device__ uint32_t load_u32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ bool hash160_matches_prefix_then_full(const uint8_t* h, const uint8_t* target, uint32_t target_prefix_le) {
    if (load_u32_le(h) != target_prefix_le) return false;
    #pragma unroll
    for (int k = 4; k < 20; ++k) {
        if (h[k] != target[k]) return false;
    }
    return true;
}

__device__ bool eq256_u64(const uint64_t a[4], uint64_t b) {
    return (a[0] == b) & (a[1] == 0ull) & (a[2] == 0ull) & (a[3] == 0ull);
}

__device__ bool hash160_prefix_equals(const uint8_t* h, uint32_t target_prefix) {
    return load_u32_le(h) == target_prefix;
}

__device__ bool ge256_u64(const uint64_t a[4], uint64_t b) {
    if (a[3] | a[2] | a[1]) return true;
    return a[0] >= b;
}

__device__ void sub256_u64_inplace(uint64_t a[4], uint64_t dec) {
    uint64_t borrow = (a[0] < dec) ? 1ull : 0ull;
    a[0] = a[0] - dec;
    #pragma unroll
    for (int i = 1; i < 4; ++i) {
        uint64_t ai = a[i];
        uint64_t bi = borrow;
        a[i] = ai - bi;
        borrow = (ai < bi) ? 1ull : 0ull;
        if (!borrow) break;
    }
}

__device__ unsigned long long warp_reduce_add_ull(unsigned long long v) {
    unsigned mask = 0xFFFFFFFFu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

std::string human_bytes(double bytes) {
    static const char* u[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    int k = 0;
    while (bytes >= 1024.0 && k < 5) { bytes /= 1024.0; ++k; }
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o << std::setprecision(bytes < 10 ? 2 : 1) << bytes << " " << u[k];
    return o.str();
}

long double ld_from_u256(const uint64_t v[4]) {
    return std::ldexp((long double)v[3], 192) + std::ldexp((long double)v[2], 128) +
           std::ldexp((long double)v[1], 64) + (long double)v[0];
}

bool decode_p2pkh_address(const std::string& addr, uint8_t hash160[20]) {
    // Placeholder for address decoding (requires base58 and SHA256)
    // Implement actual decoding logic here or use an external library
    return hexToHash160(addr, hash160); // Simplified for now
}