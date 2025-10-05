#include "CUDAMath.h"
#include "CUDAUtils.h"
#include "sha256.h"
#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <stdexcept>

__host__ void add256_u64(const unsigned long long a[4], unsigned long long b, unsigned long long out[4]) {
    __uint128_t sum = (__uint128_t)a[0] + b;
    out[0] = (unsigned long long)sum;
    unsigned long long carry = (unsigned long long)(sum >> 64);
    for (int i = 1; i < 4; ++i) {
        sum = (__uint128_t)a[i] + carry;
        out[i] = (unsigned long long)sum;
        carry = (unsigned long long)(sum >> 64);
    }
}

__host__ void add256(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]) {
    __uint128_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t s = (__uint128_t)a[i] + b[i] + carry;
        out[i] = (unsigned long long)s;
        carry = s >> 64;
    }
}

__host__ void sub256(const unsigned long long a[4], const unsigned long long b[4], unsigned long long out[4]) {
    unsigned long long borrow = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned long long bi = b[i] + borrow;
        if (a[i] < bi) {
            out[i] = (unsigned long long)(((__uint128_t(1) << 64) + a[i]) - bi);
            borrow = 1;
        } else {
            out[i] = a[i] - bi;
            borrow = 0;
        }
    }
}

__host__ void inc256(unsigned long long a[4], unsigned long long inc) {
    __uint128_t cur = (__uint128_t)a[0] + inc;
    a[0] = (unsigned long long)cur;
    unsigned long long carry = (unsigned long long)(cur >> 64);
    for (int i = 1; i < 4 && carry; ++i) {
        cur = (__uint128_t)a[i] + carry;
        a[i] = (unsigned long long)cur;
        carry = (unsigned long long)(cur >> 64);
    }
}

__host__ void divmod_256_by_u64(const unsigned long long value[4], unsigned long long divisor, unsigned long long quotient[4], unsigned long long &remainder) {
    remainder = 0;
    for (int i = 3; i >= 0; --i) {
        __uint128_t cur = (__uint128_t(remainder) << 64) | value[i];
        quotient[i] = (unsigned long long)(cur / divisor);
        remainder = (unsigned long long)(cur % divisor);
    }
}

bool hexToLE64(const std::string& h_in, unsigned long long w[4]) {
    std::string h = h_in;
    if (h.size() >= 2 && (h[0] == '0') && (h[1] == 'x' || h[1] == 'X')) h = h.substr(2);
    if (h.size() > 64) return false;
    if (h.size() < 64) h = std::string(64 - h.size(), '0') + h;
    if (h.size() != 64) return false;
    for (int i = 0; i < 4; ++i) {
        std::string part = h.substr(i * 16, 16);
        try {
            w[3 - i] = std::stoull(part, nullptr, 16);
        } catch (...) {
            return false;
        }
    }
    return true;
}

bool hexToHash160(const std::string& h, uint8_t hash160[20]) {
    if (h.size() != 40) return false;
    for (int i = 0; i < 20; ++i) {
        std::string byteStr = h.substr(i * 2, 2);
        try {
            hash160[i] = (uint8_t)std::stoul(byteStr, nullptr, 16);
        } catch (...) {
            return false;
        }
    }
    return true;
}

__device__ void inc256_device(unsigned long long a[4], unsigned long long inc) {
    unsigned __int128 cur = (unsigned __int128)a[0] + inc;
    a[0] = (unsigned long long)cur;
    unsigned long long carry = (unsigned long long)(cur >> 64);
    for (int i = 1; i < 4 && carry; ++i) {
        cur = (unsigned __int128)a[i] + carry;
        a[i] = (unsigned long long)cur;
        carry = (unsigned long long)(cur >> 64);
    }
}

__device__ uint32_t load_u32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ bool hash160_matches_prefix_then_full(const uint8_t* h, const uint8_t* target, uint32_t target_prefix_le) {
    uint32_t h_prefix = load_u32_le(h);
    if (h_prefix != target_prefix_le) return false;
    for (int i = 0; i < 20; ++i) {
        if (h[i] != target[i]) return false;
    }
    return true;
}

__device__ bool eq256_u64(const unsigned long long a[4], unsigned long long b) {
    return a[0] == b && a[1] == 0ULL && a[2] == 0ULL && a[3] == 0ULL;
}

__device__ bool hash160_prefix_equals(const uint8_t* h, uint32_t target_prefix) {
    return load_u32_le(h) == target_prefix;
}

__device__ void sub256_u64_inplace(unsigned long long a[4], unsigned long long dec) {
    unsigned long long borrow = 0;
    unsigned long long temp = a[0];
    a[0] = temp - dec;
    borrow = (a[0] > temp) ? 1 : 0;
    for (int i = 1; i < 4 && borrow; ++i) {
        temp = a[i];
        a[i] = temp - borrow;
        borrow = (a[i] > temp) ? 1 : 0;
    }
}

__device__ unsigned long long warp_reduce_add_ull(unsigned long long v) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xFFFFFFFF, v, offset);
    }
    return v;
}

std::string human_bytes(double bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = bytes;
    while (size >= 1024 && unit_idx < 4) {
        size /= 1024;
        unit_idx++;
    }
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    return ss.str();
}

long double ld_from_u256(const unsigned long long v[4]) {
    long double result = 0.0L;
    for (int i = 3; i >= 0; --i) {
        result = result * 18446744073709551616.0L + (long double)v[i];
    }
    return result;
}