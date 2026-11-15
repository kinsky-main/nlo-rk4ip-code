/**
 * @file sha256.c
 * @brief Minimal streaming SHA-256 (FIPS 180-4).
 */

#include "io/sha256.h"
#include <string.h>

static const uint32_t k_sha256_round[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static uint32_t rotr32(uint32_t value, unsigned bits)
{
    return (value >> bits) | (value << (32u - bits));
}

static void sha256_compress(sha256_ctx* ctx, const unsigned char block[64])
{
    uint32_t w[64];
    for (unsigned i = 0u; i < 16u; ++i) {
        w[i] = ((uint32_t)block[4u * i] << 24) |
               ((uint32_t)block[4u * i + 1u] << 16) |
               ((uint32_t)block[4u * i + 2u] << 8) |
               (uint32_t)block[4u * i + 3u];
    }
    for (unsigned i = 16u; i < 64u; ++i) {
        const uint32_t s0 = rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^ (w[i - 15u] >> 3);
        const uint32_t s1 = rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^ (w[i - 2u] >> 10);
        w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
    }

    uint32_t a = ctx->h[0];
    uint32_t b = ctx->h[1];
    uint32_t c = ctx->h[2];
    uint32_t d = ctx->h[3];
    uint32_t e = ctx->h[4];
    uint32_t f = ctx->h[5];
    uint32_t g = ctx->h[6];
    uint32_t h = ctx->h[7];
    for (unsigned i = 0u; i < 64u; ++i) {
        const uint32_t s1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
        const uint32_t ch = (e & f) ^ ((~e) & g);
        const uint32_t t1 = h + s1 + ch + k_sha256_round[i] + w[i];
        const uint32_t s0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    ctx->h[0] += a;
    ctx->h[1] += b;
    ctx->h[2] += c;
    ctx->h[3] += d;
    ctx->h[4] += e;
    ctx->h[5] += f;
    ctx->h[6] += g;
    ctx->h[7] += h;
}

void sha256_init(sha256_ctx* ctx)
{
    static const uint32_t initial[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    memcpy(ctx->h, initial, sizeof(initial));
    ctx->total_bytes = 0u;
    ctx->block_fill = 0u;
}

void sha256_update(sha256_ctx* ctx, const void* data, size_t len)
{
    const unsigned char* bytes = (const unsigned char*)data;
    ctx->total_bytes += (uint64_t)len;
    while (len > 0u) {
        const size_t take = (64u - ctx->block_fill < len) ? (64u - ctx->block_fill) : len;
        memcpy(ctx->block + ctx->block_fill, bytes, take);
        ctx->block_fill += take;
        bytes += take;
        len -= take;
        if (ctx->block_fill == 64u) {
            sha256_compress(ctx, ctx->block);
            ctx->block_fill = 0u;
        }
    }
}

void sha256_final(sha256_ctx* ctx, unsigned char out[32])
{
    const uint64_t bit_len = ctx->total_bytes * 8u;
    const unsigned char pad_start = 0x80u;
    const unsigned char zero = 0u;
    sha256_update(ctx, &pad_start, 1u);
    while (ctx->block_fill != 56u) {
        sha256_update(ctx, &zero, 1u);
    }
    unsigned char length_be[8];
    for (unsigned i = 0u; i < 8u; ++i) {
        length_be[i] = (unsigned char)(bit_len >> (56u - 8u * i));
    }
    sha256_update(ctx, length_be, 8u);
    for (unsigned i = 0u; i < 8u; ++i) {
        out[4u * i] = (unsigned char)(ctx->h[i] >> 24);
        out[4u * i + 1u] = (unsigned char)(ctx->h[i] >> 16);
        out[4u * i + 2u] = (unsigned char)(ctx->h[i] >> 8);
        out[4u * i + 3u] = (unsigned char)ctx->h[i];
    }
}

void sha256_final_hex(sha256_ctx* ctx, char out_hex[65])
{
    static const char digits[] = "0123456789abcdef";
    unsigned char raw[32];
    sha256_final(ctx, raw);
    for (unsigned i = 0u; i < 32u; ++i) {
        out_hex[2u * i] = digits[raw[i] >> 4];
        out_hex[2u * i + 1u] = digits[raw[i] & 0x0fu];
    }
    out_hex[64] = '\0';
}
