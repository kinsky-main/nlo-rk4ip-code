/**
 * @file sha256.h
 * @brief Minimal streaming SHA-256 (FIPS 180-4) used for run keys and array digests.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Streaming SHA-256 state. */
typedef struct {
    uint32_t h[8];
    uint64_t total_bytes;
    unsigned char block[64];
    size_t block_fill;
} sha256_ctx;

/** @brief Reset @p ctx to the SHA-256 initial state. */
void sha256_init(sha256_ctx* ctx);

/** @brief Absorb @p len bytes from @p data. */
void sha256_update(sha256_ctx* ctx, const void* data, size_t len);

/** @brief Finish the digest and write 32 raw bytes to @p out. */
void sha256_final(sha256_ctx* ctx, unsigned char out[32]);

/** @brief Finish the digest and write 64 lowercase hex digits plus NUL to @p out_hex. */
void sha256_final_hex(sha256_ctx* ctx, char out_hex[65]);

#ifdef __cplusplus
}
#endif
