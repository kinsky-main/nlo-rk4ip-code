/**
 * @file run_store.h
 * @brief Run keys, staging databases and locked merges into a shared run database.
 *
 * A stored run streams into a per-run staging database in
 * `<dir of shared db>/staging/` and is merged into the shared database in a
 * short transaction while holding `<shared db>.lock`. See
 * docs/nlolib_run_database.md for the layout and the lock protocol.
 */
#pragma once

#include "core/state.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Maximum UTF-8 path length handled by the run store. */
#define RUN_STORE_PATH_MAX 4096u

/** @brief Result codes of run-store operations. */
typedef enum {
    RUN_STORE_OK = 0,
    /** The lock could not be taken before the timeout; nothing was changed. */
    RUN_STORE_LOCK_TIMEOUT = 1,
    /** The staging database holds no complete run (no `runs` row). */
    RUN_STORE_INCOMPLETE = 2,
    RUN_STORE_ERROR = 3
} run_store_status;

/** @brief Digests of the large inputs that enter a run key. */
typedef struct {
    char key[STORAGE_KEY_MAX];
    char input_sha256[65];
    char potential_sha256[65];
} run_store_key_info;

/**
 * @brief Compute the run key and input digests of a propagation request.
 *
 * The key is the first 16 hex digits of SHA-256 over a canonical text that
 * starts with `kind=nlolib` and covers grids, solver controls, operators,
 * constants, nonlinear/Raman settings, digests of every caller-supplied array
 * (input field, potential, frequency grids, axes, Raman response, record z
 * schedule) and the record specification. Backend and device settings are
 * excluded.
 *
 * @param config Merged simulation/physics configuration.
 * @param num_time_samples Flattened sample count of @p input_field.
 * @param input_field Input field.
 * @param num_recorded_samples Effective record count.
 * @param explicit_record_z Optional explicit record schedule (NULL: uniform).
 * @param explicit_record_z_count Entries in @p explicit_record_z.
 * @param storage Storage options (record mode, strides, final-field flag); may be NULL.
 * @param out_info Destination key and digests.
 * @return int 0 on success, nonzero on invalid input.
 */
int run_store_compute_key(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    size_t num_recorded_samples,
    const double* explicit_record_z,
    size_t explicit_record_z_count,
    const storage_options* storage,
    run_store_key_info* out_info
);

/**
 * @brief Create the staging folder of @p shared_path and choose a unique staging file.
 *
 * @param shared_path Shared run database path.
 * @param key Run key.
 * @param out_path Destination staging database path.
 * @param out_capacity Capacity of @p out_path in bytes.
 * @return int 0 on success.
 */
int run_store_make_staging_path(const char* shared_path, const char* key, char* out_path, size_t out_capacity);

/**
 * @brief Text stored in the `file` column: `<parent folder>/<file name>` of @p shared_path.
 */
void run_store_index_file(const char* shared_path, char* out_file, size_t out_capacity);

/** @brief Current UTC time as `YYYY-MM-DDTHH:MM:SS.sss`. */
void run_store_utc_now(char out_text[32]);

/**
 * @brief Merge a complete staging database into the shared database and delete it.
 *
 * Takes the lock, copies the run in one short transaction (raw chunks larger
 * than a few tens of MB go first in their own transactions), writes the
 * `runs` row in the last transaction, releases the lock and removes the
 * staging file. On any failure the staging file is kept.
 */
run_store_status run_store_merge(const char* staging_path, const char* shared_path);

/** @brief Remove a staging database and its journal. */
void run_store_discard(const char* staging_path);

/**
 * @brief Merge complete and optionally discard incomplete leftover staging files.
 *
 * Incomplete staging files whose writer is a live process on this host are kept;
 * complete ones are merged whoever wrote them.
 *
 * @param shared_path Shared run database path.
 * @param discard_incomplete Nonzero deletes staging files without a `runs` row.
 * @param min_age_seconds Incomplete files younger than this are kept.
 */
run_store_status run_store_recover(
    const char* shared_path,
    int discard_incomplete,
    double min_age_seconds,
    size_t* out_merged,
    size_t* out_discarded,
    size_t* out_skipped
);

/** @brief Set @p out_present to 1 when the shared database has a `runs` row for @p key. */
run_store_status run_store_has_key(const char* shared_path, const char* key, int* out_present);

/** @brief Read the layout of a stored run. */
run_store_status run_store_read_layout(const char* db_path, const char* key, storage_run_layout* out_layout);

/**
 * @brief Read one stored array of a run as float64 values.
 *
 * Names: z, t, omega, x, y, power, spectrum, fluence_x, fluence_y,
 * final_field, raw. Complex arrays (final_field, raw) are returned as
 * interleaved (re, im) pairs. With @p out NULL or @p capacity too small only
 * @p out_count is set.
 */
run_store_status run_store_read_array(
    const char* db_path,
    const char* key,
    const char* name,
    double* out,
    size_t capacity,
    size_t* out_count
);

/**
 * @brief SQLite VFS used to write staging and shared databases, or NULL for SQLite's default.
 *
 * Default: "unix-none" on POSIX systems, SQLite's native VFS on Windows.
 * SQLite's default Unix locking cannot commit on Linux SMB/CIFS mounts (SMB
 * refuses the read-to-write lock upgrade), so POSIX writers take no SQLite
 * locks and rely on the run store's own writer lock file. Override with the
 * environment variable NLOLIB_STORE_VFS (a VFS name such as "unix",
 * "unix-dotfile", "unix-none", or "default").
 */
const char* run_store_sqlite_vfs(void);

/**
 * @brief Lock-protocol tuning for tests: timeout to wait for the lock and the
 *        age after which an unchanged lock is stale (seconds; <= 0 keeps the default).
 *
 * Defaults: 600 s timeout, 120 s stale age, overridable by the environment
 * variables NLOLIB_STORE_LOCK_TIMEOUT_S and NLOLIB_STORE_LOCK_STALE_S.
 */
void run_store_set_lock_timing(double timeout_seconds, double stale_seconds);

#ifdef __cplusplus
}
#endif
