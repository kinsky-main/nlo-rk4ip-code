/**
 * @file test_run_store.c
 * @brief Stored runs: staging, locked merge, the contract `runs` row and reduced maps.
 *
 * Checks the `runs` row of a merged run against the ModelStore index contract
 * (version 1): the minimal DDL of the fixed columns, the key format, the
 * timestamp format, column names and value types (no BLOB parameters). Also
 * covers two concurrent writers, a writer killed mid-run, a stale lock, a lock
 * timeout with later recovery, an aborted run, migration of an old database,
 * key stability across backends, and reduced maps against the raw records in
 * 1D and 3D.
 *
 * POSIX only (fork, kill).
 */

#include "nlolib.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define PI 3.14159265358979323846

static int g_failures = 0;

#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);      \
            fprintf(stderr, __VA_ARGS__);                             \
            fprintf(stderr, "\n");                                    \
            g_failures += 1;                                          \
        }                                                             \
    } while (0)

/* -------------------------------------------------------------------------
 * Run fixtures
 * ---------------------------------------------------------------------- */

typedef struct {
    size_t nt;
    size_t nx;
    size_t ny;
    size_t n;
    simulation_config sim;
    physics_config phys;
    execution_options exec;
    storage_options storage;
    propagate_options options;
    nlo_complex* field;
    nlo_complex* omega;
    nlo_complex* potential;
} run_case;

static void case_free(run_case* rc)
{
    free(rc->field);
    free(rc->omega);
    free(rc->potential);
    memset(rc, 0, sizeof(*rc));
}

/* 1D soliton-like pulse: D = i c0 w^2, N = i A (c1 |A|^2). */
static void case_init_1d(run_case* rc, size_t nt, double amplitude, size_t records, const char* db_path)
{
    memset(rc, 0, sizeof(*rc));
    rc->nt = nt;
    rc->nx = 1u;
    rc->ny = 1u;
    rc->n = nt;
    const double dt = 0.05;
    rc->field = (nlo_complex*)calloc(nt, sizeof(nlo_complex));
    rc->omega = (nlo_complex*)calloc(nt, sizeof(nlo_complex));
    for (size_t i = 0u; i < nt; ++i) {
        const double t = ((double)i - 0.5 * (double)(nt - 1u)) * dt;
        const double env = amplitude / cosh(t / 0.3);
        rc->field[i] = make(env * cos(1.5 * t), env * sin(1.5 * t));
        const double step = 2.0 * PI / ((double)nt * dt);
        const double w = (i <= (nt - 1u) / 2u) ? (double)i * step : -((double)(nt - i)) * step;
        rc->omega[i] = make(w, 0.0);
    }
    rc->sim.propagation.propagation_distance = 0.4;
    rc->sim.propagation.starting_step_size = 1e-3;
    rc->sim.propagation.max_step_size = 2e-2;
    rc->sim.propagation.min_step_size = 1e-6;
    rc->sim.propagation.error_tolerance = 1e-7;
    rc->sim.time.pulse_period = (double)nt * dt;
    rc->sim.time.delta_time = dt;
    rc->sim.frequency.frequency_grid = rc->omega;
    rc->sim.spatial.nx = nt;
    rc->sim.spatial.ny = 1u;
    rc->sim.spatial.delta_x = 1.0;
    rc->sim.spatial.delta_y = 1.0;
    rc->phys.linear_factor_expr = "i*c0*w*w";
    rc->phys.nonlinear_expr = "i*A*(c1*I)";
    rc->phys.num_constants = 2u;
    rc->phys.constants[0] = -0.5 * 0.08;
    rc->phys.constants[1] = 1.0;

    rc->exec = (execution_options){0};
    rc->exec.backend_type = VECTOR_BACKEND_CPU;
    rc->exec.fft_backend = FFT_BACKEND_FFTW;
    rc->exec.device_heap_fraction = 0.5;

    rc->storage = (storage_options){0};
    rc->storage.sqlite_path = db_path;
    rc->storage.log_final_output_field_to_db = 1;
    rc->storage.record_mode = STORAGE_RECORD_MODE_REDUCED;
    rc->storage.binding = "c-test";

    rc->options = nlolib_propagate_options_default();
    rc->options.num_recorded_samples = records;
    rc->options.exec_options = &rc->exec;
    rc->options.storage_options = &rc->storage;
    rc->options.return_records = 0;
}

/* Small (t, x, y) GRIN-like run: diffraction, a parabolic potential and Kerr. */
static void case_init_3d(run_case* rc, size_t nt, size_t nx, size_t ny, size_t records, const char* db_path)
{
    case_init_1d(rc, nt, 1.0, records, db_path);
    free(rc->field);
    free(rc->omega);
    rc->omega = NULL;
    rc->nx = nx;
    rc->ny = ny;
    rc->n = nt * nx * ny;
    const double dt = 0.06;
    const double dx = 0.4;
    rc->field = (nlo_complex*)calloc(rc->n, sizeof(nlo_complex));
    rc->potential = (nlo_complex*)calloc(rc->n, sizeof(nlo_complex));
    for (size_t x = 0u; x < nx; ++x) {
        for (size_t y = 0u; y < ny; ++y) {
            const double xv = ((double)x - 0.5 * (double)(nx - 1u)) * dx;
            const double yv = ((double)y - 0.5 * (double)(ny - 1u)) * dx;
            const double r2 = xv * xv + yv * yv;
            for (size_t t = 0u; t < nt; ++t) {
                const double tv = ((double)t - 0.5 * (double)(nt - 1u)) * dt;
                const double env = exp(-0.5 * r2) / cosh(tv / 0.3);
                rc->field[((x * ny) + y) * nt + t] = make(env * cos(0.3 * xv), env * sin(0.2 * yv));
                rc->potential[((x * ny) + y) * nt + t] = make(0.05 * r2, 0.0);
            }
        }
    }
    rc->sim.time.pulse_period = (double)nt * dt;
    rc->sim.time.delta_time = dt;
    rc->sim.frequency.frequency_grid = NULL;
    rc->sim.tensor.nt = nt;
    rc->sim.tensor.nx = nx;
    rc->sim.tensor.ny = ny;
    rc->sim.tensor.layout = TENSOR_LAYOUT_XYT_T_FAST;
    rc->sim.spatial.nx = nx;
    rc->sim.spatial.ny = ny;
    rc->sim.spatial.delta_x = dx;
    rc->sim.spatial.delta_y = dx;
    rc->sim.spatial.potential_grid = rc->potential;
    rc->sim.propagation.propagation_distance = 0.2;
    rc->phys.linear_factor_expr = "i*(c0*(wt*wt) + c1*((kx*kx)+(ky*ky)))";
    rc->phys.nonlinear_expr = "i*A*(c2*I + V)";
    rc->phys.num_constants = 3u;
    rc->phys.constants[0] = -0.04;
    rc->phys.constants[1] = -0.08;
    rc->phys.constants[2] = 1.0;
}

static nlolib_status case_run(run_case* rc, nlo_complex* out_records, size_t capacity, storage_result* out_result)
{
    propagate_output output = nlolib_propagate_output_default();
    size_t written = 0u;
    output.records_written = &written;
    output.storage_result = out_result;
    if (out_records != NULL) {
        rc->options.return_records = 1;
        output.output_records = out_records;
        output.output_record_capacity = capacity;
    }
    return nlolib_propagate(&rc->sim, &rc->phys, rc->n, rc->field, &rc->options, &output);
}

/* -------------------------------------------------------------------------
 * File helpers
 * ---------------------------------------------------------------------- */

static void path_join(char* out, size_t cap, const char* a, const char* b)
{
    if (snprintf(out, cap, "%s/%s", a, b) >= (int)cap) {
        out[0] = '\0';
    }
}

static void make_store(const char* root, char* out_db, size_t cap)
{
    char dir[1024];
    (void)mkdir(root, 0775);
    path_join(dir, sizeof(dir), root, "nlolib");
    (void)mkdir(dir, 0775);
    path_join(out_db, cap, dir, "nlolib.sqlite");
}

static size_t count_files(const char* dir, const char* suffix)
{
    DIR* d = opendir(dir);
    if (d == NULL) {
        return 0u;
    }
    size_t count = 0u;
    struct dirent* ent = NULL;
    while ((ent = readdir(d)) != NULL) {
        const size_t len = strlen(ent->d_name);
        const size_t slen = strlen(suffix);
        if (len >= slen && strcmp(ent->d_name + len - slen, suffix) == 0) {
            count += 1u;
        }
    }
    closedir(d);
    return count;
}

static void staging_dir(const char* db, char* out, size_t cap)
{
    const char* slash = strrchr(db, '/');
    const int len = (slash != NULL) ? (int)(slash - db) : 0;
    (void)snprintf(out, cap, "%.*s/staging", len, db);
}

static long long sql_int(sqlite3* db, const char* sql)
{
    sqlite3_stmt* stmt = NULL;
    long long value = -1;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW) {
        value = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return value;
}

static void sql_text(sqlite3* db, const char* sql, char* out, size_t cap)
{
    sqlite3_stmt* stmt = NULL;
    out[0] = '\0';
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW &&
        sqlite3_column_type(stmt, 0) != SQLITE_NULL) {
        (void)snprintf(out, cap, "%s", (const char*)sqlite3_column_text(stmt, 0));
    }
    sqlite3_finalize(stmt);
}

static sqlite3* open_ro(const char* path)
{
    sqlite3* db = NULL;
    if (sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, NULL) != SQLITE_OK) {
        sqlite3_close(db);
        return NULL;
    }
    sqlite3_busy_timeout(db, 60000);
    return db;
}

/* -------------------------------------------------------------------------
 * Contract checks
 * ---------------------------------------------------------------------- */

static int is_key(const char* text)
{
    if (text == NULL || strlen(text) != 16u) {
        return 0;
    }
    for (size_t i = 0u; i < 16u; ++i) {
        if (!((text[i] >= '0' && text[i] <= '9') || (text[i] >= 'a' && text[i] <= 'f'))) {
            return 0;
        }
    }
    return 1;
}

/* YYYY-MM-DDTHH:MM:SS or YYYY-MM-DDTHH:MM:SS.sss */
static int is_contract_timestamp(const char* text)
{
    static const char pattern[] = "dddd-dd-ddTdd:dd:dd";
    const size_t len = strlen(text);
    if (len != 19u && len != 23u) {
        return 0;
    }
    for (size_t i = 0u; i < 19u; ++i) {
        if (pattern[i] == 'd' ? !isdigit((unsigned char)text[i]) : text[i] != pattern[i]) {
            return 0;
        }
    }
    if (len == 23u) {
        return text[19] == '.' && isdigit((unsigned char)text[20]) && isdigit((unsigned char)text[21]) &&
               isdigit((unsigned char)text[22]);
    }
    return 1;
}

/* Every rule of the contract that a `runs` table and its rows can be checked against. */
static void check_runs_contract(const char* db_path)
{
    sqlite3* db = open_ro(db_path);
    CHECK(db != NULL, "cannot open %s", db_path);
    if (db == NULL) {
        return;
    }

    char mode[32];
    sql_text(db, "PRAGMA journal_mode;", mode, sizeof(mode));
    CHECK(strcmp(mode, "delete") == 0, "journal mode is '%s', expected rollback journal 'delete'", mode);

    /* Fixed columns exactly as the minimal DDL. */
    static const struct {
        const char* name;
        const char* type;
        int notnull;
        int pk;
    } fixed[] = {
        {"key", "TEXT", 0, 1},
        {"kind", "TEXT", 1, 0},
        {"file", "TEXT", 1, 0},
        {"created_at", "TEXT", 0, 0},
        {"git_commit", "TEXT", 0, 0},
        {"julia_version", "TEXT", 0, 0},
    };
    sqlite3_stmt* stmt = NULL;
    CHECK(sqlite3_prepare_v2(db, "PRAGMA table_info(runs);", -1, &stmt, NULL) == SQLITE_OK, "table_info(runs)");
    char names[128][64];
    char types[128][16];
    size_t ncols = 0u;
    while (stmt != NULL && sqlite3_step(stmt) == SQLITE_ROW && ncols < 128u) {
        const int cid = sqlite3_column_int(stmt, 0);
        const char* name = (const char*)sqlite3_column_text(stmt, 1);
        const char* type = (const char*)sqlite3_column_text(stmt, 2);
        const int notnull = sqlite3_column_int(stmt, 3);
        const int pk = sqlite3_column_int(stmt, 5);
        (void)snprintf(names[ncols], sizeof(names[ncols]), "%s", name);
        (void)snprintf(types[ncols], sizeof(types[ncols]), "%s", type != NULL ? type : "");
        if (cid < 6) {
            CHECK(strcmp(name, fixed[cid].name) == 0, "fixed column %d is '%s', expected '%s'", cid, name, fixed[cid].name);
            CHECK(strcmp(type, fixed[cid].type) == 0, "column %s type '%s'", name, type);
            CHECK(notnull == fixed[cid].notnull, "column %s notnull %d", name, notnull);
            CHECK(pk == fixed[cid].pk, "column %s pk %d", name, pk);
        } else {
            const size_t len = strlen(name);
            CHECK(len >= 1u && len <= 63u, "parameter name '%s' length %zu", name, len);
            CHECK(strncmp(name, "nlolib_", 7u) == 0, "parameter '%s' lacks the nlolib_ prefix", name);
            for (size_t i = 0u; i < len; ++i) {
                CHECK(!isupper((unsigned char)name[i]), "parameter '%s' is not lowercase", name);
            }
            CHECK(strcmp(type, "INTEGER") == 0 || strcmp(type, "REAL") == 0 || strcmp(type, "TEXT") == 0,
                  "parameter '%s' declared '%s'", name, type);
        }
        ncols += 1u;
    }
    sqlite3_finalize(stmt);
    CHECK(ncols > 6u, "runs has no parameter columns");
    for (size_t i = 0u; i < ncols; ++i) {
        for (size_t j = i + 1u; j < ncols; ++j) {
            CHECK(sqlite3_stricmp(names[i], names[j]) != 0, "columns '%s' and '%s' differ only in case", names[i], names[j]);
        }
    }

    /* Values: fixed columns text, key format, timestamp, parameters never BLOB and of their declared type. */
    CHECK(sqlite3_prepare_v2(db, "SELECT * FROM runs;", -1, &stmt, NULL) == SQLITE_OK, "select runs");
    size_t rows = 0u;
    while (stmt != NULL && sqlite3_step(stmt) == SQLITE_ROW) {
        rows += 1u;
        const char* key = (const char*)sqlite3_column_text(stmt, 0);
        CHECK(sqlite3_column_type(stmt, 0) == SQLITE_TEXT && is_key(key), "key '%s' is not 16 lowercase hex", key);
        CHECK(sqlite3_column_type(stmt, 1) == SQLITE_TEXT &&
                  strcmp((const char*)sqlite3_column_text(stmt, 1), "nlolib") == 0,
              "kind is not 'nlolib'");
        CHECK(sqlite3_column_type(stmt, 2) == SQLITE_TEXT &&
                  strcmp((const char*)sqlite3_column_text(stmt, 2), "nlolib/nlolib.sqlite") == 0,
              "file is '%s'", (const char*)sqlite3_column_text(stmt, 2));
        CHECK(sqlite3_column_type(stmt, 3) == SQLITE_TEXT &&
                  is_contract_timestamp((const char*)sqlite3_column_text(stmt, 3)),
              "created_at '%s' is not contract ISO form", (const char*)sqlite3_column_text(stmt, 3));
        const char* commit = (const char*)sqlite3_column_text(stmt, 4);
        CHECK(sqlite3_column_type(stmt, 4) == SQLITE_TEXT && commit != NULL && commit[0] != '\0', "git_commit missing");
        if (commit != NULL && strncmp(commit, "nlolib-", 7u) != 0) {
            size_t hex = 0u;
            while (commit[hex] != '\0' && isxdigit((unsigned char)commit[hex])) {
                hex += 1u;
            }
            CHECK(hex == 40u && (commit[40] == '\0' || strcmp(commit + 40, "-dirty") == 0),
                  "git_commit '%s' is neither a commit hash nor nlolib-<release>", commit);
        }
        CHECK(sqlite3_column_type(stmt, 5) == SQLITE_NULL, "julia_version must be NULL for nlolib");
        for (int c = 6; c < sqlite3_column_count(stmt); ++c) {
            const int type = sqlite3_column_type(stmt, c);
            CHECK(type != SQLITE_BLOB, "parameter %s holds a BLOB", names[c]);
            if (type == SQLITE_INTEGER) {
                CHECK(strcmp(types[c], "INTEGER") == 0, "%s holds an integer but is %s", names[c], types[c]);
            } else if (type == SQLITE_FLOAT) {
                CHECK(strcmp(types[c], "REAL") == 0, "%s holds a real but is %s", names[c], types[c]);
            } else if (type == SQLITE_TEXT) {
                CHECK(strcmp(types[c], "TEXT") == 0, "%s holds text but is %s", names[c], types[c]);
            }
        }
    }
    sqlite3_finalize(stmt);
    CHECK(rows > 0u, "runs is empty");

    /* A row promises a complete run: every key has its data. */
    CHECK(sql_int(db, "SELECT COUNT(*) FROM runs r WHERE NOT EXISTS "
                      "(SELECT 1 FROM io_runs i JOIN io_record_layout l ON l.run_id = i.run_id WHERE i.key = r.key);") == 0,
          "a runs row has no data");
    char integrity[64];
    sql_text(db, "PRAGMA integrity_check;", integrity, sizeof(integrity));
    CHECK(strcmp(integrity, "ok") == 0, "integrity_check: %s", integrity);
    sqlite3_close(db);
}

/* -------------------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------------- */

static void test_basic_run(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    run_case rc;
    case_init_1d(&rc, 128u, 1.0, 9u, db_path);

    char key[STORAGE_KEY_MAX] = {0};
    CHECK(nlolib_storage_run_key(&rc.sim, &rc.phys, rc.n, rc.field, &rc.options, key) == NLOLIB_STATUS_OK, "run_key");
    CHECK(is_key(key), "key '%s'", key);
    int present = 1;
    CHECK(nlolib_storage_has_run(db_path, key, &present) == NLOLIB_STATUS_OK && present == 0, "has_run before");

    storage_result result;
    memset(&result, 0, sizeof(result));
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK, "propagate");
    CHECK(result.merged == 1, "not merged");
    CHECK(strcmp(result.key, key) == 0, "stored key %s != computed %s", result.key, key);
    CHECK(strcmp(result.run_id, key) == 0, "default run_id should be the key");
    CHECK(nlolib_storage_has_run(db_path, key, &present) == NLOLIB_STATUS_OK && present == 1, "has_run after");

    char sdir[1024];
    staging_dir(db_path, sdir, sizeof(sdir));
    CHECK(count_files(sdir, ".sqlite") == 0u, "staging file left behind");
    char lock[1100];
    (void)snprintf(lock, sizeof(lock), "%s.lock", db_path);
    CHECK(access(lock, F_OK) != 0, "lock file left behind");

    check_runs_contract(db_path);

    sqlite3* db = open_ro(db_path);
    char text[256];
    sql_text(db, "SELECT nlolib_binding FROM runs;", text, sizeof(text));
    CHECK(strcmp(text, "c-test") == 0, "nlolib_binding '%s'", text);
    sql_text(db, "SELECT nlolib_backend FROM runs;", text, sizeof(text));
    CHECK(strcmp(text, "cpu") == 0, "nlolib_backend '%s'", text);
    sql_text(db, "SELECT nlolib_record_mode FROM runs;", text, sizeof(text));
    CHECK(strcmp(text, "reduced") == 0, "nlolib_record_mode '%s'", text);
    CHECK(sql_int(db, "SELECT nlolib_nt FROM runs;") == 128, "nlolib_nt");
    CHECK(sql_int(db, "SELECT COUNT(*) FROM io_record_maps;") == 9, "map rows");
    CHECK(sql_int(db, "SELECT COUNT(*) FROM io_record_chunks;") == 0, "reduced mode must not store raw chunks");
    CHECK(sql_int(db, "SELECT COUNT(*) FROM io_final_output_fields;") == 1, "final field row");
    CHECK(sql_int(db, "SELECT length(power) FROM io_record_maps LIMIT 1;") == 128 * 8, "power blob bytes");
    CHECK(sql_int(db, "SELECT length(fluence_x) FROM io_record_maps LIMIT 1;") == 8, "fluence_x blob bytes");
    sqlite3_close(db);

    storage_run_layout layout;
    CHECK(nlolib_storage_read_run_layout(db_path, key, &layout) == NLOLIB_STATUS_OK, "read layout");
    CHECK(layout.nt == 128u && layout.nx == 1u && layout.ny == 1u && layout.num_records == 9u &&
              layout.has_final_field == 1 && layout.record_mode == STORAGE_RECORD_MODE_REDUCED,
          "layout fields");
    double z[9];
    size_t count = 0u;
    CHECK(nlolib_storage_read_run_array(db_path, key, "z", z, 9u, &count) == NLOLIB_STATUS_OK && count == 9u, "z");
    CHECK(fabs(z[0]) < 1e-15 && fabs(z[8] - 0.4) < 1e-12 && fabs(z[4] - 0.2) < 1e-12, "z axis values");

    /* Same run again replaces its row; a changed constant gets a new key. */
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.merged == 1, "second propagate");
    rc.phys.constants[1] = 1.1;
    char key2[STORAGE_KEY_MAX];
    CHECK(nlolib_storage_run_key(&rc.sim, &rc.phys, rc.n, rc.field, &rc.options, key2) == NLOLIB_STATUS_OK, "key2");
    CHECK(strcmp(key, key2) != 0, "constant change kept the key");
    rc.phys.constants[1] = 1.0;
    rc.field[3].re += 1e-12;
    CHECK(nlolib_storage_run_key(&rc.sim, &rc.phys, rc.n, rc.field, &rc.options, key2) == NLOLIB_STATUS_OK, "key3");
    CHECK(strcmp(key, key2) != 0, "input change kept the key");
    rc.field[3].re -= 1e-12;
    db = open_ro(db_path);
    CHECK(sql_int(db, "SELECT COUNT(*) FROM runs;") == 1, "rerun added a row");
    CHECK(sql_int(db, "SELECT COUNT(*) FROM io_runs;") == 1, "rerun left old data");
    CHECK(sql_int(db, "SELECT COUNT(*) FROM io_record_maps;") == 9, "rerun duplicated maps");
    sqlite3_close(db);

    /* The key ignores the backend: Vulkan writes the same key, replacing the CPU row. */
    rc.exec.backend_type = VECTOR_BACKEND_VULKAN;
    rc.exec.fft_backend = FFT_BACKEND_AUTO;
    char key_vk[STORAGE_KEY_MAX];
    CHECK(nlolib_storage_run_key(&rc.sim, &rc.phys, rc.n, rc.field, &rc.options, key_vk) == NLOLIB_STATUS_OK &&
              strcmp(key, key_vk) == 0,
          "key depends on backend");
    if (case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK) {
        CHECK(strcmp(result.key, key) == 0 && result.merged == 1, "vulkan run key");
        db = open_ro(db_path);
        sql_text(db, "SELECT nlolib_backend FROM runs;", text, sizeof(text));
        CHECK(strcmp(text, "vulkan") == 0, "backend column after vulkan run: %s", text);
        CHECK(sql_int(db, "SELECT COUNT(*) FROM runs;") == 1, "vulkan run added a second row");
        sqlite3_close(db);
    } else {
        printf("  (vulkan backend unavailable; skipped the vulkan half of the key check)\n");
    }
    case_free(&rc);
}

static double max_rel_diff(const double* a, const double* b, size_t n)
{
    double scale = 0.0;
    double diff = 0.0;
    for (size_t i = 0u; i < n; ++i) {
        scale = fmax(scale, fabs(b[i]));
        diff = fmax(diff, fabs(a[i] - b[i]));
    }
    return diff / fmax(scale, 1e-300);
}

/* Reduced maps against the same quantities computed here from the raw records (naive DFT). */
static void check_maps_against_records(const char* db_path, const char* key, const run_case* rc,
                                       const nlo_complex* records, size_t num_records)
{
    const size_t nt = rc->nt;
    const size_t nx = rc->nx;
    const size_t ny = rc->ny;
    double* power = (double*)calloc(num_records * nt, sizeof(double));
    double* spectrum = (double*)calloc(num_records * nt, sizeof(double));
    double* fx = (double*)calloc(num_records * nx, sizeof(double));
    double* fy = (double*)calloc(num_records * ny, sizeof(double));
    double* ref = (double*)calloc(num_records * nt, sizeof(double));
    double* omega = (double*)calloc(nt, sizeof(double));
    size_t count = 0u;
    CHECK(nlolib_storage_read_run_array(db_path, key, "power", power, num_records * nt, &count) == NLOLIB_STATUS_OK &&
              count == num_records * nt, "read power (%zu)", count);
    CHECK(nlolib_storage_read_run_array(db_path, key, "spectrum", spectrum, num_records * nt, &count) == NLOLIB_STATUS_OK &&
              count == num_records * nt, "read spectrum");
    CHECK(nlolib_storage_read_run_array(db_path, key, "fluence_x", fx, num_records * nx, &count) == NLOLIB_STATUS_OK &&
              count == num_records * nx, "read fluence_x");
    CHECK(nlolib_storage_read_run_array(db_path, key, "fluence_y", fy, num_records * ny, &count) == NLOLIB_STATUS_OK &&
              count == num_records * ny, "read fluence_y");
    CHECK(nlolib_storage_read_run_array(db_path, key, "omega", omega, nt, &count) == NLOLIB_STATUS_OK && count == nt,
          "read omega");

    /* omega ascending, step 2 pi / (nt dt), zero at index nt/2. */
    const double dw = 2.0 * PI / ((double)nt * rc->sim.time.delta_time);
    CHECK(fabs(omega[nt / 2u]) < 1e-12 && fabs(omega[1] - omega[0] - dw) < 1e-9 * dw, "omega axis");

    for (size_t r = 0u; r < num_records; ++r) {
        const nlo_complex* rec = records + r * rc->n;
        double* p = ref + r * nt;
        for (size_t i = 0u; i < nt; ++i) {
            p[i] = 0.0;
        }
        for (size_t l = 0u; l < nx * ny; ++l) {
            for (size_t t = 0u; t < nt; ++t) {
                p[t] += rec[l * nt + t].re * rec[l * nt + t].re + rec[l * nt + t].im * rec[l * nt + t].im;
            }
        }
    }
    double err = max_rel_diff(power, ref, num_records * nt);
    CHECK(err < 1e-12, "power map differs from records: %g", err);

    for (size_t r = 0u; r < num_records; ++r) {
        const nlo_complex* rec = records + r * rc->n;
        double* s = ref + r * nt;
        for (size_t j = 0u; j < nt; ++j) {
            /* ascending index j <-> FFT bin k = (j + nt - nt/2) mod nt, forward sign e^{-i 2 pi k n / nt} */
            const size_t k = (j + nt - nt / 2u) % nt;
            double acc = 0.0;
            for (size_t l = 0u; l < nx * ny; ++l) {
                double re = 0.0;
                double im = 0.0;
                for (size_t n = 0u; n < nt; ++n) {
                    const double ang = -2.0 * PI * (double)((k * n) % nt) / (double)nt;
                    const nlo_complex a = rec[l * nt + n];
                    re += a.re * cos(ang) - a.im * sin(ang);
                    im += a.re * sin(ang) + a.im * cos(ang);
                }
                acc += re * re + im * im;
            }
            s[j] = acc;
        }
    }
    err = max_rel_diff(spectrum, ref, num_records * nt);
    CHECK(err < 1e-10, "spectrum map differs from records: %g", err);

    for (size_t r = 0u; r < num_records; ++r) {
        const nlo_complex* rec = records + r * rc->n;
        for (size_t x = 0u; x < nx; ++x) {
            double acc = 0.0;
            for (size_t y = 0u; y < ny; ++y) {
                for (size_t t = 0u; t < nt; ++t) {
                    const nlo_complex a = rec[((x * ny) + y) * nt + t];
                    acc += a.re * a.re + a.im * a.im;
                }
            }
            ref[r * nx + x] = acc;
        }
    }
    err = max_rel_diff(fx, ref, num_records * nx);
    CHECK(err < 1e-12, "fluence_x differs: %g", err);
    for (size_t r = 0u; r < num_records; ++r) {
        const nlo_complex* rec = records + r * rc->n;
        for (size_t y = 0u; y < ny; ++y) {
            double acc = 0.0;
            for (size_t x = 0u; x < nx; ++x) {
                for (size_t t = 0u; t < nt; ++t) {
                    const nlo_complex a = rec[((x * ny) + y) * nt + t];
                    acc += a.re * a.re + a.im * a.im;
                }
            }
            ref[r * ny + y] = acc;
        }
    }
    err = max_rel_diff(fy, ref, num_records * ny);
    CHECK(err < 1e-12, "fluence_y differs: %g", err);

    free(power);
    free(spectrum);
    free(fx);
    free(fy);
    free(ref);
    free(omega);
}

static void test_maps(const char* root, int three_d)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    run_case rc;
    const size_t records = 5u;
    if (three_d) {
        case_init_3d(&rc, 32u, 6u, 4u, records, db_path);
    } else {
        case_init_1d(&rc, 96u, 1.6, records, db_path);
    }
    rc.storage.record_mode = STORAGE_RECORD_MODE_REDUCED_AND_RAW;
    nlo_complex* out = (nlo_complex*)calloc(records * rc.n, sizeof(nlo_complex));
    storage_result result;
    CHECK(case_run(&rc, out, records, &result) == NLOLIB_STATUS_OK && result.merged == 1, "propagate (%s)",
          three_d ? "3d" : "1d");
    check_maps_against_records(db_path, result.key, &rc, out, records);

    /* Raw records (stride 1) are the returned records, and the final field is the last one. */
    double* raw = (double*)calloc(2u * records * rc.n, sizeof(double));
    size_t count = 0u;
    CHECK(nlolib_storage_read_run_array(db_path, result.key, "raw", raw, 2u * records * rc.n, &count) ==
                  NLOLIB_STATUS_OK &&
              count == 2u * records * rc.n,
          "read raw (%zu)", count);
    CHECK(memcmp(raw, out, records * rc.n * sizeof(nlo_complex)) == 0, "raw records differ from returned records");
    CHECK(nlolib_storage_read_run_array(db_path, result.key, "final_field", raw, 2u * rc.n, &count) ==
                  NLOLIB_STATUS_OK &&
              count == 2u * rc.n,
          "read final field");
    CHECK(memcmp(raw, out + (records - 1u) * rc.n, rc.n * sizeof(nlo_complex)) == 0, "final field differs");

    if (three_d) {
        /* Downsampled raw records: every 2nd t and x, every y. */
        rc.storage.raw_stride_t = 2u;
        rc.storage.raw_stride_x = 2u;
        CHECK(case_run(&rc, out, records, &result) == NLOLIB_STATUS_OK && result.merged == 1, "strided run");
        storage_run_layout layout;
        CHECK(nlolib_storage_read_run_layout(db_path, result.key, &layout) == NLOLIB_STATUS_OK &&
                  layout.raw_nt == 16u && layout.raw_nx == 3u && layout.raw_ny == 4u && layout.raw_records == records,
              "strided layout %zu %zu %zu", layout.raw_nt, layout.raw_nx, layout.raw_ny);
        const size_t raw_samples = 16u * 3u * 4u;
        CHECK(nlolib_storage_read_run_array(db_path, result.key, "raw", raw, 2u * records * raw_samples, &count) ==
                      NLOLIB_STATUS_OK &&
                  count == 2u * records * raw_samples,
              "strided raw count %zu", count);
        int same = 1;
        for (size_t r = 0u; r < records && same; ++r) {
            size_t o = 0u;
            for (size_t x = 0u; x < 6u; x += 2u) {
                for (size_t y = 0u; y < 4u; ++y) {
                    for (size_t t = 0u; t < 32u; t += 2u) {
                        const nlo_complex a = out[r * rc.n + ((x * 4u) + y) * 32u + t];
                        const double* b = raw + 2u * (r * raw_samples + o);
                        same = same && a.re == b[0] && a.im == b[1];
                        o += 1u;
                    }
                }
            }
        }
        CHECK(same, "strided raw samples differ");
        check_runs_contract(db_path);
    }
    free(raw);
    free(out);
    case_free(&rc);
}

static void test_concurrent_writers(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    const int writers = 3;
    const int runs_each = 3;
    pid_t pids[8];
    for (int w = 0; w < writers; ++w) {
        pids[w] = fork();
        if (pids[w] == 0) {
            int bad = 0;
            for (int r = 0; r < runs_each; ++r) {
                run_case rc;
                case_init_1d(&rc, 128u, 0.5 + 0.1 * (double)(w * runs_each + r), 6u, db_path);
                storage_result result;
                if (case_run(&rc, NULL, 0u, &result) != NLOLIB_STATUS_OK || result.merged != 1) {
                    bad = 1;
                }
                case_free(&rc);
            }
            _exit(bad);
        }
    }
    for (int w = 0; w < writers; ++w) {
        int status = 0;
        (void)waitpid(pids[w], &status, 0);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "writer %d failed", w);
    }
    sqlite3* db = open_ro(db_path);
    CHECK(sql_int(db, "SELECT COUNT(*) FROM runs;") == writers * runs_each, "concurrent runs rows");
    CHECK(sql_int(db, "SELECT COUNT(*) FROM io_record_maps;") == writers * runs_each * 6, "concurrent map rows");
    sqlite3_close(db);
    char sdir[1024];
    staging_dir(db_path, sdir, sizeof(sdir));
    CHECK(count_files(sdir, ".sqlite") == 0u, "staging left after concurrent writers");
    check_runs_contract(db_path);
}

static int kill_after_steps(const progress_info* info, void* user)
{
    (void)user;
    if (info->step_index >= 20u) {
        (void)raise(SIGKILL);
    }
    return 1;
}

static int abort_after_steps(const progress_info* info, void* user)
{
    (void)user;
    return info->step_index < 20u;
}

static void file_digest(const char* path, unsigned long long* out_hash, long* out_size)
{
    FILE* f = fopen(path, "rb");
    unsigned long long h = 1469598103934665603ull;
    long size = 0;
    if (f != NULL) {
        int c = 0;
        while ((c = fgetc(f)) != EOF) {
            h = (h ^ (unsigned)c) * 1099511628211ull;
            size += 1;
        }
        fclose(f);
    }
    *out_hash = h;
    *out_size = size;
}

static void test_killed_and_aborted_writer(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    run_case rc;
    case_init_1d(&rc, 128u, 1.0, 6u, db_path);
    storage_result result;
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.merged == 1, "seed run");
    case_free(&rc);

    unsigned long long before = 0u;
    long before_size = 0;
    file_digest(db_path, &before, &before_size);

    pid_t pid = fork();
    if (pid == 0) {
        run_case child;
        case_init_1d(&child, 128u, 1.3, 50u, db_path);
        child.sim.propagation.max_step_size = 1e-3;
        child.options.progress_callback = kill_after_steps;
        (void)case_run(&child, NULL, 0u, NULL);
        _exit(3); /* not reached when the kill works */
    }
    int status = 0;
    (void)waitpid(pid, &status, 0);
    CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL, "child was not killed mid-run (status %d)", status);

    unsigned long long after = 0u;
    long after_size = 0;
    file_digest(db_path, &after, &after_size);
    CHECK(before == after && before_size == after_size, "killed writer changed the shared database");
    char sdir[1024];
    staging_dir(db_path, sdir, sizeof(sdir));
    CHECK(count_files(sdir, ".sqlite") == 1u, "killed writer should leave one staging file");

    size_t merged = 9u;
    size_t discarded = 9u;
    size_t skipped = 9u;
    CHECK(nlolib_storage_recover_staging(db_path, 0, 0.0, &merged, &discarded, &skipped) == NLOLIB_STATUS_OK &&
              merged == 0u && discarded == 0u && skipped == 1u,
          "recover without discard: %zu %zu %zu", merged, discarded, skipped);
    CHECK(nlolib_storage_recover_staging(db_path, 1, 0.0, &merged, &discarded, &skipped) == NLOLIB_STATUS_OK &&
              merged == 0u && discarded == 1u,
          "recover with discard: %zu %zu %zu", merged, discarded, skipped);
    CHECK(count_files(sdir, ".sqlite") == 0u && count_files(sdir, "-journal") == 0u, "discard left files");
    file_digest(db_path, &after, &after_size);
    CHECK(before == after, "recovery of an incomplete run changed the shared database");

    /* A progress-callback abort is an interrupted run: nothing reaches the shared database. */
    run_case ab;
    case_init_1d(&ab, 128u, 1.4, 50u, db_path);
    ab.sim.propagation.max_step_size = 1e-3;
    ab.options.progress_callback = abort_after_steps;
    CHECK(case_run(&ab, NULL, 0u, &result) == NLOLIB_STATUS_ABORTED, "abort status");
    CHECK(result.merged == 0, "aborted run merged");
    case_free(&ab);
    file_digest(db_path, &after, &after_size);
    CHECK(before == after, "aborted run changed the shared database");
    CHECK(count_files(sdir, ".sqlite") == 0u, "aborted run left a staging file");
    check_runs_contract(db_path);
}

static void write_lock(const char* db_path, const char* host, unsigned long pid)
{
    char lock[1100];
    (void)snprintf(lock, sizeof(lock), "%s.lock", db_path);
    FILE* f = fopen(lock, "wb");
    fprintf(f, "nlolib-lock 1\nhost=%s\npid=%lu\ncreated=2026-01-01T00:00:00.000\nupdated=x\nbeat=0\ntoken=feedfeedfeedfeed\n",
            host, pid);
    fclose(f);
}

static void this_host(char* out, size_t cap)
{
    char raw[256] = {0};
    (void)gethostname(raw, sizeof(raw) - 1u);
    size_t j = 0u;
    for (size_t i = 0u; raw[i] != '\0' && raw[i] != '.' && j + 1u < cap; ++i) {
        const char c = raw[i];
        out[j++] = (isalnum((unsigned char)c) || c == '-' || c == '_') ? c : '_';
    }
    out[j] = '\0';
}

static void test_locks(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    char lock[1100];
    (void)snprintf(lock, sizeof(lock), "%s.lock", db_path);
    char sdir[1024];
    staging_dir(db_path, sdir, sizeof(sdir));
    char host[128];
    this_host(host, sizeof(host));

    /* Stale lock of a dead process on this host: taken over at once. */
    pid_t dead = fork();
    if (dead == 0) {
        _exit(0);
    }
    (void)waitpid(dead, NULL, 0);
    write_lock(db_path, host, (unsigned long)dead);
    run_case rc;
    case_init_1d(&rc, 64u, 1.0, 4u, db_path);
    storage_result result;
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.merged == 1, "stale local lock not broken");
    CHECK(access(lock, F_OK) != 0, "lock left after stale takeover");

    /* A live lock of another machine: the writer times out, keeps its complete staging file,
     * and the shared database is untouched; recovery merges it later. */
    setenv("NLOLIB_STORE_LOCK_TIMEOUT_S", "1.5", 1);
    setenv("NLOLIB_STORE_LOCK_STALE_S", "600", 1);
    write_lock(db_path, "some-other-host", 4242u);
    rc.field[0].re += 0.25;
    char key[STORAGE_KEY_MAX];
    CHECK(nlolib_storage_run_key(&rc.sim, &rc.phys, rc.n, rc.field, &rc.options, key) == NLOLIB_STATUS_OK, "key");
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK, "propagate under foreign lock");
    CHECK(result.merged == 0, "merged although another machine held the lock");
    CHECK(count_files(sdir, ".sqlite") == 1u, "complete staging file not kept");
    int present = 1;
    CHECK(nlolib_storage_has_run(db_path, key, &present) == NLOLIB_STATUS_OK && present == 0, "row written under foreign lock");
    size_t merged = 0u;
    CHECK(nlolib_storage_recover_staging(db_path, 1, 0.0, &merged, NULL, NULL) != NLOLIB_STATUS_OK && merged == 0u,
          "recovery must also wait for the foreign lock");
    CHECK(count_files(sdir, ".sqlite") == 1u, "recovery under foreign lock lost the staging file");

    /* The foreign lock goes stale (unchanged for longer than the stale age): it is broken. */
    setenv("NLOLIB_STORE_LOCK_STALE_S", "0.5", 1);
    setenv("NLOLIB_STORE_LOCK_TIMEOUT_S", "10", 1);
    CHECK(nlolib_storage_recover_staging(db_path, 0, 0.0, &merged, NULL, NULL) == NLOLIB_STATUS_OK && merged == 1u,
          "recovery after stale foreign lock merged %zu", merged);
    CHECK(nlolib_storage_has_run(db_path, key, &present) == NLOLIB_STATUS_OK && present == 1, "recovered row missing");
    CHECK(count_files(sdir, ".sqlite") == 0u && access(lock, F_OK) != 0, "recovery left files");

    /* A lock released while waiting: the writer proceeds after it. */
    setenv("NLOLIB_STORE_LOCK_STALE_S", "600", 1);
    write_lock(db_path, "some-other-host", 4242u);
    pid_t releaser = fork();
    if (releaser == 0) {
        usleep(700000);
        unlink(lock);
        _exit(0);
    }
    rc.field[0].re += 0.25;
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.merged == 1, "writer did not wait for release");
    (void)waitpid(releaser, NULL, 0);
    unsetenv("NLOLIB_STORE_LOCK_TIMEOUT_S");
    unsetenv("NLOLIB_STORE_LOCK_STALE_S");
    case_free(&rc);
    check_runs_contract(db_path);
}

/* A database written by an older nlolib (WAL mode, io_runs without `key`) stays readable and gains runs. */
static void test_old_database(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    sqlite3* db = NULL;
    CHECK(sqlite3_open(db_path, &db) == SQLITE_OK, "create old db");
    const char* old_schema =
        "PRAGMA journal_mode=WAL;"
        "CREATE TABLE io_system_config (system_hash TEXT PRIMARY KEY, backend_type INTEGER NOT NULL,"
        " fft_backend INTEGER NOT NULL, num_time_samples INTEGER NOT NULL, nt INTEGER NOT NULL,"
        " nx INTEGER NOT NULL, ny INTEGER NOT NULL, device_heap_fraction REAL NOT NULL,"
        " record_ring_target INTEGER NOT NULL, forced_device_budget_bytes INTEGER NOT NULL,"
        " created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);"
        "CREATE TABLE io_sim_config (config_hash TEXT PRIMARY KEY, z_end REAL NOT NULL, h_start REAL NOT NULL,"
        " h_max REAL NOT NULL, h_min REAL NOT NULL, tolerance REAL NOT NULL, pulse_period REAL NOT NULL,"
        " delta_time REAL NOT NULL, disp_factor_expr TEXT, disp_expr TEXT, nonlinear_expr TEXT,"
        " constants_blob BLOB, created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);"
        "CREATE TABLE io_runs (run_id TEXT PRIMARY KEY, system_hash TEXT NOT NULL, config_hash TEXT NOT NULL,"
        " num_recorded_samples INTEGER NOT NULL, num_time_samples INTEGER NOT NULL, chunk_records INTEGER NOT NULL,"
        " cap_policy INTEGER NOT NULL, max_bytes INTEGER NOT NULL, records_written INTEGER NOT NULL DEFAULT 0,"
        " chunks_written INTEGER NOT NULL DEFAULT 0, db_size_bytes INTEGER NOT NULL DEFAULT 0,"
        " truncated INTEGER NOT NULL DEFAULT 0, created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);"
        "CREATE TABLE io_record_chunks (run_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,"
        " record_start INTEGER NOT NULL, record_count INTEGER NOT NULL, payload BLOB NOT NULL,"
        " PRIMARY KEY(run_id, chunk_index));"
        "INSERT INTO io_system_config VALUES('s',0,0,4,4,1,1,0.5,0,0,'x');"
        "INSERT INTO io_sim_config(config_hash,z_end,h_start,h_max,h_min,tolerance,pulse_period,delta_time)"
        " VALUES('c',1,1,1,1,1,1,1);"
        "INSERT INTO io_runs(run_id,system_hash,config_hash,num_recorded_samples,num_time_samples,chunk_records,"
        " cap_policy,max_bytes) VALUES('old-run','s','c',1,4,1,0,0);"
        "INSERT INTO io_record_chunks VALUES('old-run',0,0,1,zeroblob(64));";
    CHECK(sqlite3_exec(db, old_schema, NULL, NULL, NULL) == SQLITE_OK, "old schema");
    sqlite3_close(db);

    run_case rc;
    case_init_1d(&rc, 64u, 1.0, 3u, db_path);
    storage_result result;
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.merged == 1, "store into old database");
    case_free(&rc);
    db = open_ro(db_path);
    CHECK(sql_int(db, "SELECT COUNT(*) FROM io_record_chunks WHERE run_id='old-run';") == 1, "old data lost");
    CHECK(sql_int(db, "SELECT COUNT(*) FROM io_runs;") == 2, "io_runs rows after migration");
    sqlite3_close(db);
    check_runs_contract(db_path);
}

int main(int argc, char** argv)
{
    char root[1024];
    const char* base = (argc > 1) ? argv[1] : ".";
    (void)snprintf(root, sizeof(root), "%s/test_run_store_%ld", base, (long)getpid());
    (void)mkdir(root, 0775);
    (void)nlolib_set_log_level(NLOLIB_LOG_LEVEL_WARN);
    (void)nlolib_set_progress_options(0, 100, 0);

    char sub[1100];
    struct {
        const char* name;
        void (*fn)(const char*);
    } tests[] = {
        {"basic", test_basic_run},
        {"concurrent", test_concurrent_writers},
        {"killed", test_killed_and_aborted_writer},
        {"locks", test_locks},
        {"old_database", test_old_database},
    };
    for (size_t i = 0u; i < sizeof(tests) / sizeof(tests[0]); ++i) {
        path_join(sub, sizeof(sub), root, tests[i].name);
        const int before = g_failures;
        tests[i].fn(sub);
        printf("%s: %s\n", tests[i].name, g_failures == before ? "ok" : "FAILED");
    }
    path_join(sub, sizeof(sub), root, "maps_1d");
    {
        const int before = g_failures;
        test_maps(sub, 0);
        printf("maps_1d: %s\n", g_failures == before ? "ok" : "FAILED");
    }
    path_join(sub, sizeof(sub), root, "maps_3d");
    {
        const int before = g_failures;
        test_maps(sub, 1);
        printf("maps_3d: %s\n", g_failures == before ? "ok" : "FAILED");
    }

    if (g_failures == 0 && getenv("NLOLIB_TEST_KEEP") == NULL) {
        char cmd[1200];
        (void)snprintf(cmd, sizeof(cmd), "rm -rf '%s'", root);
        if (system(cmd) != 0) {
            fprintf(stderr, "could not remove %s\n", root);
        }
    } else {
        fprintf(stderr, "%d check(s) failed; files kept in %s\n", g_failures, root);
    }
    return g_failures == 0 ? 0 : 1;
}
