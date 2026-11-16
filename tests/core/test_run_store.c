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
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
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
    (void)snprintf(lock, sizeof(lock), "%s.writer-lock", db_path);
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

    /* t, x, y centered on the grid's middle with the run's steps (layout document). */
    const char* axis_names[3] = {"t", "x", "y"};
    const size_t axis_len[3] = {nt, nx, ny};
    const double axis_step[3] = {rc->sim.time.delta_time,
                                 (rc->sim.spatial.delta_x > 0.0) ? rc->sim.spatial.delta_x : 1.0,
                                 (rc->sim.spatial.delta_y > 0.0) ? rc->sim.spatial.delta_y : 1.0};
    for (int a = 0; a < 3; ++a) {
        double axis[1024];
        size_t m = 0u;
        CHECK(nlolib_storage_read_run_array(db_path, key, axis_names[a], axis, 1024u, &m) == NLOLIB_STATUS_OK &&
                  m == axis_len[a], "read %s axis", axis_names[a]);
        for (size_t i = 0u; i < m; ++i) {
            const double expected = ((double)i - 0.5 * (double)(axis_len[a] - 1u)) * axis_step[a];
            CHECK(fabs(axis[i] - expected) <= 1e-12 * fmax(1.0, fabs(expected)), "%s axis[%zu]", axis_names[a], i);
        }
    }
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

/*
 * Readers while other runs merge. Writers take no SQLite locks by default, so a
 * read-only reader that opens during a merge sees its journal as hot and gets
 * "attempt to write a readonly database". has_run and the read functions must
 * wait that out, not report a stored run as missing or fail.
 */
static void test_readers_during_merge(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    run_case first;
    case_init_1d(&first, 256u, 0.4, 4u, db_path);
    storage_result stored;
    CHECK(case_run(&first, NULL, 0u, &stored) == NLOLIB_STATUS_OK && stored.merged == 1, "first run merged");
    case_free(&first);

    const int writers = 2;
    const int runs_each = 6;
    pid_t pids[2];
    for (int w = 0; w < writers; ++w) {
        pids[w] = fork();
        if (pids[w] == 0) {
            int bad = 0;
            for (int r = 0; r < runs_each; ++r) {
                run_case rc;
                case_init_1d(&rc, 4096u, 0.5 + 0.01 * (double)(w * runs_each + r), 32u, db_path);
                rc.storage.record_mode = STORAGE_RECORD_MODE_REDUCED_AND_RAW;
                storage_result result;
                if (case_run(&rc, NULL, 0u, &result) != NLOLIB_STATUS_OK || result.merged != 1) {
                    bad = 1;
                }
                case_free(&rc);
            }
            _exit(bad);
        }
    }
    long reads = 0;
    long missing = 0;
    long failed = 0;
    int live = writers;
    while (live > 0) {
        int present = 0;
        const nlolib_status has = nlolib_storage_has_run(db_path, stored.key, &present);
        storage_run_layout layout;
        const nlolib_status read = nlolib_storage_read_run_layout(db_path, stored.key, &layout);
        reads += 1;
        failed += (has != NLOLIB_STATUS_OK) + (read != NLOLIB_STATUS_OK);
        missing += (has == NLOLIB_STATUS_OK && !present);
        live = 0;
        for (int w = 0; w < writers; ++w) {
            if (pids[w] > 0) {
                int status = 0;
                if (waitpid(pids[w], &status, WNOHANG) == pids[w]) {
                    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "writer %d failed", w);
                    pids[w] = 0;
                } else {
                    live += 1;
                }
            }
        }
    }
    printf("readers_during_merge: %ld reads, %ld reported missing, %ld failed\n", reads, missing, failed);
    CHECK(missing == 0, "has_run reported a stored run missing %ld times", missing);
    CHECK(failed == 0, "reads failed %ld times during merges", failed);
    sqlite3* db = open_ro(db_path);
    CHECK(sql_int(db, "SELECT COUNT(*) FROM runs;") == 1 + writers * runs_each, "runs rows after readers test");
    sqlite3_close(db);
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
    (void)snprintf(lock, sizeof(lock), "%s.writer-lock", db_path);
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
    (void)snprintf(lock, sizeof(lock), "%s.writer-lock", db_path);
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


static const char* g_merge_tool = NULL;

static long long count_where_key(const char* db_path, const char* table, const char* key)
{
    sqlite3* db = open_ro(db_path);
    if (db == NULL) {
        return -1;
    }
    char sql[256];
    (void)snprintf(sql, sizeof(sql),
                   "SELECT COUNT(*) FROM %s WHERE run_id IN (SELECT run_id FROM io_runs WHERE key='%s');", table, key);
    const long long n = sql_int(db, sql);
    sqlite3_close(db);
    return n;
}

/* Other SQLite VFS choices still produce a valid shared database. */
static void test_vfs_choices(const char* root)
{
    static const char* vfs[] = {"unix-dotfile", "unix", "default"};
    (void)mkdir(root, 0775);
    for (size_t i = 0u; i < sizeof(vfs) / sizeof(vfs[0]); ++i) {
        char sub[1100];
        path_join(sub, sizeof(sub), root, vfs[i]);
        char db_path[1024];
        make_store(sub, db_path, sizeof(db_path));
        setenv("NLOLIB_STORE_VFS", vfs[i], 1);
        run_case rc;
        case_init_1d(&rc, 64u, 1.0, 4u, db_path);
        storage_result result;
        CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.merged == 1, "vfs %s", vfs[i]);
        case_free(&rc);
        unsetenv("NLOLIB_STORE_VFS");
        check_runs_contract(db_path);
    }
}

/* defer_merge leaves the complete run in staging/; the merge tool merges it. */
static void test_deferred_merge(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    char sdir[1024];
    staging_dir(db_path, sdir, sizeof(sdir));
    run_case rc;
    case_init_1d(&rc, 64u, 1.0, 4u, db_path);
    rc.storage.defer_merge = 1;
    storage_result result;
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK, "deferred run");
    CHECK(result.merged == 0 && result.deferred == 1, "deferred flags %d %d", result.merged, result.deferred);
    int present = 1;
    CHECK(nlolib_storage_has_run(db_path, result.key, &present) == NLOLIB_STATUS_OK && present == 0, "deferred run merged");
    CHECK(count_files(sdir, ".sqlite") == 1u, "deferred staging file missing");
    char key1[STORAGE_KEY_MAX];
    (void)snprintf(key1, sizeof(key1), "%s", result.key);

    /* The environment variable defers too. */
    rc.storage.defer_merge = 0;
    rc.field[1].re += 0.5;
    setenv("NLOLIB_STORE_DEFER_MERGE", "1", 1);
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.deferred == 1, "env deferred run");
    unsetenv("NLOLIB_STORE_DEFER_MERGE");
    CHECK(count_files(sdir, ".sqlite") == 2u, "env deferred staging file missing");
    case_free(&rc);

    if (g_merge_tool != NULL) {
        char store[1024];
        (void)snprintf(store, sizeof(store), "%s", root);
        char cmd[3000];
        (void)snprintf(cmd, sizeof(cmd), "'%s' -q '%s'", g_merge_tool, store);
        CHECK(system(cmd) == 0, "merge tool failed: %s", cmd);
    } else {
        size_t merged = 0u;
        CHECK(nlolib_storage_recover_staging(db_path, 0, 0.0, &merged, NULL, NULL) == NLOLIB_STATUS_OK && merged == 2u,
              "recover merged %zu", merged);
    }
    CHECK(nlolib_storage_has_run(db_path, key1, &present) == NLOLIB_STATUS_OK && present == 1, "first deferred run not merged");
    CHECK(nlolib_storage_has_run(db_path, result.key, &present) == NLOLIB_STATUS_OK && present == 1, "second deferred run not merged");
    CHECK(count_files(sdir, ".sqlite") == 0u, "staging left after the merge tool");
    check_runs_contract(db_path);
}

/* Recovery merges a pass of staging files under one lock and settles the view once, not once per file. */
static void test_recovery_pass_settles_once(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    char sdir[1024];
    staging_dir(db_path, sdir, sizeof(sdir));
    run_case rc;
    case_init_1d(&rc, 64u, 1.0, 4u, db_path);
    rc.storage.defer_merge = 1;
    const size_t runs = 6u;
    for (size_t i = 0u; i < runs; ++i) {
        rc.field[1].re += 0.25;
        CHECK(case_run(&rc, NULL, 0u, NULL) == NLOLIB_STATUS_OK, "deferred run %zu", i);
    }
    case_free(&rc);
    CHECK(count_files(sdir, ".sqlite") == runs, "deferred staging files");

    /* A settle of 0.5 s per file would make the pass take at least 3 s. */
    setenv("NLOLIB_STORE_SETTLE_S", "0.5", 1);
    struct timespec t0;
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    size_t merged = 0u;
    const nlolib_status status = nlolib_storage_recover_staging(db_path, 0, 0.0, &merged, NULL, NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    unsetenv("NLOLIB_STORE_SETTLE_S");
    const double elapsed = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    CHECK(status == NLOLIB_STATUS_OK && merged == runs, "pass merged %zu of %zu", merged, runs);
    CHECK(elapsed >= 0.5 && elapsed < 2.0, "pass took %.2f s; expected one 0.5 s settle", elapsed);
    CHECK(count_files(sdir, ".sqlite") == 0u, "staging left after the pass");
    char lock[1100];
    (void)snprintf(lock, sizeof(lock), "%s.writer-lock", db_path);
    CHECK(access(lock, F_OK) != 0, "writer lock kept after the pass");
    check_runs_contract(db_path);
}

/*
 * Two runs with different keys and one explicit run_id: the data tables are
 * keyed by run_id, so the second must never replace the first run's data. A
 * new run is refused before it starts; one staged before the first was merged
 * is refused at merge time and kept in staging. A rerun of the same key with
 * the same run_id still replaces its own run.
 */
static void test_run_id_conflict(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    char sdir[1024];
    staging_dir(db_path, sdir, sizeof(sdir));
    run_case rc;
    case_init_1d(&rc, 64u, 1.0, 4u, db_path);
    rc.storage.run_id = "shared-label";

    /* B is staged first and left for a later merge. */
    const nlo_complex a_sample = rc.field[3];
    rc.storage.defer_merge = 1;
    rc.field[3].re += 0.5;
    storage_result b;
    CHECK(case_run(&rc, NULL, 0u, &b) == NLOLIB_STATUS_OK && b.deferred == 1, "deferred run B");
    rc.field[3] = a_sample;

    /* A takes the label in the shared database. */
    rc.storage.defer_merge = 0;
    storage_result a;
    CHECK(case_run(&rc, NULL, 0u, &a) == NLOLIB_STATUS_OK && a.merged == 1, "run A");
    CHECK(strcmp(a.key, b.key) != 0, "A and B must have different keys");
    const long long a_maps = count_where_key(db_path, "io_record_maps", a.key);
    CHECK(a_maps == 4, "A maps %lld", a_maps);

    /* Merging B would put B's data under A's runs row: refused, B kept. */
    size_t merged = 9u;
    size_t skipped = 0u;
    CHECK(nlolib_storage_recover_staging(db_path, 0, 0.0, &merged, NULL, &skipped) != NLOLIB_STATUS_OK,
          "recovery merged a run whose run_id belongs to another key");
    CHECK(merged == 0u && skipped == 1u, "conflict pass merged %zu skipped %zu", merged, skipped);
    CHECK(count_files(sdir, ".sqlite") == 1u, "conflicting staging file not kept");
    int present = 0;
    CHECK(nlolib_storage_has_run(db_path, a.key, &present) == NLOLIB_STATUS_OK && present == 1, "A's runs row lost");
    CHECK(nlolib_storage_has_run(db_path, b.key, &present) == NLOLIB_STATUS_OK && present == 0, "B merged");
    CHECK(count_where_key(db_path, "io_record_maps", a.key) == a_maps, "A's maps changed");
    CHECK(count_where_key(db_path, "io_record_layout", a.key) == 1, "A's layout lost");
    CHECK(count_where_key(db_path, "io_final_output_fields", a.key) == 1, "A's final field lost");

    /* A new run with the taken label is refused before it starts and stages nothing. */
    rc.field[3].re += 1.0;
    CHECK(case_run(&rc, NULL, 0u, NULL) == NLOLIB_STATUS_INVALID_ARGUMENT, "new run with a taken run_id not refused");
    rc.field[3] = a_sample;
    CHECK(count_files(sdir, ".sqlite") == 1u, "refused run left a staging file");

    /* Rerunning A itself is allowed and replaces it. */
    storage_result again;
    memset(&again, 0, sizeof(again));
    const nlolib_status again_status = case_run(&rc, NULL, 0u, &again);
    CHECK(again_status == NLOLIB_STATUS_OK && again.merged == 1 && strcmp(again.key, a.key) == 0,
          "rerun of A: status %d merged %d key %s (A %s)", (int)again_status, again.merged, again.key, a.key);
    CHECK(count_where_key(db_path, "io_record_maps", a.key) == a_maps, "A's maps after rerun");
    case_free(&rc);
    check_runs_contract(db_path);
}

/*
 * A staging write that fails at the OS level during the run (here a file
 * size limit, EFBIG) fails the run, and the log names the SQLite result and OS error behind it,
 * not only the propagate stage.
 */
static void test_write_failure_logged(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    pid_t pid = fork();
    if (pid == 0) {
        signal(SIGXFSZ, SIG_IGN);
        /* Room for the schema and a few records, not for 32 raw records of 16 KiB. */
        struct rlimit limit = {256u * 1024u, 256u * 1024u};
        (void)setrlimit(RLIMIT_FSIZE, &limit);
        (void)nlolib_set_log_buffer(1u << 16);
        run_case rc;
        case_init_1d(&rc, 1024u, 1.0, 32u, db_path);
        rc.storage.record_mode = STORAGE_RECORD_MODE_REDUCED_AND_RAW;
        const nlolib_status status = case_run(&rc, NULL, 0u, NULL);
        char text[1u << 16];
        size_t written = 0u;
        (void)nlolib_read_log_buffer(text, sizeof(text), &written, 0);
        text[(written < sizeof(text)) ? written : sizeof(text) - 1u] = '\0';
        char errno_line[64];
        (void)snprintf(errno_line, sizeof(errno_line), "os errno: %d", EFBIG);
        const int named = strstr(text, "sqlite ") != NULL && strstr(text, errno_line) != NULL &&
                          strstr(text, "stage: snapshot_capture") != NULL;
        if (!named) {
            fprintf(stderr, "log without the cause:\n%s\n", text);
        }
        _exit(status == NLOLIB_STATUS_OK ? 2 : (named ? 0 : 3));
    }
    int status = 0;
    (void)waitpid(pid, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "failed staging write: exit %d (2: run succeeded, 3: log does not name the OS error)",
          WIFEXITED(status) ? WEXITSTATUS(status) : -1);
}

/* A merge split over many small transactions, a writer killed between two of them, and the clean-up. */
static void test_batched_and_interrupted_merge(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    char sdir[1024];
    staging_dir(db_path, sdir, sizeof(sdir));

    setenv("NLOLIB_STORE_MERGE_BATCH_BYTES", "3000", 1);
    run_case rc;
    case_init_1d(&rc, 128u, 1.0, 12u, db_path);
    rc.storage.record_mode = STORAGE_RECORD_MODE_REDUCED_AND_RAW;
    rc.storage.chunk_records = 2u;
    storage_result result;
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.merged == 1, "batched merge");
    CHECK(count_where_key(db_path, "io_record_maps", result.key) == 12, "batched maps");
    CHECK(count_where_key(db_path, "io_record_chunks", result.key) == 6, "batched chunks");
    check_runs_contract(db_path);

    /* Writer killed after the second merge transaction. */
    rc.field[2].re += 0.5;
    char key[STORAGE_KEY_MAX];
    CHECK(nlolib_storage_run_key(&rc.sim, &rc.phys, rc.n, rc.field, &rc.options, key) == NLOLIB_STATUS_OK, "key");
    pid_t pid = fork();
    if (pid == 0) {
        setenv("NLOLIB_STORE_TEST_CRASH_AFTER_COMMITS", "2", 1);
        (void)case_run(&rc, NULL, 0u, NULL);
        _exit(3);
    }
    int status = 0;
    (void)waitpid(pid, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 77, "writer did not die mid-merge (status %d)", status);
    int present = 1;
    CHECK(nlolib_storage_has_run(db_path, key, &present) == NLOLIB_STATUS_OK && present == 0, "interrupted merge has a runs row");
    CHECK(count_where_key(db_path, "io_record_layout", key) == 1, "interrupted merge left no partial data");
    CHECK(count_files(sdir, ".sqlite") == 1u, "interrupted merge lost its staging file");
    check_runs_contract(db_path);

    /* Resume: merging the staging file again starts over and completes. */
    size_t merged = 0u;
    size_t discarded = 0u;
    CHECK(nlolib_storage_recover_staging(db_path, 0, 0.0, &merged, NULL, NULL) == NLOLIB_STATUS_OK && merged == 1u,
          "resume merged %zu", merged);
    CHECK(nlolib_storage_has_run(db_path, key, &present) == NLOLIB_STATUS_OK && present == 1, "resumed run missing");
    CHECK(count_where_key(db_path, "io_record_maps", key) == 12, "resumed maps");
    CHECK(count_where_key(db_path, "io_record_chunks", key) == 6, "resumed chunks");

    /* Discard: interrupted again, staging file lost, recovery removes the partial data. */
    rc.field[3].re += 0.5;
    CHECK(nlolib_storage_run_key(&rc.sim, &rc.phys, rc.n, rc.field, &rc.options, key) == NLOLIB_STATUS_OK, "key");
    pid = fork();
    if (pid == 0) {
        setenv("NLOLIB_STORE_TEST_CRASH_AFTER_COMMITS", "2", 1);
        (void)case_run(&rc, NULL, 0u, NULL);
        _exit(3);
    }
    (void)waitpid(pid, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 77, "second writer did not die mid-merge");
    char cmd[1200];
    (void)snprintf(cmd, sizeof(cmd), "rm -f '%s'/*.sqlite", sdir);
    CHECK(system(cmd) == 0, "rm staging");
    CHECK(count_where_key(db_path, "io_record_layout", key) == 1, "partial data before discard");
    CHECK(nlolib_storage_recover_staging(db_path, 1, 0.0, &merged, &discarded, NULL) == NLOLIB_STATUS_OK && discarded == 1u,
          "orphan discard %zu", discarded);
    CHECK(count_where_key(db_path, "io_record_layout", key) == 0, "orphaned partial data kept");
    sqlite3* db = open_ro(db_path);
    char q[256];
    (void)snprintf(q, sizeof(q), "SELECT COUNT(*) FROM io_runs WHERE key='%s';", key);
    CHECK(sql_int(db, q) == 0, "orphaned io_runs row kept");
    CHECK(sql_int(db, "SELECT COUNT(*) FROM runs;") == 2, "runs rows after clean-up");
    sqlite3_close(db);
    unsetenv("NLOLIB_STORE_MERGE_BATCH_BYTES");
    case_free(&rc);
    check_runs_contract(db_path);
}

/* A writer that sees an older database than the last merge recorded refuses to write. */
static void test_stale_view_guard(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    run_case rc;
    case_init_1d(&rc, 64u, 1.0, 4u, db_path);
    storage_result result;
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.merged == 1, "first run");

    char state[1100];
    (void)snprintf(state, sizeof(state), "%s.writer-state", db_path);
    char content[512];
    FILE* f = fopen(state, "rb");
    size_t n = (f != NULL) ? fread(content, 1u, sizeof(content) - 1u, f) : 0u;
    if (f != NULL) {
        fclose(f);
    }
    content[n] = '\0';
    const char* at = strstr(content, "\ncounter=");
    CHECK(at != NULL, "state file has no counter: %s", content);
    unsigned long counter = (at != NULL) ? strtoul(at + 9, NULL, 10) : 0u;
    /* The recorded counter must be the database's current one. */
    FILE* dbf = fopen(db_path, "rb");
    unsigned char h[100] = {0};
    if (dbf != NULL) {
        CHECK(fread(h, 1u, sizeof(h), dbf) == sizeof(h), "read header");
        fclose(dbf);
    }
    const unsigned long actual = ((unsigned long)h[24] << 24) | ((unsigned long)h[25] << 16) |
                                 ((unsigned long)h[26] << 8) | (unsigned long)h[27];
    CHECK(counter == actual, "state counter %lu != header %lu", counter, actual);

    /* Pretend a later merge (by another host) committed more than this host can see. */
    f = fopen(state, "wb");
    fprintf(f, "nlolib-state 1\ncounter=%lu\nhost=other\n", actual + 5u);
    fclose(f);
    unsigned long long before = 0u;
    long before_size = 0;
    file_digest(db_path, &before, &before_size);
    rc.field[0].re += 0.5;
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.merged == 0, "wrote on a stale view");
    unsigned long long after = 0u;
    long after_size = 0;
    file_digest(db_path, &after, &after_size);
    CHECK(before == after, "stale-view writer changed the database");
    unlink(state);
    size_t merged = 0u;
    CHECK(nlolib_storage_recover_staging(db_path, 0, 0.0, &merged, NULL, NULL) == NLOLIB_STATUS_OK && merged == 1u,
          "merge after the view caught up");
    case_free(&rc);
    check_runs_contract(db_path);
}

/* A writer killed inside a commit leaves a hot journal; the next merge rolls it back and proceeds. */
static void test_hot_journal(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    run_case rc;
    case_init_1d(&rc, 64u, 1.0, 4u, db_path);
    storage_result result;
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.merged == 1, "first run");
    char first_key[STORAGE_KEY_MAX];
    (void)snprintf(first_key, sizeof(first_key), "%s", result.key);

    pid_t pid = fork();
    if (pid == 0) {
        sqlite3* db = NULL;
        if (sqlite3_open_v2(db_path, &db, SQLITE_OPEN_READWRITE, "unix-none") != SQLITE_OK) {
            _exit(2);
        }
        /* A tiny cache makes SQLite spill changed pages into the file before the commit. */
        sqlite3_exec(db, "PRAGMA cache_size=2; BEGIN; CREATE TABLE junk(x BLOB);"
                         "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<400) "
                         "INSERT INTO junk SELECT randomblob(4000) FROM n;"
                         "DELETE FROM io_record_maps;", NULL, NULL, NULL);
        _exit(0); /* dies inside the transaction */
    }
    int status = 0;
    (void)waitpid(pid, &status, 0);
    char journal[1100];
    (void)snprintf(journal, sizeof(journal), "%s-journal", db_path);
    struct stat st;
    CHECK(stat(journal, &st) == 0 && st.st_size > 0, "no hot journal was left");

    rc.field[0].re += 0.5;
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.merged == 1, "merge after hot journal");
    CHECK(stat(journal, &st) != 0 || st.st_size == 0, "hot journal not rolled back");
    sqlite3* db = open_ro(db_path);
    CHECK(sql_int(db, "SELECT COUNT(*) FROM sqlite_master WHERE name='junk';") == 0, "rolled-back table present");
    CHECK(sql_int(db, "SELECT COUNT(*) FROM io_record_maps;") == 8, "maps lost by the rollback");
    sqlite3_close(db);
    int present = 0;
    CHECK(nlolib_storage_has_run(db_path, first_key, &present) == NLOLIB_STATUS_OK && present == 1, "first run lost");
    case_free(&rc);
    check_runs_contract(db_path);
}

/* -------------------------------------------------------------------------
 * Verification pass: copied staging files, same keys, key stability,
 * truncation and a merge tool killed mid-merge
 * ---------------------------------------------------------------------- */

/* Copy @p from to @p to; keep only the first @p keep bytes, or zero the rest when @p zero_tail. */
static int copy_file(const char* from, const char* to, long keep, int zero_tail)
{
    FILE* in = fopen(from, "rb");
    FILE* out = fopen(to, "wb");
    if (in == NULL || out == NULL) {
        if (in != NULL) {
            fclose(in);
        }
        if (out != NULL) {
            fclose(out);
        }
        return -1;
    }
    long n = 0;
    int c = 0;
    while ((c = fgetc(in)) != EOF) {
        if (keep >= 0 && n >= keep) {
            if (!zero_tail) {
                break;
            }
            c = 0;
        }
        fputc(c, out);
        n += 1;
    }
    fclose(in);
    fclose(out);
    return 0;
}

/* Name of the only staging file in @p dir. */
static int only_staging_file(const char* dir, char* out_name, size_t cap)
{
    DIR* d = opendir(dir);
    int found = 0;
    struct dirent* ent = NULL;
    while (d != NULL && (ent = readdir(d)) != NULL) {
        const size_t len = strlen(ent->d_name);
        if (len > 7u && strcmp(ent->d_name + len - 7u, ".sqlite") == 0) {
            (void)snprintf(out_name, cap, "%s", ent->d_name);
            found += 1;
        }
    }
    if (d != NULL) {
        closedir(d);
    }
    return found == 1 ? 0 : -1;
}

static long file_size(const char* path)
{
    struct stat st;
    return (stat(path, &st) == 0) ? (long)st.st_size : -1;
}

static int run_tool(const char* args, const char* target)
{
    if (g_merge_tool == NULL) {
        return -1;
    }
    char cmd[3000];
    (void)snprintf(cmd, sizeof(cmd), "'%s' -q %s '%s' 2>/dev/null", g_merge_tool, args, target);
    return system(cmd);
}

static double* read_array(const char* db, const char* key, const char* name, size_t* out_n)
{
    size_t n = 0u;
    if (nlolib_storage_read_run_array(db, key, name, NULL, 0u, &n) != NLOLIB_STATUS_OK) {
        *out_n = 0u;
        return NULL;
    }
    double* values = (double*)calloc(n > 0u ? n : 1u, sizeof(double));
    if (nlolib_storage_read_run_array(db, key, name, values, n, out_n) != NLOLIB_STATUS_OK) {
        *out_n = 0u;
    }
    return values;
}

/*
 * A deferred run staged for one store, copied into another store's staging/
 * and merged there with nlolib_store_merge. Half-copied and zero-filled copies
 * are neither merged nor deleted, even with --discard-incomplete --min-age 0.
 */
static void test_copied_staging(const char* root)
{
    char a_root[1100];
    char a_dir[1100];
    char a_db[1200];
    char a_staging[1200];
    path_join(a_root, sizeof(a_root), root, "a");
    path_join(a_dir, sizeof(a_dir), a_root, "otherfolder");
    (void)mkdir(root, 0775);
    (void)mkdir(a_root, 0775);
    (void)mkdir(a_dir, 0775);
    path_join(a_db, sizeof(a_db), a_dir, "nlolib.sqlite");
    path_join(a_staging, sizeof(a_staging), a_dir, "staging");

    run_case rc;
    case_init_1d(&rc, 128u, 1.0, 6u, a_db);
    rc.storage.defer_merge = 1;
    rc.storage.record_mode = STORAGE_RECORD_MODE_REDUCED_AND_RAW;
    storage_result result;
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.deferred == 1, "deferred run for store a");
    case_free(&rc);
    char name[600];
    CHECK(only_staging_file(a_staging, name, sizeof(name)) == 0, "no single staging file in store a");
    char src[1900];
    path_join(src, sizeof(src), a_staging, name);

    static const char* arrays[] = {"z", "t", "omega", "power", "spectrum", "fluence_x", "fluence_y", "final_field", "raw"};
    double* original[9];
    size_t original_n[9];
    for (size_t i = 0u; i < 9u; ++i) {
        original[i] = read_array(src, result.key, arrays[i], &original_n[i]);
        CHECK(original_n[i] > 0u, "staging array %s empty", arrays[i]);
    }

    char b_db[1100];
    char b_store[1100];
    path_join(b_store, sizeof(b_store), root, "b");
    make_store(b_store, b_db, sizeof(b_db));
    char b_staging[1200];
    staging_dir(b_db, b_staging, sizeof(b_staging));
    (void)mkdir(b_staging, 0775);
    char dst[1900];
    path_join(dst, sizeof(dst), b_staging, name);
    const long full = file_size(src);
    int present = 1;

    /* Half copied: a truncated file under the final name. */
    CHECK(copy_file(src, dst, full / 2, 0) == 0, "half copy");
    (void)run_tool("--discard-incomplete --min-age 0", b_store);
    CHECK(file_size(dst) == full / 2, "half-copied staging file was changed or deleted");
    CHECK(nlolib_storage_has_run(b_db, result.key, &present) == NLOLIB_STATUS_OK && present == 0, "half copy merged");

    /* Preallocated copy: full size, second half still zeros. */
    CHECK(copy_file(src, dst, full / 2, 1) == 0, "zero-tail copy");
    (void)run_tool("--discard-incomplete --min-age 0", b_store);
    CHECK(file_size(dst) == full, "zero-filled staging file was changed or deleted");
    CHECK(nlolib_storage_has_run(b_db, result.key, &present) == NLOLIB_STATUS_OK && present == 0, "zero-filled copy merged");
    unlink(dst);

    /* Safe protocol: copy under another name, then rename into place. */
    char part[2000];
    (void)snprintf(part, sizeof(part), "%s.part", dst);
    CHECK(copy_file(src, part, -1, 0) == 0, "part copy");
    CHECK(run_tool("--discard-incomplete --min-age 0", b_store) == 0, "tool with a .part file present");
    CHECK(file_size(part) == full, ".part file touched");
    CHECK(rename(part, dst) == 0, "rename into place");
    CHECK(run_tool("", b_store) == 0, "merge tool failed");
    CHECK(nlolib_storage_has_run(b_db, result.key, &present) == NLOLIB_STATUS_OK && present == 1, "copied run not merged");
    CHECK(count_files(b_staging, ".sqlite") == 0u, "merged copy not removed");
    CHECK(file_size(src) == full, "source staging file changed");

    sqlite3* db = open_ro(b_db);
    char text[256];
    sql_text(db, "SELECT file FROM runs;", text, sizeof(text));
    CHECK(strcmp(text, "nlolib/nlolib.sqlite") == 0, "file of copied run is '%s'", text);
    sql_text(db, "SELECT key FROM runs;", text, sizeof(text));
    CHECK(strcmp(text, result.key) == 0, "key of copied run is '%s'", text);
    sqlite3_close(db);
    for (size_t i = 0u; i < 9u; ++i) {
        size_t n = 0u;
        double* merged = read_array(b_db, result.key, arrays[i], &n);
        CHECK(n == original_n[i] && memcmp(merged, original[i], n * sizeof(double)) == 0,
              "merged %s differs from the staging file (%zu vs %zu values)", arrays[i], n, original_n[i]);
        free(merged);
        free(original[i]);
    }
    check_runs_contract(b_db);
}

static void staging_created_at(const char* path, char* out, size_t cap)
{
    sqlite3* db = open_ro(path);
    out[0] = '\0';
    if (db != NULL) {
        sql_text(db, "SELECT created_at FROM runs;", out, cap);
        sqlite3_close(db);
    }
}

/* Two staged copies of one key, a staged run whose key is already stored, and an older copy arriving late. */
static void test_same_key(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    char sdir[1024];
    staging_dir(db_path, sdir, sizeof(sdir));
    run_case rc;
    case_init_1d(&rc, 64u, 1.0, 5u, db_path);
    rc.storage.defer_merge = 1;
    storage_result first;
    storage_result second;
    CHECK(case_run(&rc, NULL, 0u, &first) == NLOLIB_STATUS_OK && first.deferred == 1, "first staged copy");
    usleep(5000);
    CHECK(case_run(&rc, NULL, 0u, &second) == NLOLIB_STATUS_OK && second.deferred == 1, "second staged copy");
    CHECK(strcmp(first.key, second.key) == 0, "same parameters gave different keys");
    CHECK(count_files(sdir, ".sqlite") == 2u, "two staging files expected");
    int present = 1;
    CHECK(nlolib_storage_has_run(db_path, first.key, &present) == NLOLIB_STATUS_OK && present == 0,
          "has_run is true for staged runs");

    /* Newest created_at wins whatever order the files are merged in. */
    char newest[32] = "";
    DIR* d = opendir(sdir);
    struct dirent* ent = NULL;
    while (d != NULL && (ent = readdir(d)) != NULL) {
        if (strstr(ent->d_name, ".sqlite") == NULL) {
            continue;
        }
        char path[1400];
        char created[32];
        path_join(path, sizeof(path), sdir, ent->d_name);
        staging_created_at(path, created, sizeof(created));
        if (strcmp(created, newest) > 0) {
            (void)snprintf(newest, sizeof(newest), "%s", created);
        }
    }
    if (d != NULL) {
        closedir(d);
    }
    size_t merged = 0u;
    CHECK(nlolib_storage_recover_staging(db_path, 0, 0.0, &merged, NULL, NULL) == NLOLIB_STATUS_OK && merged == 2u,
          "both copies resolved (%zu)", merged);
    CHECK(nlolib_storage_has_run(db_path, first.key, &present) == NLOLIB_STATUS_OK && present == 1, "has_run after merge");
    CHECK(count_files(sdir, ".sqlite") == 0u, "a same-key copy was left in staging");
    sqlite3* db = open_ro(db_path);
    char text[64];
    sql_text(db, "SELECT created_at FROM runs;", text, sizeof(text));
    CHECK(strcmp(text, newest) == 0, "stored created_at %s is not the newest copy %s", text, newest);
    CHECK(sql_int(db, "SELECT COUNT(*) FROM runs;") == 1 && sql_int(db, "SELECT COUNT(*) FROM io_runs;") == 1 &&
              sql_int(db, "SELECT COUNT(*) FROM io_record_maps;") == 5,
          "same key duplicated");
    sqlite3_close(db);

    /* A staged run whose key is already stored replaces it (it is newer). */
    storage_result third;
    CHECK(case_run(&rc, NULL, 0u, &third) == NLOLIB_STATUS_OK, "third staged copy");
    char third_path[1400];
    char third_name[600];
    CHECK(only_staging_file(sdir, third_name, sizeof(third_name)) == 0, "third staging file");
    path_join(third_path, sizeof(third_path), sdir, third_name);
    char third_created[32];
    staging_created_at(third_path, third_created, sizeof(third_created));
    /* Keep a copy of it to bring back later as a late, older copy. */
    char late[1500];
    (void)snprintf(late, sizeof(late), "%s/late-copy.bak", root);
    CHECK(copy_file(third_path, late, -1, 0) == 0, "keep late copy");
    CHECK(nlolib_storage_recover_staging(db_path, 0, 0.0, &merged, NULL, NULL) == NLOLIB_STATUS_OK && merged == 1u,
          "third copy merged");
    db = open_ro(db_path);
    sql_text(db, "SELECT created_at FROM runs;", text, sizeof(text));
    CHECK(strcmp(text, third_created) == 0, "stored run not replaced by the newer copy");
    sqlite3_close(db);

    /* A newer run is merged, then the older copy arrives: it is superseded and removed, not merged. */
    usleep(5000);
    storage_result fourth;
    CHECK(case_run(&rc, NULL, 0u, &fourth) == NLOLIB_STATUS_OK, "fourth staged copy");
    CHECK(nlolib_storage_recover_staging(db_path, 0, 0.0, &merged, NULL, NULL) == NLOLIB_STATUS_OK && merged == 1u,
          "fourth copy merged");
    db = open_ro(db_path);
    char fourth_created[32];
    sql_text(db, "SELECT created_at FROM runs;", fourth_created, sizeof(fourth_created));
    sqlite3_close(db);
    CHECK(strcmp(fourth_created, third_created) > 0, "fourth run is not newer");
    CHECK(copy_file(late, third_path, -1, 0) == 0, "restore late copy");
    CHECK(nlolib_storage_recover_staging(db_path, 0, 0.0, &merged, NULL, NULL) == NLOLIB_STATUS_OK && merged == 1u,
          "late copy resolved");
    CHECK(count_files(sdir, ".sqlite") == 0u, "superseded copy left in staging");
    db = open_ro(db_path);
    sql_text(db, "SELECT created_at FROM runs;", text, sizeof(text));
    CHECK(strcmp(text, fourth_created) == 0, "an older copy replaced a newer run (%s)", text);
    CHECK(sql_int(db, "SELECT COUNT(*) FROM runs;") == 1 && sql_int(db, "SELECT COUNT(*) FROM io_record_maps;") == 5,
          "rows duplicated after the late copy");
    sqlite3_close(db);
    unlink(late);
    case_free(&rc);
    check_runs_contract(db_path);
}

/* Exactly representable run used for the cross-binding key check (same values in the Julia and Python tests). */
#define KEY_REFERENCE_N 16u
static const char* const k_reference_key = "a29aa53a4be7d5f7";

static void reference_case(run_case* rc, nlo_complex* field, nlo_complex* omega)
{
    memset(rc, 0, sizeof(*rc));
    rc->nt = KEY_REFERENCE_N;
    rc->nx = 1u;
    rc->ny = 1u;
    rc->n = KEY_REFERENCE_N;
    for (size_t i = 0u; i < KEY_REFERENCE_N; ++i) {
        field[i] = make(((double)i - 8.0) / 16.0, (double)(i % 4u) / 8.0);
        omega[i] = make(0.5 * (double)((i <= 7u) ? (long)i : (long)i - 16), 0.0);
    }
    rc->field = field;
    rc->sim.propagation.propagation_distance = 0.25;
    rc->sim.propagation.starting_step_size = 0.0625;
    rc->sim.propagation.max_step_size = 0.125;
    rc->sim.propagation.min_step_size = 0.0009765625;
    rc->sim.propagation.error_tolerance = 1e-6;
    rc->sim.time.pulse_period = 2.0;
    rc->sim.time.delta_time = 0.125;
    rc->sim.frequency.frequency_grid = omega;
    rc->sim.spatial.delta_x = 1.0;
    rc->sim.spatial.delta_y = 1.0;
    rc->phys.dispersion_factor_expr = "i*c0*w*w";
    rc->phys.nonlinear_expr = "i*A*(c1*I)";
    rc->phys.num_constants = 2u;
    rc->phys.constants[0] = -0.04;
    rc->phys.constants[1] = 1.0;
    rc->storage = (storage_options){0};
    rc->storage.log_final_output_field_to_db = 1;
    rc->storage.record_mode = STORAGE_RECORD_MODE_REDUCED;
    rc->options = nlolib_propagate_options_default();
    rc->options.num_recorded_samples = 3u;
    rc->options.storage_options = &rc->storage;
}

static void key_of(const run_case* rc, char out[STORAGE_KEY_MAX])
{
    out[0] = '\0';
    CHECK(nlolib_storage_run_key(&rc->sim, &rc->phys, rc->n, rc->field, &rc->options, out) == NLOLIB_STATUS_OK,
          "run_key failed");
}

/* The key is the same for equivalent spellings of one run and different for every parameter that defines it. */
static void test_key_stability(const char* root)
{
    (void)root;
    nlo_complex field[KEY_REFERENCE_N];
    nlo_complex omega[KEY_REFERENCE_N];
    nlo_complex alt[KEY_REFERENCE_N];
    run_case rc;
    reference_case(&rc, field, omega);
    char base[STORAGE_KEY_MAX];
    key_of(&rc, base);
    printf("  reference key: %s\n", base);
    if (strcmp(k_reference_key, "REFERENCE_KEY_PLACEHOLDER") != 0) {
        CHECK(strcmp(base, k_reference_key) == 0, "reference key %s, expected %s", base, k_reference_key);
    }

    /* Same key: backend, aliases, irrelevant fields, equivalent spellings of dt. */
    char key[STORAGE_KEY_MAX];
    execution_options exec = (execution_options){0};
    exec.backend_type = VECTOR_BACKEND_VULKAN;
    rc.options.exec_options = &exec;
    key_of(&rc, key);
    CHECK(strcmp(key, base) == 0, "exec options changed the key");
    exec.backend_type = VECTOR_BACKEND_CPU;
    exec.fft_backend = FFT_BACKEND_FFTW;
    key_of(&rc, key);
    CHECK(strcmp(key, base) == 0, "cpu backend changed the key");
    rc.options.exec_options = NULL;
    rc.phys.linear_factor_expr = "i*c0*w*w"; /* Julia fills both aliases; time-only runs use dispersion_factor */
    key_of(&rc, key);
    CHECK(strcmp(key, base) == 0, "unused linear_factor_expr changed a time-only key");
    rc.phys.linear_factor_expr = NULL;
    rc.phys.nonlinear_gamma = 3.0;
    rc.phys.raman_tau1 = 0.0122;
    key_of(&rc, key);
    CHECK(strcmp(key, base) == 0, "Raman settings changed an expression-model key");
    rc.phys.nonlinear_gamma = 0.0;
    rc.phys.raman_tau1 = 0.0;
    rc.sim.spatial.delta_x = 7.0;
    key_of(&rc, key);
    CHECK(strcmp(key, base) == 0, "dx changed a time-only key");
    rc.sim.spatial.delta_x = 1.0;
    rc.sim.time.pulse_period = 0.0;
    key_of(&rc, key);
    CHECK(strcmp(key, base) == 0, "pulse_period changed the key although delta_time is set");
    rc.sim.time.pulse_period = 2.0;
    rc.sim.time.delta_time = 0.0;
    key_of(&rc, key);
    CHECK(strcmp(key, base) == 0, "dt from pulse_period differs from the same delta_time");
    rc.sim.time.delta_time = 0.125;
    rc.phys.num_constants = 4u;
    rc.phys.constants[2] = 1.0; /* the defaults of c2, c3 */
    rc.phys.constants[3] = 0.0;
    key_of(&rc, key);
    CHECK(strcmp(key, base) == 0, "explicit default constants changed the key");
    rc.phys.num_constants = 2u;
    rc.phys.constants[2] = 0.0;
    rc.options.storage_options = &rc.storage;
    rc.storage.sqlite_path = "/somewhere/else.sqlite";
    rc.storage.chunk_records = 3u;
    rc.storage.run_id = "label";
    rc.storage.defer_merge = 1;
    key_of(&rc, key);
    CHECK(strcmp(key, base) == 0, "path, chunk size, run_id or defer_merge changed the key");
    rc.storage.sqlite_path = NULL;
    rc.storage.chunk_records = 0u;
    rc.storage.run_id = NULL;
    rc.storage.defer_merge = 0;

    /* Different key: every parameter that defines the run or its stored output. */
    char keys[40][STORAGE_KEY_MAX];
    const char* labels[40];
    size_t nk = 0u;
#define VARY(label, change, restore) \
    do {                              \
        change;                       \
        key_of(&rc, keys[nk]);        \
        labels[nk++] = (label);       \
        restore;                      \
    } while (0)
    VARY("dt", rc.sim.time.delta_time = 0.25, rc.sim.time.delta_time = 0.125);
    VARY("z", rc.sim.propagation.propagation_distance = 0.5, rc.sim.propagation.propagation_distance = 0.25);
    VARY("h_start", rc.sim.propagation.starting_step_size = 0.03125, rc.sim.propagation.starting_step_size = 0.0625);
    VARY("h_max", rc.sim.propagation.max_step_size = 0.25, rc.sim.propagation.max_step_size = 0.125);
    VARY("h_min", rc.sim.propagation.min_step_size = 0.001953125, rc.sim.propagation.min_step_size = 0.0009765625);
    VARY("tol", rc.sim.propagation.error_tolerance = 1e-7, rc.sim.propagation.error_tolerance = 1e-6);
    VARY("dispersion_factor_expr", rc.phys.dispersion_factor_expr = "i*c0*w*w-c1", rc.phys.dispersion_factor_expr = "i*c0*w*w");
    VARY("dispersion_expr", rc.phys.dispersion_expr = "exp(h*D)*1", rc.phys.dispersion_expr = NULL);
    VARY("nonlinear_expr", rc.phys.nonlinear_expr = "i*A*(c1*I+V)", rc.phys.nonlinear_expr = "i*A*(c1*I)");
    VARY("c0", rc.phys.constants[0] = -0.05, rc.phys.constants[0] = -0.04);
    VARY("c1", rc.phys.constants[1] = 1.5, rc.phys.constants[1] = 1.0);
    VARY("c4", (rc.phys.num_constants = 5u, rc.phys.constants[4] = 2.0), (rc.phys.num_constants = 2u, rc.phys.constants[4] = 0.0));
    VARY("raman model", (rc.phys.nonlinear_model = NONLINEAR_MODEL_KERR_RAMAN, rc.phys.nonlinear_gamma = 1.0),
         (rc.phys.nonlinear_model = NONLINEAR_MODEL_EXPR, rc.phys.nonlinear_gamma = 0.0));
    rc.phys.nonlinear_model = NONLINEAR_MODEL_KERR_RAMAN;
    rc.phys.nonlinear_gamma = 1.0;
    char raman_base[STORAGE_KEY_MAX];
    key_of(&rc, raman_base);
    VARY("gamma", rc.phys.nonlinear_gamma = 2.0, rc.phys.nonlinear_gamma = 1.0);
    VARY("raman_fraction", rc.phys.raman_fraction = 0.18, rc.phys.raman_fraction = 0.0);
    VARY("raman_tau1", rc.phys.raman_tau1 = 0.02, rc.phys.raman_tau1 = 0.0);
    VARY("raman_tau2", rc.phys.raman_tau2 = 0.05, rc.phys.raman_tau2 = 0.0);
    VARY("shock_omega0", rc.phys.shock_omega0 = 10.0, rc.phys.shock_omega0 = 0.0);
    rc.phys.raman_tau1 = 0.0122; /* the default value: same as unset */
    key_of(&rc, key);
    CHECK(strcmp(key, raman_base) == 0, "explicit default raman_tau1 changed the key");
    rc.phys.raman_tau1 = 0.0;
    rc.phys.nonlinear_model = NONLINEAR_MODEL_EXPR;
    rc.phys.nonlinear_gamma = 0.0;
    VARY("input sample", field[3].re += 1.0 / 1024.0, field[3].re -= 1.0 / 1024.0);
    VARY("frequency grid", omega[5].re += 0.5, omega[5].re -= 0.5);
    memcpy(alt, field, sizeof(alt));
    VARY("potential grid", rc.sim.spatial.potential_grid = alt, rc.sim.spatial.potential_grid = NULL);
    VARY("records", rc.options.num_recorded_samples = 4u, rc.options.num_recorded_samples = 3u);
    double zs[3] = {0.0, 0.125, 0.25};
    VARY("explicit z", (rc.options.explicit_record_z = zs, rc.options.explicit_record_z_count = 3u),
         (rc.options.explicit_record_z = NULL, rc.options.explicit_record_z_count = 0u));
    VARY("record_mode", rc.storage.record_mode = STORAGE_RECORD_MODE_REDUCED_AND_RAW,
         rc.storage.record_mode = STORAGE_RECORD_MODE_REDUCED);
    VARY("final_field", rc.storage.log_final_output_field_to_db = 0, rc.storage.log_final_output_field_to_db = 1);
    rc.storage.record_mode = STORAGE_RECORD_MODE_RAW;
    VARY("raw (stride 1)", (void)0, (void)0);
    VARY("raw_stride_t", rc.storage.raw_stride_t = 2u, rc.storage.raw_stride_t = 0u);
    rc.storage.record_mode = STORAGE_RECORD_MODE_REDUCED;
    key_of(&rc, key);
    CHECK(strcmp(key, base) == 0, "restoring every parameter did not restore the key");

    /* Tensor runs: transverse steps, potential and linear expressions. */
#define VARY_T(label, change, restore) \
    do {                                \
        change;                         \
        key_of(&t, keys[nk]);           \
        labels[nk++] = (label);         \
        restore;                        \
    } while (0)
    run_case t;
    case_init_3d(&t, 16u, 4u, 4u, 3u, NULL);
    char tbase[STORAGE_KEY_MAX];
    key_of(&t, tbase);
    VARY_T("3d dx", t.sim.spatial.delta_x = 0.5, t.sim.spatial.delta_x = 0.4);
    VARY_T("3d dy", t.sim.spatial.delta_y = 0.5, t.sim.spatial.delta_y = 0.4);
    VARY_T("3d potential_expr", t.phys.potential_expr = "0.5", t.phys.potential_expr = NULL);
    VARY_T("3d linear_factor_expr", t.phys.linear_factor_expr = "i*c0*(wt*wt)", t.phys.linear_factor_expr =
         "i*(c0*(wt*wt) + c1*((kx*kx)+(ky*ky)))");
    VARY_T("3d potential grid", t.potential[0].re += 1.0, t.potential[0].re -= 1.0);
#undef VARY
#undef VARY_T
    case_free(&t);

    for (size_t i = 0u; i < nk; ++i) {
        CHECK(strcmp(keys[i], base) != 0 && strcmp(keys[i], tbase) != 0, "%s did not change the key", labels[i]);
        for (size_t j = i + 1u; j < nk; ++j) {
            CHECK(strcmp(keys[i], keys[j]) != 0, "%s and %s gave the same key", labels[i], labels[j]);
        }
    }
}

/* A run stopped early by the size cap: nlolib_records_written and nlolib_truncated tell the truth. */
static void test_truncated_run(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    static const storage_record_mode modes[2] = {STORAGE_RECORD_MODE_RAW, STORAGE_RECORD_MODE_REDUCED_AND_RAW};
    for (int m = 0; m < 2; ++m) {
        run_case rc;
        case_init_1d(&rc, 1024u, 1.0 + 0.1 * m, 20u, db_path);
        rc.storage.record_mode = modes[m];
        rc.storage.chunk_records = 2u;
        rc.storage.sqlite_max_bytes = (1u << 20) + 200u * 1024u; /* room for a few 32 KiB chunks */
        storage_result result;
        CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.merged == 1, "capped run (mode %d)", m);
        CHECK(result.truncated == 1, "capped run not truncated (mode %d)", m);
        storage_run_layout layout;
        CHECK(nlolib_storage_read_run_layout(db_path, result.key, &layout) == NLOLIB_STATUS_OK, "layout");
        CHECK(layout.raw_records > 0u && layout.raw_records < 20u, "raw records %zu", layout.raw_records);
        sqlite3* db = open_ro(db_path);
        char q[256];
        (void)snprintf(q, sizeof(q), "SELECT nlolib_records_written FROM runs WHERE key='%s';", result.key);
        const long long written = sql_int(db, q);
        (void)snprintf(q, sizeof(q), "SELECT nlolib_truncated FROM runs WHERE key='%s';", result.key);
        CHECK(sql_int(db, q) == 1, "nlolib_truncated not set (mode %d)", m);
        (void)snprintf(q, sizeof(q), "SELECT nlolib_num_records FROM runs WHERE key='%s';", result.key);
        CHECK(sql_int(db, q) == 20, "nlolib_num_records (mode %d)", m);
        sqlite3_close(db);
        if (modes[m] == STORAGE_RECORD_MODE_RAW) {
            CHECK(written == (long long)layout.raw_records, "raw-only records_written %lld, stored %zu", written,
                  layout.raw_records);
        } else {
            CHECK(written == 20 && layout.num_records == 20u, "maps stored for every record: %lld", written);
        }
        CHECK(count_where_key(db_path, "io_record_chunks", result.key) > 0, "no raw chunks kept");
        case_free(&rc);
    }
    check_runs_contract(db_path);
}

/*
 * nlolib_store_merge killed with SIGKILL at many points of a multi-transaction
 * merge: after every kill the shared database is intact, every `runs` row has
 * all its data, and the next merge finishes the job.
 */
static void test_merge_tool_killed(const char* root)
{
    if (g_merge_tool == NULL) {
        printf("  (merge tool path not given; skipped)\n");
        return;
    }
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    char sdir[1024];
    staging_dir(db_path, sdir, sizeof(sdir));
    const int runs = 4;
    char keys[4][STORAGE_KEY_MAX];
    for (int r = 0; r < runs; ++r) {
        run_case rc;
        case_init_3d(&rc, 64u, 8u, 8u, 16u, db_path);
        rc.field[0].re += 0.25 * r;
        rc.storage.defer_merge = 1;
        rc.storage.record_mode = STORAGE_RECORD_MODE_REDUCED_AND_RAW;
        rc.storage.chunk_records = 2u;
        storage_result result;
        CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.deferred == 1, "staged run %d", r);
        (void)snprintf(keys[r], sizeof(keys[r]), "%s", result.key);
        case_free(&rc);
    }
    setenv("NLOLIB_STORE_MERGE_BATCH_BYTES", "65536", 1);
    static const unsigned delays_ms[] = {5, 20, 40, 60, 80, 110, 150, 200, 260, 330, 420, 520};
    int kills = 0;
    int journals = 0;
    for (size_t i = 0u; i < 60u && count_files(sdir, ".sqlite") > 0u; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            execl(g_merge_tool, g_merge_tool, "-q", db_path, (char*)NULL);
            _exit(127);
        }
        usleep(1000u * delays_ms[i % (sizeof(delays_ms) / sizeof(delays_ms[0]))]);
        if (kill(pid, SIGKILL) == 0) {
            kills += 1;
        }
        (void)waitpid(pid, NULL, 0);
        char journal[1100];
        (void)snprintf(journal, sizeof(journal), "%s-journal", db_path);
        if (file_size(journal) > 0) {
            journals += 1;
        }
        /* The writer lock of the killed tool is stale (dead pid); the next merger breaks it. A reader may
         * meet the hot journal: that is the documented case a writer has to roll back first. */
        if (file_size(journal) <= 0) {
            sqlite3* db = open_ro(db_path);
            if (db != NULL) {
                char integrity[64];
                sql_text(db, "PRAGMA integrity_check;", integrity, sizeof(integrity));
                CHECK(integrity[0] == '\0' || strcmp(integrity, "ok") == 0, "integrity after kill %zu: %s", i, integrity);
                CHECK(sql_int(db, "SELECT COUNT(*) FROM runs r WHERE "
                                  "(SELECT COUNT(*) FROM io_record_maps m WHERE m.run_id = r.nlolib_run_id) != 16 OR "
                                  "(SELECT COUNT(*) FROM io_record_chunks c WHERE c.run_id = r.nlolib_run_id) != 8 OR "
                                  "(SELECT COUNT(*) FROM io_final_output_fields f WHERE f.run_id = r.nlolib_run_id) != 1;") <= 0,
                      "a runs row without all its data after kill %zu", i);
                sqlite3_close(db);
            }
        }
    }
    unsetenv("NLOLIB_STORE_MERGE_BATCH_BYTES");
    printf("  killed the merge tool %d times, %d left a hot journal\n", kills, journals);
    CHECK(run_tool("", db_path) == 0, "final merge after kills failed");
    CHECK(count_files(sdir, ".sqlite") == 0u, "staging files left after the final merge");
    for (int r = 0; r < runs; ++r) {
        int present = 0;
        CHECK(nlolib_storage_has_run(db_path, keys[r], &present) == NLOLIB_STATUS_OK && present == 1, "run %d lost", r);
        CHECK(count_where_key(db_path, "io_record_maps", keys[r]) == 16 &&
                  count_where_key(db_path, "io_record_chunks", keys[r]) == 8,
              "run %d incomplete after kills", r);
    }
    sqlite3* db = open_ro(db_path);
    CHECK(sql_int(db, "SELECT COUNT(*) FROM io_runs;") == runs, "leftover partial merges in io_runs");
    sqlite3_close(db);
    check_runs_contract(db_path);
}

/*
 * Run one stored run in a child process with injected I/O failures
 * (NLOLIB_STORE_TEST_IO_FAULTS) and a retry budget; report its status,
 * whether it merged, how long it took, and whether each of up to two
 * strings appears in its log. Exit code: bit 0 run OK, bit 1 merged,
 * bit 2 first string found, bit 3 second string found. The duration is
 * passed back through a file next to the database.
 */
static int faulted_run(const char* db_path, const char* faults, const char* budget,
                       const char* want1, const char* want2, double* out_seconds)
{
    char timing[1100];
    (void)snprintf(timing, sizeof(timing), "%s.timing", db_path);
    pid_t pid = fork();
    if (pid == 0) {
        setenv("NLOLIB_STORE_TEST_IO_FAULTS", faults, 1);
        if (budget != NULL) {
            setenv("NLOLIB_STORE_IO_RETRY_S", budget, 1);
        }
        (void)nlolib_set_log_buffer(1u << 18);
        run_case rc;
        case_init_1d(&rc, 256u, 1.0, 8u, db_path);
        rc.storage.chunk_records = 2u;
        rc.storage.record_mode = STORAGE_RECORD_MODE_REDUCED_AND_RAW;
        storage_result result;
        memset(&result, 0, sizeof(result));
        struct timespec t0;
        struct timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        const nlolib_status status = case_run(&rc, NULL, 0u, &result);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        static char text[1u << 18];
        size_t written = 0u;
        (void)nlolib_read_log_buffer(text, sizeof(text), &written, 0);
        text[(written < sizeof(text)) ? written : sizeof(text) - 1u] = '\0';
        FILE* f = fopen(timing, "w");
        if (f != NULL) {
            fprintf(f, "%f\n", (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec));
            fclose(f);
        }
        int code = 0;
        code |= (status == NLOLIB_STATUS_OK) ? 1 : 0;
        code |= (result.merged == 1) ? 2 : 0;
        code |= (want1 == NULL || strstr(text, want1) != NULL) ? 4 : 0;
        code |= (want2 == NULL || strstr(text, want2) != NULL) ? 8 : 0;
        if ((code & 12) != 12) {
            fprintf(stderr, "log of the faulted run (%s):\n%s\n", faults, text);
        }
        _exit(code);
    }
    int status = 0;
    (void)waitpid(pid, &status, 0);
    *out_seconds = -1.0;
    FILE* f = fopen(timing, "r");
    if (f != NULL) {
        if (fscanf(f, "%lf", out_seconds) != 1) {
            *out_seconds = -1.0;
        }
        fclose(f);
        unlink(timing);
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

/*
 * The failures a loaded SMB share returned to staging writers (CANTOPEN with
 * ENOENT on a re-created file, EAGAIN from fsync, truncate and write) are
 * retried until the call succeeds, so the run completes and merges. A failure
 * that outlasts the retry budget fails the run and the log names it; one that
 * is not transient (EIO) is not retried.
 */
static void test_io_faults_retried(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    char sdir[1024];
    staging_dir(db_path, sdir, sizeof(sdir));
    double seconds = 0.0;

    int code = faulted_run(db_path, "open:4:2,sync:3:11,truncate:3:11,write:3:11", NULL,
                           "open of ", "succeeded after", &seconds);
    CHECK(code == 15, "transient faults: exit %d (1 ok, 2 merged, 4/8 retries logged)", code);
    CHECK(count_files(sdir, ".sqlite") == 0u, "staging file left after a retried run");
    check_runs_contract(db_path);

    code = faulted_run(db_path, "sync:1000000:11", "0.3", "giving up", "os errno: 11", &seconds);
    CHECK((code & 3) == 0 && (code & 12) == 12, "endless EAGAIN: exit %d (expected a failed run naming errno 11)", code);
    CHECK(seconds >= 0.3 && seconds < 5.0, "endless EAGAIN gave up after %.2f s; budget 0.3 s", seconds);
    CHECK(count_files(sdir, ".sqlite") == 0u, "failed run left its staging file");

    code = faulted_run(db_path, "sync:1:5", NULL, "os errno: 5", NULL, &seconds);
    CHECK((code & 1) == 0 && (code & 4) == 4, "EIO: exit %d (expected a failed run naming errno 5)", code);
    CHECK(seconds >= 0.0 && seconds < 2.0, "EIO took %.2f s; it must not be retried", seconds);
}

/* Path of the one staging file in @p sdir (empty when there is none or several). */
static void staging_file_path(const char* sdir, char* out, size_t cap)
{
    out[0] = '\0';
    DIR* d = opendir(sdir);
    if (d == NULL) {
        return;
    }
    size_t found = 0u;
    struct dirent* ent = NULL;
    while ((ent = readdir(d)) != NULL) {
        const size_t len = strlen(ent->d_name);
        if (len > 7u && strcmp(ent->d_name + len - 7u, ".sqlite") == 0) {
            found += 1u;
            path_join(out, cap, sdir, ent->d_name);
        }
    }
    closedir(d);
    if (found != 1u) {
        out[0] = '\0';
    }
}

static void write_zero_journal(const char* db, size_t bytes)
{
    char journal[1100];
    (void)snprintf(journal, sizeof(journal), "%s-journal", db);
    FILE* f = fopen(journal, "wb");
    for (size_t i = 0u; f != NULL && i < bytes; ++i) {
        fputc(0, f);
    }
    if (f != NULL) {
        fclose(f);
    }
}

/*
 * Writers use journal_mode TRUNCATE: a commit truncates the journal instead
 * of deleting it, so a zero-length journal stays beside a written file. Such
 * a journal, or one with a zeroed header as PERSIST leaves it, is never taken
 * for a hot journal: the staging file is still complete, recovery merges it,
 * a merge does not roll anything back, and a read-only reader is not refused.
 * Merged staging files leave no journal behind.
 */
static void test_truncate_journal(const char* root)
{
    char db_path[1024];
    make_store(root, db_path, sizeof(db_path));
    char sdir[1024];
    staging_dir(db_path, sdir, sizeof(sdir));
    char journal[1100];
    (void)snprintf(journal, sizeof(journal), "%s-journal", db_path);

    run_case rc;
    case_init_1d(&rc, 64u, 1.0, 4u, db_path);
    rc.storage.chunk_records = 1u;
    rc.storage.record_mode = STORAGE_RECORD_MODE_REDUCED_AND_RAW;
    storage_result result;
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.merged == 1, "merged run");
    CHECK(file_size(journal) == 0, "shared journal is %ld bytes; TRUNCATE leaves it at 0", file_size(journal));
    CHECK(count_files(sdir, ".sqlite") == 0u && count_files(sdir, "-journal") == 0u, "merge left staging files");

    /* A deferred run keeps its zero-length journal until it is merged. */
    rc.storage.defer_merge = 1;
    rc.field[2].re += 0.5;
    CHECK(case_run(&rc, NULL, 0u, &result) == NLOLIB_STATUS_OK && result.deferred == 1, "deferred run");
    char staged[1100];
    staging_file_path(sdir, staged, sizeof(staged));
    char staged_journal[1200];
    (void)snprintf(staged_journal, sizeof(staged_journal), "%s-journal", staged);
    CHECK(staged[0] != '\0' && file_size(staged_journal) == 0, "staging journal is %ld bytes", file_size(staged_journal));

    /* Header-zeroed journals (as PERSIST leaves them) beside both files. */
    write_zero_journal(staged, 4096u);
    write_zero_journal(db_path, 4096u);
    sqlite3* ro = open_ro(db_path);
    CHECK(ro != NULL && sql_int(ro, "SELECT COUNT(*) FROM runs;") == 1, "read-only reader refused beside a zeroed journal");
    if (ro != NULL) {
        sqlite3_close(ro);
    }
    int present = 0;
    CHECK(nlolib_storage_has_run(db_path, result.key, &present) == NLOLIB_STATUS_OK && present == 0, "has_run");
    size_t merged = 0u;
    size_t skipped = 0u;
    CHECK(nlolib_storage_recover_staging(db_path, 0, 0.0, &merged, NULL, &skipped) == NLOLIB_STATUS_OK &&
              merged == 1u && skipped == 0u,
          "staging file beside a zeroed journal: merged %zu skipped %zu", merged, skipped);
    CHECK(nlolib_storage_has_run(db_path, result.key, &present) == NLOLIB_STATUS_OK && present == 1, "deferred run missing");
    CHECK(count_files(sdir, ".sqlite") == 0u && count_files(sdir, "-journal") == 0u, "recovery left staging files");
    CHECK(file_size(journal) == 0, "shared journal after the merge is %ld bytes", file_size(journal));

    /* A journal whose staging file was deleted by hand is removed by recovery with discard_incomplete. */
    char orphan[1200];
    (void)snprintf(orphan, sizeof(orphan), "%s-journal", staged);
    write_zero_journal(staged, 0u);
    CHECK(file_size(orphan) == 0, "orphan journal not created");
    CHECK(nlolib_storage_recover_staging(db_path, 1, 0.0, NULL, NULL, NULL) == NLOLIB_STATUS_OK, "sweep");
    CHECK(file_size(orphan) < 0, "orphan journal kept");
    case_free(&rc);
    check_runs_contract(db_path);
}

int main(int argc, char** argv)
{
    char root[1024];
    const char* base = (argc > 1) ? argv[1] : ".";
    g_merge_tool = (argc > 2) ? argv[2] : NULL;
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
        {"readers_during_merge", test_readers_during_merge},
        {"killed", test_killed_and_aborted_writer},
        {"locks", test_locks},
        {"old_database", test_old_database},
        {"vfs_choices", test_vfs_choices},
        {"deferred_merge", test_deferred_merge},
        {"recovery_pass_settles_once", test_recovery_pass_settles_once},
        {"run_id_conflict", test_run_id_conflict},
        {"write_failure_logged", test_write_failure_logged},
        {"truncate_journal", test_truncate_journal},
        {"io_faults_retried", test_io_faults_retried},
        {"batched_interrupted_merge", test_batched_and_interrupted_merge},
        {"stale_view_guard", test_stale_view_guard},
        {"hot_journal", test_hot_journal},
        {"copied_staging", test_copied_staging},
        {"same_key", test_same_key},
        {"key_stability", test_key_stability},
        {"truncated_run", test_truncated_run},
        {"merge_tool_killed", test_merge_tool_killed},
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
