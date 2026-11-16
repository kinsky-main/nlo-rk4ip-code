/**
 * @file run_store.c
 * @brief Run keys, staging databases and locked merges into a shared run database.
 */

#if !defined(_WIN32)
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE 1
#endif
#endif

#include "io/run_store.h"
#include "io/sha256.h"
#include "io/snapshot_store.h"
#include "io/log_sink.h"
#include "core/sim_dimensions_internal.h"
#include "core/runtime_defaults.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>
#else
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/vfs.h>
#elif defined(__APPLE__)
#include <sys/mount.h>
#include <sys/param.h>
#endif
#endif

#if defined(HAVE_SQLITE3)
#include <sqlite3.h>
#endif

/* Writer lock next to the shared database. Not "<db>.lock": SQLite's unix-dotfile VFS uses that name. */
#define RUN_STORE_LOCK_SUFFIX ".writer-lock"
/* Header state the last merge committed, used to detect a stale view on the next writer's host. */
#define RUN_STORE_STATE_SUFFIX ".writer-state"

// MARK: Run key

static void key_put_text(sha256_ctx* ctx, const char* name, const char* text)
{
    char head[128];
    if (text == NULL) {
        (void)snprintf(head, sizeof(head), "%s=-\n", name);
        sha256_update(ctx, head, strlen(head));
        return;
    }
    const size_t len = strlen(text);
    (void)snprintf(head, sizeof(head), "%s=%zu:", name, len);
    sha256_update(ctx, head, strlen(head));
    sha256_update(ctx, text, len);
    sha256_update(ctx, "\n", 1u);
}

static void key_put_u64(sha256_ctx* ctx, const char* name, uint64_t value)
{
    char line[128];
    (void)snprintf(line, sizeof(line), "%s=%llu\n", name, (unsigned long long)value);
    sha256_update(ctx, line, strlen(line));
}

/* Doubles enter the key as their IEEE-754 bit pattern, so the text is exact and locale-free. */
static void key_put_f64(sha256_ctx* ctx, const char* name, double value)
{
    uint64_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    char line[128];
    (void)snprintf(line, sizeof(line), "%s=f64:%016llx\n", name, (unsigned long long)bits);
    sha256_update(ctx, line, strlen(line));
}

static void sha256_update_f64_le(sha256_ctx* ctx, const double* values, size_t count)
{
    unsigned char bytes[8 * 64];
    size_t fill = 0u;
    for (size_t i = 0u; i < count; ++i) {
        uint64_t bits = 0u;
        memcpy(&bits, &values[i], sizeof(bits));
        for (unsigned b = 0u; b < 8u; ++b) {
            bytes[fill++] = (unsigned char)(bits >> (8u * b));
        }
        if (fill == sizeof(bytes)) {
            sha256_update(ctx, bytes, fill);
            fill = 0u;
        }
    }
    if (fill > 0u) {
        sha256_update(ctx, bytes, fill);
    }
}

static void complex_array_sha256(const nlo_complex* values, size_t count, char out_hex[65])
{
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update_f64_le(&ctx, (const double*)values, 2u * count);
    sha256_final_hex(&ctx, out_hex);
}

static void key_put_complex_array(sha256_ctx* ctx, const char* name, const nlo_complex* values, size_t count)
{
    if (values == NULL || count == 0u) {
        key_put_text(ctx, name, NULL);
        return;
    }
    char hex[65];
    complex_array_sha256(values, count, hex);
    char line[160];
    (void)snprintf(line, sizeof(line), "%s=sha256:%zu:%s\n", name, count, hex);
    sha256_update(ctx, line, strlen(line));
}

static const char* key_expr(const char* expr, const char* fallback)
{
    return (expr != NULL && expr[0] != '\0') ? expr : fallback;
}

int run_store_compute_key(
    const sim_config* config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    size_t num_recorded_samples,
    const double* explicit_record_z,
    size_t explicit_record_z_count,
    const storage_options* storage,
    run_store_key_info* out_info
)
{
    if (config == NULL || input_field == NULL || num_time_samples == 0u || out_info == NULL) {
        return -1;
    }
    memset(out_info, 0, sizeof(*out_info));

    size_t nt = 0u;
    size_t nx = 0u;
    size_t ny = 0u;
    int explicit_nd = 0;
    if (resolve_sim_dimensions_internal(config, num_time_samples, &nt, &nx, &ny, &explicit_nd) != 0) {
        return -1;
    }
    const int tensor_mode = (config->tensor.nt > 0u) ? 1 : 0;

    /*
     * The key describes the physics the solver runs, so every value enters as
     * the core resolves it: unset expressions take their defaults, aliases are
     * merged, constants are the effective c0..cN, dt is the resolved step,
     * transverse steps count only for axes with more than one sample, and the
     * Raman settings only for the Kerr-Raman model.
     */
    const runtime_operator_params* rt = &config->runtime;
    const double dt = (config->time.delta_time > 0.0) ? config->time.delta_time
                      : (config->time.pulse_period > 0.0) ? config->time.pulse_period / (double)nt
                      : 1.0;

    sha256_ctx ctx;
    sha256_init(&ctx);
    key_put_text(&ctx, "kind", "nlolib");
    key_put_u64(&ctx, "key_version", 2u);

    key_put_u64(&ctx, "num_time_samples", num_time_samples);
    key_put_u64(&ctx, "nt", nt);
    key_put_u64(&ctx, "nx", nx);
    key_put_u64(&ctx, "ny", ny);
    key_put_u64(&ctx, "tensor_mode", (uint64_t)tensor_mode);
    key_put_f64(&ctx, "dt", dt);
    if (nx > 1u) {
        key_put_f64(&ctx, "dx", (config->spatial.delta_x > 0.0) ? config->spatial.delta_x : 1.0);
    }
    if (ny > 1u) {
        key_put_f64(&ctx, "dy", (config->spatial.delta_y > 0.0) ? config->spatial.delta_y : 1.0);
    }

    key_put_f64(&ctx, "propagation_distance", config->propagation.propagation_distance);
    key_put_f64(&ctx, "starting_step_size", config->propagation.starting_step_size);
    key_put_f64(&ctx, "max_step_size", config->propagation.max_step_size);
    key_put_f64(&ctx, "min_step_size", config->propagation.min_step_size);
    key_put_f64(&ctx, "error_tolerance", config->propagation.error_tolerance);

    /* Only the programs the solver compiles for this mode (see init_state_builder.c). */
    if (tensor_mode) {
        key_put_text(&ctx, "linear_factor_expr",
                     key_expr(rt->linear_factor_expr, key_expr(rt->dispersion_factor_expr, DEFAULT_LINEAR_FACTOR_EXPR)));
        key_put_text(&ctx, "linear_expr", key_expr(rt->linear_expr, key_expr(rt->dispersion_expr, DEFAULT_LINEAR_EXPR)));
        key_put_text(&ctx, "potential_expr", key_expr(rt->potential_expr, DEFAULT_POTENTIAL_EXPR));
    } else {
        key_put_text(&ctx, "dispersion_factor_expr",
                     key_expr(rt->dispersion_factor_expr, DEFAULT_DISPERSION_FACTOR_EXPR));
        key_put_text(&ctx, "dispersion_expr", key_expr(rt->dispersion_expr, DEFAULT_DISPERSION_EXPR));
    }
    key_put_text(&ctx, "nonlinear_expr", key_expr(rt->nonlinear_expr, DEFAULT_NONLINEAR_EXPR));

    double constants[RUNTIME_OPERATOR_CONSTANTS_MAX] = {DEFAULT_C0, DEFAULT_C1, DEFAULT_C2, DEFAULT_C3};
    size_t num_constants = 4u;
    if (rt->num_constants > RUNTIME_OPERATOR_CONSTANTS_MAX) {
        return -1;
    }
    for (size_t i = 0u; i < rt->num_constants; ++i) {
        constants[i] = rt->constants[i];
    }
    if (rt->num_constants > num_constants) {
        num_constants = rt->num_constants;
    }
    key_put_u64(&ctx, "num_constants", num_constants);
    for (size_t i = 0u; i < num_constants; ++i) {
        char name[32];
        (void)snprintf(name, sizeof(name), "c%zu", i);
        key_put_f64(&ctx, name, constants[i]);
    }
    const int raman = (rt->nonlinear_model == NONLINEAR_MODEL_KERR_RAMAN);
    key_put_u64(&ctx, "nonlinear_model", raman ? 1u : 0u);
    if (raman) {
        key_put_f64(&ctx, "nonlinear_gamma", rt->nonlinear_gamma);
        key_put_f64(&ctx, "raman_fraction", rt->raman_fraction);
        key_put_f64(&ctx, "raman_tau1", (rt->raman_tau1 > 0.0) ? rt->raman_tau1 : DEFAULT_RAMAN_TAU1);
        key_put_f64(&ctx, "raman_tau2", (rt->raman_tau2 > 0.0) ? rt->raman_tau2 : DEFAULT_RAMAN_TAU2);
        key_put_f64(&ctx, "shock_omega0", rt->shock_omega0);
        key_put_complex_array(&ctx, "raman_response_time", rt->raman_response_time, rt->raman_response_len);
    }

    complex_array_sha256(input_field, num_time_samples, out_info->input_sha256);
    key_put_text(&ctx, "input_field_sha256", out_info->input_sha256);

    const size_t potential_count = (tensor_mode || explicit_nd == 0) ? num_time_samples : 1u;
    if (config->spatial.potential_grid != NULL) {
        complex_array_sha256(config->spatial.potential_grid, potential_count, out_info->potential_sha256);
        key_put_text(&ctx, "potential_grid_sha256", out_info->potential_sha256);
    } else {
        key_put_text(&ctx, "potential_grid_sha256", NULL);
    }
    key_put_complex_array(&ctx, "frequency_grid", config->frequency.frequency_grid, nt);
    key_put_complex_array(&ctx, "wt_axis", config->time.wt_axis, nt);
    key_put_complex_array(&ctx, "kx_axis", config->spatial.kx_axis, nx);
    key_put_complex_array(&ctx, "ky_axis", config->spatial.ky_axis, ny);
    key_put_complex_array(&ctx, "spatial_frequency_grid", config->spatial.spatial_frequency_grid, num_time_samples);

    key_put_u64(&ctx, "num_recorded_samples", num_recorded_samples);
    if (explicit_record_z != NULL && explicit_record_z_count > 0u) {
        sha256_ctx zctx;
        sha256_init(&zctx);
        sha256_update_f64_le(&zctx, explicit_record_z, explicit_record_z_count);
        char zhex[65];
        sha256_final_hex(&zctx, zhex);
        key_put_text(&ctx, "explicit_record_z_sha256", zhex);
    } else {
        key_put_text(&ctx, "explicit_record_z_sha256", NULL);
    }

    const int record_mode = (storage != NULL) ? (int)storage->record_mode : (int)STORAGE_RECORD_MODE_REDUCED;
    const int final_field = (storage != NULL) ? (storage->log_final_output_field_to_db != 0) : 1;
    const size_t stride_t = (storage != NULL && storage->raw_stride_t > 1u) ? storage->raw_stride_t : 1u;
    const size_t stride_x = (storage != NULL && storage->raw_stride_x > 1u) ? storage->raw_stride_x : 1u;
    const size_t stride_y = (storage != NULL && storage->raw_stride_y > 1u) ? storage->raw_stride_y : 1u;
    key_put_u64(&ctx, "record_mode", (uint64_t)record_mode);
    key_put_u64(&ctx, "final_field", (uint64_t)final_field);
    if (record_mode != (int)STORAGE_RECORD_MODE_REDUCED) {
        key_put_u64(&ctx, "raw_stride_t", stride_t);
        key_put_u64(&ctx, "raw_stride_x", stride_x);
        key_put_u64(&ctx, "raw_stride_y", stride_y);
    }

    char full_hex[65];
    sha256_final_hex(&ctx, full_hex);
    memcpy(out_info->key, full_hex, 16u);
    out_info->key[16] = '\0';
    return 0;
}

// MARK: Platform helpers

static int rs_join(char* out, size_t cap, const char* a, const char* b)
{
    const size_t la = strlen(a);
    const int need_sep = (la > 0u && a[la - 1u] != '/' && a[la - 1u] != '\\');
    const int written = snprintf(out, cap, "%s%s%s", a, need_sep ? "/" : "", b);
    return (written < 0 || (size_t)written >= cap) ? -1 : 0;
}

static size_t rs_last_sep(const char* path)
{
    size_t pos = (size_t)-1;
    for (size_t i = 0u; path[i] != '\0'; ++i) {
        if (path[i] == '/' || path[i] == '\\') {
            pos = i;
        }
    }
    return pos;
}

static void rs_dirname(const char* path, char* out, size_t cap)
{
    const size_t sep = rs_last_sep(path);
    if (sep == (size_t)-1) {
        (void)snprintf(out, cap, ".");
        return;
    }
    const size_t len = (sep == 0u) ? 1u : sep;
    (void)snprintf(out, cap, "%.*s", (int)len, path);
}

static const char* rs_basename(const char* path)
{
    const size_t sep = rs_last_sep(path);
    return (sep == (size_t)-1) ? path : path + sep + 1u;
}

/* File name without a trailing ".sqlite" / ".db" / ".sqlite3" extension. */
static void rs_stem(const char* path, char* out, size_t cap)
{
    const char* base = rs_basename(path);
    const char* dot = strrchr(base, '.');
    const size_t len = (dot != NULL && dot != base) ? (size_t)(dot - base) : strlen(base);
    (void)snprintf(out, cap, "%.*s", (int)len, base);
}

#if defined(_WIN32)
static wchar_t* rs_wide(const char* utf8)
{
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
    if (n <= 0) {
        return NULL;
    }
    wchar_t* out = (wchar_t*)malloc((size_t)n * sizeof(wchar_t));
    if (out != NULL && MultiByteToWideChar(CP_UTF8, 0, utf8, -1, out, n) <= 0) {
        free(out);
        out = NULL;
    }
    return out;
}
#endif

static double rs_monotonic_seconds(void)
{
#if defined(_WIN32)
    return (double)GetTickCount64() / 1000.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#endif
}

static void rs_sleep_ms(unsigned ms)
{
#if defined(_WIN32)
    Sleep(ms);
#else
    struct timespec ts;
    ts.tv_sec = (time_t)(ms / 1000u);
    ts.tv_nsec = (long)(ms % 1000u) * 1000000L;
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {
    }
#endif
}

static unsigned long rs_pid(void)
{
#if defined(_WIN32)
    return (unsigned long)GetCurrentProcessId();
#else
    return (unsigned long)getpid();
#endif
}

/* 1 alive, 0 gone, -1 unknown. */
static int rs_pid_alive(unsigned long pid)
{
#if defined(_WIN32)
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid);
    if (h == NULL) {
        return (GetLastError() == ERROR_INVALID_PARAMETER) ? 0 : -1;
    }
    DWORD code = 0;
    const BOOL ok = GetExitCodeProcess(h, &code);
    CloseHandle(h);
    if (!ok) {
        return -1;
    }
    return (code == STILL_ACTIVE) ? 1 : 0;
#else
    if (kill((pid_t)pid, 0) == 0) {
        return 1;
    }
    return (errno == ESRCH) ? 0 : ((errno == EPERM) ? 1 : -1);
#endif
}

static void rs_hostname(char* out, size_t cap)
{
    char raw[256] = {0};
#if defined(_WIN32)
    DWORD len = (DWORD)sizeof(raw);
    if (!GetComputerNameA(raw, &len)) {
        raw[0] = '\0';
    }
#else
    if (gethostname(raw, sizeof(raw) - 1u) != 0) {
        raw[0] = '\0';
    }
#endif
    size_t j = 0u;
    for (size_t i = 0u; raw[i] != '\0' && raw[i] != '.' && j + 1u < cap; ++i) {
        const char c = raw[i];
        const int ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
        out[j++] = ok ? c : '_';
    }
    if (j == 0u && cap > 8u) {
        memcpy(out, "unknown", 7u);
        j = 7u;
    }
    out[j] = '\0';
}

static void rs_token(char out_hex[17])
{
    static unsigned long counter = 0u;
    counter += 1u;
    sha256_ctx ctx;
    sha256_init(&ctx);
    const unsigned long pid = rs_pid();
    const double mono = rs_monotonic_seconds();
    const time_t now = time(NULL);
    const void* stack_addr = (const void*)&ctx;
    char host[128];
    rs_hostname(host, sizeof(host));
    sha256_update(&ctx, &pid, sizeof(pid));
    sha256_update(&ctx, &mono, sizeof(mono));
    sha256_update(&ctx, &now, sizeof(now));
    sha256_update(&ctx, &counter, sizeof(counter));
    sha256_update(&ctx, &stack_addr, sizeof(stack_addr));
    sha256_update(&ctx, host, strlen(host));
    char hex[65];
    sha256_final_hex(&ctx, hex);
    memcpy(out_hex, hex, 16u);
    out_hex[16] = '\0';
}

static int rs_mkdir(const char* path)
{
#if defined(_WIN32)
    wchar_t* w = rs_wide(path);
    if (w == NULL) {
        return -1;
    }
    const int rc = _wmkdir(w);
    const int err = errno;
    free(w);
    return (rc == 0 || err == EEXIST) ? 0 : -1;
#else
    return (mkdir(path, 0775) == 0 || errno == EEXIST) ? 0 : -1;
#endif
}

/* 0 created, 1 exists, -1 error. */
static int rs_create_exclusive(const char* path, const char* content)
{
#if defined(_WIN32)
    wchar_t* w = rs_wide(path);
    if (w == NULL) {
        return -1;
    }
    const int fd = _wopen(w, _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
    const int err = errno;
    free(w);
    if (fd < 0) {
        return (err == EEXIST) ? 1 : -1;
    }
    const size_t len = strlen(content);
    const int ok = (_write(fd, content, (unsigned)len) == (int)len) && (_commit(fd) == 0);
    _close(fd);
#else
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0664);
    if (fd < 0) {
        return (errno == EEXIST) ? 1 : -1;
    }
    const size_t len = strlen(content);
    const int ok = (write(fd, content, len) == (ssize_t)len) && (fsync(fd) == 0);
    close(fd);
#endif
    if (!ok) {
        return -1;
    }
    return 0;
}

/* Overwrite an existing file in place; fails when the file is gone. */
static int rs_overwrite_existing(const char* path, const char* content)
{
#if defined(_WIN32)
    wchar_t* w = rs_wide(path);
    if (w == NULL) {
        return -1;
    }
    const int fd = _wopen(w, _O_WRONLY | _O_TRUNC | _O_BINARY);
    free(w);
    if (fd < 0) {
        return -1;
    }
    const size_t len = strlen(content);
    const int ok = (_write(fd, content, (unsigned)len) == (int)len) && (_commit(fd) == 0);
    _close(fd);
#else
    const int fd = open(path, O_WRONLY | O_TRUNC);
    if (fd < 0) {
        return -1;
    }
    const size_t len = strlen(content);
    const int ok = (write(fd, content, len) == (ssize_t)len) && (fsync(fd) == 0);
    close(fd);
#endif
    return ok ? 0 : -1;
}

/* Bytes read (NUL-terminated), -2 when missing, -1 on error. */
static long rs_read_small(const char* path, char* out, size_t cap)
{
#if defined(_WIN32)
    wchar_t* w = rs_wide(path);
    if (w == NULL) {
        return -1;
    }
    const int fd = _wopen(w, _O_RDONLY | _O_BINARY);
    const int err = errno;
    free(w);
    if (fd < 0) {
        return (err == ENOENT) ? -2 : -1;
    }
    const int n = _read(fd, out, (unsigned)(cap - 1u));
    _close(fd);
#else
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return (errno == ENOENT) ? -2 : -1;
    }
    const ssize_t n = read(fd, out, cap - 1u);
    close(fd);
#endif
    if (n < 0) {
        return -1;
    }
    out[n] = '\0';
    return (long)n;
}

static int rs_unlink(const char* path)
{
#if defined(_WIN32)
    wchar_t* w = rs_wide(path);
    if (w == NULL) {
        return -1;
    }
    const int rc = _wunlink(w);
    free(w);
    return rc;
#else
    return unlink(path);
#endif
}

/* Rename that never replaces an existing target. */
static int rs_rename_noreplace(const char* from, const char* to)
{
#if defined(_WIN32)
    wchar_t* wf = rs_wide(from);
    wchar_t* wt = rs_wide(to);
    const int ok = (wf != NULL && wt != NULL) ? (MoveFileExW(wf, wt, 0) != 0) : 0;
    free(wf);
    free(wt);
    return ok ? 0 : -1;
#else
    if (link(from, to) == 0) {
        (void)unlink(from);
        return 0;
    }
    if (errno == EEXIST) {
        return -1;
    }
    /* File systems without hard links: fall back to rename onto a fresh unique name. */
    struct stat st;
    if (stat(to, &st) == 0) {
        return -1;
    }
    return rename(from, to);
#endif
}

static int rs_file_exists(const char* path)
{
#if defined(_WIN32)
    wchar_t* w = rs_wide(path);
    if (w == NULL) {
        return 0;
    }
    struct _stat64 st;
    const int rc = _wstat64(w, &st);
    free(w);
    return rc == 0;
#else
    struct stat st;
    return stat(path, &st) == 0;
#endif
}

/* Age of the file's modification time against this machine's clock. */
static int rs_file_age_seconds(const char* path, double* out_age)
{
#if defined(_WIN32)
    wchar_t* w = rs_wide(path);
    if (w == NULL) {
        return -1;
    }
    struct _stat64 st;
    const int rc = _wstat64(w, &st);
    free(w);
    if (rc != 0) {
        return -1;
    }
    *out_age = difftime(time(NULL), (time_t)st.st_mtime);
#else
    struct stat st;
    if (stat(path, &st) != 0) {
        return -1;
    }
    *out_age = difftime(time(NULL), st.st_mtime);
#endif
    return 0;
}

/* Rename that replaces an existing target (used for the state sidecar only). */
static int rs_rename_replace(const char* from, const char* to)
{
#if defined(_WIN32)
    wchar_t* wf = rs_wide(from);
    wchar_t* wt = rs_wide(to);
    const int ok = (wf != NULL && wt != NULL) ? (MoveFileExW(wf, wt, MOVEFILE_REPLACE_EXISTING) != 0) : 0;
    free(wf);
    free(wt);
    return ok ? 0 : -1;
#else
    return rename(from, to);
#endif
}

/* 1 when @p dir is on a network file system (SMB/CIFS or NFS), 0 when local or unknown. */
static int rs_is_network_dir(const char* dir)
{
#if defined(_WIN32)
    if ((dir[0] == '\\' || dir[0] == '/') && (dir[1] == '\\' || dir[1] == '/')) {
        return 1;
    }
    if (dir[0] != '\0' && dir[1] == ':') {
        wchar_t root[4] = {(wchar_t)dir[0], L':', L'\\', 0};
        return GetDriveTypeW(root) == DRIVE_REMOTE;
    }
    return 0;
#elif defined(__linux__)
    struct statfs st;
    if (statfs(dir, &st) != 0) {
        return 0;
    }
    const unsigned long type = (unsigned long)st.f_type;
    return type == 0xFF534D42ul /* CIFS */ || type == 0xFE534D42ul /* SMB2 */ ||
           type == 0x517Bul /* SMB */ || type == 0x6969ul /* NFS */;
#elif defined(__APPLE__)
    struct statfs st;
    if (statfs(dir, &st) != 0) {
        return 0;
    }
    return strcmp(st.f_fstypename, "smbfs") == 0 || strcmp(st.f_fstypename, "nfs") == 0;
#else
    (void)dir;
    return 0;
#endif
}

typedef int (*rs_dir_visit)(const char* name, void* user);

static int rs_list_dir(const char* dir, rs_dir_visit visit, void* user)
{
#if defined(_WIN32)
    char pattern[RUN_STORE_PATH_MAX];
    if (rs_join(pattern, sizeof(pattern), dir, "*") != 0) {
        return -1;
    }
    wchar_t* w = rs_wide(pattern);
    if (w == NULL) {
        return -1;
    }
    WIN32_FIND_DATAW data;
    HANDLE h = FindFirstFileW(w, &data);
    free(w);
    if (h == INVALID_HANDLE_VALUE) {
        return (GetLastError() == ERROR_FILE_NOT_FOUND || GetLastError() == ERROR_PATH_NOT_FOUND) ? 0 : -1;
    }
    do {
        char name[1024];
        if (WideCharToMultiByte(CP_UTF8, 0, data.cFileName, -1, name, (int)sizeof(name), NULL, NULL) > 0) {
            if (visit(name, user) != 0) {
                break;
            }
        }
    } while (FindNextFileW(h, &data));
    FindClose(h);
    return 0;
#else
    DIR* d = opendir(dir);
    if (d == NULL) {
        return (errno == ENOENT) ? 0 : -1;
    }
    struct dirent* ent = NULL;
    while ((ent = readdir(d)) != NULL) {
        if (visit(ent->d_name, user) != 0) {
            break;
        }
    }
    closedir(d);
    return 0;
#endif
}

typedef struct {
    const char* prefix;
    int found;
} prefix_search;

static int prefix_visit(const char* name, void* user)
{
    prefix_search* search = (prefix_search*)user;
    if (strncmp(name, search->prefix, strlen(search->prefix)) == 0) {
        search->found = 1;
        return 1;
    }
    return 0;
}

/* 1 when @p dir has a file whose name starts with @p prefix. */
static int staging_has_prefix(const char* dir, const char* prefix)
{
    prefix_search search = {prefix, 0};
    (void)rs_list_dir(dir, prefix_visit, &search);
    return search.found;
}

void run_store_utc_now(char out_text[32])
{
    time_t seconds = 0;
    long millis = 0;
#if defined(_WIN32)
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    const unsigned long long ticks = ((unsigned long long)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    const unsigned long long unix_100ns = ticks - 116444736000000000ull;
    seconds = (time_t)(unix_100ns / 10000000ull);
    millis = (long)((unix_100ns / 10000ull) % 1000ull);
    struct tm tm_utc;
    gmtime_s(&tm_utc, &seconds);
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
        seconds = ts.tv_sec;
        millis = ts.tv_nsec / 1000000L;
    } else {
        seconds = time(NULL);
    }
    struct tm tm_utc;
    gmtime_r(&seconds, &tm_utc);
#endif
    (void)snprintf(out_text,
                   32u,
                   "%04d-%02d-%02dT%02d:%02d:%02d.%03ld",
                   (tm_utc.tm_year + 1900) % 10000,
                   (tm_utc.tm_mon + 1) % 100,
                   tm_utc.tm_mday % 100,
                   tm_utc.tm_hour % 100,
                   tm_utc.tm_min % 100,
                   tm_utc.tm_sec % 100,
                   millis % 1000L);
}

void run_store_index_file(const char* shared_path, char* out_file, size_t out_capacity)
{
    char dir[RUN_STORE_PATH_MAX];
    rs_dirname(shared_path, dir, sizeof(dir));
    const char* parent = rs_basename(dir);
    const char* base = rs_basename(shared_path);
    if (parent[0] == '\0' || strcmp(parent, ".") == 0 || strcmp(parent, "..") == 0 ||
        strcmp(dir, "/") == 0 || (strlen(dir) == 2u && dir[1] == ':')) {
        (void)snprintf(out_file, out_capacity, "%s", base);
    } else {
        (void)snprintf(out_file, out_capacity, "%s/%s", parent, base);
    }
}

static int staging_dir_of(const char* shared_path, char* out, size_t cap)
{
    char dir[RUN_STORE_PATH_MAX];
    rs_dirname(shared_path, dir, sizeof(dir));
    return rs_join(out, cap, dir, "staging");
}

int run_store_make_staging_path(const char* shared_path, const char* key, char* out_path, size_t out_capacity)
{
    if (shared_path == NULL || key == NULL || out_path == NULL) {
        return -1;
    }
    char dir[RUN_STORE_PATH_MAX];
    rs_dirname(shared_path, dir, sizeof(dir));
    if (rs_mkdir(dir) != 0) {
        return -1;
    }
    char staging[RUN_STORE_PATH_MAX];
    if (staging_dir_of(shared_path, staging, sizeof(staging)) != 0 || rs_mkdir(staging) != 0) {
        return -1;
    }
    char stem[256];
    char host[128];
    char token[17];
    rs_stem(shared_path, stem, sizeof(stem));
    rs_hostname(host, sizeof(host));
    rs_token(token);
    char name[512];
    (void)snprintf(name, sizeof(name), "%s.%s.%lu.%s.%s.sqlite", stem, key, rs_pid(), token, host);
    return rs_join(out_path, out_capacity, staging, name);
}

void run_store_discard(const char* staging_path)
{
    if (staging_path == NULL || staging_path[0] == '\0') {
        return;
    }
    char journal[RUN_STORE_PATH_MAX];
    (void)rs_unlink(staging_path);
    if (snprintf(journal, sizeof(journal), "%s-journal", staging_path) < (int)sizeof(journal)) {
        (void)rs_unlink(journal);
    }
}

// MARK: Lock file

static double g_lock_timeout_s = 0.0;
static double g_lock_stale_s = 0.0;

void run_store_set_lock_timing(double timeout_seconds, double stale_seconds)
{
    g_lock_timeout_s = timeout_seconds;
    g_lock_stale_s = stale_seconds;
}

static double env_seconds(const char* name, double fallback)
{
    const char* text = getenv(name);
    if (text == NULL || text[0] == '\0') {
        return fallback;
    }
    char* end = NULL;
    const double value = strtod(text, &end);
    return (end != text && value > 0.0) ? value : fallback;
}

static double env_seconds_allow_zero(const char* name, double fallback)
{
    const char* text = getenv(name);
    if (text == NULL || text[0] == '\0') {
        return fallback;
    }
    char* end = NULL;
    const double value = strtod(text, &end);
    return (end != text && value >= 0.0) ? value : fallback;
}

static double lock_timeout_seconds(void)
{
    return (g_lock_timeout_s > 0.0) ? g_lock_timeout_s : env_seconds("NLOLIB_STORE_LOCK_TIMEOUT_S", 600.0);
}

static double lock_stale_seconds(void)
{
    return (g_lock_stale_s > 0.0) ? g_lock_stale_s : env_seconds("NLOLIB_STORE_LOCK_STALE_S", 120.0);
}

typedef struct {
    char path[RUN_STORE_PATH_MAX];
    char host[128];
    char token[17];
    char created[32];
    unsigned long beat;
    int held;
} run_lock;

static void lock_format(const run_lock* lock, char* out, size_t cap)
{
    char now[32];
    run_store_utc_now(now);
    (void)snprintf(out,
                   cap,
                   "nlolib-lock 1\nhost=%s\npid=%lu\ncreated=%s\nupdated=%s\nbeat=%lu\ntoken=%s\n",
                   lock->host,
                   rs_pid(),
                   lock->created,
                   now,
                   lock->beat,
                   lock->token);
}

static int lock_field(const char* content, const char* name, char* out, size_t cap)
{
    char needle[32];
    (void)snprintf(needle, sizeof(needle), "\n%s=", name);
    const char* at = strstr(content, needle);
    if (at == NULL) {
        return -1;
    }
    at += strlen(needle);
    size_t n = 0u;
    while (at[n] != '\0' && at[n] != '\n' && at[n] != '\r' && n + 1u < cap) {
        out[n] = at[n];
        n += 1u;
    }
    out[n] = '\0';
    return 0;
}

static int lock_owned(const run_lock* lock)
{
    char content[1024];
    char token[64];
    if (rs_read_small(lock->path, content, sizeof(content)) < 0 ||
        lock_field(content, "token", token, sizeof(token)) != 0) {
        return 0;
    }
    return strcmp(token, lock->token) == 0;
}

/* Rewrite the lock with a new heartbeat; -1 when the lock is no longer ours. */
static int lock_refresh(run_lock* lock)
{
    if (!lock->held || !lock_owned(lock)) {
        return -1;
    }
    lock->beat += 1u;
    char content[1024];
    lock_format(lock, content, sizeof(content));
    return rs_overwrite_existing(lock->path, content);
}

static void lock_release(run_lock* lock)
{
    if (lock->held && lock_owned(lock)) {
        (void)rs_unlink(lock->path);
    }
    lock->held = 0;
}

/*
 * Remove a lock judged stale. Breakers serialise on "<lock>.break"; the stale
 * file is renamed away only if it still holds exactly the content that was
 * judged stale, and the renamed file is checked again afterwards.
 */
static void lock_break_stale(const run_lock* self, const char* observed)
{
    char brk[RUN_STORE_PATH_MAX];
    if (snprintf(brk, sizeof(brk), "%s.break", self->path) >= (int)sizeof(brk)) {
        return;
    }
    char mine[1024];
    lock_format(self, mine, sizeof(mine));
    const int rc = rs_create_exclusive(brk, mine);
    if (rc == 1) {
        double age = 0.0;
        if (rs_file_age_seconds(brk, &age) == 0 && age > 2.0 * lock_stale_seconds()) {
            (void)rs_unlink(brk);
        }
        return;
    }
    if (rc != 0) {
        return;
    }

    char current[1024];
    if (rs_read_small(self->path, current, sizeof(current)) >= 0 && strcmp(current, observed) == 0) {
        char moved[RUN_STORE_PATH_MAX];
        if (snprintf(moved, sizeof(moved), "%s.stale-%s", self->path, self->token) < (int)sizeof(moved) &&
            rs_rename_noreplace(self->path, moved) == 0) {
            char check[1024];
            if (rs_read_small(moved, check, sizeof(check)) >= 0 && strcmp(check, observed) != 0) {
                /* A fresh lock was taken in between: hand it back. */
                (void)rs_rename_noreplace(moved, self->path);
            }
            (void)rs_unlink(moved);
            log_emit(LOG_LEVEL_WARN, "[nlolib][store] removed stale lock %s", self->path);
        }
    }
    (void)rs_unlink(brk);
}

static run_store_status lock_acquire(run_lock* lock, const char* shared_path)
{
    memset(lock, 0, sizeof(*lock));
    if (snprintf(lock->path, sizeof(lock->path), "%s" RUN_STORE_LOCK_SUFFIX, shared_path) >= (int)sizeof(lock->path)) {
        return RUN_STORE_ERROR;
    }
    rs_hostname(lock->host, sizeof(lock->host));
    rs_token(lock->token);
    run_store_utc_now(lock->created);

    const double timeout = lock_timeout_seconds();
    const double stale = lock_stale_seconds();
    const double start = rs_monotonic_seconds();
    char observed[1024] = {0};
    double observed_since = start;
    unsigned attempt = 0u;

    for (;;) {
        char content[1024];
        lock_format(lock, content, sizeof(content));
        const int rc = rs_create_exclusive(lock->path, content);
        if (rc == 0) {
            lock->held = 1;
            return RUN_STORE_OK;
        }
        if (rc < 0) {
            log_emit(LOG_LEVEL_ERROR, "[nlolib][store] cannot create lock file %s", lock->path);
            return RUN_STORE_ERROR;
        }

        char current[1024];
        const long n = rs_read_small(lock->path, current, sizeof(current));
        const double now = rs_monotonic_seconds();
        if (n >= 0) {
            int is_stale = 0;
            char host[128] = {0};
            char pid_text[32] = {0};
            if (lock_field(current, "host", host, sizeof(host)) == 0 &&
                lock_field(current, "pid", pid_text, sizeof(pid_text)) == 0 &&
                strcmp(host, lock->host) == 0) {
                const unsigned long pid = strtoul(pid_text, NULL, 10);
                if (pid != rs_pid() && rs_pid_alive(pid) == 0) {
                    is_stale = 1;
                }
            }
            if (strcmp(current, observed) != 0) {
                (void)snprintf(observed, sizeof(observed), "%s", current);
                observed_since = now;
            } else if (now - observed_since >= stale) {
                is_stale = 1;
            }
            if (is_stale) {
                lock_break_stale(lock, current);
                observed[0] = '\0';
                continue;
            }
        }
        if (now - start >= timeout) {
            log_emit(LOG_LEVEL_WARN,
                     "[nlolib][store] timed out after %.0f s waiting for lock %s",
                     timeout,
                     lock->path);
            return RUN_STORE_LOCK_TIMEOUT;
        }
        attempt += 1u;
        rs_sleep_ms(100u + (unsigned)((rs_pid() + attempt * 37u) % 200u));
    }
}

#if defined(HAVE_SQLITE3)

// MARK: SQLite helpers

static int db_exec(sqlite3* db, const char* sql)
{
    char* err = NULL;
    if (sqlite3_exec(db, sql, NULL, NULL, &err) != SQLITE_OK) {
        log_emit(LOG_LEVEL_ERROR, "[nlolib][store] sqlite error: %s (%s)", err != NULL ? err : "?", sql);
        sqlite3_free(err);
        return -1;
    }
    return 0;
}

const char* run_store_sqlite_vfs(void)
{
    const char* env = getenv("NLOLIB_STORE_VFS");
    if (env != NULL && env[0] != '\0') {
        return (strcmp(env, "default") == 0) ? NULL : env;
    }
#if defined(_WIN32)
    return NULL;
#else
    return "unix-none";
#endif
}

/* Readers use SQLite's default VFS (read locks work on SMB); writers use run_store_sqlite_vfs(). */
static int db_open(const char* path, int readonly, sqlite3** out_db)
{
    *out_db = NULL;
    const int flags = readonly ? SQLITE_OPEN_READONLY : (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
    const char* vfs = readonly ? NULL : run_store_sqlite_vfs();
    if (vfs != NULL && sqlite3_vfs_find(vfs) == NULL) {
        log_emit(LOG_LEVEL_ERROR, "[nlolib][store] SQLite VFS '%s' is not available", vfs);
        return -1;
    }
    if (sqlite3_open_v2(path, out_db, flags, vfs) != SQLITE_OK) {
        if (*out_db != NULL) {
            sqlite3_close(*out_db);
            *out_db = NULL;
        }
        return -1;
    }
    sqlite3_busy_timeout(*out_db, 60000);
    return 0;
}

/*
 * Open @p path read-only and start a read transaction, waiting out a merge in
 * progress. Writers take no SQLite locks by default (unix-none), so a reader
 * that opens during a merge sees the merge's journal as hot; a read-only
 * connection cannot roll it back and gets SQLITE_READONLY ("attempt to write a
 * readonly database") until the merge's transaction ends. Retried like a busy
 * timeout (60 s).
 */
static int reader_open(const char* path, sqlite3** out_db)
{
    const double deadline = rs_monotonic_seconds() + 60.0;
    for (;;) {
        if (db_open(path, 1, out_db) != 0) {
            return -1;
        }
        const int rc = sqlite3_exec(*out_db, "BEGIN; SELECT COUNT(*) FROM sqlite_master;", NULL, NULL, NULL) & 0xff;
        if (rc == SQLITE_OK) {
            return 0;
        }
        sqlite3_close(*out_db);
        *out_db = NULL;
        if ((rc != SQLITE_READONLY && rc != SQLITE_BUSY) || rs_monotonic_seconds() >= deadline) {
            return -1;
        }
        rs_sleep_ms(20u);
    }
}

static int db_table_exists(sqlite3* db, const char* schema, const char* table)
{
    char sql[256];
    (void)snprintf(sql, sizeof(sql), "SELECT 1 FROM \"%s\".sqlite_master WHERE type='table' AND name=?;", schema);
    sqlite3_stmt* stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        return 0;
    }
    sqlite3_bind_text(stmt, 1, table, -1, SQLITE_TRANSIENT);
    const int found = (sqlite3_step(stmt) == SQLITE_ROW);
    sqlite3_finalize(stmt);
    return found;
}

static int db_text_query(sqlite3* db, const char* sql, const char* bind1, char* out, size_t cap)
{
    sqlite3_stmt* stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        return -1;
    }
    if (bind1 != NULL) {
        sqlite3_bind_text(stmt, 1, bind1, -1, SQLITE_TRANSIENT);
    }
    int rc = -1;
    if (sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_type(stmt, 0) != SQLITE_NULL) {
        (void)snprintf(out, cap, "%s", (const char*)sqlite3_column_text(stmt, 0));
        rc = 0;
    }
    sqlite3_finalize(stmt);
    return rc;
}

typedef struct {
    char (*names)[128];
    char (*types)[32];
    size_t count;
} column_list;

static void column_list_free(column_list* list)
{
    free(list->names);
    free(list->types);
    memset(list, 0, sizeof(*list));
}

static int column_list_load(sqlite3* db, const char* schema, const char* table, column_list* out)
{
    memset(out, 0, sizeof(*out));
    char sql[256];
    (void)snprintf(sql, sizeof(sql), "PRAGMA \"%s\".table_info(\"%s\");", schema, table);
    sqlite3_stmt* stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
        return -1;
    }
    size_t cap = 0u;
    int rc = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        if (out->count == cap) {
            cap = (cap == 0u) ? 32u : 2u * cap;
            void* names = realloc(out->names, cap * sizeof(*out->names));
            void* types = realloc(out->types, cap * sizeof(*out->types));
            if (names == NULL || types == NULL) {
                free(names != NULL ? names : out->names);
                free(types != NULL ? types : out->types);
                out->names = NULL;
                out->types = NULL;
                out->count = 0u;
                rc = -1;
                break;
            }
            out->names = (char (*)[128])names;
            out->types = (char (*)[32])types;
        }
        const char* name = (const char*)sqlite3_column_text(stmt, 1);
        const char* type = (const char*)sqlite3_column_text(stmt, 2);
        (void)snprintf(out->names[out->count], 128u, "%s", name != NULL ? name : "");
        (void)snprintf(out->types[out->count], 32u, "%s", type != NULL ? type : "");
        out->count += 1u;
    }
    sqlite3_finalize(stmt);
    return rc;
}

static int column_list_has(const column_list* list, const char* name)
{
    for (size_t i = 0u; i < list->count; ++i) {
        if (sqlite3_stricmp(list->names[i], name) == 0) {
            return 1;
        }
    }
    return 0;
}

/* Copy rows of stg.<table> matching @p where (with ?1 bound to run_id and optional ?2/?3) into main. */
static int copy_rows(
    sqlite3* db,
    const char* table,
    const char* verb,
    const char* where,
    const char* run_id,
    sqlite3_int64 lo,
    sqlite3_int64 hi
)
{
    if (!db_table_exists(db, "stg", table)) {
        return 0;
    }
    column_list src;
    column_list dst;
    if (column_list_load(db, "stg", table, &src) != 0 || column_list_load(db, "main", table, &dst) != 0) {
        column_list_free(&src);
        return -1;
    }
    size_t cap = 256u + 2u * src.count * 132u + strlen(where);
    char* cols = (char*)calloc(cap, 1u);
    char* sql = (char*)calloc(2u * cap, 1u);
    int rc = (cols == NULL || sql == NULL) ? -1 : 0;
    size_t used = 0u;
    for (size_t i = 0u; rc == 0 && i < src.count; ++i) {
        if (!column_list_has(&dst, src.names[i])) {
            continue;
        }
        char* quoted = sqlite3_mprintf("%s\"%w\"", used > 0u ? "," : "", src.names[i]);
        if (quoted == NULL) {
            rc = -1;
            break;
        }
        const size_t len = strlen(quoted);
        memcpy(cols + used, quoted, len);
        used += len;
        sqlite3_free(quoted);
    }
    if (rc == 0 && used > 0u) {
        (void)snprintf(sql, 2u * cap, "%s INTO main.\"%s\"(%s) SELECT %s FROM stg.\"%s\" WHERE %s;",
                       verb, table, cols, cols, table, where);
        sqlite3_stmt* stmt = NULL;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
            log_emit(LOG_LEVEL_ERROR, "[nlolib][store] merge prepare failed: %s", sqlite3_errmsg(db));
            rc = -1;
        } else {
            sqlite3_bind_text(stmt, 1, run_id, -1, SQLITE_TRANSIENT);
            if (sqlite3_bind_parameter_count(stmt) >= 3) {
                sqlite3_bind_int64(stmt, 2, lo);
                sqlite3_bind_int64(stmt, 3, hi);
            }
            if (sqlite3_step(stmt) != SQLITE_DONE) {
                log_emit(LOG_LEVEL_ERROR, "[nlolib][store] merge copy of %s failed: %s", table, sqlite3_errmsg(db));
                rc = -1;
            }
            sqlite3_finalize(stmt);
        }
    }
    free(cols);
    free(sql);
    column_list_free(&src);
    column_list_free(&dst);
    return rc;
}

/* Add parameter columns of stg.runs that main.runs does not have yet, with their declared type. */
static int merge_runs_columns(sqlite3* db)
{
    column_list src;
    column_list dst;
    if (column_list_load(db, "stg", "runs", &src) != 0 || column_list_load(db, "main", "runs", &dst) != 0) {
        column_list_free(&src);
        return -1;
    }
    int rc = 0;
    for (size_t i = 0u; rc == 0 && i < src.count; ++i) {
        if (column_list_has(&dst, src.names[i])) {
            continue;
        }
        char* sql = sqlite3_mprintf("ALTER TABLE main.runs ADD COLUMN \"%w\" %s;",
                                    src.names[i],
                                    src.types[i][0] != '\0' ? src.types[i] : "TEXT");
        rc = (sql != NULL) ? db_exec(db, sql) : -1;
        sqlite3_free(sql);
    }
    column_list_free(&src);
    column_list_free(&dst);
    return rc;
}

static int delete_run(sqlite3* db, const char* key, const char* run_id)
{
    static const char* tables[] = {
        "io_record_maps", "io_record_layout", "io_record_chunks", "io_final_output_fields"
    };
    sqlite3_stmt* stmt = NULL;
    if (sqlite3_prepare_v2(db, "DELETE FROM main.runs WHERE key=?1;", -1, &stmt, NULL) != SQLITE_OK) {
        return -1;
    }
    sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT);
    int rc = (sqlite3_step(stmt) == SQLITE_DONE) ? 0 : -1;
    sqlite3_finalize(stmt);
    for (size_t i = 0u; rc == 0 && i < sizeof(tables) / sizeof(tables[0]); ++i) {
        if (!db_table_exists(db, "main", tables[i])) {
            continue;
        }
        char sql[512];
        (void)snprintf(sql, sizeof(sql),
                       "DELETE FROM main.\"%s\" WHERE run_id=?2 OR run_id IN "
                       "(SELECT run_id FROM main.io_runs WHERE key=?1);",
                       tables[i]);
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
            return -1;
        }
        sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, run_id, -1, SQLITE_TRANSIENT);
        rc = (sqlite3_step(stmt) == SQLITE_DONE) ? 0 : -1;
        sqlite3_finalize(stmt);
    }
    if (rc == 0 &&
        sqlite3_prepare_v2(db, "DELETE FROM main.io_runs WHERE run_id=?2 OR key=?1;", -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, run_id, -1, SQLITE_TRANSIENT);
        rc = (sqlite3_step(stmt) == SQLITE_DONE) ? 0 : -1;
        sqlite3_finalize(stmt);
    }
    return rc;
}

/* 1 when the file's header says WAL (read/write versions 2), 0 otherwise or when unreadable. */
static int file_is_wal(const char* path)
{
#if defined(_WIN32)
    wchar_t* w = rs_wide(path);
    FILE* f = (w != NULL) ? _wfopen(w, L"rb") : NULL;
    free(w);
#else
    FILE* f = fopen(path, "rb");
#endif
    if (f == NULL) {
        return 0;
    }
    unsigned char h[20];
    const size_t n = fread(h, 1u, sizeof(h), f);
    fclose(f);
    return n == sizeof(h) && memcmp(h, "SQLite format 3", 16) == 0 && (h[18] == 2u || h[19] == 2u);
}

/*
 * Databases written by older nlolib releases used WAL. Switching them needs
 * SQLite's own locking and shared memory, so it is done once with the default
 * VFS; on a Linux SMB mount that cannot commit, and the file has to be
 * converted once on Windows or a local disk.
 */
static int convert_from_wal(const char* path)
{
    if (!file_is_wal(path)) {
        return 0;
    }
    sqlite3* db = NULL;
    char mode[32] = {0};
    int rc = -1;
    if (sqlite3_open_v2(path, &db, SQLITE_OPEN_READWRITE, NULL) == SQLITE_OK) {
        sqlite3_busy_timeout(db, 60000);
        if (db_text_query(db, "PRAGMA journal_mode=DELETE;", NULL, mode, sizeof(mode)) == 0 &&
            sqlite3_stricmp(mode, "delete") == 0) {
            rc = 0;
        }
    }
    sqlite3_close(db);
    if (rc != 0) {
        log_emit(LOG_LEVEL_ERROR,
                 "[nlolib][store] %s is a WAL database and could not be switched to the rollback journal "
                 "here; run `PRAGMA journal_mode=DELETE;` on it once from Windows or a local disk",
                 path);
    } else {
        log_emit(LOG_LEVEL_INFO, "[nlolib][store] switched %s from WAL to the rollback journal", path);
    }
    return rc;
}

static int open_shared(const char* shared_path, sqlite3** out_db)
{
    if (convert_from_wal(shared_path) != 0) {
        *out_db = NULL;
        return -1;
    }
    if (db_open(shared_path, 0, out_db) != 0) {
        log_emit(LOG_LEVEL_ERROR, "[nlolib][store] cannot open shared database %s", shared_path);
        return -1;
    }
    char mode[32] = {0};
    if (db_text_query(*out_db, "PRAGMA journal_mode=DELETE;", NULL, mode, sizeof(mode)) != 0 ||
        sqlite3_stricmp(mode, "delete") != 0 ||
        db_exec(*out_db, "PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;") != 0 ||
        snapshot_store_ensure_schema(*out_db) != 0) {
        log_emit(LOG_LEVEL_ERROR,
                 "[nlolib][store] shared database %s could not be prepared (journal mode '%s')",
                 shared_path,
                 mode);
        sqlite3_close(*out_db);
        *out_db = NULL;
        return -1;
    }
    return 0;
}

/* Upper bound on the payload bytes one merge transaction writes (NLOLIB_STORE_MERGE_BATCH_BYTES). */
#define RUN_STORE_MERGE_BATCH_BYTES (16ll * 1024ll * 1024ll)

static sqlite3_int64 merge_batch_bytes(void)
{
    const double value = env_seconds("NLOLIB_STORE_MERGE_BATCH_BYTES", (double)RUN_STORE_MERGE_BATCH_BYTES);
    return (value >= 1.0) ? (sqlite3_int64)value : RUN_STORE_MERGE_BATCH_BYTES;
}

typedef struct {
    sqlite3* db;
    run_lock* lock;
    const char* run_id;
    int in_txn;
    sqlite3_int64 txn_bytes;
    sqlite3_int64 budget;
    unsigned commits;
} merge_txn;

static int merge_begin(merge_txn* m)
{
    if (m->in_txn) {
        return 0;
    }
    if (db_exec(m->db, "BEGIN IMMEDIATE;") != 0) {
        return -1;
    }
    m->in_txn = 1;
    m->txn_bytes = 0;
    return 0;
}

/* Commit the open transaction, checking that the lock is still ours first. */
static int merge_commit(merge_txn* m)
{
    if (!m->in_txn) {
        return 0;
    }
    if (!lock_owned(m->lock)) {
        log_emit(LOG_LEVEL_ERROR, "[nlolib][store] lost lock %s before commit; merge rolled back", m->lock->path);
        return -1;
    }
    if (db_exec(m->db, "COMMIT;") != 0) {
        return -1;
    }
    m->in_txn = 0;
    m->commits += 1u;
    const char* crash = getenv("NLOLIB_STORE_TEST_CRASH_AFTER_COMMITS");
    if (crash != NULL && crash[0] != '\0' && (unsigned)strtoul(crash, NULL, 10) == m->commits) {
        /* Test hook: die like a killed writer between two merge transactions. */
        _exit(77);
    }
    return lock_refresh(m->lock);
}

/*
 * Copy the rows of stg.<table> for the run in index order, splitting them over
 * transactions so that none writes much more than the batch budget.
 */
static int merge_table_batched(merge_txn* m, const char* table, const char* index_col, const char* bytes_expr)
{
    if (!db_table_exists(m->db, "stg", table)) {
        return 0;
    }
    char sql[512];
    (void)snprintf(sql, sizeof(sql), "SELECT \"%s\", %s FROM stg.\"%s\" WHERE run_id=?1 ORDER BY \"%s\";",
                   index_col, bytes_expr, table, index_col);
    char where[256];
    (void)snprintf(where, sizeof(where), "run_id=?1 AND \"%s\" BETWEEN ?2 AND ?3", index_col);
    sqlite3_stmt* list = NULL;
    if (sqlite3_prepare_v2(m->db, sql, -1, &list, NULL) != SQLITE_OK) {
        return -1;
    }
    sqlite3_bind_text(list, 1, m->run_id, -1, SQLITE_TRANSIENT);

    /* Collect (index, bytes) first: the list statement must not stay open across commits. */
    size_t count = 0u;
    size_t cap = 0u;
    sqlite3_int64* idx = NULL;
    sqlite3_int64* bytes = NULL;
    int rc = 0;
    while (sqlite3_step(list) == SQLITE_ROW) {
        if (count == cap) {
            cap = (cap == 0u) ? 64u : 2u * cap;
            sqlite3_int64* ni = (sqlite3_int64*)realloc(idx, cap * sizeof(*idx));
            sqlite3_int64* nb = (sqlite3_int64*)realloc(bytes, cap * sizeof(*bytes));
            if (ni == NULL || nb == NULL) {
                free(ni != NULL ? ni : idx);
                free(nb != NULL ? nb : bytes);
                sqlite3_finalize(list);
                return -1;
            }
            idx = ni;
            bytes = nb;
        }
        idx[count] = sqlite3_column_int64(list, 0);
        bytes[count] = sqlite3_column_int64(list, 1);
        count += 1u;
    }
    sqlite3_finalize(list);

    size_t first = 0u;
    while (rc == 0 && first < count) {
        if (merge_begin(m) != 0) {
            rc = -1;
            break;
        }
        size_t last = first;
        sqlite3_int64 batch = bytes[first];
        while (last + 1u < count && m->txn_bytes + batch + bytes[last + 1u] <= m->budget) {
            last += 1u;
            batch += bytes[last];
        }
        if (m->txn_bytes > 0 && m->txn_bytes + batch > m->budget) {
            /* Start this batch in a fresh transaction. */
            rc = merge_commit(m);
            continue;
        }
        rc = copy_rows(m->db, table, "INSERT", where, m->run_id, idx[first], idx[last]);
        m->txn_bytes += batch;
        first = last + 1u;
        if (rc == 0 && m->txn_bytes >= m->budget) {
            rc = merge_commit(m);
        }
    }
    free(idx);
    free(bytes);
    return rc;
}

/* `file` names the database the run is in now, whatever store the staging file was made for. */
static int set_runs_file(sqlite3* db, const char* key, const char* index_file)
{
    sqlite3_stmt* stmt = NULL;
    if (sqlite3_prepare_v2(db, "UPDATE main.runs SET file=?2 WHERE key=?1;", -1, &stmt, NULL) != SQLITE_OK) {
        return -1;
    }
    sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, index_file, -1, SQLITE_TRANSIENT);
    const int rc = (sqlite3_step(stmt) == SQLITE_DONE) ? 0 : -1;
    sqlite3_finalize(stmt);
    return rc;
}

static long long stg_count(sqlite3* db, const char* schema, const char* table, const char* run_id)
{
    if (!db_table_exists(db, schema, table)) {
        return 0;
    }
    char sql[256];
    (void)snprintf(sql, sizeof(sql), "SELECT COUNT(*) FROM \"%s\".\"%s\" WHERE run_id=?1;", schema, table);
    sqlite3_stmt* stmt = NULL;
    long long n = -1;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, run_id, -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            n = sqlite3_column_int64(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    return n;
}

/*
 * Merge in transactions of bounded size:
 *   1. delete any earlier run with this key (its `runs` row first) and copy the run's metadata;
 *   2. raw chunks, then record maps, each split by the batch budget;
 *   3. the final field and the `runs` row, in the last transaction.
 * An interrupted merge leaves data without a `runs` row; merging the same
 * staging file again starts over from step 1, and nlolib_storage_recover_staging
 * deletes such data when its staging file is gone.
 */
static run_store_status merge_locked(sqlite3* db, run_lock* lock, const char* key, const char* run_id,
                                     const char* index_file)
{
    merge_txn m;
    memset(&m, 0, sizeof(m));
    m.db = db;
    m.lock = lock;
    m.run_id = run_id;
    m.budget = merge_batch_bytes();

    int ok = merge_begin(&m) == 0 &&
             delete_run(db, key, run_id) == 0 &&
             copy_rows(db, "io_system_config", "INSERT OR IGNORE",
                       "system_hash IN (SELECT system_hash FROM stg.io_runs WHERE run_id=?1)", run_id, 0, 0) == 0 &&
             copy_rows(db, "io_sim_config", "INSERT OR IGNORE",
                       "config_hash IN (SELECT config_hash FROM stg.io_runs WHERE run_id=?1)", run_id, 0, 0) == 0 &&
             copy_rows(db, "io_runs", "INSERT", "run_id=?1", run_id, 0, 0) == 0 &&
             copy_rows(db, "io_record_layout", "INSERT", "run_id=?1", run_id, 0, 0) == 0;
    ok = ok &&
         merge_table_batched(&m, "io_record_chunks", "chunk_index", "length(payload)") == 0 &&
         merge_table_batched(&m, "io_record_maps", "record_index",
                             "length(power)+length(spectrum)+length(fluence_x)+length(fluence_y)") == 0;

    sqlite3_int64 final_bytes = 0;
    if (ok && db_table_exists(db, "stg", "io_final_output_fields")) {
        sqlite3_stmt* stmt = NULL;
        if (sqlite3_prepare_v2(db, "SELECT COALESCE(SUM(length(payload)),0) FROM stg.io_final_output_fields WHERE run_id=?1;",
                               -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, run_id, -1, SQLITE_TRANSIENT);
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                final_bytes = sqlite3_column_int64(stmt, 0);
            }
            sqlite3_finalize(stmt);
        }
    }
    if (ok && m.in_txn && m.txn_bytes > 0 && m.txn_bytes + final_bytes > m.budget) {
        ok = merge_commit(&m) == 0;
    }
    ok = ok &&
         merge_begin(&m) == 0 &&
         copy_rows(db, "io_final_output_fields", "INSERT", "run_id=?1", run_id, 0, 0) == 0 &&
         merge_runs_columns(db) == 0 &&
         copy_rows(db, "runs", "INSERT", "key=(SELECT key FROM stg.io_runs WHERE run_id=?1)", run_id, 0, 0) == 0 &&
         set_runs_file(db, key, index_file) == 0 &&
         merge_commit(&m) == 0;
    if (!ok) {
        if (m.in_txn) {
            (void)sqlite3_exec(db, "ROLLBACK;", NULL, NULL, NULL);
        }
        return RUN_STORE_ERROR;
    }
    if (m.commits > 1u) {
        log_emit(LOG_LEVEL_INFO, "[nlolib][store] merged run %s in %u transactions", key, m.commits);
    }
    return RUN_STORE_OK;
}

// MARK: Fresh view of the shared database

/* Fields of the 100-byte SQLite header that change on every commit. */
typedef struct {
    int exists;
    unsigned long change_counter;
    unsigned long page_count;
    unsigned long page_size;
    unsigned long version_valid_for;
    long long file_bytes;
} db_header;

static unsigned long be32(const unsigned char* p)
{
    return ((unsigned long)p[0] << 24) | ((unsigned long)p[1] << 16) | ((unsigned long)p[2] << 8) | (unsigned long)p[3];
}

/* Read the header with a fresh open of the file (not through SQLite's cache). */
static int read_db_header(const char* path, db_header* out)
{
    memset(out, 0, sizeof(*out));
#if defined(_WIN32)
    wchar_t* w = rs_wide(path);
    FILE* f = (w != NULL) ? _wfopen(w, L"rb") : NULL;
    free(w);
#else
    FILE* f = fopen(path, "rb");
#endif
    if (f == NULL) {
        return 0;
    }
    unsigned char h[100];
    const size_t n = fread(h, 1u, sizeof(h), f);
    long long size = 0;
    if (fseek(f, 0, SEEK_END) == 0) {
#if defined(_WIN32)
        size = (long long)_ftelli64(f);
#else
        size = (long long)ftello(f);
#endif
    }
    fclose(f);
    out->exists = 1;
    out->file_bytes = size;
    if (n < sizeof(h)) {
        return (n == 0u) ? 0 : -1;
    }
    if (memcmp(h, "SQLite format 3", 16) != 0) {
        return -1;
    }
    const unsigned long raw_page = ((unsigned long)h[16] << 8) | (unsigned long)h[17];
    out->page_size = (raw_page == 1u) ? 65536u : raw_page;
    out->change_counter = be32(h + 24);
    out->page_count = be32(h + 28);
    out->version_valid_for = be32(h + 92);
    return 0;
}

/* A header is self-consistent when its page count is current and matches the file size. */
static int header_consistent(const db_header* h)
{
    if (!h->exists || h->file_bytes == 0) {
        return 1;
    }
    return h->version_valid_for == h->change_counter &&
           (long long)h->page_count * (long long)h->page_size == h->file_bytes;
}

static void state_path_of(const char* shared_path, char* out, size_t cap)
{
    (void)snprintf(out, cap, "%s" RUN_STORE_STATE_SUFFIX, shared_path);
}

/* Change counter recorded by the last merge (0 when there is no state file). */
static unsigned long read_state_counter(const char* shared_path, int* out_found)
{
    char path[RUN_STORE_PATH_MAX];
    state_path_of(shared_path, path, sizeof(path));
    char content[512];
    char value[32];
    *out_found = 0;
    if (rs_read_small(path, content, sizeof(content)) < 0 || lock_field(content, "counter", value, sizeof(value)) != 0) {
        return 0u;
    }
    *out_found = 1;
    return strtoul(value, NULL, 10);
}

static void write_state(const char* shared_path, const run_lock* lock, const db_header* h, const char* key)
{
    char path[RUN_STORE_PATH_MAX];
    char tmp[RUN_STORE_PATH_MAX + 32];
    state_path_of(shared_path, path, sizeof(path));
    (void)snprintf(tmp, sizeof(tmp), "%s.tmp-%s", path, lock->token);
    char now[32];
    run_store_utc_now(now);
    char content[512];
    (void)snprintf(content, sizeof(content),
                   "nlolib-state 1\ncounter=%lu\npages=%lu\npage_size=%lu\nbytes=%lld\nhost=%s\nupdated=%s\nkey=%s\n",
                   h->change_counter, h->page_count, h->page_size, h->file_bytes, lock->host, now, key);
    if (rs_create_exclusive(tmp, content) == 0 && rs_rename_replace(tmp, path) != 0) {
        (void)rs_unlink(tmp);
    }
}

static double settle_seconds(const char* shared_path)
{
    char dir[RUN_STORE_PATH_MAX];
    rs_dirname(shared_path, dir, sizeof(dir));
    return env_seconds_allow_zero("NLOLIB_STORE_SETTLE_S", rs_is_network_dir(dir) ? 2.5 : 0.0);
}

/*
 * After taking the lock, wait until this host sees the state the previous
 * writer committed: first the settle time (longer than the mount's attribute
 * and close caches), then until the file header is self-consistent and its
 * change counter has reached the one the previous merge recorded.
 */
static run_store_status await_fresh_view(const char* shared_path)
{
    const double settle = settle_seconds(shared_path);
    if (settle > 0.0) {
        rs_sleep_ms((unsigned)(settle * 1000.0));
    }
    char journal[RUN_STORE_PATH_MAX + 16];
    (void)snprintf(journal, sizeof(journal), "%s-journal", shared_path);
    db_header jh;
    (void)read_db_header(journal, &jh);
    if (jh.exists && jh.file_bytes > 0) {
        /*
         * A writer died during a commit and left a hot journal. We hold the
         * writer lock, so no commit is in progress: let SQLite roll it back,
         * then check the restored file as usual.
         */
        log_emit(LOG_LEVEL_WARN, "[nlolib][store] rolling back the hot journal of %s", shared_path);
        sqlite3* db = NULL;
        if (db_open(shared_path, 0, &db) == 0) {
            (void)db_exec(db, "SELECT COUNT(*) FROM sqlite_master;");
            sqlite3_close(db);
        }
    }
    const double deadline = rs_monotonic_seconds() + 10.0 + 4.0 * settle;
    for (;;) {
        db_header h;
        int found = 0;
        const unsigned long recorded = read_state_counter(shared_path, &found);
        const int read_ok = read_db_header(shared_path, &h) == 0;
        const int fresh = read_ok && header_consistent(&h) &&
                          (!found || !h.exists || h.change_counter >= recorded);
        if (fresh) {
            return RUN_STORE_OK;
        }
        if (rs_monotonic_seconds() >= deadline) {
            log_emit(LOG_LEVEL_ERROR,
                     "[nlolib][store] %s still looks stale or inconsistent on this host "
                     "(header counter %lu, recorded %lu, %lld bytes for %lu pages of %lu); not writing",
                     shared_path, h.change_counter, recorded, h.file_bytes, h.page_count, h.page_size);
            return RUN_STORE_ERROR;
        }
        rs_sleep_ms(250u);
    }
}

/* Re-read the merged run with a new connection: header consistent, `runs` row and row counts present. */
static int verify_merge(const char* shared_path, const char* key, const char* run_id,
                        long long maps, long long chunks, long long finals)
{
    db_header h;
    if (read_db_header(shared_path, &h) != 0 || !header_consistent(&h)) {
        return -1;
    }
    sqlite3* db = NULL;
    if (db_open(shared_path, 0, &db) != 0) {
        return -1;
    }
    char found[STORAGE_KEY_MAX] = {0};
    int ok = db_text_query(db, "SELECT key FROM runs WHERE key=?1;", key, found, sizeof(found)) == 0 &&
             stg_count(db, "main", "io_record_maps", run_id) == maps &&
             stg_count(db, "main", "io_record_chunks", run_id) == chunks &&
             stg_count(db, "main", "io_final_output_fields", run_id) == finals &&
             stg_count(db, "main", "io_record_layout", run_id) == 1;
    sqlite3_close(db);
    return ok ? 0 : -1;
}

/* Inspect a staging database: key and run id of its complete run, if any. */
/*
 * Classify a staging file:
 *   RUN_STORE_OK         a complete run (its `runs` row is present);
 *   RUN_STORE_INCOMPLETE a valid database without a `runs` row (a run in
 *                        progress or killed), or one with a hot journal (its
 *                        writer died during a commit);
 *   RUN_STORE_ERROR      damaged, truncated or still being copied: its header
 *                        does not match its size, or (with @p deep) SQLite's
 *                        quick_check fails. Such files are never merged or
 *                        deleted automatically.
 */
static run_store_status staging_inspect(const char* staging_path, int deep,
                                        char key[STORAGE_KEY_MAX], char run_id[STORAGE_RUN_ID_MAX])
{
    char journal[RUN_STORE_PATH_MAX + 16];
    (void)snprintf(journal, sizeof(journal), "%s-journal", staging_path);
    db_header h;
    (void)read_db_header(journal, &h);
    if (h.exists && h.file_bytes > 0) {
        return RUN_STORE_INCOMPLETE;
    }
    if (read_db_header(staging_path, &h) != 0 || !h.exists) {
        return RUN_STORE_ERROR;
    }
    if (h.file_bytes == 0) {
        return RUN_STORE_INCOMPLETE;
    }
    if (!header_consistent(&h)) {
        return RUN_STORE_ERROR;
    }
    sqlite3* db = NULL;
    if (db_open(staging_path, 1, &db) != 0) {
        return RUN_STORE_ERROR;
    }
    run_store_status status = RUN_STORE_INCOMPLETE;
    if (deep) {
        char check[64] = {0};
        if (db_text_query(db, "PRAGMA quick_check;", NULL, check, sizeof(check)) != 0 || strcmp(check, "ok") != 0) {
            sqlite3_close(db);
            return RUN_STORE_ERROR;
        }
    }
    if (db_table_exists(db, "main", "runs") && db_table_exists(db, "main", "io_runs")) {
        sqlite3_stmt* stmt = NULL;
        if (sqlite3_prepare_v2(db,
                               "SELECT r.key, i.run_id FROM runs r JOIN io_runs i ON i.key = r.key LIMIT 1;",
                               -1, &stmt, NULL) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                (void)snprintf(key, STORAGE_KEY_MAX, "%s", (const char*)sqlite3_column_text(stmt, 0));
                (void)snprintf(run_id, STORAGE_RUN_ID_MAX, "%s", (const char*)sqlite3_column_text(stmt, 1));
                status = RUN_STORE_OK;
            }
            sqlite3_finalize(stmt);
        } else {
            status = RUN_STORE_ERROR;
        }
    }
    sqlite3_close(db);
    return status;
}

/* 1 when the shared database already holds a run with this key that was created after the staged one. */
static int staged_run_superseded(sqlite3* db, const char* key)
{
    sqlite3_stmt* stmt = NULL;
    int superseded = 0;
    if (sqlite3_prepare_v2(db,
                           "SELECT (SELECT created_at FROM main.runs WHERE key=?1), "
                           "(SELECT created_at FROM stg.runs WHERE key=?1);",
                           -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_type(stmt, 0) == SQLITE_TEXT &&
            sqlite3_column_type(stmt, 1) == SQLITE_TEXT) {
            superseded = strcmp((const char*)sqlite3_column_text(stmt, 0), (const char*)sqlite3_column_text(stmt, 1)) > 0;
        }
        sqlite3_finalize(stmt);
    }
    return superseded;
}

/*
 * The writer lock held over several merges of one pass. The view of the shared
 * database is settled once after the lock is taken: while the lock stays ours
 * no other host commits, so later merges of the pass need no settle wait. The
 * lock is given up and taken again after RUN_STORE_PASS_HOLD_S, so that
 * writers finishing a run during a long pass are not held up for long.
 */
#define RUN_STORE_PASS_HOLD_S 30.0

typedef struct {
    const char* shared_path;
    run_lock lock;
    double taken_at;
    int fresh;
} merge_session;

static run_store_status session_lock(merge_session* s)
{
    if (s->lock.held) {
        if (rs_monotonic_seconds() - s->taken_at < RUN_STORE_PASS_HOLD_S && lock_refresh(&s->lock) == 0) {
            return RUN_STORE_OK;
        }
        lock_release(&s->lock);
    }
    s->fresh = 0;
    const run_store_status status = lock_acquire(&s->lock, s->shared_path);
    s->taken_at = rs_monotonic_seconds();
    return status;
}

static run_store_status session_fresh(merge_session* s)
{
    if (!s->fresh) {
        s->fresh = (await_fresh_view(s->shared_path) == RUN_STORE_OK);
    }
    return s->fresh ? RUN_STORE_OK : RUN_STORE_ERROR;
}

static void session_end(merge_session* s)
{
    lock_release(&s->lock);
    s->fresh = 0;
}

/* Merge one complete staging file within a session; the lock is kept on return. */
static run_store_status session_merge(merge_session* s, const char* staging_path)
{
    const char* shared_path = s->shared_path;
    char key[STORAGE_KEY_MAX] = {0};
    char run_id[STORAGE_RUN_ID_MAX] = {0};
    const run_store_status inspect = staging_inspect(staging_path, 0, key, run_id);
    if (inspect != RUN_STORE_OK) {
        return inspect;
    }
    const run_store_status lock_status = session_lock(s);
    if (lock_status != RUN_STORE_OK) {
        return lock_status;
    }
    /* Another writer may have merged this file while we waited: staging files are only deleted under the lock. */
    if (!rs_file_exists(staging_path)) {
        int present = 0;
        if (run_store_has_key(shared_path, key, &present) == RUN_STORE_OK && present) {
            log_emit(LOG_LEVEL_INFO, "[nlolib][store] run %s was already merged into %s", key, shared_path);
            return RUN_STORE_OK;
        }
        return RUN_STORE_ERROR;
    }
    /* Full check before anything is copied: a damaged or partly copied file is left alone. */
    const run_store_status deep = staging_inspect(staging_path, 1, key, run_id);
    if (deep != RUN_STORE_OK) {
        log_emit(LOG_LEVEL_WARN,
                 "[nlolib][store] %s is %s; not merged and left in place",
                 staging_path,
                 deep == RUN_STORE_INCOMPLETE ? "incomplete" : "damaged or still being copied");
        return deep;
    }
    char index_file[RUN_STORE_PATH_MAX];
    run_store_index_file(shared_path, index_file, sizeof(index_file));
    int superseded = 0;

    run_store_status status = session_fresh(s);
    sqlite3* db = NULL;
    long long maps = -1;
    long long chunks = -1;
    long long finals = -1;
    if (status == RUN_STORE_OK) {
        status = RUN_STORE_ERROR;
        if (open_shared(shared_path, &db) == 0) {
            sqlite3_stmt* attach = NULL;
            if (sqlite3_prepare_v2(db, "ATTACH DATABASE ?1 AS stg;", -1, &attach, NULL) == SQLITE_OK) {
                sqlite3_bind_text(attach, 1, staging_path, -1, SQLITE_TRANSIENT);
                const int attached = (sqlite3_step(attach) == SQLITE_DONE);
                sqlite3_finalize(attach);
                if (attached) {
                    maps = stg_count(db, "stg", "io_record_maps", run_id);
                    chunks = stg_count(db, "stg", "io_record_chunks", run_id);
                    finals = stg_count(db, "stg", "io_final_output_fields", run_id);
                    superseded = staged_run_superseded(db, key);
                    status = superseded ? RUN_STORE_OK : merge_locked(db, &s->lock, key, run_id, index_file);
                    (void)sqlite3_exec(db, "DETACH DATABASE stg;", NULL, NULL, NULL);
                }
            }
            sqlite3_close(db);
        }
    }
    if (superseded) {
        /* The shared database holds a newer run with the same key: the newest run wins. */
        run_store_discard(staging_path);
        log_emit(LOG_LEVEL_INFO,
                 "[nlolib][store] %s holds a newer run %s; staged copy %s discarded",
                 shared_path, key, staging_path);
        return RUN_STORE_OK;
    }
    if (status == RUN_STORE_OK && verify_merge(shared_path, key, run_id, maps, chunks, finals) != 0) {
        log_emit(LOG_LEVEL_ERROR, "[nlolib][store] run %s did not read back from %s after the merge", key, shared_path);
        status = RUN_STORE_ERROR;
    }
    if (status == RUN_STORE_OK) {
        db_header h;
        if (read_db_header(shared_path, &h) == 0) {
            write_state(shared_path, &s->lock, &h, key);
        }
        run_store_discard(staging_path);
    } else {
        /* Check the view again before the next merge of the pass. */
        s->fresh = 0;
    }

    if (status == RUN_STORE_OK) {
        log_emit(LOG_LEVEL_INFO, "[nlolib][store] merged run %s into %s", key, shared_path);
    } else {
        log_emit(LOG_LEVEL_ERROR,
                 "[nlolib][store] merge of %s into %s failed; staging file kept",
                 staging_path,
                 shared_path);
    }
    return status;
}

run_store_status run_store_merge(const char* staging_path, const char* shared_path)
{
    if (staging_path == NULL || shared_path == NULL) {
        return RUN_STORE_ERROR;
    }
    merge_session session;
    memset(&session, 0, sizeof(session));
    session.shared_path = shared_path;
    const run_store_status status = session_merge(&session, staging_path);
    session_end(&session);
    return status;
}

/*
 * Under the lock, delete data of interrupted merges: io_runs rows with a key
 * but no `runs` row whose staging file no longer exists. Rows without a key
 * were written before run keys existed and are kept.
 */
static size_t discard_orphans(merge_session* s, const char* staging_dir, const char* stem)
{
    const char* shared_path = s->shared_path;
    if (!rs_file_exists(shared_path)) {
        return 0u;
    }
    if (session_lock(s) != RUN_STORE_OK) {
        return 0u;
    }
    size_t removed = 0u;
    sqlite3* db = NULL;
    if (session_fresh(s) == RUN_STORE_OK && open_shared(shared_path, &db) == 0) {
        sqlite3_stmt* stmt = NULL;
        char keys[64][STORAGE_KEY_MAX];
        char ids[64][STORAGE_RUN_ID_MAX];
        size_t n = 0u;
        if (sqlite3_prepare_v2(db,
                               "SELECT key, run_id FROM io_runs WHERE key IS NOT NULL "
                               "AND key NOT IN (SELECT key FROM runs) LIMIT 64;",
                               -1, &stmt, NULL) == SQLITE_OK) {
            while (n < 64u && sqlite3_step(stmt) == SQLITE_ROW) {
                (void)snprintf(keys[n], STORAGE_KEY_MAX, "%s", (const char*)sqlite3_column_text(stmt, 0));
                (void)snprintf(ids[n], STORAGE_RUN_ID_MAX, "%s", (const char*)sqlite3_column_text(stmt, 1));
                n += 1u;
            }
            sqlite3_finalize(stmt);
        }
        for (size_t i = 0u; i < n; ++i) {
            char pattern[RUN_STORE_PATH_MAX];
            (void)snprintf(pattern, sizeof(pattern), "%.255s.%s.", stem, keys[i]);
            if (staging_has_prefix(staging_dir, pattern)) {
                continue;
            }
            if (db_exec(db, "BEGIN IMMEDIATE;") == 0 && delete_run(db, keys[i], ids[i]) == 0 &&
                lock_owned(&s->lock) && db_exec(db, "COMMIT;") == 0) {
                removed += 1u;
                log_emit(LOG_LEVEL_INFO, "[nlolib][store] removed data of interrupted merge %s", keys[i]);
            } else {
                (void)sqlite3_exec(db, "ROLLBACK;", NULL, NULL, NULL);
            }
        }
        sqlite3_close(db);
        if (removed > 0u) {
            db_header h;
            if (read_db_header(shared_path, &h) == 0) {
                write_state(shared_path, &s->lock, &h, "-");
            }
        }
    }
    return removed;
}

typedef struct {
    const char* shared_path;
    merge_session session;
    char stem[256];
    char staging_dir[RUN_STORE_PATH_MAX];
    char host[128];
    int discard_incomplete;
    double min_age_seconds;
    size_t merged;
    size_t discarded;
    size_t skipped;
    int failed;
} recover_ctx;

/* Parse "<stem>.<key>.<pid>.<token>.<host>.sqlite" from the right. */
static int parse_staging_name(const char* name, const char* stem, unsigned long* out_pid, char* out_host, size_t host_cap)
{
    const size_t len = strlen(name);
    const size_t stem_len = strlen(stem);
    static const char suffix[] = ".sqlite";
    if (len <= stem_len + 1u + sizeof(suffix) || strcmp(name + len - (sizeof(suffix) - 1u), suffix) != 0 ||
        strncmp(name, stem, stem_len) != 0 || name[stem_len] != '.') {
        return -1;
    }
    char rest[512];
    (void)snprintf(rest, sizeof(rest), "%.*s", (int)(len - stem_len - 1u - (sizeof(suffix) - 1u)), name + stem_len + 1u);
    char* parts[4] = {0};
    char* cursor = rest;
    for (int i = 0; i < 3; ++i) {
        char* dot = strchr(cursor, '.');
        if (dot == NULL) {
            return -1;
        }
        *dot = '\0';
        parts[i] = cursor;
        cursor = dot + 1;
    }
    parts[3] = cursor;
    if (strlen(parts[0]) != 16u || strchr(parts[3], '.') != NULL) {
        return -1;
    }
    *out_pid = strtoul(parts[1], NULL, 10);
    (void)snprintf(out_host, host_cap, "%s", parts[3]);
    return 0;
}

typedef struct {
    const char* stem;
    char (*names)[512];
    size_t count;
    size_t capacity;
    int failed;
} name_list;

static int collect_visit(const char* name, void* user)
{
    name_list* list = (name_list*)user;
    unsigned long pid = 0u;
    char host[128];
    if (parse_staging_name(name, list->stem, &pid, host, sizeof(host)) != 0 || strlen(name) >= 512u) {
        return 0;
    }
    if (list->count == list->capacity) {
        const size_t cap = (list->capacity == 0u) ? 16u : 2u * list->capacity;
        void* grown = realloc(list->names, cap * sizeof(*list->names));
        if (grown == NULL) {
            list->failed = 1;
            return 1;
        }
        list->names = (char (*)[512])grown;
        list->capacity = cap;
    }
    (void)snprintf(list->names[list->count], 512u, "%s", name);
    list->count += 1u;
    return 0;
}

static void recover_one(recover_ctx* ctx, const char* name)
{
    unsigned long pid = 0u;
    char host[128];
    if (parse_staging_name(name, ctx->stem, &pid, host, sizeof(host)) != 0) {
        return;
    }
    char path[RUN_STORE_PATH_MAX];
    if (rs_join(path, sizeof(path), ctx->staging_dir, name) != 0 || !rs_file_exists(path)) {
        return;
    }
    const int same_host = (strcmp(host, ctx->host) == 0);
    char key[STORAGE_KEY_MAX];
    char run_id[STORAGE_RUN_ID_MAX];
    const run_store_status inspect = staging_inspect(path, 0, key, run_id);
    if (inspect == RUN_STORE_ERROR) {
        /* Damaged, truncated or still being copied: never merged or deleted here. */
        log_emit(LOG_LEVEL_WARN, "[nlolib][store] %s is damaged or still being copied; left in place", path);
        ctx->skipped += 1u;
        return;
    }
    /* Complete files are merged whoever wrote them; incomplete ones of live local processes are runs in progress. */
    if (inspect != RUN_STORE_OK && same_host && pid != rs_pid() && rs_pid_alive(pid) != 0) {
        ctx->skipped += 1u;
        return;
    }
    if (inspect == RUN_STORE_OK) {
        const run_store_status merged = session_merge(&ctx->session, path);
        if (merged == RUN_STORE_OK) {
            ctx->merged += 1u;
        } else if (merged == RUN_STORE_INCOMPLETE || merged == RUN_STORE_ERROR) {
            ctx->skipped += 1u;
            ctx->failed = ctx->failed || (merged == RUN_STORE_ERROR && rs_file_exists(path) &&
                                          staging_inspect(path, 0, key, run_id) == RUN_STORE_OK);
        } else {
            ctx->skipped += 1u;
            ctx->failed = 1;
        }
        return;
    }
    /* Never discard an incomplete file of this process: it may be a run in progress. */
    double age = 0.0;
    if (ctx->discard_incomplete && !(same_host && pid == rs_pid()) &&
        rs_file_age_seconds(path, &age) == 0 && age >= ctx->min_age_seconds) {
        run_store_discard(path);
        ctx->discarded += 1u;
        log_emit(LOG_LEVEL_INFO, "[nlolib][store] discarded incomplete staging file %s", path);
    } else {
        ctx->skipped += 1u;
    }
}

run_store_status run_store_recover(
    const char* shared_path,
    int discard_incomplete,
    double min_age_seconds,
    size_t* out_merged,
    size_t* out_discarded,
    size_t* out_skipped
)
{
    if (shared_path == NULL || shared_path[0] == '\0') {
        return RUN_STORE_ERROR;
    }
    recover_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.shared_path = shared_path;
    ctx.session.shared_path = shared_path;
    ctx.discard_incomplete = discard_incomplete;
    ctx.min_age_seconds = (min_age_seconds > 0.0) ? min_age_seconds : 0.0;
    rs_stem(shared_path, ctx.stem, sizeof(ctx.stem));
    rs_hostname(ctx.host, sizeof(ctx.host));
    name_list list;
    memset(&list, 0, sizeof(list));
    list.stem = ctx.stem;
    if (staging_dir_of(shared_path, ctx.staging_dir, sizeof(ctx.staging_dir)) != 0 ||
        rs_list_dir(ctx.staging_dir, collect_visit, &list) != 0 || list.failed) {
        free(list.names);
        return RUN_STORE_ERROR;
    }
    for (size_t i = 0u; i < list.count; ++i) {
        recover_one(&ctx, list.names[i]);
    }
    free(list.names);
    if (discard_incomplete) {
        ctx.discarded += discard_orphans(&ctx.session, ctx.staging_dir, ctx.stem);
    }
    session_end(&ctx.session);
    if (out_merged != NULL) {
        *out_merged = ctx.merged;
    }
    if (out_discarded != NULL) {
        *out_discarded = ctx.discarded;
    }
    if (out_skipped != NULL) {
        *out_skipped = ctx.skipped;
    }
    return ctx.failed ? RUN_STORE_ERROR : RUN_STORE_OK;
}

run_store_status run_store_has_key(const char* shared_path, const char* key, int* out_present)
{
    if (shared_path == NULL || key == NULL || out_present == NULL) {
        return RUN_STORE_ERROR;
    }
    *out_present = 0;
    if (!rs_file_exists(shared_path)) {
        return RUN_STORE_OK;
    }
    sqlite3* db = NULL;
    if (reader_open(shared_path, &db) != 0) {
        return RUN_STORE_ERROR;
    }
    /* A failed query is an error, never "not stored". */
    run_store_status status = RUN_STORE_OK;
    if (db_table_exists(db, "main", "runs")) {
        sqlite3_stmt* stmt = NULL;
        status = RUN_STORE_ERROR;
        if (sqlite3_prepare_v2(db, "SELECT 1 FROM runs WHERE key=?1;", -1, &stmt, NULL) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, key, -1, SQLITE_TRANSIENT);
            const int rc = sqlite3_step(stmt);
            if (rc == SQLITE_ROW || rc == SQLITE_DONE) {
                *out_present = (rc == SQLITE_ROW);
                status = RUN_STORE_OK;
            }
        }
        sqlite3_finalize(stmt);
    }
    sqlite3_close(db);
    return status;
}

static int read_run_id(sqlite3* db, const char* key, char run_id[STORAGE_RUN_ID_MAX])
{
    return db_text_query(db, "SELECT run_id FROM io_runs WHERE key=?1 ORDER BY rowid DESC LIMIT 1;", key,
                         run_id, STORAGE_RUN_ID_MAX);
}

run_store_status run_store_read_layout(const char* db_path, const char* key, storage_run_layout* out_layout)
{
    if (db_path == NULL || key == NULL || out_layout == NULL) {
        return RUN_STORE_ERROR;
    }
    memset(out_layout, 0, sizeof(*out_layout));
    sqlite3* db = NULL;
    if (reader_open(db_path, &db) != 0) {
        return RUN_STORE_ERROR;
    }
    run_store_status status = RUN_STORE_ERROR;
    sqlite3_stmt* stmt = NULL;
    if (read_run_id(db, key, out_layout->run_id) == 0 &&
        sqlite3_prepare_v2(db,
                           "SELECT record_mode, nt, nx, ny, num_records, raw_nt, raw_nx, raw_ny, raw_records, "
                           "dt, dx, dy, raw_stride_t, raw_stride_x, raw_stride_y, "
                           "(SELECT COUNT(*) FROM io_final_output_fields f WHERE f.run_id = l.run_id) "
                           "FROM io_record_layout l WHERE run_id=?1;",
                           -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, out_layout->run_id, -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char* mode = (const char*)sqlite3_column_text(stmt, 0);
            out_layout->record_mode =
                (mode != NULL && strcmp(mode, "raw") == 0) ? STORAGE_RECORD_MODE_RAW
                : (mode != NULL && strcmp(mode, "reduced+raw") == 0) ? STORAGE_RECORD_MODE_REDUCED_AND_RAW
                : STORAGE_RECORD_MODE_REDUCED;
            out_layout->nt = (size_t)sqlite3_column_int64(stmt, 1);
            out_layout->nx = (size_t)sqlite3_column_int64(stmt, 2);
            out_layout->ny = (size_t)sqlite3_column_int64(stmt, 3);
            out_layout->num_records = (size_t)sqlite3_column_int64(stmt, 4);
            out_layout->raw_nt = (size_t)sqlite3_column_int64(stmt, 5);
            out_layout->raw_nx = (size_t)sqlite3_column_int64(stmt, 6);
            out_layout->raw_ny = (size_t)sqlite3_column_int64(stmt, 7);
            out_layout->raw_records = (size_t)sqlite3_column_int64(stmt, 8);
            out_layout->delta_time = sqlite3_column_double(stmt, 9);
            out_layout->delta_x = sqlite3_column_double(stmt, 10);
            out_layout->delta_y = sqlite3_column_double(stmt, 11);
            out_layout->raw_stride_t = (size_t)sqlite3_column_int64(stmt, 12);
            out_layout->raw_stride_x = (size_t)sqlite3_column_int64(stmt, 13);
            out_layout->raw_stride_y = (size_t)sqlite3_column_int64(stmt, 14);
            out_layout->has_final_field = sqlite3_column_int(stmt, 15) > 0;
            status = RUN_STORE_OK;
        }
        sqlite3_finalize(stmt);
    }
    sqlite3_close(db);
    return status;
}

static void copy_out(const void* blob, size_t bytes, double* out, size_t capacity, size_t* written)
{
    const size_t count = bytes / sizeof(double);
    if (out != NULL && *written + count <= capacity) {
        memcpy(out + *written, blob, count * sizeof(double));
    }
    *written += count;
}

run_store_status run_store_read_array(
    const char* db_path,
    const char* key,
    const char* name,
    double* out,
    size_t capacity,
    size_t* out_count
)
{
    if (db_path == NULL || key == NULL || name == NULL || out_count == NULL) {
        return RUN_STORE_ERROR;
    }
    *out_count = 0u;
    const char* sql = NULL;
    if (strcmp(name, "z") == 0 || strcmp(name, "t") == 0 || strcmp(name, "omega") == 0 ||
        strcmp(name, "x") == 0 || strcmp(name, "y") == 0) {
        static const char* layout_sql[] = {
            "SELECT record_z FROM io_record_layout WHERE run_id=?1;",
            "SELECT t_axis FROM io_record_layout WHERE run_id=?1;",
            "SELECT omega_axis FROM io_record_layout WHERE run_id=?1;",
            "SELECT x_axis FROM io_record_layout WHERE run_id=?1;",
            "SELECT y_axis FROM io_record_layout WHERE run_id=?1;"
        };
        const int idx = (name[0] == 'z') ? 0 : (name[0] == 't') ? 1 : (name[0] == 'o') ? 2 : (name[0] == 'x') ? 3 : 4;
        sql = layout_sql[idx];
    } else if (strcmp(name, "power") == 0) {
        sql = "SELECT power FROM io_record_maps WHERE run_id=?1 ORDER BY record_index;";
    } else if (strcmp(name, "spectrum") == 0) {
        sql = "SELECT spectrum FROM io_record_maps WHERE run_id=?1 ORDER BY record_index;";
    } else if (strcmp(name, "fluence_x") == 0) {
        sql = "SELECT fluence_x FROM io_record_maps WHERE run_id=?1 ORDER BY record_index;";
    } else if (strcmp(name, "fluence_y") == 0) {
        sql = "SELECT fluence_y FROM io_record_maps WHERE run_id=?1 ORDER BY record_index;";
    } else if (strcmp(name, "final_field") == 0) {
        sql = "SELECT payload FROM io_final_output_fields WHERE run_id=?1;";
    } else if (strcmp(name, "raw") == 0) {
        sql = "SELECT payload FROM io_record_chunks WHERE run_id=?1 ORDER BY record_start;";
    } else {
        return RUN_STORE_ERROR;
    }

    sqlite3* db = NULL;
    if (reader_open(db_path, &db) != 0) {
        return RUN_STORE_ERROR;
    }
    run_store_status status = RUN_STORE_ERROR;
    char run_id[STORAGE_RUN_ID_MAX];
    sqlite3_stmt* stmt = NULL;
    if (read_run_id(db, key, run_id) == 0 && sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, run_id, -1, SQLITE_TRANSIENT);
        size_t written = 0u;
        int rc = SQLITE_ROW;
        while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
            const void* blob = sqlite3_column_blob(stmt, 0);
            const int bytes = sqlite3_column_bytes(stmt, 0);
            if (blob != NULL && bytes > 0) {
                copy_out(blob, (size_t)bytes, out, capacity, &written);
            }
        }
        sqlite3_finalize(stmt);
        if (rc == SQLITE_DONE) {
            *out_count = written;
            status = RUN_STORE_OK;
        }
    }
    sqlite3_close(db);
    return status;
}

#else

run_store_status run_store_merge(const char* staging_path, const char* shared_path)
{
    (void)staging_path;
    (void)shared_path;
    return RUN_STORE_ERROR;
}

run_store_status run_store_recover(const char* shared_path, int discard_incomplete, double min_age_seconds,
                                   size_t* out_merged, size_t* out_discarded, size_t* out_skipped)
{
    (void)shared_path;
    (void)discard_incomplete;
    (void)min_age_seconds;
    (void)out_merged;
    (void)out_discarded;
    (void)out_skipped;
    return RUN_STORE_ERROR;
}

run_store_status run_store_has_key(const char* shared_path, const char* key, int* out_present)
{
    (void)shared_path;
    (void)key;
    if (out_present != NULL) {
        *out_present = 0;
    }
    return RUN_STORE_ERROR;
}

run_store_status run_store_read_layout(const char* db_path, const char* key, storage_run_layout* out_layout)
{
    (void)db_path;
    (void)key;
    (void)out_layout;
    return RUN_STORE_ERROR;
}

run_store_status run_store_read_array(const char* db_path, const char* key, const char* name,
                                      double* out, size_t capacity, size_t* out_count)
{
    (void)db_path;
    (void)key;
    (void)name;
    (void)out;
    (void)capacity;
    (void)out_count;
    return RUN_STORE_ERROR;
}

#endif
