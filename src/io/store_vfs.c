/**
 * @file store_vfs.c
 * @brief SQLite VFS for run-store writers that retries transient I/O failures.
 *
 * A thin wrapper around the writer VFS (run_store_sqlite_vfs()): every call is
 * passed through, and the calls that a loaded SMB share fails transiently
 * (open, write, sync, truncate, delete) are retried with a bounded backoff
 * before SQLite sees the error, so one hiccup of the share does not fail a run.
 */
#if !defined(_WIN32)
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE 1
#endif
#endif

#include "io/store_vfs.h"
#include "io/run_store.h"

#if defined(HAVE_SQLITE3) && !defined(_WIN32)

#include "io/log_sink.h"
#include <errno.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    sqlite3_file base;
    sqlite3_file* real;
    const char* name;
} rv_file;

typedef struct {
    sqlite3_vfs vfs;
    char name[96];
    char wrapped[64];
} rv_entry;

#define RV_MAX_ENTRIES 8
static rv_entry g_entries[RV_MAX_ENTRIES];
static int g_entry_count = 0;

enum { RV_OPEN = 0, RV_WRITE, RV_SYNC, RV_TRUNCATE, RV_DELETE, RV_OPS };
static const char* const g_op_names[RV_OPS] = {"open", "write", "sync", "truncate", "delete"};

// MARK: Test fault injection

static char g_fault_spec[256];
static long g_fault_count[RV_OPS];
static int g_fault_errno[RV_OPS];

static void parse_faults(const char* spec)
{
    memset(g_fault_count, 0, sizeof(g_fault_count));
    memset(g_fault_errno, 0, sizeof(g_fault_errno));
    (void)snprintf(g_fault_spec, sizeof(g_fault_spec), "%s", spec);
    char buf[256];
    (void)snprintf(buf, sizeof(buf), "%s", spec);
    char* save = NULL;
    for (char* item = strtok_r(buf, ",", &save); item != NULL; item = strtok_r(NULL, ",", &save)) {
        char op[16] = {0};
        long count = 0;
        int err = 0;
        if (sscanf(item, "%15[a-z]:%ld:%d", op, &count, &err) != 3) {
            continue;
        }
        for (int i = 0; i < RV_OPS; ++i) {
            if (strcmp(op, g_op_names[i]) == 0) {
                g_fault_count[i] = count;
                g_fault_errno[i] = err;
            }
        }
    }
}

/* errno of an injected failure for the next call of @p op, or 0. */
static int take_fault(int op)
{
    const char* spec = getenv("NLOLIB_STORE_TEST_IO_FAULTS");
    if ((spec == NULL || spec[0] == '\0') && g_fault_spec[0] == '\0') {
        return 0;
    }
    sqlite3_mutex* mutex = sqlite3_mutex_alloc(SQLITE_MUTEX_STATIC_APP1);
    sqlite3_mutex_enter(mutex);
    if (spec == NULL || spec[0] == '\0') {
        parse_faults("");
    } else if (strcmp(spec, g_fault_spec) != 0) {
        parse_faults(spec);
    }
    int err = 0;
    if (g_fault_count[op] > 0) {
        g_fault_count[op] -= 1;
        err = g_fault_errno[op];
    }
    sqlite3_mutex_leave(mutex);
    return err;
}

/* Milliseconds NLOLIB_STORE_TEST_IO_DELAY_MS ("op:ms,...") adds before each call of @p op. */
static long take_delay_ms(int op)
{
    const char* spec = getenv("NLOLIB_STORE_TEST_IO_DELAY_MS");
    if (spec == NULL || spec[0] == '\0') {
        return 0;
    }
    const size_t len = strlen(g_op_names[op]);
    for (const char* item = spec; item != NULL && *item != '\0';) {
        if (strncmp(item, g_op_names[op], len) == 0 && item[len] == ':') {
            return strtol(item + len + 1u, NULL, 10);
        }
        item = strchr(item, ',');
        item = (item != NULL) ? item + 1 : NULL;
    }
    return 0;
}

static int fault_result(int op)
{
    switch (op) {
        case RV_OPEN:
            return SQLITE_CANTOPEN;
        case RV_WRITE:
            return SQLITE_IOERR_WRITE;
        case RV_SYNC:
            return SQLITE_IOERR_FSYNC;
        case RV_TRUNCATE:
            return SQLITE_IOERR_TRUNCATE;
        default:
            return SQLITE_IOERR_DELETE;
    }
}

// MARK: Retry loop

static int is_transient(int op, int err, int creating)
{
    if (err == EAGAIN || err == EINTR || err == EBUSY || err == ETIMEDOUT) {
        return 1;
    }
#if defined(EWOULDBLOCK) && EWOULDBLOCK != EAGAIN
    if (err == EWOULDBLOCK) {
        return 1;
    }
#endif
    /* On SMB a file deleted a moment ago can stay delete-pending, and creating one of that name fails. */
    return op == RV_OPEN && creating && (err == ENOENT || err == ESTALE);
}

static double retry_budget_seconds(void)
{
    const char* text = getenv("NLOLIB_STORE_IO_RETRY_S");
    if (text == NULL || text[0] == '\0') {
        return 30.0;
    }
    char* end = NULL;
    const double value = strtod(text, &end);
    return (end != text && value >= 0.0) ? value : 30.0;
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void sleep_seconds(double seconds)
{
    struct timespec ts;
    ts.tv_sec = (time_t)seconds;
    ts.tv_nsec = (long)((seconds - (double)ts.tv_sec) * 1e9);
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {
    }
}

/*
 * Run @p call until it returns SQLITE_OK, a non-transient failure, or the
 * retry budget is spent. errno holds the last failure's errno on return, so
 * that SQLite's xGetLastError (sqlite3_system_errno) reports it.
 */
static int run_with_retry(int op, const char* path, int creating, int (*call)(void*), void* ctx)
{
    double started = -1.0;
    double budget = 0.0;
    double delay = 0.02;
    int attempts = 0;
    int first_err = 0;
    const long delay_ms = take_delay_ms(op);
    if (delay_ms > 0) {
        sleep_seconds(1e-3 * (double)delay_ms);
    }
    for (;;) {
        int err = take_fault(op);
        int rc = 0;
        if (err != 0) {
            rc = fault_result(op);
        } else {
            errno = 0;
            rc = call(ctx);
            err = errno;
        }
        if (rc == SQLITE_OK) {
            if (attempts > 0) {
                log_emit(LOG_LEVEL_WARN,
                         "[nlolib][store] %s of %s succeeded after %d attempts (first failure: errno %d, %s)",
                         g_op_names[op], path, attempts + 1, first_err, strerror(first_err));
            }
            return rc;
        }
        attempts += 1;
        if (first_err == 0) {
            first_err = err;
        }
        if (!is_transient(op, err, creating)) {
            errno = err;
            return rc;
        }
        if (started < 0.0) {
            started = now_seconds();
            budget = retry_budget_seconds();
        }
        const double elapsed = now_seconds() - started;
        if (elapsed >= budget) {
            log_emit(LOG_LEVEL_ERROR,
                     "[nlolib][store] %s of %s still failing after %d attempts in %.1f s (errno %d, %s); giving up",
                     g_op_names[op], path, attempts, elapsed, err, strerror(err));
            errno = err;
            return rc;
        }
        sleep_seconds((delay < budget - elapsed) ? delay : budget - elapsed);
        delay = (delay * 2.0 < 1.0) ? delay * 2.0 : 1.0;
    }
}

// MARK: File methods

#define REAL(f) (((rv_file*)(f))->real)
#define NAME(f) ((((rv_file*)(f))->name != NULL) ? ((rv_file*)(f))->name : "(temporary file)")

typedef struct {
    sqlite3_file* real;
    const void* buf;
    int amount;
    sqlite3_int64 offset;
    int flags;
} rv_io_args;

static int call_write(void* v)
{
    rv_io_args* a = (rv_io_args*)v;
    return a->real->pMethods->xWrite(a->real, a->buf, a->amount, a->offset);
}

static int call_truncate(void* v)
{
    rv_io_args* a = (rv_io_args*)v;
    return a->real->pMethods->xTruncate(a->real, a->offset);
}

static int call_sync(void* v)
{
    rv_io_args* a = (rv_io_args*)v;
    return a->real->pMethods->xSync(a->real, a->flags);
}

static int rv_close(sqlite3_file* f)
{
    sqlite3_file* real = REAL(f);
    int rc = SQLITE_OK;
    if (real->pMethods != NULL) {
        rc = real->pMethods->xClose(real);
        real->pMethods = NULL;
    }
    return rc;
}

static int rv_read(sqlite3_file* f, void* buf, int amount, sqlite3_int64 offset)
{
    return REAL(f)->pMethods->xRead(REAL(f), buf, amount, offset);
}

static int rv_write(sqlite3_file* f, const void* buf, int amount, sqlite3_int64 offset)
{
    rv_io_args a = {REAL(f), buf, amount, offset, 0};
    return run_with_retry(RV_WRITE, NAME(f), 0, call_write, &a);
}

static int rv_truncate(sqlite3_file* f, sqlite3_int64 size)
{
    rv_io_args a = {REAL(f), NULL, 0, size, 0};
    return run_with_retry(RV_TRUNCATE, NAME(f), 0, call_truncate, &a);
}

static int rv_sync(sqlite3_file* f, int flags)
{
    rv_io_args a = {REAL(f), NULL, 0, 0, flags};
    return run_with_retry(RV_SYNC, NAME(f), 0, call_sync, &a);
}

static int rv_file_size(sqlite3_file* f, sqlite3_int64* size)
{
    return REAL(f)->pMethods->xFileSize(REAL(f), size);
}

static int rv_lock(sqlite3_file* f, int level)
{
    return REAL(f)->pMethods->xLock(REAL(f), level);
}

static int rv_unlock(sqlite3_file* f, int level)
{
    return REAL(f)->pMethods->xUnlock(REAL(f), level);
}

static int rv_check_reserved(sqlite3_file* f, int* out)
{
    return REAL(f)->pMethods->xCheckReservedLock(REAL(f), out);
}

static int rv_file_control(sqlite3_file* f, int op, void* arg)
{
    return REAL(f)->pMethods->xFileControl(REAL(f), op, arg);
}

static int rv_sector_size(sqlite3_file* f)
{
    return REAL(f)->pMethods->xSectorSize(REAL(f));
}

static int rv_device_characteristics(sqlite3_file* f)
{
    return REAL(f)->pMethods->xDeviceCharacteristics(REAL(f));
}

static int rv_shm_map(sqlite3_file* f, int region, int size, int extend, void volatile** out)
{
    sqlite3_file* real = REAL(f);
    return (real->pMethods->iVersion >= 2) ? real->pMethods->xShmMap(real, region, size, extend, out)
                                           : SQLITE_IOERR_SHMMAP;
}

static int rv_shm_lock(sqlite3_file* f, int offset, int n, int flags)
{
    sqlite3_file* real = REAL(f);
    return (real->pMethods->iVersion >= 2) ? real->pMethods->xShmLock(real, offset, n, flags) : SQLITE_IOERR_SHMLOCK;
}

static void rv_shm_barrier(sqlite3_file* f)
{
    sqlite3_file* real = REAL(f);
    if (real->pMethods->iVersion >= 2) {
        real->pMethods->xShmBarrier(real);
    }
}

static int rv_shm_unmap(sqlite3_file* f, int delete_flag)
{
    sqlite3_file* real = REAL(f);
    return (real->pMethods->iVersion >= 2) ? real->pMethods->xShmUnmap(real, delete_flag) : SQLITE_OK;
}

static int rv_fetch(sqlite3_file* f, sqlite3_int64 offset, int amount, void** out)
{
    sqlite3_file* real = REAL(f);
    if (real->pMethods->iVersion >= 3) {
        return real->pMethods->xFetch(real, offset, amount, out);
    }
    *out = NULL;
    return SQLITE_OK;
}

static int rv_unfetch(sqlite3_file* f, sqlite3_int64 offset, void* page)
{
    sqlite3_file* real = REAL(f);
    return (real->pMethods->iVersion >= 3) ? real->pMethods->xUnfetch(real, offset, page) : SQLITE_OK;
}

static const sqlite3_io_methods g_io_methods = {
    3,
    rv_close,
    rv_read,
    rv_write,
    rv_truncate,
    rv_sync,
    rv_file_size,
    rv_lock,
    rv_unlock,
    rv_check_reserved,
    rv_file_control,
    rv_sector_size,
    rv_device_characteristics,
    rv_shm_map,
    rv_shm_lock,
    rv_shm_barrier,
    rv_shm_unmap,
    rv_fetch,
    rv_unfetch,
};

// MARK: VFS methods

#define WRAPPED(vfs) ((sqlite3_vfs*)(vfs)->pAppData)

typedef struct {
    sqlite3_vfs* vfs;
    const char* name;
    sqlite3_file* real;
    int flags;
    int* out_flags;
    int dir_sync;
} rv_vfs_args;

static int call_open(void* v)
{
    rv_vfs_args* a = (rv_vfs_args*)v;
    if (a->real->pMethods != NULL) {
        /* A failed open may leave methods set; close before trying again. */
        a->real->pMethods->xClose(a->real);
        a->real->pMethods = NULL;
    }
    return a->vfs->xOpen(a->vfs, a->name, a->real, a->flags, a->out_flags);
}

static int call_delete(void* v)
{
    rv_vfs_args* a = (rv_vfs_args*)v;
    return a->vfs->xDelete(a->vfs, a->name, a->dir_sync);
}

static int rv_open(sqlite3_vfs* vfs, const char* name, sqlite3_file* f, int flags, int* out_flags)
{
    rv_file* p = (rv_file*)f;
    memset(p, 0, (size_t)vfs->szOsFile);
    p->real = (sqlite3_file*)&p[1];
    p->name = name;
    rv_vfs_args a = {WRAPPED(vfs), name, p->real, flags, out_flags, 0};
    const int rc = run_with_retry(RV_OPEN, (name != NULL) ? name : "(temporary file)",
                                  (flags & SQLITE_OPEN_CREATE) != 0, call_open, &a);
    if (rc == SQLITE_OK) {
        p->base.pMethods = &g_io_methods;
    } else if (p->real->pMethods != NULL) {
        const int saved = errno;
        p->real->pMethods->xClose(p->real);
        p->real->pMethods = NULL;
        errno = saved;
    }
    return rc;
}

static int rv_delete(sqlite3_vfs* vfs, const char* name, int dir_sync)
{
    rv_vfs_args a = {WRAPPED(vfs), name, NULL, 0, NULL, dir_sync};
    return run_with_retry(RV_DELETE, name, 0, call_delete, &a);
}

static int rv_access(sqlite3_vfs* vfs, const char* name, int flags, int* out)
{
    return WRAPPED(vfs)->xAccess(WRAPPED(vfs), name, flags, out);
}

static int rv_full_pathname(sqlite3_vfs* vfs, const char* name, int n, char* out)
{
    return WRAPPED(vfs)->xFullPathname(WRAPPED(vfs), name, n, out);
}

static void* rv_dl_open(sqlite3_vfs* vfs, const char* name)
{
    return WRAPPED(vfs)->xDlOpen(WRAPPED(vfs), name);
}

static void rv_dl_error(sqlite3_vfs* vfs, int n, char* out)
{
    WRAPPED(vfs)->xDlError(WRAPPED(vfs), n, out);
}

static void (*rv_dl_sym(sqlite3_vfs* vfs, void* handle, const char* symbol))(void)
{
    return WRAPPED(vfs)->xDlSym(WRAPPED(vfs), handle, symbol);
}

static void rv_dl_close(sqlite3_vfs* vfs, void* handle)
{
    WRAPPED(vfs)->xDlClose(WRAPPED(vfs), handle);
}

static int rv_randomness(sqlite3_vfs* vfs, int n, char* out)
{
    return WRAPPED(vfs)->xRandomness(WRAPPED(vfs), n, out);
}

static int rv_sleep(sqlite3_vfs* vfs, int microseconds)
{
    return WRAPPED(vfs)->xSleep(WRAPPED(vfs), microseconds);
}

static int rv_current_time(sqlite3_vfs* vfs, double* out)
{
    return WRAPPED(vfs)->xCurrentTime(WRAPPED(vfs), out);
}

static int rv_get_last_error(sqlite3_vfs* vfs, int n, char* out)
{
    return WRAPPED(vfs)->xGetLastError(WRAPPED(vfs), n, out);
}

static int rv_current_time_int64(sqlite3_vfs* vfs, sqlite3_int64* out)
{
    return WRAPPED(vfs)->xCurrentTimeInt64(WRAPPED(vfs), out);
}

static int rv_set_system_call(sqlite3_vfs* vfs, const char* name, sqlite3_syscall_ptr ptr)
{
    return WRAPPED(vfs)->xSetSystemCall(WRAPPED(vfs), name, ptr);
}

static sqlite3_syscall_ptr rv_get_system_call(sqlite3_vfs* vfs, const char* name)
{
    return WRAPPED(vfs)->xGetSystemCall(WRAPPED(vfs), name);
}

static const char* rv_next_system_call(sqlite3_vfs* vfs, const char* name)
{
    return WRAPPED(vfs)->xNextSystemCall(WRAPPED(vfs), name);
}

const char* store_vfs_writer_name(void)
{
    const char* wrapped = run_store_sqlite_vfs();
    sqlite3_vfs* real = sqlite3_vfs_find(wrapped);
    const char* key = (wrapped != NULL) ? wrapped : "";
    if (real == NULL || strlen(key) >= sizeof(g_entries[0].wrapped) ||
        strncmp(key, "nlolib-retry-", 13u) == 0) {
        return wrapped;
    }
    const char* result = NULL;
    sqlite3_mutex* mutex = sqlite3_mutex_alloc(SQLITE_MUTEX_STATIC_APP1);
    sqlite3_mutex_enter(mutex);
    for (int i = 0; i < g_entry_count && result == NULL; ++i) {
        if (strcmp(g_entries[i].wrapped, key) == 0) {
            result = g_entries[i].name;
        }
    }
    if (result == NULL && g_entry_count < RV_MAX_ENTRIES) {
        rv_entry* e = &g_entries[g_entry_count];
        memset(e, 0, sizeof(*e));
        (void)snprintf(e->wrapped, sizeof(e->wrapped), "%s", key);
        (void)snprintf(e->name, sizeof(e->name), "nlolib-retry-%s", (key[0] != '\0') ? key : "default");
        e->vfs.iVersion = (real->iVersion < 3) ? real->iVersion : 3;
        e->vfs.szOsFile = (int)sizeof(rv_file) + real->szOsFile;
        e->vfs.mxPathname = real->mxPathname;
        e->vfs.zName = e->name;
        e->vfs.pAppData = real;
        e->vfs.xOpen = rv_open;
        e->vfs.xDelete = rv_delete;
        e->vfs.xAccess = rv_access;
        e->vfs.xFullPathname = rv_full_pathname;
        e->vfs.xDlOpen = rv_dl_open;
        e->vfs.xDlError = rv_dl_error;
        e->vfs.xDlSym = rv_dl_sym;
        e->vfs.xDlClose = rv_dl_close;
        e->vfs.xRandomness = rv_randomness;
        e->vfs.xSleep = rv_sleep;
        e->vfs.xCurrentTime = rv_current_time;
        e->vfs.xGetLastError = rv_get_last_error;
        if (e->vfs.iVersion >= 2) {
            e->vfs.xCurrentTimeInt64 = rv_current_time_int64;
        }
        if (e->vfs.iVersion >= 3) {
            e->vfs.xSetSystemCall = rv_set_system_call;
            e->vfs.xGetSystemCall = rv_get_system_call;
            e->vfs.xNextSystemCall = rv_next_system_call;
        }
        if (sqlite3_vfs_register(&e->vfs, 0) == SQLITE_OK) {
            g_entry_count += 1;
            result = e->name;
        }
    }
    sqlite3_mutex_leave(mutex);
    return (result != NULL) ? result : wrapped;
}

#else

const char* store_vfs_writer_name(void)
{
#if defined(HAVE_SQLITE3)
    return run_store_sqlite_vfs();
#else
    return NULL;
#endif
}

#endif
