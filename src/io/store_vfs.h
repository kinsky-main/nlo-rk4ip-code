/**
 * @file store_vfs.h
 * @brief SQLite VFS for run-store writers that retries transient I/O failures.
 */
#pragma once

/**
 * @brief Name of the SQLite VFS that staging and shared run databases are written with.
 *
 * On POSIX systems this is run_store_sqlite_vfs() wrapped by a VFS that
 * retries the transient failures a network share reports under load: an open
 * of a file being created that fails with ENOENT or ESTALE (the previous file
 * of that name still delete-pending on the server), and an open, write, sync,
 * truncate or delete that fails with EAGAIN, EINTR, EBUSY or ETIMEDOUT. Each
 * call is retried with a backoff from 20 ms, doubling to 1 s, until it
 * succeeds or NLOLIB_STORE_IO_RETRY_S seconds (default 30; 0 disables
 * retries) have been spent on it; only then does SQLite see the failure.
 * Other errors (EIO, ENOSPC, EFBIG, EACCES, ...) are returned at once. Each
 * retried call that succeeds is logged as a warning, and each one that gives
 * up is logged as an error, both with the file and errno.
 *
 * The wrapper is registered with SQLite on first use, once per wrapped VFS.
 * On Windows, or when the wrapped VFS does not exist, the wrapped name is
 * returned unchanged (NULL meaning SQLite's default VFS).
 *
 * Tests can inject failures with NLOLIB_STORE_TEST_IO_FAULTS, a comma-separated
 * list of `op:count:errno` (op one of open, write, sync, truncate, delete): the
 * next @c count calls of that kind fail with @c errno before reaching the file.
 *
 * @return VFS name for sqlite3_open_v2(); NULL for SQLite's default VFS.
 */
const char* store_vfs_writer_name(void);
