# Nlolib run database

This document specifies the SQLite file that Nlolib writes when a run is stored,
for programs that read it with plain SQL and no Nlolib code (for example a web
run browser). It also states how Nlolib writes the file, so that a reader knows
what it can rely on.

The file is a *run database* in the sense of the ModelStore index contract
(version 1): it holds a table `runs` with one row per complete run, which
ModelStore copies into its index, plus Nlolib's own data tables, described here.

Everything below refers to layout version 1 of these tables (Nlolib 1.19.0 and
later).

## Contents

- [Files and folders](#files-and-folders)
- [Reading a run](#reading-a-run)
- [Blob encoding](#blob-encoding)
- [Grids and axes](#grids-and-axes)
- [Table `runs`](#table-runs)
- [Table `io_record_layout`](#table-io_record_layout)
- [Table `io_record_maps`](#table-io_record_maps)
- [Table `io_final_output_fields`](#table-io_final_output_fields)
- [Table `io_record_chunks`](#table-io_record_chunks)
- [Other tables](#other-tables)
- [Example queries](#example-queries)
- [How Nlolib writes the file](#how-nlolib-writes-the-file)
- [The run key](#the-run-key)
- [API](#api)

## Files and folders

| Path | What it is |
|---|---|
| `<store>/nlolib/nlolib.sqlite` | The shared run database: every stored Nlolib run. This is the `sqlite_path` given to Nlolib. |
| `<store>/nlolib/staging/` | Per-run staging databases of runs in progress (see [writing](#how-nlolib-writes-the-file)). Never read these as run databases. |
| `<store>/nlolib/nlolib.sqlite.lock` | Lock file held by the writer that is merging a run. |
| `<store>/nlolib/nlolib.sqlite.lock.break` | Short-lived file held while a stale lock is removed. |
| `<store>/nlolib/nlolib.sqlite-journal` | SQLite's rollback journal, present only during a write transaction. |

Any `sqlite_path` works the same way: the staging folder is `staging/` next to
it and the lock is `<sqlite_path>.lock`.

The database uses SQLite's rollback journal (`journal_mode=DELETE`), never WAL,
so it can be read on a network share. Open it read-only (`mode=ro`), set a busy
timeout (Nlolib uses 60 s), read in one transaction and close it.

## Reading a run

1. Start from `runs`. A row there promises that the run is complete; data rows
   without a `runs` row belong to a merge that has not finished and must be
   ignored.
2. `runs.nlolib_run_id` (equivalently `io_runs.run_id` where `io_runs.key =
   runs.key`) is the `run_id` that every data table below is keyed by. By
   default it equals `runs.key`.
3. `io_record_layout` gives the grid, record counts and axes of the run.
4. `io_record_maps` holds the reduced maps, one row per record.
5. `io_final_output_fields` holds the final field, `io_record_chunks` the raw
   records when they were stored.

## Blob encoding

Every array is a BLOB of packed IEEE-754 binary64 values in **little-endian**
byte order, with no header or padding. The element count is `length(blob) / 8`.

- A real array of `n` elements is `8 n` bytes.
- A complex array of `n` elements is `16 n` bytes: `n` pairs `(re, im)`, real
  part first.

Multi-dimensional arrays are flattened with the **time index fastest**, then y,
then x: the sample at time index `t`, x index `x`, y index `y` of an
`nt x nx x ny` field is element

```
idx = ((x * ny) + y) * nt + t
```

In NumPy: `np.frombuffer(blob, '<c16').reshape(nx, ny, nt)`; in Julia:
`reshape(reinterpret(ComplexF64, blob), nt, ny, nx)`.

## Grids and axes

A run has `nt` time samples, `nx` x samples and `ny` y samples (`nx = ny = 1` for
a time-only run). Spacings `dt`, `dx`, `dy` are those of the run.

| Axis | Values, index `i = 0 … n-1` | Blob in `io_record_layout` |
|---|---|---|
| time `t` | `t_i = (i - (nt - 1)/2) dt` (centered) | `t_axis`, `nt` values |
| x | `x_i = (i - (nx - 1)/2) dx` | `x_axis`, `nx` values |
| y | `y_i = (i - (ny - 1)/2) dy` | `y_axis`, `ny` values |
| angular frequency `ω` | `ω_i = (i - floor(nt/2)) · 2π / (nt dt)`, ascending | `omega_axis`, `nt` values |
| propagation `z` | position of record `r` | `record_z`, `num_records` values |

`ω = 0` is at index `floor(nt/2)`. These are the axes the solver itself uses for
runtime expressions (`t`, `x`, `y`, `wt`/`w`), with `ω` reordered ascending.

**Units.** Nlolib does not fix units: `t`, `x`, `y`, `z` are in whatever units the
caller used for `dt`, `dx`, `dy` and the propagation distance, `ω` is in radians
per time unit, and `|A|²` is in the units of the squared field (for example W
when `A` is in √W). The maps below are plain sums over grid points with no
`dt`, `dx`, `dy` weights; multiply by the spacings to get integrals (see each
map).

## Table `runs`

The table of the ModelStore index contract. The fixed columns are exactly its
minimal DDL:

```sql
CREATE TABLE runs (
    key TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    file TEXT NOT NULL,
    created_at TEXT,
    git_commit TEXT,
    julia_version TEXT
);
```

| Column | Value written by Nlolib |
|---|---|
| `key` | 16 lowercase hex digits, see [the run key](#the-run-key). |
| `kind` | `nlolib` |
| `file` | `<parent folder>/<file name>` of the shared database, e.g. `nlolib/nlolib.sqlite` (ModelStore rewrites it on sync). |
| `created_at` | UTC time the run finished, `YYYY-MM-DDTHH:MM:SS.sss`. |
| `git_commit` | Commit of the Nlolib build (40 hex digits, `-dirty` appended if the build tree had uncommitted changes to tracked files), or `nlolib-<version>` for a build without a repository. |
| `julia_version` | Always NULL. The binding is in `nlolib_binding`. |

Parameter columns, all prefixed `nlolib_` so they cannot collide with other
kinds' parameters in the shared index. Each has one type; a NULL means "not
applicable to this run".

| Column | Type | Meaning |
|---|---|---|
| `nlolib_version` | TEXT | Nlolib release, e.g. `1.19.0`. |
| `nlolib_binding` | TEXT | What made the run: `c` (C API), `julia`, `python`, `matlab`, or the caller's own tag. |
| `nlolib_backend` | TEXT | Backend that ran it: `cpu` or `vulkan`. Not part of the key. |
| `nlolib_run_id` | TEXT | `run_id` of the run in the data tables. |
| `nlolib_nt` | INTEGER | Time samples. |
| `nlolib_nx` | INTEGER | x samples (1 for time-only runs). |
| `nlolib_ny` | INTEGER | y samples (1 for time-only runs). |
| `nlolib_dt` | REAL | Time step. |
| `nlolib_dx` | REAL | x step; NULL when `nx = 1`. |
| `nlolib_dy` | REAL | y step; NULL when `ny = 1`. |
| `nlolib_z_end` | REAL | Propagation distance. |
| `nlolib_h_start` | REAL | Initial step size. |
| `nlolib_h_min` | REAL | Minimum step size. |
| `nlolib_h_max` | REAL | Maximum step size. |
| `nlolib_tol` | REAL | Adaptive error tolerance. |
| `nlolib_linear_factor_expr` | TEXT | Linear operator factor expression, or NULL. |
| `nlolib_linear_expr` | TEXT | Linear operator expression, or NULL. |
| `nlolib_potential_expr` | TEXT | Potential expression, or NULL. |
| `nlolib_dispersion_factor_expr` | TEXT | Dispersion factor expression, or NULL. |
| `nlolib_dispersion_expr` | TEXT | Dispersion operator expression, or NULL. |
| `nlolib_nonlinear_expr` | TEXT | Nonlinear expression, or NULL. |
| `nlolib_constants` | TEXT | JSON array of the runtime constants `c0, c1, …` (decimal, `.` separator). |
| `nlolib_nonlinear_model` | TEXT | `expr` or `kerr_raman`. |
| `nlolib_gamma` | REAL | Kerr coefficient of the `kerr_raman` model; NULL for `expr`. |
| `nlolib_raman_fraction` | REAL | Raman fraction `f_R`; NULL for `expr`. |
| `nlolib_raman_tau1` | REAL | Raman `τ1`; NULL for `expr`. |
| `nlolib_raman_tau2` | REAL | Raman `τ2`; NULL for `expr`. |
| `nlolib_shock_omega0` | REAL | Self-steepening `ω0` (0: off); NULL for `expr`. |
| `nlolib_input_sha256` | TEXT | SHA-256 of the input field (blob encoding above), 64 hex digits. |
| `nlolib_potential_sha256` | TEXT | SHA-256 of the potential grid, or NULL when none was given. |
| `nlolib_num_records` | INTEGER | Records requested. |
| `nlolib_records_written` | INTEGER | Records stored (`io_record_layout.num_records`). |
| `nlolib_record_mode` | TEXT | `reduced`, `raw` or `reduced+raw`. |
| `nlolib_raw_stride_t` | INTEGER | Raw time stride; NULL in `reduced` mode. |
| `nlolib_raw_stride_x` | INTEGER | Raw x stride; NULL in `reduced` mode. |
| `nlolib_raw_stride_y` | INTEGER | Raw y stride; NULL in `reduced` mode. |
| `nlolib_final_field` | INTEGER | 1 when the final field is stored, else 0. |
| `nlolib_truncated` | INTEGER | 1 when raw records stopped at the size cap, else 0. |

## Table `io_record_layout`

One row per run.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | TEXT PRIMARY KEY | Run id. |
| `record_mode` | TEXT | `reduced`, `raw` or `reduced+raw`. |
| `nt`, `nx`, `ny` | INTEGER | Grid of the run. |
| `dt`, `dx`, `dy` | REAL | Spacings (1.0 when not applicable). |
| `num_records` | INTEGER | Records stored: rows in `io_record_maps` (when maps are stored) and entries of `record_z`. |
| `raw_stride_t`, `raw_stride_x`, `raw_stride_y` | INTEGER | Raw downsampling strides (1: every sample). |
| `raw_nt`, `raw_nx`, `raw_ny` | INTEGER | Raw record grid: `ceil(n / stride)` each; 0 when raw records are not stored. |
| `raw_records` | INTEGER | Raw records stored (less than `num_records` only when truncated). |
| `record_z` | BLOB | `num_records` reals: z of each record. |
| `t_axis` | BLOB | `nt` reals. |
| `omega_axis` | BLOB | `nt` reals, ascending. |
| `x_axis` | BLOB | `nx` reals. |
| `y_axis` | BLOB | `ny` reals. |

## Table `io_record_maps`

Reduced maps, one row per record, present in modes `reduced` (the default) and
`reduced+raw`. `A_r(t, x, y)` is the field of record `r`.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | TEXT | Run id. |
| `record_index` | INTEGER | Record `r`, from 0. Primary key with `run_id`. |
| `z` | REAL | z of the record. |
| `power` | BLOB | `nt` reals: `P_r[t] = Σ_x Σ_y |A_r(t, x, y)|²`. |
| `spectrum` | BLOB | `nt` reals on `omega_axis`: `S_r[j] = Σ_x Σ_y |Ã_r(k_j, x, y)|²`. |
| `fluence_x` | BLOB | `nx` reals: `F_r[x] = Σ_y Σ_t |A_r(t, x, y)|²`. |
| `fluence_y` | BLOB | `ny` reals: `G_r[y] = Σ_x Σ_t |A_r(t, x, y)|²`. |

The spectrum uses Nlolib's forward transform along time, unnormalised,

```
Ã(k) = Σ_{n=0}^{nt-1} A(t_n) · exp(-2πi k n / nt)
```

with DFT bin `k_j = (j + nt - floor(nt/2)) mod nt` stored at index `j`, so that
index `j` corresponds to `omega_axis[j]`. This is the frequency grid on which
the solver evaluates its linear operator. Parseval: `Σ_j S_r[j] = nt · Σ_t P_r[t]`.

Integrals from the sums:

- power `∫∫|A|² dx dy` = `P_r[t] · dx · dy` (for a time-only run, `P_r[t] = |A(t)|²`);
- energy `∫|A|² dt dx dy` = `Σ_t P_r[t] · dt · dx · dy`;
- spectral density: `S_r[j] · dt²` is `|Â(ω_j)|²` for `Â(ω) = ∫A(t) e^{-iωt} dt`
  (summed over x, y), and `Σ_j S_r[j] dt² · Δω / 2π` with `Δω = 2π/(nt dt)` is
  the energy;
- fluence profiles `∫∫|A|² dt dy` = `F_r[x] · dt · dy`, `∫∫|A|² dt dx` = `G_r[y] · dt · dx`.

For a time-only run `fluence_x` and `fluence_y` each hold one value, the record's
`Σ_t |A|²`.

## Table `io_final_output_fields`

The final field (at `z = nlolib_z_end`), stored unless the caller turned it off
(`nlolib_final_field = 0`).

| Column | Type | Meaning |
|---|---|---|
| `run_id` | TEXT PRIMARY KEY | Run id. |
| `num_time_samples` | INTEGER | Complex samples: `nt · nx · ny`. |
| `payload` | BLOB | The full field, complex, time fastest (`idx = ((x*ny)+y)*nt + t`). |
| `created_utc` | TEXT | SQLite `CURRENT_TIMESTAMP` of the row (not contract form). |

## Table `io_record_chunks`

Raw records, present in modes `raw` and `reduced+raw`, in chunks of consecutive
records.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | TEXT | Run id. |
| `chunk_index` | INTEGER | Chunk number, from 0. Primary key with `run_id`. |
| `record_start` | INTEGER | Index of the chunk's first record. |
| `record_count` | INTEGER | Records in the chunk. |
| `payload` | BLOB | `record_count` records one after another, each `raw_nt · raw_nx · raw_ny` complex samples. |

Within a raw record the sample at raw indices `(ti, xi, yi)` is element
`((xi * raw_ny) + yi) * raw_nt + ti` and holds the field at the full-grid indices
`t = ti · raw_stride_t`, `x = xi · raw_stride_x`, `y = yi · raw_stride_y`, i.e. at
`t_axis[t]`, `x_axis[x]`, `y_axis[y]`. With all strides 1 a raw record is the full
field. Record `record_start + k` is at `record_z[record_start + k]`.

Databases written before layout version 1 have raw records only, always at full
resolution, and no `io_record_layout` or `runs` rows for those runs.

## Other tables

`io_runs` (one row per run: `run_id`, `key`, record and chunk counters, size
cap), `io_sim_config` and `io_system_config` (solver and execution settings,
keyed by hashes) are kept for Nlolib's own bookkeeping and older readers. Their
`created_utc` columns use SQLite's `YYYY-MM-DD HH:MM:SS` form, not the contract's.

## Example queries

All runs with their grid:

```sql
SELECT r.key, r.created_at, r.nlolib_nt, r.nlolib_nx, r.nlolib_ny, r.nlolib_z_end,
       r.nlolib_record_mode, r.nlolib_backend
FROM runs r ORDER BY r.created_at DESC;
```

Axes of one run:

```sql
SELECT l.nt, l.nx, l.ny, l.num_records, l.record_z, l.t_axis, l.omega_axis, l.x_axis, l.y_axis
FROM runs r JOIN io_record_layout l ON l.run_id = r.nlolib_run_id
WHERE r.key = :key;
```

Power map (`num_records` rows of `nt` reals; stack them for a z–t image):

```sql
SELECT m.record_index, m.z, m.power
FROM runs r JOIN io_record_maps m ON m.run_id = r.nlolib_run_id
WHERE r.key = :key ORDER BY m.record_index;
```

Final field:

```sql
SELECT f.payload FROM runs r JOIN io_final_output_fields f ON f.run_id = r.nlolib_run_id
WHERE r.key = :key;
```

## How Nlolib writes the file

**Staging.** A run streams into its own staging database
`staging/<stem>.<key>.<pid>.<token>.<host>.sqlite` (`<stem>` is the shared file's
name without `.sqlite`, `<pid>` the writer's process id, `<token>` 16 random hex
digits, `<host>` the writer's host name up to its first dot). Records are
committed in batches of `chunk_records`. When the run completes, the last data,
the layout and the `runs` row are written in one transaction; the staging file
then holds one complete run.

**Merge.** Nlolib then takes the lock, opens the shared database with a 60 s busy
timeout in rollback-journal mode, attaches the staging file, and in one
`BEGIN IMMEDIATE` transaction deletes any earlier run with the same key (its
`runs` row and data), copies the new run's rows and writes its `runs` row. Raw
chunks totalling more than 32 MiB are copied first in transactions of at most
that size (no `runs` row yet), so readers are never locked out for long. After
the commit it releases the lock and deletes the staging file. A row in `runs`
therefore never appears before all of its data.

**Interrupted runs.** A run that is aborted (progress callback), fails, or whose
process is killed never touches the shared database. An aborted or failed run
deletes its staging file; a killed one leaves an incomplete staging file (no
`runs` row). A run that completes but cannot get the lock in time (default
600 s) keeps its complete staging file and reports `merged = 0`.
`nlolib_storage_recover_staging` merges complete leftovers and, on request,
deletes incomplete ones older than a given age; it never touches files of live
processes on the same host.

**Lock protocol.** The lock file `<sqlite_path>.lock` is created with an
exclusive create (`O_CREAT | O_EXCL`), which SMB 3 performs atomically on the
server, and holds text lines:

```
nlolib-lock 1
host=<host>
pid=<pid>
created=<UTC time>
updated=<UTC time>
beat=<counter>
token=<16 hex digits>
```

The holder rewrites it (incrementing `beat`) between merge transactions and
checks before its final commit that the file still holds its token, rolling back
otherwise. A waiting writer treats the lock as stale when its host is the
writer's own and that process no longer exists, or when the file's content has
not changed for the stale age (default 120 s) as measured on the waiter's own
clock (so clock skew between machines does not matter). To remove a stale lock
a waiter first creates `<lock>.break` exclusively, re-reads the lock, and only
if it still holds exactly the content judged stale renames it to a unique name,
checks the renamed content once more (handing a fresh lock back if it caught
one), deletes it and removes `.break`. Writers poll every 0.1–0.3 s. The
timeout and stale age can be changed with the environment variables
`NLOLIB_STORE_LOCK_TIMEOUT_S` and `NLOLIB_STORE_LOCK_STALE_S`.

SQLite's own locks still guard every transaction, so even a writer that lost its
lock while paused for longer than the stale age commits atomically; the lock
file keeps merges from interleaving and avoids long `SQLITE_BUSY` waits.

**Older files.** Opening an existing file migrates it: missing tables are
created, missing columns are added (`PRAGMA table_info` is inspected), and a
WAL-mode file is switched to the rollback journal. Existing rows are kept.

## The run key

`key` is the first 16 hex digits of the SHA-256 of a canonical text. Readers
never parse it; this section only says what makes two runs the same run. The
text is a sequence of lines `name=value\n`, starting with `kind=6:nlolib` and
`key_version=1`, then:

- grid: total samples, `nt`, `nx`, `ny`, tensor mode and layout, `dt`, pulse
  period, `dx`, `dy`;
- solver: propagation distance, initial, maximum and minimum step, tolerance;
- operators: every expression string (length-prefixed, `-` when unset), the
  number of constants and each constant, nonlinear model, `γ`, `f_R`, `τ1`, `τ2`,
  `ω0`, and the SHA-256 of a custom Raman response;
- inputs: SHA-256 of the input field, of the potential grid and of any
  caller-supplied frequency grid, `wt`, `kx`, `ky` axes or spatial frequency grid;
- record specification: record count, SHA-256 of an explicit z schedule, record
  mode, final-field flag and, unless the mode is `reduced`, the raw strides.

Doubles enter as the hex of their IEEE-754 bits, arrays as SHA-256 of their
little-endian bytes. The execution backend, FFT backend, device settings, chunk
size, size cap, `run_id` and file path are not part of the key: the same run on
CPU and on Vulkan has one key, and storing it again replaces its row.

Operators given as host-language callables never reach the key as code: the
Python binding translates them into runtime expressions (capturing their
constants) before calling the C library, MATLAB translates function handles the
same way, and the Julia binding accepts expressions only. The key hashes the
translated expressions and constants, so no caller-supplied label is needed.

## API

| C | Julia | Python | MATLAB |
|---|---|---|---|
| `storage_options.record_mode`, `raw_stride_t/x/y`, `binding` | `record_mode`, `raw_stride_t/x/y` keywords | same keywords | `storage.record_mode`, `storage.raw_stride_*` |
| `nlolib_storage_run_key` | `storage_run_key` | `storage_run_key` | (via `calllib`) |
| `nlolib_storage_has_run` | `storage_has_run` | `storage_has_run` | `api.storage_has_run` |
| `nlolib_storage_recover_staging` | `storage_recover_staging` | `storage_recover_staging` | `api.storage_recover_staging` |
| `nlolib_storage_read_run_layout`, `nlolib_storage_read_run_array` | `read_stored_run` | (via ctypes) | (via `calllib`) |

`storage_result.key` and `storage_result.merged` report the stored key and
whether the run reached the shared database.
