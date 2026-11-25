/**
 * @file nlolib_async.c
 * @dir src
 * @brief Asynchronous propagation jobs: nlolib_propagate() on a worker thread.
 *
 * A job runs one nlolib_propagate() call on its own thread and keeps the most
 * recent progress event, so a host that cannot receive C callbacks (MATLAB
 * loadlibrary) can read progress, logs and a cancel request at its own pace.
 */

#include "nlolib.h"
#include "io/log_sink.h"
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <process.h>
typedef SRWLOCK job_lock;
#else
  #include <errno.h>
  #include <pthread.h>
  #include <time.h>
typedef pthread_mutex_t job_lock;
#endif

struct nlolib_propagate_job
{
    /* Private copies of every struct the caller passed, so that only the
     * buffers they point to have to outlive nlolib_propagate_start(). */
    simulation_config simulation;
    physics_config physics;
    execution_options exec_options;
    storage_options storage;
    propagate_options options;
    propagate_output output;
    int has_simulation;
    int has_physics;
    int has_output;
    size_t num_time_samples;
    const nlo_complex* input_field;

    progress_callback user_callback;
    void* user_data;

    /* Guarded by lock. */
    progress_info info;
    int cancel_requested;
    int done;
    nlolib_status status;

    job_lock lock;
#if defined(_WIN32)
    HANDLE done_event;
    HANDLE thread;
#else
    pthread_cond_t done_cond;
    pthread_t thread;
#endif
};

/* The log sink, progress state and perf counters are process-wide, so one job
 * runs at a time. */
#if defined(_WIN32)
static volatile LONG g_job_active = 0;
static int job_slot_acquire(void) { return InterlockedCompareExchange(&g_job_active, 1, 0) == 0; }
static void job_slot_release(void) { InterlockedExchange(&g_job_active, 0); }
static void job_lock_init(job_lock* lock) { InitializeSRWLock(lock); }
static void job_lock_destroy(job_lock* lock) { (void)lock; }
static void job_lock_acquire(job_lock* lock) { AcquireSRWLockExclusive(lock); }
static void job_lock_release(job_lock* lock) { ReleaseSRWLockExclusive(lock); }
#else
static pthread_mutex_t g_job_slot_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_job_active = 0;
static int job_slot_acquire(void)
{
    (void)pthread_mutex_lock(&g_job_slot_lock);
    const int acquired = (g_job_active == 0) ? 1 : 0;
    g_job_active = 1;
    (void)pthread_mutex_unlock(&g_job_slot_lock);
    return acquired;
}
static void job_slot_release(void)
{
    (void)pthread_mutex_lock(&g_job_slot_lock);
    g_job_active = 0;
    (void)pthread_mutex_unlock(&g_job_slot_lock);
}
static void job_lock_init(job_lock* lock) { (void)pthread_mutex_init(lock, NULL); }
static void job_lock_destroy(job_lock* lock) { (void)pthread_mutex_destroy(lock); }
static void job_lock_acquire(job_lock* lock) { (void)pthread_mutex_lock(lock); }
static void job_lock_release(job_lock* lock) { (void)pthread_mutex_unlock(lock); }
#endif

static int job_progress(const progress_info* info, void* user_data)
{
    nlolib_propagate_job* job = (nlolib_propagate_job*)user_data;
    job_lock_acquire(&job->lock);
    job->info = *info;
    int keep_going = (job->cancel_requested == 0) ? 1 : 0;
    job_lock_release(&job->lock);

    if (keep_going != 0 && job->user_callback != NULL &&
        job->user_callback(info, job->user_data) == 0) {
        keep_going = 0;
    }
    return keep_going;
}

static void job_run(nlolib_propagate_job* job)
{
    const nlolib_status status =
        nlolib_propagate((job->has_simulation != 0) ? &job->simulation : NULL,
                         (job->has_physics != 0) ? &job->physics : NULL,
                         job->num_time_samples,
                         job->input_field,
                         &job->options,
                         (job->has_output != 0) ? &job->output : NULL);

    job_lock_acquire(&job->lock);
    job->status = status;
    job->done = 1;
#if !defined(_WIN32)
    (void)pthread_cond_broadcast(&job->done_cond);
#endif
    job_lock_release(&job->lock);
#if defined(_WIN32)
    (void)SetEvent(job->done_event);
#endif
}

#if defined(_WIN32)
static unsigned __stdcall job_thread_main(void* arg)
{
    job_run((nlolib_propagate_job*)arg);
    return 0u;
}
#else
static void* job_thread_main(void* arg)
{
    job_run((nlolib_propagate_job*)arg);
    return NULL;
}
#endif

/* Block until the job is done or timeout_ms elapses. Returns nonzero when done. */
static int job_wait_done(nlolib_propagate_job* job, uint32_t timeout_ms, int forever)
{
#if defined(_WIN32)
    (void)WaitForSingleObject(job->done_event, (forever != 0) ? INFINITE : (DWORD)timeout_ms);
    job_lock_acquire(&job->lock);
    const int done = job->done;
    job_lock_release(&job->lock);
    return done;
#else
    struct timespec deadline;
    (void)clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += (time_t)(timeout_ms / 1000u);
    deadline.tv_nsec += (long)(timeout_ms % 1000u) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000L;
    }

    job_lock_acquire(&job->lock);
    while (job->done == 0) {
        if (forever != 0) {
            (void)pthread_cond_wait(&job->done_cond, &job->lock);
        } else if (pthread_cond_timedwait(&job->done_cond, &job->lock, &deadline) == ETIMEDOUT) {
            break;
        }
    }
    const int done = job->done;
    job_lock_release(&job->lock);
    return done;
#endif
}

static void job_destroy(nlolib_propagate_job* job)
{
#if defined(_WIN32)
    if (job->thread != NULL) {
        (void)CloseHandle(job->thread);
    }
    if (job->done_event != NULL) {
        (void)CloseHandle(job->done_event);
    }
#else
    (void)pthread_cond_destroy(&job->done_cond);
#endif
    job_lock_destroy(&job->lock);
    free(job);
}

NLOLIB_API nlolib_status nlolib_propagate_start(
    const simulation_config* simulation_config,
    const physics_config* physics_config,
    size_t num_time_samples,
    const nlo_complex* input_field,
    const propagate_options* options,
    propagate_output* output,
    nlolib_propagate_job** out_job
)
{
    if (out_job == NULL) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }
    *out_job = NULL;

    if (job_slot_acquire() == 0) {
        log_emit(LOG_LEVEL_ERROR, "[nlolib] propagate_start: another propagation job is still running.");
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }

    nlolib_propagate_job* job = (nlolib_propagate_job*)calloc(1u, sizeof(*job));
    if (job == NULL) {
        job_slot_release();
        return NLOLIB_STATUS_ALLOCATION_FAILED;
    }

    if (simulation_config != NULL) {
        job->simulation = *simulation_config;
        job->has_simulation = 1;
    }
    if (physics_config != NULL) {
        job->physics = *physics_config;
        job->has_physics = 1;
    }
    job->options = (options != NULL) ? *options : nlolib_propagate_options_default();
    if (job->options.exec_options != NULL) {
        job->exec_options = *job->options.exec_options;
        job->options.exec_options = &job->exec_options;
    }
    if (job->options.storage_options != NULL) {
        job->storage = *job->options.storage_options;
        job->options.storage_options = &job->storage;
    }
    if (output != NULL) {
        job->output = *output;
        job->has_output = 1;
    }
    job->num_time_samples = num_time_samples;
    job->input_field = input_field;
    job->user_callback = job->options.progress_callback;
    job->user_data = job->options.progress_user_data;
    job->options.progress_callback = job_progress;
    job->options.progress_user_data = job;
    job->status = NLOLIB_STATUS_OK;
    job_lock_init(&job->lock);

#if defined(_WIN32)
    job->done_event = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (job->done_event != NULL) {
        job->thread = (HANDLE)_beginthreadex(NULL, 0u, job_thread_main, job, 0u, NULL);
    }
    const int started = (job->thread != NULL) ? 1 : 0;
#else
    (void)pthread_cond_init(&job->done_cond, NULL);
    const int started = (pthread_create(&job->thread, NULL, job_thread_main, job) == 0) ? 1 : 0;
#endif
    if (started == 0) {
        job_destroy(job);
        job_slot_release();
        return NLOLIB_STATUS_ALLOCATION_FAILED;
    }

    *out_job = job;
    return NLOLIB_STATUS_OK;
}

NLOLIB_API nlolib_status nlolib_propagate_poll(
    nlolib_propagate_job* job,
    uint32_t timeout_ms,
    progress_info* out_info,
    int* out_done
)
{
    if (job == NULL) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }

    const int done = job_wait_done(job, timeout_ms, 0);
    if (out_info != NULL) {
        job_lock_acquire(&job->lock);
        *out_info = job->info;
        job_lock_release(&job->lock);
    }
    if (out_done != NULL) {
        *out_done = done;
    }
    return NLOLIB_STATUS_OK;
}

NLOLIB_API nlolib_status nlolib_propagate_cancel(nlolib_propagate_job* job)
{
    if (job == NULL) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }

    job_lock_acquire(&job->lock);
    job->cancel_requested = 1;
    job_lock_release(&job->lock);
    return NLOLIB_STATUS_OK;
}

NLOLIB_API nlolib_status nlolib_propagate_finish(nlolib_propagate_job* job)
{
    if (job == NULL) {
        return NLOLIB_STATUS_INVALID_ARGUMENT;
    }

    (void)job_wait_done(job, 0u, 1);
#if defined(_WIN32)
    (void)WaitForSingleObject(job->thread, INFINITE);
#else
    (void)pthread_join(job->thread, NULL);
#endif
    const nlolib_status status = job->status;
    job_destroy(job);
    job_slot_release();
    return status;
}
