/**
 * @file store_merge.c
 * @brief `nlolib_store_merge`: merge completed staging files into a shared run database.
 *
 * Meant for the host that merges runs left in `staging/` by writers that ran
 * with defer_merge (or that could not take the lock). See
 * docs/nlolib_run_database.md.
 */

#include "nlolib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

static void usage(FILE* out)
{
    fprintf(out,
            "usage: nlolib_store_merge [options] <store-folder | run-database.sqlite>...\n"
            "\n"
            "Merge every completed staging file of each run database into it, under the\n"
            "database's writer lock. A store folder means <folder>/nlolib/nlolib.sqlite.\n"
            "\n"
            "options:\n"
            "  --discard-incomplete   also delete incomplete staging files (crashed or killed runs)\n"
            "                         and the data of interrupted merges whose staging file is gone\n"
            "  --min-age SECONDS      only discard incomplete files at least this old (default 86400)\n"
            "  --watch SECONDS        repeat every SECONDS until interrupted\n"
            "  -q, --quiet            print only errors\n"
            "  -h, --help             show this help\n");
}

static int ends_with(const char* text, const char* suffix)
{
    const size_t n = strlen(text);
    const size_t m = strlen(suffix);
    return n >= m && strcmp(text + n - m, suffix) == 0;
}

static void resolve_db(const char* arg, char* out, size_t cap)
{
    if (ends_with(arg, ".sqlite")) {
        (void)snprintf(out, cap, "%s", arg);
        return;
    }
    const size_t n = strlen(arg);
    const int sep = (n > 0u && (arg[n - 1u] == '/' || arg[n - 1u] == '\\'));
    (void)snprintf(out, cap, "%s%snlolib/nlolib.sqlite", arg, sep ? "" : "/");
}

static void sleep_seconds(double seconds)
{
#if defined(_WIN32)
    Sleep((DWORD)(seconds * 1000.0));
#else
    usleep((useconds_t)(seconds * 1e6));
#endif
}

int main(int argc, char** argv)
{
    int discard = 0;
    int quiet = 0;
    double min_age = 86400.0;
    double watch = 0.0;
    int first_path = argc;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--discard-incomplete") == 0) {
            discard = 1;
        } else if (strcmp(argv[i], "--min-age") == 0 && i + 1 < argc) {
            min_age = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--watch") == 0 && i + 1 < argc) {
            watch = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            quiet = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(stdout);
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "nlolib_store_merge: unknown option %s\n", argv[i]);
            usage(stderr);
            return 2;
        } else {
            first_path = i;
            break;
        }
    }
    if (first_path >= argc) {
        usage(stderr);
        return 2;
    }
    (void)nlolib_set_log_level(quiet ? NLOLIB_LOG_LEVEL_ERROR : NLOLIB_LOG_LEVEL_WARN);

    int failures = 0;
    for (;;) {
        for (int i = first_path; i < argc; ++i) {
            char db[4096];
            resolve_db(argv[i], db, sizeof(db));
            size_t merged = 0u;
            size_t discarded = 0u;
            size_t skipped = 0u;
            const nlolib_status status =
                nlolib_storage_recover_staging(db, discard, min_age, &merged, &discarded, &skipped);
            if (status != NLOLIB_STATUS_OK) {
                failures += 1;
                fprintf(stderr, "nlolib_store_merge: %s: status %d (merged %zu, discarded %zu, skipped %zu)\n",
                        db, (int)status, merged, discarded, skipped);
            } else if (!quiet) {
                printf("%s: merged %zu, discarded %zu, skipped %zu\n", db, merged, discarded, skipped);
                fflush(stdout);
            }
        }
        if (watch <= 0.0) {
            break;
        }
        sleep_seconds(watch);
    }
    return failures == 0 ? 0 : 1;
}
