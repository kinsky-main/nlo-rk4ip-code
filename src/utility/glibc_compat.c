/**
 * @file glibc_compat.c
 * @brief Private stand-ins for glibc symbols newer than 2.35, Linux builds only.
 *
 * The release builds run on a newer glibc than the hosts they ship to. C++
 * code compiled against glibc 2.38 headers calls strtol/strtoul through
 * __isoc23_strtol/__isoc23_strtoul (glslang does), and the static libstdc++
 * calls arc4random (glibc 2.36). Defining them here, hidden, binds those
 * references inside the library, so it loads on glibc 2.35 (Ubuntu 22.04).
 * Nothing here is exported.
 */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/random.h>
#include <sys/types.h>
#include <unistd.h>

#define NLO_COMPAT_HIDDEN __attribute__((visibility("hidden")))

/* The plain symbols, whatever the headers would redirect them to. */
extern long nlo_compat_strtol(const char* text, char** end, int base) __asm__("strtol");
extern unsigned long nlo_compat_strtoul(const char* text, char** end, int base) __asm__("strtoul");
extern void abort(void);

long __isoc23_strtol(const char* text, char** end, int base);
unsigned long __isoc23_strtoul(const char* text, char** end, int base);
uint32_t arc4random(void);

/* C23 also accepts a 0b prefix for base 0 and 2; no caller here parses one. */
NLO_COMPAT_HIDDEN long __isoc23_strtol(const char* text, char** end, int base)
{
    return nlo_compat_strtol(text, end, base);
}

NLO_COMPAT_HIDDEN unsigned long __isoc23_strtoul(const char* text, char** end, int base)
{
    return nlo_compat_strtoul(text, end, base);
}

/* Like glibc's, this aborts when the kernel has no randomness to give. */
NLO_COMPAT_HIDDEN uint32_t arc4random(void)
{
    uint32_t value = 0;
    ssize_t got;
    do {
        got = getrandom(&value, sizeof(value), 0);
    } while (got < 0 && errno == EINTR);
    if (got == (ssize_t)sizeof(value)) {
        return value;
    }

    int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        do {
            got = read(fd, &value, sizeof(value));
        } while (got < 0 && errno == EINTR);
        close(fd);
        if (got == (ssize_t)sizeof(value)) {
            return value;
        }
    }
    abort();
    return 0;
}
