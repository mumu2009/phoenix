/* Minimal glibc 2.38 symbol shims for cross-linking GCC 15 libstdc++ against
 * Ubuntu 22.04 / glibc 2.35 RDK sysroots.  Only linked into aarch64
 * llama-server; not used on the Windows host build. */
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base) {
    return strtoul(nptr, endptr, base);
}

unsigned long long __isoc23_strtoull(const char *nptr, char **endptr, int base) {
    return strtoull(nptr, endptr, base);
}

long __isoc23_strtol(const char *nptr, char **endptr, int base) {
    return strtol(nptr, endptr, base);
}

long long __isoc23_strtoll(const char *nptr, char **endptr, int base) {
    return strtoll(nptr, endptr, base);
}

static uint32_t arc4random_state;

static void arc4_seed_once(void) {
    if (arc4random_state != 0) {
        return;
    }
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        (void)read(fd, &arc4random_state, sizeof(arc4random_state));
        close(fd);
    }
    if (arc4random_state == 0) {
        arc4random_state = 0x9e3779b9u;
    }
}

uint32_t arc4random(void) {
    arc4_seed_once();
    uint32_t x = arc4random_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    arc4random_state = x ? x : 1u;
    return x;
}

void arc4random_buf(void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    while (n-- > 0) {
        *p++ = (uint8_t)arc4random();
    }
}
