/* Copyright The Lightning AI team. Licensed under the Apache License, Version 2.0. */
#define _GNU_SOURCE
#define _LARGEFILE64_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <unistd.h>

static int selected_suffix(const char *path, const char *suffixes) {
    size_t length = strlen(path);
    for (const char *p = suffixes; *p;) {
        const char *end = strchr(p, ':');
        size_t n = end ? (size_t)(end - p) : strlen(p);
        if (n && length >= n && !memcmp(path + length - n, p, n)) return 1;
        if (!end) break;
        p = end + 1;
    }
    return 0;
}

static int configure(int fd, int flags) {
    int saved = errno;
    const char *root = getenv("LITDATA_DIRECT_IO_ROOT");
    const char *suffixes = getenv("LITDATA_DIRECT_IO_SUFFIXES");
    if (fd < 0 || !root || !*root || !suffixes ||
        (flags & O_ACCMODE) != O_RDONLY || (flags & (O_DIRECTORY | O_PATH))) return fd;
    char link[64], path[PATH_MAX + 1];
    snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);
    ssize_t count = readlink(link, path, sizeof(path) - 1);
    if (count < 0) goto fail;
    if (count >= PATH_MAX) { errno = ENAMETOOLONG; goto fail; }
    path[count] = '\0';
    size_t n = strlen(root);
    if ((size_t)count <= n || strncmp(path, root, n) || path[n] != '/' || !selected_suffix(path, suffixes)) {
        errno = saved;
        return fd;
    }
    struct stat st;
    if (fstat(fd, &st)) goto fail;
    if (!S_ISREG(st.st_mode)) { errno = EINVAL; goto fail; }
    struct statfs fs;
    if (fstatfs(fd, &fs)) goto fail;
    if (fs.f_type != 0x6969) { errno = EOPNOTSUPP; goto fail; }
    int current = fcntl(fd, F_GETFL);
    if (current < 0 || fcntl(fd, F_SETFL, current | O_DIRECT) < 0) goto fail;
    errno = saved;
    return fd;
fail: {
    int error = errno;
    close(fd);
    errno = error;
    return -1;
}
}

static mode_t get_mode(int flags, va_list args) {
    if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) return va_arg(args, mode_t);
    return 0;
}

#define WRAP_OPEN(name) \
int name(const char *path, int flags, ...) { \
    va_list args; va_start(args, flags); \
    mode_t mode = get_mode(flags, args); va_end(args); \
    int (*original)(const char *, int, ...) = dlsym(RTLD_NEXT, #name); \
    if (!original) { errno = ENOSYS; return -1; } \
    return configure(original(path, flags, mode), flags); \
}

#define WRAP_OPENAT(name) \
int name(int dirfd, const char *path, int flags, ...) { \
    va_list args; va_start(args, flags); \
    mode_t mode = get_mode(flags, args); va_end(args); \
    int (*original)(int, const char *, int, ...) = dlsym(RTLD_NEXT, #name); \
    if (!original) { errno = ENOSYS; return -1; } \
    return configure(original(dirfd, path, flags, mode), flags); \
}

WRAP_OPEN(open)
WRAP_OPEN(open64)
WRAP_OPENAT(openat)
WRAP_OPENAT(openat64)

/* glibc's fortified two-argument variants may bypass the public open symbols. */
#define WRAP_OPEN2(name) \
int name(const char *path, int flags) { \
    int (*original)(const char *, int) = dlsym(RTLD_NEXT, #name); \
    if (!original) { errno = ENOSYS; return -1; } \
    return configure(original(path, flags), flags); \
}
#define WRAP_OPENAT2(name) \
int name(int dirfd, const char *path, int flags) { \
    int (*original)(int, const char *, int) = dlsym(RTLD_NEXT, #name); \
    if (!original) { errno = ENOSYS; return -1; } \
    return configure(original(dirfd, path, flags), flags); \
}
WRAP_OPEN2(__open_2)
WRAP_OPEN2(__open64_2)
WRAP_OPENAT2(__openat_2)
WRAP_OPENAT2(__openat64_2)
