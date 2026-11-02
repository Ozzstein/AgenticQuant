"""POSIX descriptor-relative private storage; no pathname check/open pairs.

Public ancestors may be shared (e.g. /tmp); every component is opened without
following links, and the final directory must be owned and private. Opened
objects stay pinned across ancestor renames. This does not isolate hostile code
running as the same uid, which can chmod or mutate already opened objects.
"""

import os
import stat
from contextlib import contextmanager
from pathlib import Path


class PrivateFileError(ValueError):
    pass


_DIR_FD_SUPPORTED = all(fn in os.supports_dir_fd for fn in (os.open, os.mkdir, os.stat))


def _capabilities():
    if (
        not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"))
        or not _DIR_FD_SUPPORTED
        or os.listdir not in os.supports_fd
    ):
        raise PrivateFileError("private_source_invalid")


def _name(name):
    if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\0" in name:
        raise PrivateFileError("private_source_invalid")


def _directory(fd):
    info = os.fstat(fd)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise PrivateFileError("private_source_permissions")


@contextmanager
def private_directory(path: Path, *, create=False):
    _capabilities()
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or "\0" in str(path) or path == Path("/"):
        raise PrivateFileError("private_source_invalid")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        _directory(fd)
        yield fd
    finally:
        os.close(fd)


@contextmanager
def private_child(parent, name):
    _name(name)
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    try:
        _directory(fd)
        yield fd
    finally:
        os.close(fd)


def _file(fd, limit):
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
        or info.st_size > limit
    ):
        raise PrivateFileError("private_source_invalid")


def read_at(parent, name, limit):
    _name(name)
    _directory(parent)
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        _file(fd, limit)
        chunks, remaining = [], limit + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        _file(fd, limit)
        raw = b"".join(chunks)
        if len(raw) > limit:
            raise PrivateFileError("private_source_invalid")
        return raw
    finally:
        os.close(fd)


def read_private(path, limit):
    with private_directory(Path(path).parent) as parent:
        return read_at(parent, Path(path).name, limit)


def write_private(path, data):
    path = Path(path)
    with private_directory(path.parent) as parent:
        _name(path.name)
        fd = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
            dir_fd=parent,
        )
        try:
            _file(fd, len(data))
            remaining = memoryview(data)
            while remaining:
                count = os.write(fd, remaining)
                if count <= 0:
                    raise PrivateFileError("private_source_invalid")
                remaining = remaining[count:]
            _file(fd, len(data))
            os.fsync(fd)
        finally:
            os.close(fd)
