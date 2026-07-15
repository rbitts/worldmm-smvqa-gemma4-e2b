"""Fail-closed Linux x86_64 ``openat2(2)`` sealed-root resolution."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
from pathlib import Path
from typing import Final, cast, final

RESOLVE_NO_XDEV: Final = 0x01
RESOLVE_NO_MAGICLINKS: Final = 0x02
RESOLVE_NO_SYMLINKS: Final = 0x04
RESOLVE_BENEATH: Final = 0x08
RESOLVE_IN_ROOT: Final = 0x10
SEALED_RESOLVE: Final = (
    RESOLVE_IN_ROOT | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV
)
_OPENAT2_SYSCALL_X86_64: Final = 437


class Openat2UnsupportedError(RuntimeError):
    """Raised when the mandatory sealed-root kernel primitive is unavailable."""


@final
class _OpenHow(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_ulonglong),
        ("mode", ctypes.c_ulonglong),
        ("resolve", ctypes.c_ulonglong),
    ]


def openat2_sealed(
    dir_fd: int,
    relative_path: str | os.PathLike[str],
    flags: int,
) -> int:
    """Open one syntactically relative path under ``dir_fd`` using the exact mask."""
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        message = "openat2 sealed-root resolution requires Linux x86_64"
        raise Openat2UnsupportedError(message)
    path = os.fspath(relative_path)
    if (
        not path
        or Path(path).is_absolute()
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        message = (
            "sealed-root path must be a non-empty relative path without dot components"
        )
        raise ValueError(message)
    how = _OpenHow(flags=flags, mode=0, resolve=SEALED_RESOLVE)
    libc = ctypes.CDLL(None, use_errno=True)
    result = cast(
        "int",
        libc.syscall(
            _OPENAT2_SYSCALL_X86_64,
            ctypes.c_int(dir_fd),
            ctypes.c_char_p(os.fsencode(path)),
            ctypes.byref(how),
            ctypes.sizeof(how),
        ),
    )
    if result == -1:
        error_number = ctypes.get_errno()
        if error_number in {errno.ENOSYS, errno.EINVAL}:
            message = "openat2 sealed-root resolution is unavailable on this kernel"
            raise Openat2UnsupportedError(message)
        raise OSError(error_number, os.strerror(error_number), path)
    return result
