"""Windows console safeguards for long-running commands."""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Callable

STD_INPUT_HANDLE = -10
ENABLE_QUICK_EDIT_MODE = 0x0040
ENABLE_EXTENDED_FLAGS = 0x0080
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def _load_kernel32():
    if os.name != "nt":
        return None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetConsoleMode.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetConsoleMode.restype = wintypes.BOOL
    kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetConsoleMode.restype = wintypes.BOOL
    return kernel32


def disable_quick_edit() -> Callable[[], None] | None:
    """Disable selection pauses in the legacy Windows console.

    Returns an idempotent callback that restores the original input mode. If
    stdin is redirected or the host does not expose a Windows console, this is
    a no-op and returns ``None``.
    """

    kernel32 = _load_kernel32()
    if kernel32 is None:
        return None

    handle = kernel32.GetStdHandle(
        wintypes.DWORD(STD_INPUT_HANDLE & 0xFFFFFFFF)
    )
    if handle in (None, 0, _INVALID_HANDLE_VALUE):
        return None

    original = wintypes.DWORD()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(original)):
        return None

    protected = (
        original.value | ENABLE_EXTENDED_FLAGS
    ) & ~ENABLE_QUICK_EDIT_MODE
    if protected == original.value:
        return None
    if not kernel32.SetConsoleMode(handle, protected):
        return None

    restored = False

    def restore() -> None:
        nonlocal restored
        if not restored:
            restored = True
            kernel32.SetConsoleMode(handle, original.value)

    return restore


__all__ = ["disable_quick_edit"]
