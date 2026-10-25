"""Locate the ONNX Runtime DLLs used by the compiled ``_5dchess`` extension.

The extension links against the ONNX Runtime C API but does not ship a copy of
``onnxruntime.dll``. Windows resolves that dependency through the process-wide
DLL search path, so the runtime directory has to be registered before the
extension is imported. Prefer an explicit ``SUZERO_ORT_RUNTIME_DIR`` override,
then the ``capi`` directory of the installed ``onnxruntime`` package (a GPU
package keeps ``onnxruntime_providers_cuda.dll`` beside ``onnxruntime.dll``,
which is where ONNX Runtime expects to find its providers).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterator

# Windows drops a directory from the search path once its handle is closed,
# so keep every handle alive for the lifetime of the process.
_dll_handles: list = []


def _candidate_dirs() -> Iterator[Path]:
    override = os.environ.get("SUZERO_ORT_RUNTIME_DIR")
    if override:
        yield Path(override)

    # find_spec locates the package without importing it, so training code that
    # never touches ONNX does not pay for the onnxruntime import.
    try:
        spec = importlib.util.find_spec("onnxruntime")
    except (ImportError, ValueError):
        return
    if spec is not None and spec.origin:
        yield Path(spec.origin).resolve().parent / "capi"


def configure_onnxruntime_dlls() -> bool:
    """Register the ONNX Runtime directory for the current process.

    Returns ``True`` when a directory containing ``onnxruntime.dll`` was
    registered, or when the platform does not need the fallback (non-Windows).
    """
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return True

    for candidate in _candidate_dirs():
        if not (candidate / "onnxruntime.dll").is_file():
            continue
        try:
            _dll_handles.append(os.add_dll_directory(str(candidate)))
        except OSError:
            continue
        return True
    return False
