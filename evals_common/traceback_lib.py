"""ctypes loader for the compiled matrix-traceback shared library.

The library (``libtraceback_matrix.so`` / ``.dylib``) is compiled on first use
from ``baselines/traceback/matrix_traceback.cpp`` — a dependency-free C++ file
that exposes four ``extern "C"`` functions.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import threading
from pathlib import Path
from typing import Optional

import numpy as np

_lock = threading.Lock()
_lib: Optional[ctypes.CDLL] = None
_lib_loaded = False


def _src_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "baselines" / "traceback" / "matrix_traceback.cpp"
    )


def _lib_path() -> Path:
    override = os.environ.get("VECREC_EVALS_BIN")
    bin_dir = Path(override) if override else Path(__file__).resolve().parents[1] / "bin"
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    return bin_dir / f"libtraceback_matrix{suffix}"


def _compiler_flags() -> list[str]:
    """Extra flags for the current platform (mirrors harness.platform_compiler_flags)."""
    if platform.system() != "Darwin":
        return []
    try:
        sdk = subprocess.run(
            ["xcrun", "--show-sdk-path"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return ["-isysroot", sdk] if sdk else []
    except (OSError, subprocess.CalledProcessError):
        return []


def _build_lib() -> bool:
    """Compile the traceback shared library; return True on success."""
    src = _src_path()
    out = _lib_path()
    if not src.exists():
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "c++", "-std=c++17", "-O3", "-fPIC", "-shared",
        *_compiler_flags(),
        "-o", str(out), str(src),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except OSError:
        return False


def _load() -> Optional[ctypes.CDLL]:
    p = _lib_path()
    if not p.exists():
        if not _build_lib():
            return None
    if not _lib_path().exists():
        return None
    try:
        lib = ctypes.CDLL(str(_lib_path()))
    except OSError:
        return None

    lib.nw_traceback_c.restype = ctypes.c_int
    lib.nw_traceback_c.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ]

    lib.sw_traceback_c.restype = ctypes.c_int
    lib.sw_traceback_c.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ]

    lib.affine_traceback_c.restype = ctypes.c_int
    lib.affine_traceback_c.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ]

    lib.levenshtein_traceback_c.restype = ctypes.c_int
    lib.levenshtein_traceback_c.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ]

    return lib


def _get_lib() -> Optional[ctypes.CDLL]:
    global _lib, _lib_loaded
    if _lib_loaded:
        return _lib
    with _lock:
        if not _lib_loaded:
            _lib = _load()
            _lib_loaded = True
    return _lib


def require_traceback_lib() -> ctypes.CDLL:
    """Return the compiled traceback shared library or raise ``AssertionError``."""
    lib = _get_lib()
    assert lib is not None, (
        f"matrix traceback C++ library not available at {_lib_path()}. "
        "Ensure a C++ compiler is installed and "
        "baselines/traceback/matrix_traceback.cpp exists."
    )
    return lib


def _bufs(rows: int, cols: int) -> tuple:
    size = rows + cols + 1
    return (
        ctypes.create_string_buffer(size),
        ctypes.create_string_buffer(size),
        ctypes.create_string_buffer(size),
    )


def _decode(q_buf, r_buf, c_buf) -> tuple[str, str, str]:
    return q_buf.value.decode(), r_buf.value.decode(), c_buf.value.decode()


def _h_ptr(arr: np.ndarray):
    return arr.astype(np.int64, copy=False).ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


def _s_ptr(arr: np.ndarray):
    return arr.astype(np.int32, copy=False).ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def nw_traceback(
    matrix_flat: np.ndarray, rows: int, cols: int,
    s: np.ndarray, x: str, y: str, gap: int,
) -> tuple[str, str, str]:
    lib = require_traceback_lib()
    q, r, c = _bufs(rows, cols)
    lib.nw_traceback_c(
        _h_ptr(matrix_flat), rows, cols,
        _s_ptr(s), x.encode(), y.encode(), gap,
        q, r, c,
    )
    return _decode(q, r, c)


def sw_traceback(
    matrix_flat: np.ndarray, rows: int, cols: int,
    s: np.ndarray, x: str, y: str, gap: int,
) -> tuple[str, str, str]:
    lib = require_traceback_lib()
    q, r, c = _bufs(rows, cols)
    lib.sw_traceback_c(
        _h_ptr(matrix_flat), rows, cols,
        _s_ptr(s), x.encode(), y.encode(), gap,
        q, r, c,
    )
    return _decode(q, r, c)


def affine_traceback(
    matrix_flat: np.ndarray, rows: int, cols: int,
    s: np.ndarray, x: str, y: str, gap_open: int, gap_extend: int,
) -> tuple[str, str, str]:
    lib = require_traceback_lib()
    q, r, c = _bufs(rows, cols)
    lib.affine_traceback_c(
        _h_ptr(matrix_flat), rows, cols,
        _s_ptr(s), x.encode(), y.encode(), gap_open, gap_extend,
        q, r, c,
    )
    return _decode(q, r, c)


def levenshtein_traceback(
    matrix_flat: np.ndarray, rows: int, cols: int,
    x: str, y: str,
) -> tuple[str, str, str]:
    lib = require_traceback_lib()
    q, r, c = _bufs(rows, cols)
    lib.levenshtein_traceback_c(
        _h_ptr(matrix_flat), rows, cols,
        x.encode(), y.encode(),
        q, r, c,
    )
    return _decode(q, r, c)
