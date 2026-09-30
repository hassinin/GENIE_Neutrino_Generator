"""
Python interface for the Portable C++/HIP GPU Cross-Section Spline Engine.
Uses ctypes to interface directly with libgpuspline.so.
Supports NumPy arrays for high-throughput batch evaluation.
"""

import os
import ctypes
import numpy as np

# Find libgpuspline.so relative to this file
_dir = os.path.dirname(os.path.abspath(__file__))
_lib_candidates = [
    os.path.join(_dir, "../lib/libgpuspline.so"),
    os.path.join(_dir, "libgpuspline.so"),
    "libgpuspline.so"
]

_lib = None
for path in _lib_candidates:
    if os.path.exists(path):
        try:
            _lib = ctypes.CDLL(path)
            break
        except Exception:
            pass

if _lib is None:
    try:
        _lib = ctypes.CDLL("libgpuspline.so")
    except Exception:
        _lib = None


class GpuSplines:
    def __init__(self, xml_path=None, device_id=0):
        if _lib is None:
            raise RuntimeError("Could not find or load libgpuspline.so! Build it first using `make` in gpu_spline/.")

        self._setup_c_api()
        self._handle = _lib.gpu_spline_create()
        self._gpu_ready = False
        if not self._handle:
            raise MemoryError("Could not allocate spline engine")

        if xml_path:
            self.load(xml_path, device_id=device_id)

    def _setup_c_api(self):
        _lib.gpu_spline_create.restype = ctypes.c_void_p
        _lib.gpu_spline_destroy.argtypes = [ctypes.c_void_p]

        _lib.gpu_spline_load_xml.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _lib.gpu_spline_load_xml.restype = ctypes.c_int

        _lib.gpu_spline_init_gpu.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _lib.gpu_spline_init_gpu.restype = ctypes.c_int

        _lib.gpu_spline_free_gpu.argtypes = [ctypes.c_void_p]

        _lib.gpu_spline_get_num_splines.argtypes = [ctypes.c_void_p]
        _lib.gpu_spline_get_num_splines.restype = ctypes.c_int

        _lib.gpu_spline_find_id.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _lib.gpu_spline_find_id.restype = ctypes.c_int

        _lib.gpu_spline_find_id_for_tune.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        _lib.gpu_spline_find_id_for_tune.restype = ctypes.c_int

        _lib.gpu_spline_get_tune.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _lib.gpu_spline_get_tune.restype = ctypes.c_char_p

        _lib.gpu_spline_get_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _lib.gpu_spline_get_name.restype = ctypes.c_char_p

        _lib.gpu_spline_eval_batch_cpu.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]

        _lib.gpu_spline_eval_batch_gpu.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            _lib.gpu_spline_destroy(self._handle)
            self._handle = None

    def load(self, xml_path, device_id=0):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Spline file not found: {xml_path}")

        ret = _lib.gpu_spline_load_xml(self._handle, xml_path.encode("utf-8"))
        if not ret:
            raise RuntimeError(f"Failed to load spline XML file: {xml_path}")

        self._gpu_ready = False
        if device_id is not None and device_id >= 0:
            ret_gpu = _lib.gpu_spline_init_gpu(self._handle, device_id)
            self._gpu_ready = bool(ret_gpu)
            if not self._gpu_ready:
                print(f"[Warning] Failed to initialize GPU {device_id}, falling back to CPU.")

    @property
    def num_splines(self):
        return _lib.gpu_spline_get_num_splines(self._handle)

    def find_spline_id(self, name, tune=None):
        """Return -1 if missing; require a tune when the name is ambiguous."""
        if tune is not None:
            return _lib.gpu_spline_find_id_for_tune(
                self._handle, name.encode("utf-8"), tune.encode("utf-8"))
        spline_id = _lib.gpu_spline_find_id(self._handle, name.encode("utf-8"))
        if spline_id == -2:
            raise ValueError(f"Spline {name!r} exists in multiple tunes; specify tune=...")
        return spline_id

    def get_spline_tune(self, spline_id):
        tune_bytes = _lib.gpu_spline_get_tune(self._handle, spline_id)
        return tune_bytes.decode("utf-8") if tune_bytes else ""

    def get_spline_name(self, spline_id):
        name_bytes = _lib.gpu_spline_get_name(self._handle, spline_id)
        return name_bytes.decode("utf-8") if name_bytes else ""

    def evaluate(self, energies, spline_ids, use_gpu=True):
        """
        Evaluate cross-sections for arrays of energies and spline_ids.
        Parameters:
            energies: np.ndarray (float64)
            spline_ids: np.ndarray (int32)
            use_gpu: bool (prefer GPU when initialized; otherwise OpenMP CPU)
        Returns:
            np.ndarray (float64) cross sections
        """
        energies = np.ascontiguousarray(energies, dtype=np.float64)
        spline_ids = np.ascontiguousarray(spline_ids, dtype=np.int32)

        if energies.shape != spline_ids.shape:
            raise ValueError(f"Shape mismatch: energies {energies.shape} vs spline_ids {spline_ids.shape}")

        n = energies.size
        results = np.empty(n, dtype=np.float64)

        p_energies = energies.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_spline_ids = spline_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        p_results = results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        if use_gpu and self._gpu_ready:
            _lib.gpu_spline_eval_batch_gpu(self._handle, p_energies, p_spline_ids, p_results, n)
        else:
            _lib.gpu_spline_eval_batch_cpu(self._handle, p_energies, p_spline_ids, p_results, n)

        return results
