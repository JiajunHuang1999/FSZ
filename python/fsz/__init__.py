"""Python bindings for the FSZ GPU error-bounded lossy compressor.

FSZ compresses float32 and float64 arrays on the GPU under an absolute
error bound: every reconstructed value lands within that bound of the original,
up to rounding on the last bit. The library checks itself against the bound
with one percent of slack, and so should you. These bindings talk to the shared
library through ctypes and numpy, with no other dependencies.

    import numpy as np, fsz
    field = np.sin(np.linspace(0, 8, 64 * 64 * 64, dtype=np.float32)).reshape(64, 64, 64)
    blob = fsz.compress(field, 1e-3, mode="rel")     # bytes, self-describing
    back = fsz.decompress(blob)                      # float32, original shape
    print(field.nbytes / len(blob), np.abs(field - back).max())

The element type follows the input array: a float64 array compresses as
float64 and comes back as float64, with the error bound kept in full double
precision. The two are separate stream formats, and the container records
which one it holds, so :func:`decompress` needs nothing but the blob.

:func:`compress` returns a self-describing container: a 56 byte header holding
the shape, the element type, the absolute error bound and the bitstream length,
followed by the bitstream exactly as the library produced it. The layout is the
one the ``fsz`` command line tool reads and writes, so blobs move between the
two freely.

Arrays of any dimensionality are accepted, up to the three dimensions that the
header records. Anything wider should be reshaped by the caller.

The shared library is located at import time, in this order:

1. the path in the ``FSZ_LIB`` environment variable,
2. ``build/libfsz.so`` beside the package in a repository checkout,
3. whatever ``ctypes.util.find_library("fsz")`` turns up.
"""

import ctypes
import ctypes.util
import os
import struct

import numpy as np

__version__ = "1.0.0"

__all__ = [
    "compress",
    "decompress",
    "compress_file",
    "decompress_file",
    "compress_raw",
    "decompress_raw",
    "library_version",
    "num_tiles",
    "max_compressed_bytes",
    "HEADER_SIZE",
    "FORMAT_VERSION",
    "FLAG_F64",
    "EB_OFFSET",
    "__version__",
]

HEADER_FORMAT = "<4sIII3QfIQ"
HEADER_SIZE = 56
FORMAT_VERSION = 1
FLAG_F64 = 0x1
EB_OFFSET = 40
MAX_HEADER_DIMS = 3

MAGIC = b"FSZ1"

assert struct.calcsize(HEADER_FORMAT) == 56, (
    "the FSZ header layout must pack to exactly 56 bytes, got %d"
    % struct.calcsize(HEADER_FORMAT)
)



def _repo_build_candidate():
    """Path to build/libfsz.so as laid out in a repository checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "build", "libfsz.so"))


def _load_library():
    """Find and open libfsz.so, or raise ImportError naming every place tried."""
    attempts = []

    env_path = os.environ.get("FSZ_LIB")
    if env_path:
        attempts.append(("FSZ_LIB environment variable", env_path))
    else:
        attempts.append(("FSZ_LIB environment variable", None))

    bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfsz.so")
    attempts.append(("library built into the installed package",
                     bundled if os.path.exists(bundled) else None))
    attempts.append(("build directory beside the package", _repo_build_candidate()))
    attempts.append(("system search path", ctypes.util.find_library("fsz")))

    notes = []
    for source, path in attempts:
        if not path:
            notes.append("  %s: not set" % source)
            continue
        if os.sep in path and not os.path.exists(path):
            notes.append("  %s: %s (no such file)" % (source, path))
            continue
        try:
            return ctypes.CDLL(path)
        except OSError as exc:
            notes.append("  %s: %s (%s)" % (source, path, exc))

    raise ImportError(
        "the FSZ shared library could not be loaded. Tried, in order:\n"
        + "\n".join(notes)
        + "\n\nBuild it with\n"
        "  cmake -S <repo> -B <repo>/build -DFSZ_BUILD_SHARED=ON\n"
        "  cmake --build <repo>/build --target fsz_shared -j4\n"
        "or point FSZ_LIB at an existing libfsz.so. If the library exists but "
        "fails to load, check that the CUDA runtime (libcudart) is on "
        "LD_LIBRARY_PATH."
    )


_lib = _load_library()

_c_size_p = ctypes.POINTER(ctypes.c_size_t)
_c_float_p = ctypes.POINTER(ctypes.c_float)
_c_double_p = ctypes.POINTER(ctypes.c_double)
_c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)

_lib.fsz_version.argtypes = []
_lib.fsz_version.restype = ctypes.c_char_p

_lib.fsz_status_string.argtypes = [ctypes.c_int]
_lib.fsz_status_string.restype = ctypes.c_char_p

_lib.fsz_num_tiles.argtypes = [ctypes.c_size_t]
_lib.fsz_num_tiles.restype = ctypes.c_size_t

_lib.fsz_max_compressed_bytes.argtypes = [ctypes.c_size_t]
_lib.fsz_max_compressed_bytes.restype = ctypes.c_size_t

_lib.fsz_compress_hostptr.argtypes = [
    _c_float_p,      # in
    _c_ubyte_p,      # cmp
    ctypes.c_size_t,  # n_elements
    ctypes.c_float,  # abs_error_bound
    _c_size_p,       # out_cmp_size
]
_lib.fsz_compress_hostptr.restype = ctypes.c_int

_lib.fsz_decompress_hostptr.argtypes = [
    _c_float_p,      # out
    _c_ubyte_p,      # cmp
    ctypes.c_size_t,  # cmp_size
    ctypes.c_size_t,  # n_elements
    ctypes.c_float,  # abs_error_bound
]
_lib.fsz_decompress_hostptr.restype = ctypes.c_int

_lib.fsz_compress_hostptr_f64.argtypes = [
    _c_double_p,     # in
    _c_ubyte_p,      # cmp
    ctypes.c_size_t,  # n_elements
    ctypes.c_double,  # abs_error_bound
    _c_size_p,       # out_cmp_size
]
_lib.fsz_compress_hostptr_f64.restype = ctypes.c_int

_lib.fsz_decompress_hostptr_f64.argtypes = [
    _c_double_p,     # out
    _c_ubyte_p,      # cmp
    ctypes.c_size_t,  # cmp_size
    ctypes.c_size_t,  # n_elements
    ctypes.c_double,  # abs_error_bound
]
_lib.fsz_decompress_hostptr_f64.restype = ctypes.c_int

FSZ_STATUS_OK = 0


def _status_text(status):
    """Human readable text for a library status code."""
    text = _lib.fsz_status_string(ctypes.c_int(int(status)))
    if text is None:
        return "unknown status"
    return text.decode("utf-8", "replace")


def _check(status, what):
    """Raise RuntimeError unless ``status`` is the library's OK code."""
    if int(status) != FSZ_STATUS_OK:
        raise RuntimeError(
            "%s failed: %s (status %d)" % (what, _status_text(status), int(status))
        )




def _as_input_array(arr):
    """Return ``arr`` as a C-contiguous float32 or float64 array of 1 to 3 axes.

    The dtype is never converted: it selects the stream format, so an integer
    or float16 array is reported instead of being quietly cast.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if arr.dtype != np.float32 and arr.dtype != np.float64:
        raise TypeError(
            "FSZ compresses float32 and float64 data, but the input array has "
            "dtype %s. Convert it explicitly, for example "
            "arr.astype(np.float32), so the precision loss is your choice and "
            "not a silent cast." % arr.dtype
        )

    if arr.ndim == 0:
        raise ValueError(
            "the input is a scalar; pass an array with at least one axis."
        )
    if arr.ndim > MAX_HEADER_DIMS:
        raise ValueError(
            "the input array has %d dimensions, and the FSZ container records at "
            "most %d. Reshape it first, for example "
            "arr.reshape(arr.shape[0], arr.shape[1], -1), and reshape the result "
            "of decompress() back afterwards." % (arr.ndim, MAX_HEADER_DIMS)
        )
    if arr.size == 0:
        raise ValueError("the input array is empty; there is nothing to compress.")

    return np.ascontiguousarray(arr)


def _resolve_abs_eb(arr, eb, mode):
    """Turn a user error bound and mode into the absolute bound the library wants.

    ``mode="abs"`` takes ``eb`` as it stands. ``mode="rel"`` reads it as a
    fraction of the value range of ``arr``, matching the command line tool. For
    a float32 array the range is taken in float32, scaled in float64, then
    rounded back to float32; for a float64 array range and scaling both stay in
    float64, so the bound keeps full double precision. The bound comes back in
    the precision of the data, and the container records it that way.
    """
    eb = float(eb)
    if not (eb > 0.0):
        raise ValueError(
            "the error bound must be greater than 0, got %r." % eb
        )

    f64 = arr.dtype == np.float64

    if mode == "abs":
        abs_eb = float(eb) if f64 else np.float32(eb)
    elif mode == "rel":
        rng = float(arr.max() - arr.min()) if f64 \
            else np.float32(arr.max() - arr.min())
        if not (rng > 0.0):
            raise ValueError(
                "the value range of the input is %r, so a relative error bound "
                "is undefined. Constant data (and data holding non-finite "
                "values) needs mode=\"abs\" with an explicit absolute bound."
                % float(rng)
            )
        abs_eb = float(rng) * eb if f64 else np.float32(np.float64(rng) * eb)
    else:
        raise ValueError(
            "mode must be \"rel\" or \"abs\", got %r." % (mode,)
        )

    if not (abs_eb > 0.0):
        raise ValueError(
            "the absolute error bound works out to %r, which %s cannot "
            "represent as a positive number. Use a larger error bound."
            % (float(abs_eb), "float64" if f64 else "float32")
        )
    return abs_eb


def _as_byte_array(payload, what):
    """View bytes, bytearray, memoryview or a uint8 ndarray as a flat uint8 array."""
    if isinstance(payload, np.ndarray):
        if payload.dtype != np.uint8:
            raise TypeError(
                "%s must be a uint8 array when given as a numpy array, got dtype "
                "%s." % (what, payload.dtype)
            )
        return np.ascontiguousarray(payload).reshape(-1)
    try:
        return np.frombuffer(payload, dtype=np.uint8)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "%s must be bytes, bytearray, memoryview or a uint8 numpy array, got "
            "%s (%s)." % (what, type(payload).__name__, exc)
        )




def _pack_header(shape, abs_eb, payload_size, f64=False):
    """Build the 56 byte container header for an array of the given shape.

    ``f64`` sets the float64 flag and writes the bound as one double across the
    eight bytes at ``EB_OFFSET``, in place of the float32 bound and the zero
    word that follows it.
    """
    dims = tuple(int(d) for d in shape)
    padded = dims + (1,) * (MAX_HEADER_DIMS - len(dims))
    head = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        FORMAT_VERSION,
        len(dims),
        FLAG_F64 if f64 else 0,
        padded[0],
        padded[1],
        padded[2],
        0.0 if f64 else float(abs_eb),
        0,                  # reserved
        int(payload_size),
    )
    if f64:
        head = (head[:EB_OFFSET]
                + struct.pack("<d", float(abs_eb))
                + head[EB_OFFSET + 8:])
    return head


def _parse_header(data):
    """Validate and unpack a container header.

    ``data`` is the whole blob as a uint8 array. Returns
    ``(shape, abs_eb, payload_size, f64)``, where ``f64`` says whether the
    payload holds float64 values and ``abs_eb`` is read at that precision.
    """
    if data.size < HEADER_SIZE:
        raise ValueError(
            "this is not an FSZ container: %d bytes is shorter than the %d byte "
            "header." % (data.size, HEADER_SIZE)
        )

    (magic, version, ndims, flags, d0, d1, d2, abs_eb, _reserved,
     payload_size) = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE].tobytes())

    if magic != MAGIC:
        raise ValueError(
            "this is not an FSZ container: expected the magic %r, found %r. A "
            "bare bitstream has no header, so decompress it with "
            "decompress_raw(payload, n, abs_eb)." % (MAGIC, magic)
        )
    if version != FORMAT_VERSION:
        raise ValueError(
            "the container is version %d and this build reads version %d."
            % (version, FORMAT_VERSION)
        )
    if ndims < 1 or ndims > MAX_HEADER_DIMS:
        raise ValueError(
            "the container records %d dimensions, which is outside the supported "
            "range of 1 to %d." % (ndims, MAX_HEADER_DIMS)
        )

    f64 = bool(flags & FLAG_F64)
    if f64:
        abs_eb, = struct.unpack("<d", data[EB_OFFSET:EB_OFFSET + 8].tobytes())

    shape = (d0, d1, d2)[:ndims]
    n = 1
    for d in shape:
        n *= d
    if n == 0:
        raise ValueError(
            "the container records the shape %r, which holds no elements." % (shape,)
        )
    if not (abs_eb > 0.0):
        raise ValueError(
            "the container records an absolute error bound of %r, which is not "
            "greater than 0." % abs_eb
        )

    available = data.size - HEADER_SIZE
    if payload_size == 0:
        raise ValueError("the container holds an empty bitstream.")
    if payload_size > available:
        raise ValueError(
            "the container claims a %d byte bitstream but only %d bytes follow "
            "the header; the blob looks truncated."
            % (payload_size, available)
        )

    return shape, abs_eb, int(payload_size), f64




def library_version():
    """Return the version string reported by the shared library.

    >>> library_version()
    '1.0.0'
    """
    return _lib.fsz_version().decode("utf-8", "replace")


def num_tiles(n):
    """Return the number of tiles the library splits ``n`` elements into."""
    return int(_lib.fsz_num_tiles(ctypes.c_size_t(int(n))))


def max_compressed_bytes(n):
    """Return the worst case compressed size in bytes for ``n`` elements."""
    return int(_lib.fsz_max_compressed_bytes(ctypes.c_size_t(int(n))))


def compress_raw(arr, abs_eb):
    """Compress ``arr`` and return the bare bitstream, with no header.

    This is the advanced entry point. The returned bytes carry no record of the
    shape, of the element type or of the error bound, so keep all three
    yourself: decompress_raw() needs the element count, the dtype and the same
    absolute bound to reconstruct the data. Most callers want :func:`compress`
    instead.

    Args:
        arr: float32 or float64 numpy array of 1 to 3 dimensions, whose dtype
            picks the stream format. Made C-contiguous if it is not already.
        abs_eb: absolute error bound, greater than 0. Every reconstructed value
            lands within this distance of the original, up to rounding on the
            last bit. Taken at the precision of ``arr``.

    Returns:
        bytes: the compressed bitstream.

    Raises:
        TypeError: if ``arr`` is neither float32 nor float64.
        ValueError: if the shape or the bound is unusable.
        RuntimeError: if the library reports a failure.
    """
    arr = _as_input_array(arr)
    f64 = arr.dtype == np.float64
    abs_eb = float(abs_eb) if f64 else np.float32(abs_eb)
    if not (abs_eb > 0.0):
        raise ValueError(
            "the absolute error bound must be greater than 0, got %r."
            % float(abs_eb)
        )

    n = arr.size
    buf = np.empty(max_compressed_bytes(n), dtype=np.uint8)
    written = ctypes.c_size_t(0)

    if f64:
        status = _lib.fsz_compress_hostptr_f64(
            arr.ctypes.data_as(_c_double_p),
            buf.ctypes.data_as(_c_ubyte_p),
            ctypes.c_size_t(n),
            ctypes.c_double(float(abs_eb)),
            ctypes.byref(written),
        )
        _check(status, "fsz_compress_hostptr_f64")
    else:
        status = _lib.fsz_compress_hostptr(
            arr.ctypes.data_as(_c_float_p),
            buf.ctypes.data_as(_c_ubyte_p),
            ctypes.c_size_t(n),
            ctypes.c_float(float(abs_eb)),
            ctypes.byref(written),
        )
        _check(status, "fsz_compress_hostptr")

    return buf[: written.value].tobytes()


def decompress_raw(payload, n, abs_eb, dtype=np.float32):
    """Reconstruct ``n`` elements from a bare bitstream.

    The counterpart of :func:`compress_raw`. The element count, the element
    type and the absolute error bound must all be the ones used at compression;
    the bitstream carries none of them.

    Args:
        payload: the bitstream, as bytes, bytearray, memoryview or a uint8
            numpy array.
        n: number of elements to reconstruct.
        abs_eb: the absolute error bound used at compression.
        dtype: the element type the bitstream was written from, ``float32``
            (the default) or ``float64``. A float64 stream read as float32, or
            the other way round, is not detected: the two formats simply differ.

    Returns:
        numpy.ndarray: flat array of ``n`` values in ``dtype``.

    Raises:
        TypeError: if ``payload`` is not a byte buffer, or ``dtype`` is neither
            float32 nor float64.
        ValueError: if ``n`` or the bound is unusable.
        RuntimeError: if the library reports a failure.
    """
    data = _as_byte_array(payload, "the compressed payload")
    n = int(n)
    if n <= 0:
        raise ValueError("the element count must be greater than 0, got %d." % n)
    if data.size == 0:
        raise ValueError("the compressed payload is empty.")

    dtype = np.dtype(dtype)
    if dtype != np.float32 and dtype != np.float64:
        raise TypeError(
            "FSZ reconstructs float32 and float64 data, but dtype is %s." % dtype
        )
    f64 = dtype == np.float64

    abs_eb = float(abs_eb) if f64 else np.float32(abs_eb)
    if not (abs_eb > 0.0):
        raise ValueError(
            "the absolute error bound must be greater than 0, got %r."
            % float(abs_eb)
        )

    out = np.empty(n, dtype=dtype)
    if f64:
        status = _lib.fsz_decompress_hostptr_f64(
            out.ctypes.data_as(_c_double_p),
            data.ctypes.data_as(_c_ubyte_p),
            ctypes.c_size_t(int(data.size)),
            ctypes.c_size_t(n),
            ctypes.c_double(float(abs_eb)),
        )
        _check(status, "fsz_decompress_hostptr_f64")
    else:
        status = _lib.fsz_decompress_hostptr(
            out.ctypes.data_as(_c_float_p),
            data.ctypes.data_as(_c_ubyte_p),
            ctypes.c_size_t(int(data.size)),
            ctypes.c_size_t(n),
            ctypes.c_float(float(abs_eb)),
        )
        _check(status, "fsz_decompress_hostptr")
    return out


def compress(arr, eb, mode="rel"):
    """Compress ``arr`` and return a self-describing blob.

    The blob is a 56 byte header followed by the compressed bitstream. The
    header records the shape, the element type, the absolute error bound and
    the bitstream length, which is everything :func:`decompress` needs, and it
    is the same container the ``fsz`` command line tool reads.

    Args:
        arr: float32 or float64 numpy array of 1 to 3 dimensions. A
            non-contiguous array is copied into contiguous order first. The
            dtype is never converted: it picks the stream format, and any other
            dtype raises TypeError rather than being cast behind your back.
        eb: the error bound, greater than 0. Read according to ``mode``.
        mode: ``"rel"`` (the default) reads ``eb`` as a fraction of the value
            range ``arr.max() - arr.min()``, so 1e-3 keeps every value within
            0.1 percent of the range. ``"abs"`` reads ``eb`` as the absolute
            bound itself.

    Returns:
        bytes: header plus bitstream.

    Raises:
        TypeError: if ``arr`` is neither float32 nor float64.
        ValueError: for an unusable shape or bound, an unknown mode, or
            constant data with ``mode="rel"``, where a relative bound has no
            meaning.
        RuntimeError: if the library reports a failure.

    >>> field = np.zeros((4, 4), dtype=np.float32)
    >>> field[2, 2] = 1.0
    >>> blob = compress(field, 1e-2, mode="rel")
    >>> decompress(blob).shape
    (4, 4)
    """
    arr = _as_input_array(arr)
    abs_eb = _resolve_abs_eb(arr, eb, mode)
    payload = compress_raw(arr, abs_eb)
    f64 = arr.dtype == np.float64
    return _pack_header(arr.shape, abs_eb, len(payload), f64) + payload


def decompress(blob):
    """Reconstruct the array held in a blob from :func:`compress`.

    Args:
        blob: the container, as bytes, bytearray, memoryview or a uint8 numpy
            array.

    Returns:
        numpy.ndarray: array with the shape and the element type recorded at
        compression, float32 or float64. A blob whose header records a single
        dimension yields shape ``(n,)``.

    Raises:
        TypeError: if ``blob`` is not a byte buffer.
        ValueError: if the magic, the version, the recorded shape or the
            bitstream length does not check out, which is also what a truncated
            blob looks like.
        RuntimeError: if the library reports a failure.
    """
    data = _as_byte_array(blob, "the compressed blob")
    shape, abs_eb, payload_size, f64 = _parse_header(data)

    n = 1
    for d in shape:
        n *= d

    payload = data[HEADER_SIZE:HEADER_SIZE + payload_size]
    if f64:
        out = decompress_raw(payload, n, float(abs_eb), np.float64)
    else:
        out = decompress_raw(payload, n, np.float32(abs_eb), np.float32)
    return out.reshape(shape)


def compress_file(src, dst, eb, mode="rel", dims=None, dtype=np.float32):
    """Compress a raw float32 or float64 file into an FSZ container on disk.

    ``src`` is read as a flat stream of values in ``dtype``, the layout that
    raw binary dumps and the ``fsz`` command line tool use. Pass ``dims`` to
    record the shape in the header so :func:`decompress_file` hands the array
    back in its original form.

    Args:
        src: path of the raw input file.
        dst: path of the container to write.
        eb: the error bound, greater than 0, read according to ``mode``.
        mode: ``"rel"`` or ``"abs"``, as in :func:`compress`.
        dims: optional shape tuple of up to 3 entries in C order, with the
            slowest-varying axis first. Its product must equal the number of
            values in the file. Left out, the array is treated as a flat run of
            values.
        dtype: element type of the raw file, ``float32`` (the default) or
            ``float64``. It sets both how ``src`` is read and which stream
            format the container holds.

    Returns:
        dict: ``n`` the element count, ``cmp_size`` the bitstream length in
        bytes, and ``cr`` the compression ratio, the size of the raw array
        divided by ``cmp_size``.

    Raises:
        TypeError: if ``dtype`` is neither float32 nor float64.
        ValueError: if the file holds no values, or if ``dims`` disagrees with
            the number of values it does hold.
    """
    dtype = np.dtype(dtype)
    if dtype != np.float32 and dtype != np.float64:
        raise TypeError(
            "FSZ compresses float32 and float64 data, but dtype is %s." % dtype
        )

    arr = np.fromfile(src, dtype=dtype)
    if arr.size == 0:
        raise ValueError(
            "%s holds no %s values; the file is empty or truncated."
            % (src, dtype.name)
        )

    if dims is not None:
        dims = tuple(int(d) for d in dims)
        wanted = 1
        for d in dims:
            wanted *= d
        if wanted != arr.size:
            raise ValueError(
                "dims %r describes %d elements but %s holds %d %s values."
                % (dims, wanted, src, arr.size, dtype.name)
            )
        arr = arr.reshape(dims)

    blob = compress(arr, eb, mode=mode)
    with open(dst, "wb") as f:
        f.write(blob)

    cmp_size = len(blob) - HEADER_SIZE
    return {
        "n": int(arr.size),
        "cmp_size": int(cmp_size),
        "cr": float(arr.nbytes) / float(cmp_size),
    }


def decompress_file(src, dst=None):
    """Reconstruct the array in an FSZ container file.

    Args:
        src: path of the container written by :func:`compress_file` or by the
            ``fsz`` command line tool.
        dst: optional path for the reconstructed data, written raw in C order,
            in the element type the container records.

    Returns:
        numpy.ndarray: array with the shape and the element type recorded in
        the container, float32 or float64.

    Raises:
        ValueError: if the file is not a readable FSZ container.
        RuntimeError: if the library reports a failure.
    """
    with open(src, "rb") as f:
        blob = f.read()

    arr = decompress(blob)
    if dst is not None:
        arr.tofile(dst)
    return arr
