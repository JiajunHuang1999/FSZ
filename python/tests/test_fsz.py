"""Tests for the FSZ Python bindings.

A plain script, so it needs nothing beyond numpy and the bindings themselves.
Every case prints [PASS] or [FAIL]; the exit status is 0 only when all of them
pass. Run it from the repository root:

    python python/tests/test_fsz.py
"""

import os
import shutil
import struct
import sys
import tempfile
import traceback

import numpy as np

try:
    import fsz
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import fsz


SHAPE = (97, 113, 131)
ABS_EB = 1e-3
REL_EB = 1e-3
TOLERANCE = 1.01  # the slack the library itself allows on the bound

_CASES = []


def case(name):
    """Register a test function under a readable name."""
    def register(fn):
        _CASES.append((name, fn))
        return fn
    return register


def smooth_field(shape, dtype=np.float32):
    """A smooth field, the kind of data an error-bounded compressor is for."""
    dtype = np.dtype(dtype)
    axes = [np.linspace(0.0, 2.0 * np.pi, n, dtype=dtype) for n in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    z, y, x = grids
    field = (np.sin(x) * np.cos(y)
             + 0.5 * np.sin(2.0 * y + z)
             + 0.25 * np.cos(x + 3.0 * z))
    return np.ascontiguousarray(field, dtype=dtype)


def relative_bound(arr, eb_rel):
    """The absolute bound a relative one works out to, computed as the library does."""
    value_range = np.float32(arr.max() - arr.min())
    return np.float32(np.float64(value_range) * float(eb_rel))


def relative_bound_f64(arr, eb_rel):
    """The same for float64 data, where range and scaling both stay in double."""
    return float(arr.max() - arr.min()) * float(eb_rel)


def header_flags(blob):
    """The flags word of a container header."""
    return struct.unpack(fsz.HEADER_FORMAT, blob[:fsz.HEADER_SIZE])[3]


def header_eb_f64(blob):
    """The float64 error bound a container records, read at full precision."""
    return struct.unpack("<d", blob[fsz.EB_OFFSET:fsz.EB_OFFSET + 8])[0]


def expect_raises(exc_types, fn, *args, **kwargs):
    """Assert that calling fn raises one of exc_types, and return the exception."""
    try:
        fn(*args, **kwargs)
    except exc_types as exc:
        return exc
    except Exception as exc:  # noqa: BLE001 - a wrong exception type is a failure
        raise AssertionError(
            "expected %s, got %s: %s" % (exc_types, type(exc).__name__, exc))
    raise AssertionError("expected %s, but the call returned normally" % (exc_types,))


_FIELD = None
_FIELD_F64 = None


def field():
    """The shared test field, built once."""
    global _FIELD
    if _FIELD is None:
        _FIELD = smooth_field(SHAPE)
    return _FIELD


def field_f64():
    """The same field in double precision, built once."""
    global _FIELD_F64
    if _FIELD_F64 is None:
        _FIELD_F64 = smooth_field(SHAPE, dtype=np.float64)
    return _FIELD_F64


@case("abs-mode round trip on a smooth %s field, abs 1e-3" % (SHAPE,))
def test_abs_roundtrip():
    original = field()
    blob = fsz.compress(original, ABS_EB, mode="abs")
    back = fsz.decompress(blob)

    assert back.shape == original.shape, \
        "shape changed: %s became %s" % (original.shape, back.shape)
    assert back.dtype == np.float32, "dtype changed: %s" % back.dtype

    nonfinite = int(np.count_nonzero(~np.isfinite(back)))
    assert nonfinite == 0, "%d non-finite values in the reconstruction" % nonfinite

    max_err = float(np.abs(original - back).max())
    limit = TOLERANCE * ABS_EB
    assert max_err <= limit, "max error %.6e exceeds %.6e" % (max_err, limit)
    print("       max error %.6e, bound %.6e, %d non-finite"
          % (max_err, ABS_EB, nonfinite))


@case("rel-mode round trip on the same field, rel 1e-3")
def test_rel_roundtrip():
    original = field()
    blob = fsz.compress(original, REL_EB, mode="rel")
    back = fsz.decompress(blob)

    assert back.shape == original.shape, \
        "shape changed: %s became %s" % (original.shape, back.shape)
    nonfinite = int(np.count_nonzero(~np.isfinite(back)))
    assert nonfinite == 0, "%d non-finite values in the reconstruction" % nonfinite

    expected = float(relative_bound(original, REL_EB))
    max_err = float(np.abs(original - back).max())
    limit = TOLERANCE * expected
    assert max_err <= limit, "max error %.6e exceeds %.6e" % (max_err, limit)

    stored = struct.unpack(fsz.HEADER_FORMAT, blob[:fsz.HEADER_SIZE])[7]
    assert stored == expected, \
        "header records bound %.9e, expected %.9e" % (stored, expected)
    print("       max error %.6e, bound %.6e (rel %g of range %.4f)"
          % (max_err, expected, REL_EB,
             float(np.float32(original.max() - original.min()))))


@case("float64 round trip on the same field in double, rel 1e-3")
def test_f64_roundtrip():
    original = field_f64()
    assert original.dtype == np.float64 and original.ndim == 3

    blob = fsz.compress(original, REL_EB, mode="rel")
    back = fsz.decompress(blob)

    assert back.shape == original.shape, \
        "shape changed: %s became %s" % (original.shape, back.shape)
    assert back.dtype == np.float64, \
        "a float64 array came back as %s" % back.dtype

    nonfinite = int(np.count_nonzero(~np.isfinite(back)))
    assert nonfinite == 0, "%d non-finite values in the reconstruction" % nonfinite

    expected = relative_bound_f64(original, REL_EB)
    max_err = float(np.abs(original - back).max())
    limit = TOLERANCE * expected
    assert max_err <= limit, "max error %.6e exceeds %.6e" % (max_err, limit)

    flags = header_flags(blob)
    assert flags & fsz.FLAG_F64, \
        "the header flags are 0x%x, without the float64 bit" % flags
    stored = header_eb_f64(blob)
    assert stored == expected, \
        "header records bound %.17g, expected %.17g" % (stored, expected)

    narrowed = float(np.float32(expected))
    if narrowed != expected:
        assert stored != narrowed, \
            "the header narrowed the bound to float32: %.17g" % stored
    print("       max error %.6e, bound %.17g (float32 would hold %.17g)"
          % (max_err, stored, narrowed))


@case("flat array round trip, n = 1000003, abs 1e-2")
def test_flat_roundtrip():
    n = 1000003
    t = np.linspace(0.0, 40.0 * np.pi, n, dtype=np.float32)
    original = np.ascontiguousarray(
        np.sin(t) + 0.3 * np.cos(3.0 * t), dtype=np.float32)
    assert original.ndim == 1 and original.size == n

    blob = fsz.compress(original, 1e-2, mode="abs")
    back = fsz.decompress(blob)

    assert back.shape == (n,), "shape changed: %s" % (back.shape,)
    nonfinite = int(np.count_nonzero(~np.isfinite(back)))
    assert nonfinite == 0, "%d non-finite values in the reconstruction" % nonfinite

    max_err = float(np.abs(original - back).max())
    limit = TOLERANCE * 1e-2
    assert max_err <= limit, "max error %.6e exceeds %.6e" % (max_err, limit)
    print("       n = %d (odd), max error %.6e, bound %.6e"
          % (n, max_err, 1e-2))


@case("compress_file / decompress_file cycle with a recorded shape")
def test_file_cycle():
    original = field()
    workdir = tempfile.mkdtemp(prefix="fsz_test_")
    try:
        raw_path = os.path.join(workdir, "field.f32")
        cmp_path = os.path.join(workdir, "field.fsz")
        out_path = os.path.join(workdir, "field.out.f32")

        original.tofile(raw_path)
        info = fsz.compress_file(raw_path, cmp_path, ABS_EB, mode="abs", dims=SHAPE)

        assert info["n"] == original.size, \
            "compress_file counted %d elements, expected %d" % (info["n"], original.size)
        assert info["cmp_size"] == os.path.getsize(cmp_path) - fsz.HEADER_SIZE, \
            "cmp_size %d disagrees with the file size" % info["cmp_size"]
        assert info["cr"] > 1.0, "compression ratio %.3f is not above 1" % info["cr"]

        back = fsz.decompress_file(cmp_path, out_path)
        assert back.shape == SHAPE, \
            "decompress_file returned shape %s, expected %s" % (back.shape, SHAPE)

        with open(out_path, "rb") as f:
            written = f.read()
        assert written == back.tobytes(), \
            "the %d bytes written to disk differ from the in-memory result" % len(written)

        max_err = float(np.abs(original - back).max())
        assert max_err <= TOLERANCE * ABS_EB, \
            "max error %.6e exceeds %.6e" % (max_err, TOLERANCE * ABS_EB)

        print("       n = %d, cmp_size = %d, cr = %.3f, max error %.6e"
              % (info["n"], info["cmp_size"], info["cr"], max_err))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@case("float64 compress_file / decompress_file cycle on a .d64")
def test_f64_file_cycle():
    original = field_f64()
    workdir = tempfile.mkdtemp(prefix="fsz_test_")
    try:
        raw_path = os.path.join(workdir, "field.d64")
        cmp_path = os.path.join(workdir, "field.fsz")
        out_path = os.path.join(workdir, "field.out.d64")

        original.tofile(raw_path)
        assert os.path.getsize(raw_path) == original.size * 8, \
            "the raw file is not 8 bytes per value"

        info = fsz.compress_file(raw_path, cmp_path, ABS_EB, mode="abs",
                                 dims=SHAPE, dtype=np.float64)

        assert info["n"] == original.size, \
            "compress_file counted %d elements, expected %d" % (info["n"], original.size)
        assert info["cmp_size"] == os.path.getsize(cmp_path) - fsz.HEADER_SIZE, \
            "cmp_size %d disagrees with the file size" % info["cmp_size"]
        expected_cr = float(original.nbytes) / float(info["cmp_size"])
        assert abs(info["cr"] - expected_cr) < 1e-9, \
            "cr %.6f is not the float64 ratio %.6f" % (info["cr"], expected_cr)

        with open(cmp_path, "rb") as f:
            head = f.read(fsz.HEADER_SIZE)
        assert header_flags(head) & fsz.FLAG_F64, \
            "the container on disk does not flag float64 data"

        back = fsz.decompress_file(cmp_path, out_path)
        assert back.dtype == np.float64, \
            "decompress_file returned dtype %s, expected float64" % back.dtype
        assert back.shape == SHAPE, \
            "decompress_file returned shape %s, expected %s" % (back.shape, SHAPE)

        with open(out_path, "rb") as f:
            written = f.read()
        assert written == back.tobytes(), \
            "the %d bytes written to disk differ from the in-memory result" % len(written)

        max_err = float(np.abs(original - back).max())
        assert max_err <= TOLERANCE * ABS_EB, \
            "max error %.6e exceeds %.6e" % (max_err, TOLERANCE * ABS_EB)

        print("       n = %d, cmp_size = %d, cr = %.3f, max error %.6e"
              % (info["n"], info["cmp_size"], info["cr"], max_err))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@case("a float64 blob rebuilds as float64 and a float32 blob as float32")
def test_mixed_dtypes():
    f32 = field()
    f64 = field_f64()

    blob32 = fsz.compress(f32, ABS_EB, mode="abs")
    blob64 = fsz.compress(f64, ABS_EB, mode="abs")

    assert header_flags(blob32) & fsz.FLAG_F64 == 0, \
        "a float32 container has the float64 flag set"
    assert header_flags(blob64) & fsz.FLAG_F64 == fsz.FLAG_F64, \
        "a float64 container does not have the float64 flag set"
    assert blob32 != blob64, "the two element types produced the same blob"

    back32 = fsz.decompress(blob32)
    back64 = fsz.decompress(blob64)

    assert back32.dtype == np.float32, "float32 blob came back as %s" % back32.dtype
    assert back64.dtype == np.float64, "float64 blob came back as %s" % back64.dtype
    assert back32.shape == SHAPE and back64.shape == SHAPE, \
        "shapes changed: %s and %s" % (back32.shape, back64.shape)

    for name, original, back in (("float32", f32, back32), ("float64", f64, back64)):
        nonfinite = int(np.count_nonzero(~np.isfinite(back)))
        assert nonfinite == 0, "%d non-finite values in the %s result" % (nonfinite, name)
        max_err = float(np.abs(original - back).max())
        assert max_err <= TOLERANCE * ABS_EB, \
            "%s max error %.6e exceeds %.6e" % (name, max_err, TOLERANCE * ABS_EB)

    payload64 = blob64[fsz.HEADER_SIZE:]
    from_raw = fsz.decompress_raw(payload64, f64.size, ABS_EB, dtype=np.float64)
    assert from_raw.dtype == np.float64, "decompress_raw returned %s" % from_raw.dtype
    assert from_raw.tobytes() == back64.ravel().tobytes(), \
        "the raw and container paths disagree on the float64 stream"

    print("       float32 blob %d bytes, float64 blob %d bytes, both within bound"
          % (len(blob32), len(blob64)))


@case("compress_raw / decompress_raw agree with the container path")
def test_raw_consistency():
    original = field()
    blob = fsz.compress(original, ABS_EB, mode="abs")
    payload = fsz.compress_raw(original, np.float32(ABS_EB))

    assert payload == blob[fsz.HEADER_SIZE:], \
        ("the bare bitstream is %d bytes and the container payload is %d; "
         "they must be identical"
         % (len(payload), len(blob) - fsz.HEADER_SIZE))

    from_raw = fsz.decompress_raw(payload, original.size, np.float32(ABS_EB))
    from_blob = fsz.decompress(blob).ravel()

    assert from_raw.shape == from_blob.shape, \
        "shapes differ: %s and %s" % (from_raw.shape, from_blob.shape)
    assert from_raw.tobytes() == from_blob.tobytes(), \
        "the two reconstructions differ bit for bit"
    print("       payload %d bytes, identical through both paths" % len(payload))


@case("error cases are reported, not crashed on")
def test_error_cases():
    exc = expect_raises(TypeError, fsz.compress,
                        np.zeros((8, 8), dtype=np.int32), 1e-3)
    print("       int32 input    -> TypeError: %s" % str(exc).split(".")[0])

    exc = expect_raises(ValueError, fsz.compress,
                        np.zeros((2, 3, 4, 5), dtype=np.float32), 1e-3)
    print("       4 axes         -> ValueError: %s" % str(exc).split(".")[0])

    exc = expect_raises(ValueError, fsz.compress,
                        np.full((32, 32), 2.5, dtype=np.float32), 1e-3, "rel")
    print("       constant + rel -> ValueError: %s" % str(exc).split(",")[0])

    exc = expect_raises(ValueError, fsz.compress,
                        np.zeros((8, 8), dtype=np.float32), 0.0, "abs")
    print("       eb 0           -> ValueError: %s" % str(exc).split(",")[0])

    exc = expect_raises(ValueError, fsz.compress,
                        np.zeros((8, 8), dtype=np.float32), 1e-3, "sideways")
    print("       unknown mode   -> ValueError: %s" % str(exc).split(",")[0])

    blob = fsz.compress(np.asarray(np.arange(4096), dtype=np.float32), 1e-2, "abs")

    exc = expect_raises((ValueError, RuntimeError), fsz.decompress,
                        blob[:fsz.HEADER_SIZE - 20])
    print("       stub blob      -> %s" % type(exc).__name__)

    exc = expect_raises((ValueError, RuntimeError), fsz.decompress,
                        blob[:fsz.HEADER_SIZE + (len(blob) - fsz.HEADER_SIZE) // 2])
    print("       cut payload    -> %s: %s" % (type(exc).__name__, str(exc).split(";")[0]))

    exc = expect_raises((ValueError, RuntimeError), fsz.decompress,
                        b"NOPE" + blob[4:])
    print("       bad magic      -> %s" % type(exc).__name__)

    assert fsz.decompress(blob).shape == (4096,), "a good blob stopped working"


@case("compression ratio on smooth data clears 5x")
def test_compression_ratio():
    original = field()
    for mode, eb in (("abs", ABS_EB), ("rel", REL_EB)):
        blob = fsz.compress(original, eb, mode=mode)
        payload_size = len(blob) - fsz.HEADER_SIZE
        cr = (original.size * 4.0) / payload_size
        print("       %s %g: %d bytes -> %d bytes, cr = %.3f"
              % (mode, eb, original.nbytes, payload_size, cr))
        assert cr > 5.0, \
            "compression ratio %.3f in %s mode does not clear 5" % (cr, mode)


def main():
    print("FSZ Python bindings %s, library %s"
          % (fsz.__version__, fsz.library_version()))
    print("%d cases" % len(_CASES))
    print()

    failures = 0
    for name, fn in _CASES:
        print("  %s" % name)
        try:
            fn()
        except Exception:  # noqa: BLE001 - the report is the point
            failures += 1
            traceback.print_exc()
            print("[FAIL] %s" % name)
        else:
            print("[PASS] %s" % name)
        print()

    if failures:
        print("%d of %d cases FAILED" % (failures, len(_CASES)))
        return 1
    print("all %d cases passed" % len(_CASES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
