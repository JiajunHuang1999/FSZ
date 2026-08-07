#!/usr/bin/env python3
"""A short tour of the FSZ Python bindings.

Builds a smooth 128 x 128 x 128 float32 field, compresses it with a relative
error bound of 1e-3, decompresses it, and reports what that cost and what it
kept. Run it straight from the repository root:

    python python/example.py
"""

import os
import sys
import time

import numpy as np

try:
    import fsz
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fsz


def smooth_field(n=128):
    """A smooth, wavy scalar field of shape (n, n, n), the kind of data FSZ is for."""
    axis = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    field = (np.sin(x) * np.cos(y)
             + 0.5 * np.sin(2.0 * y + z)
             + 0.25 * np.cos(x + 3.0 * z))
    return np.ascontiguousarray(field, dtype=np.float32)


def main():
    print("FSZ Python bindings %s (library %s)"
          % (fsz.__version__, fsz.library_version()))
    print()

    field = smooth_field(128)
    eb_rel = 1e-3
    print("field       : shape %s, dtype %s, %.1f MiB"
          % (field.shape, field.dtype, field.nbytes / 1024.0 / 1024.0))
    print("value range : [%.4f, %.4f]" % (field.min(), field.max()))
    print("error bound : rel %g" % eb_rel)
    print()

    t0 = time.time()
    blob = fsz.compress(field, eb_rel, mode="rel")
    t_compress = time.time() - t0

    t0 = time.time()
    back = fsz.decompress(blob)
    t_decompress = time.time() - t0

    value_range = np.float32(field.max() - field.min())
    abs_eb = np.float32(np.float64(value_range) * eb_rel)

    max_err = float(np.abs(field - back).max())
    ratio = field.nbytes / float(len(blob))

    tolerance = 1.01
    within = max_err <= tolerance * float(abs_eb)

    print("compressed  : %d bytes (%d header + %d bitstream) in %.3f s"
          % (len(blob), fsz.HEADER_SIZE, len(blob) - fsz.HEADER_SIZE, t_compress))
    print("ratio       : %.2fx smaller than the raw float32 data" % ratio)
    print("decompressed: %.3f s" % t_decompress)
    print()
    print("max error   : %.6e" % max_err)
    print("bound       : %.6e (rel %g of the %.4f value range)"
          % (abs_eb, eb_rel, value_range))
    print("within bound: %s (%.6f of the bound, tolerance %.2f)"
          % ("yes" if within else "NO", max_err / float(abs_eb), tolerance))
    print("all finite  : %s" % ("yes" if np.all(np.isfinite(back)) else "NO"))
    print("shape kept  : %s (%s -> %s)"
          % ("yes" if back.shape == field.shape else "NO", field.shape, back.shape))
    print("dtype kept  : %s (%s)"
          % ("yes" if back.dtype == field.dtype else "NO", back.dtype))
    print()

    print("A blob carries its own shape and bound, so a file round trip needs")
    print("no extra bookkeeping:")
    print("    fsz.compress_file(src, dst, 1e-3, dims=(128, 128, 128))")
    print("    field = fsz.decompress_file(dst)")

    return 0 if (within and back.shape == field.shape) else 1


if __name__ == "__main__":
    sys.exit(main())
