# FSZ

FSZ is the fastest GPU lossy compressor in the SZ family, for float32 and
float64 arrays. It stands at the family's Pareto frontier: achieving
both extreme throughput and a high compression ratio simultaneously.

You set an error bound, absolute or relative. FSZ compresses at up to
1.4 TB/s and decompresses at up to 1.7 TB/s (measured on a B200 GPU with
the tool in this repository), and every reconstructed value is guaranteed
to stay within the bound. The data can be anything floating point:
simulation fields, sensor and instrument captures, checkpoints, machine
learning tensors.

FSZ is the compressor introduced in the SC'26 paper
*FSZ: Breaking the Prediction-Throughput Trade-off in GPU Lossy Compression*
([arXiv:2607.15413](https://arxiv.org/abs/2607.15413)), nominated for the
Best Paper Award at SC'26.

What you get:

- **A guaranteed bound.** Every reconstructed value stays within the
  absolute or relative error bound you set, on float32 and float64 alike,
  and the tool checks it on every run.
- **Nothing to tune.** FSZ adapts to the data on its own: no modes to
  select, no flags to sweep, and compression ratios hold up on smooth,
  sparse, and unstructured data alike.
- **Every interface.** The `fsz` binary plus C, C++, Python, and Fortran
  APIs. Python takes numpy arrays of any shape and hands them back with
  the shape restored; Fortran accepts `real(4)` and `real(8)` arrays up to
  rank four. `.fsz` files are self-describing and decompress with no
  further arguments; a bare zero-overhead bitstream mode is also available.
- **BSD 3-Clause licensed.**

## News

- **2026-08**: FSZ 1.0.0 is released.
- **2026-07**: The FSZ paper is nominated for the Best Paper Award at SC'26.
- **2026-07**: Paper accepted at SC'26.

## 1. Requirements

- NVIDIA GPU, compute capability `>= 8.0` (Ampere or newer).
- CUDA Toolkit 12.x or 13.x.
- C++17 host compiler.
- CMake `>= 3.18` (only for the CMake build path).
- Python `>= 3.8` with NumPy (only for the Python API).
- A Fortran 2008 compiler such as gfortran (only for the Fortran API).

## 2. Build and install

### CMake

```bash
git clone https://github.com/JiajunHuang1999/FSZ.git
cd FSZ
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure   # optional, needs a GPU
sudo cmake --install build                   # optional: libfsz, headers, fsz tool
```

The default build covers GPU generations from Ampere on: it embeds native
code for `sm_80` and `sm_90` (plus `sm_100` when the CUDA toolkit is 12.8
or newer) together with PTX, so compute-capability 8.6/8.9 devices run the
`sm_80` code directly and newer architectures load through the embedded
PTX. Override with `-DCMAKE_CUDA_ARCHITECTURES=90` (or `80;90;100`, etc.)
to build for specific generations only.

Toggles: `-DFSZ_BUILD_SHARED=OFF` (skip `libfsz.so`), `-DFSZ_BUILD_TOOLS=OFF`,
`-DFSZ_BUILD_EXAMPLES=OFF`, `-DFSZ_BUILD_TESTS=OFF`.

Installed projects can consume the library with CMake's `find_package`:

```cmake
find_package(fsz REQUIRED)
target_link_libraries(myapp PRIVATE fsz::fsz)
```

### Makefile (nvcc only)

```bash
make ARCH=90        # ARCH auto-detected from nvidia-smi if omitted
./build/fsz -h
```

## 3. Quick start

Compress any raw float32 array with a relative error bound of `1e-3`:

```bash
./build/fsz -i velocity.f32 -eb rel 1e-3
```

This runs an in-memory compress + decompress round trip and reports the
compression ratio, throughput, and the measured maximum error against the
bound. To actually write files:

```bash
./build/fsz -z -i velocity.f32 -o velocity.fsz -eb rel 1e-3 -d 512 512 512
./build/fsz -x -i velocity.fsz -o velocity.out.f32
```

The `.fsz` file is self-describing: decompression needs no dimensions, sizes,
or error bounds on the command line.

## 4. Command-line tool

```
fsz -i data.f32 -eb rel 1e-3 [-d D1 [D2 [D3]]]        report mode: round trip + error check, writes nothing
fsz -z -i data.f32 -o data.fsz -eb rel 1e-3 [-d ...]  compress to a self-describing .fsz file
fsz -x -i data.fsz -o data.out.f32                    decompress (no other arguments needed)
fsz -t f64 -i data.d64 -eb rel 1e-3                   any mode on float64 data
```

| Option | Meaning |
|---|---|
| `-z` / `-x` | force compress / decompress. Without either, files starting with the FSZ magic are decompressed and raw inputs get the report mode. |
| `-i`, `-o` | input / output path. Without `-o`, `-z` and `-x` run fully but write nothing. |
| `-eb abs V` | absolute error bound `V`. |
| `-eb rel V` | relative error bound: `V * (max - min)` of the input data. |
| `-t f32\|f64` | element type of a raw input file (default `f32`). Decompression of `.fsz` files reads the type from the header instead. |
| `-d D1 [D2 [D3]]` | data dimensions in C order (`D1` varies slowest, like a NumPy shape). Also accepts `-d D1xD2xD3`. Optional; the product must match the file's element count. |
| `--bare` | with `-z`: write the raw bitstream without the 56-byte header. Decompressing a bare stream needs `-n` (or `-d`) and `-eb abs`. |
| `-n N` | element count, for decompressing bare streams. |
| `--csv` | additionally print a one-line machine-readable record. |
| `--ssim` | additionally compute the windowed structural similarity of the reconstruction (slower; uses the `-d` shape). PSNR and NRMSE are always reported. |

Exit codes: `0` success (error check passed), `1` bound violated, `2` usage or
file errors.

## 5. C API

Device-pointer API (data already on the GPU; zero extra copies):

```c
#include <fsz/fsz.h>

fsz_workspace_t* ws = NULL;
fsz_workspace_create(&ws, n_max);            /* reusable across calls */

fsz_compress_result_t r;
fsz_compress(d_in, d_cmp, n, eb_abs, ws, /*stream=*/0, &r);
/* r.cmp_size bytes of d_cmp now hold the bitstream */

fsz_decompress(d_out, d_cmp, n, eb_abs, &r, ws, /*stream=*/0);

fsz_workspace_destroy(ws);
```

Host-pointer API (one call, no CUDA boilerplate; the library manages device
memory and transfers internally):

```c
size_t cmp_size = 0;
unsigned char* cmp = malloc(fsz_max_compressed_bytes(n));
fsz_compress_hostptr(h_in, cmp, n, eb_abs, &cmp_size);

fsz_decompress_hostptr(h_out, cmp, cmp_size, n, eb_abs);
```

Supporting calls:

```c
size_t fsz_max_compressed_bytes(size_t n);   /* safe d_cmp / cmp capacity  */
size_t fsz_num_tiles(size_t n);
fsz_compress_result_t fsz_make_result(size_t n, size_t cmp_size);
                                             /* rebuild the result struct
                                                for a stored bare stream   */
const char* fsz_version(void);
const char* fsz_status_string(fsz_status_t);
```

All `fsz_*` calls return `fsz_status_t`; `FSZ_STATUS_OK == 0`.

Float64 data uses the `_f64` variants of the same four entry points
(`fsz_compress_f64`, `fsz_decompress_f64`, `fsz_compress_hostptr_f64`,
`fsz_decompress_hostptr_f64`), which take `double` buffers and a `double`
bound. Workspaces and the sizing helpers are shared between element types.

Buffer rules:

- `d_in` / `d_out`: `n * sizeof(float)` device bytes.
- `d_cmp`: `fsz_max_compressed_bytes(n)` device bytes, 4-byte aligned
  (any `cudaMalloc` allocation qualifies).
- Workspace: created once with a maximum capacity and reused for any
  `n <= fsz_workspace_capacity(ws)`. Allocate one workspace per concurrent
  stream; a workspace must not be shared by two in-flight calls. Passing
  `NULL` instead is also allowed: the call then creates and destroys a
  temporary workspace internally (one-shot mode; decompression is then
  synchronous). Reuse a workspace when compressing repeatedly.
- `fsz_compress` synchronizes the stream before returning (it must deliver
  `cmp_size`); `fsz_decompress` is fully asynchronous on the given stream.

## 6. C++ API

```cpp
#include <fsz/fsz.hpp>

fsz::Workspace ws(n_max);                       // RAII
auto r = fsz::compress(d_in, d_cmp, n, eb_abs, ws);
fsz::decompress(d_out, d_cmp, n, eb_abs, r, ws);

// host-pointer one-shots
size_t cmp_size = fsz::compress_hostptr(h_in, cmp, n, eb_abs);
fsz::decompress_hostptr(h_out, cmp, cmp_size, n, eb_abs);
```

Errors are reported as `fsz::Error` (derives from `std::runtime_error`).
All of these functions also accept `double*` buffers through overloads.

## 7. Python API

```bash
pip install ./python
```

The install compiles and bundles the shared library, so nothing needs to be
built first; it only needs a CUDA toolkit on the machine. Set
`FSZ_CUDA_ARCHITECTURES` (for example `80;90;100`) to target specific GPU
generations. If no toolkit is present the install still succeeds and the
module falls back to `FSZ_LIB`, a repository build directory, or the system
search path, naming all of them if none works.

```python
import numpy as np
import fsz

arr = np.fromfile("velocity.f32", dtype=np.float32).reshape(512, 512, 512)

blob = fsz.compress(arr, eb=1e-3, mode="rel")   # bytes (self-describing)
out  = fsz.decompress(blob)                     # float32 ndarray, shape restored

print(arr.shape == out.shape, np.abs(arr - out).max())
```

Arrays of dtype float64 compress through the float64 path and decompress
back as float64; the container records which type was written.

File helpers mirror the CLI: `fsz.compress_file(src, dst, eb, mode="rel",
dims=None, dtype=np.float32)` and `fsz.decompress_file(src, dst=None)`. The
package is a thin `ctypes` wrapper over `libfsz.so`: point it at a custom
build with the `FSZ_LIB` environment variable if the library is not
installed system-wide.

## 8. Fortran API

The `fsz` module wraps the host-pointer calls with generic interfaces for
`real(4)` and `real(8)` data; arrays of rank one to four are passed
directly:

```fortran
use fsz
real(4) :: field(512, 512, 512)
integer(c_int8_t), allocatable :: cmp(:)
integer(c_size_t) :: n, cmp_size
integer(c_int) :: ierr

n = int(size(field), c_size_t)
allocate(cmp(fsz_max_compressed_bytes(n)))

call fsz_compress(field, n, 1.0e-3, cmp, cmp_size, ierr)    ! absolute bound
call fsz_decompress(field, cmp, cmp_size, n, 1.0e-3, ierr)
```

`ierr` is `FSZ_OK` (0) on success and `fsz_status_message(ierr)` describes
a failure. CMake builds the module automatically when a Fortran compiler
is present (the `fsz_fortran` library and `fsz.mod`); with the Makefile,
run `make fortran`.

For data that is already on the GPU, pass device addresses as `type(c_ptr)`
and keep a workspace across calls. This is what CUDA Fortran gives through
`c_devloc`, and what OpenACC and OpenMP give inside `host_data use_device`
and `target data use_device_ptr`:

```fortran
type(c_ptr) :: ws
type(fsz_result) :: res

call fsz_workspace_create(ws, n, ierr)
!$acc host_data use_device(field, cmp_buffer)
call fsz_compress_device(c_loc(field), c_loc(cmp_buffer), n, 1.0e-3, ws, res, ierr)
!$acc end host_data
call fsz_workspace_destroy(ws)
```

`res%cmp_size` gives the stream length. The element type follows the kind of
the error bound, so a `real(8)` bound selects the float64 path. The workspace
may be `c_null_ptr` for a one-shot call, and an optional trailing `stream`
argument places the work on a CUDA stream.

## 9. Performance

Compression ratios are deterministic and bit-reproducible across GPUs; the
ratios below are the per-dataset averages over all fields of the SDRBench
suites evaluated in the paper. Miranda is distributed in double precision
and is evaluated as float64, measured with this repository; the other rows
are the paper's float32 results.

| Dataset | CR @ rel 1e-2 | CR @ rel 1e-3 | CR @ rel 1e-4 |
|---|---:|---:|---:|
| CESM-ATM | 54.25 | 27.18 | 16.09 |
| EXAALT | 6.07 | 3.72 | 2.70 |
| HACC | 24.83 | 8.75 | 4.68 |
| Hurricane | 43.40 | 24.99 | 15.82 |
| Miranda (float64) | 74.21 | 45.47 | 27.77 |
| NYX | 70.52 | 42.47 | 24.30 |
| OpenSciVis-Truss | 13.20 | 6.54 | 4.28 |
| SCALE-LETKF | 51.68 | 30.53 | 18.46 |

Throughput measured with this repository on an NVIDIA B200 (CUDA 12.8.1)
at rel `1e-3`, steady-state average of 30 runs, strictly sequential
single-field calls:

| Field | Size | Compress GB/s | Decompress GB/s |
|---|---:|---:|---:|
| OpenSciVis-Truss | 6.9 GB | 1250 | 1502 |
| CESM-ATM/CLDICE | 674 MB | 1127 | 1323 |
| HACC/xx | 1.1 GB | 1075 | 1430 |
| NYX/temperature | 537 MB | 1036 | 1303 |
| SCALE-LETKF/T | 564 MB | 972 | 1226 |

Float64 fields, measured with the `fsz` tool on the same B200 at rel `1e-3`
(average of 5 timed runs after 3 warmups):

| Field | Size | CR | Compress GB/s | Decompress GB/s |
|---|---:|---:|---:|---:|
| S3D/CH4 (float64) | 1.0 GB | 30.29 | 1395 | 1713 |
| Miranda/density (float64) | 302 MB | 40.57 | 774 | 1218 |

## 10. Citation

```bibtex
@inproceedings{huang2026fsz,
  title     = {{FSZ}: Breaking the Prediction-Throughput Trade-off in {GPU} Lossy Compression},
  author    = {Huang, Jiajun},
  booktitle = {Proceedings of the International Conference for High Performance
               Computing, Networking, Storage and Analysis (SC '26)},
  year      = {2026},
  note      = {To appear. arXiv:2607.15413}
}
```

## 11. License

BSD 3-Clause. See [LICENSE](LICENSE).
