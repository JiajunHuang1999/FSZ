# Changelog

All notable changes to FSZ are recorded here. Versions follow
[semantic versioning](https://semver.org/): the stream format is part of the
public interface, so any change that makes an existing `.fsz` file unreadable
is a major version.

## [1.0.0] - 2026-08-07

First public release: the compressor described in the SC'26 paper
*FSZ: Breaking the Prediction-Throughput Trade-off in GPU Lossy Compression*
(arXiv:2607.15413).

- Strict pointwise absolute or relative error bound, float32 and float64.
- Interfaces: the `fsz` command-line tool, C, C++, Python, and Fortran APIs.
  Fortran offers host-array and device-pointer entry points, so CUDA Fortran,
  OpenACC, and OpenMP codes can compress data already resident on the GPU.
- The device-pointer calls accept a reusable workspace for repeated
  compression, or `NULL` for one-shot use.
- Self-describing `.fsz` container recording shape, element type, and bound;
  bare bitstream mode for zero container overhead.
- Quality reporting in the tool: PSNR and NRMSE on every round trip, windowed
  structural similarity with `--ssim`.
- `pip install` builds and bundles the shared library when a CUDA toolkit is
  present.
- CMake package export (`find_package(fsz)`), Makefile build, and a build
  matrix covering CUDA 12.4, 12.8, and 13.2, for GPU generations from Ampere
  on.
