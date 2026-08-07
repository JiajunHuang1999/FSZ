# Contributing to FSZ

Bug reports, questions, and patches are welcome.

## Reporting a problem

Please include:

- the GPU and its compute capability (`nvidia-smi --query-gpu=name,compute_cap --format=csv`),
- the CUDA toolkit version (`nvcc --version`),
- how FSZ was built (CMake or Makefile, and any options),
- the element count, dimensions, element type, and error bound,
- what you expected and what happened.

A reproducer that runs from the `fsz` tool is the most useful form, for example
`fsz -i field.f32 -eb rel 1e-3 -d 512 512 512`. If the data cannot be shared, a
synthetic array that shows the same behavior works just as well.

For anything that looks like wrong output, please say whether the error bound
was respected: the tool prints `max_err` against the bound and exits 1 when the
bound is violated.

## Building and testing

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure   # requires a GPU
```

The suites cover the C and C++ APIs (`roundtrip`, `hostptr`) and the Fortran
interface (`fortran`). The Python tests run separately:

```bash
PYTHONPATH=python python3 python/tests/test_fsz.py
```

## What a change needs

FSZ is an error-bounded compressor, so correctness has a specific meaning and
patches are held to it:

1. **The bound holds.** Every reconstructed value must be within
   `1.01 x` the absolute error bound, and no output may be non-finite. The
   1 percent slack absorbs float32 rounding at the last bit and nothing more.
2. **The stream does not move unintentionally.** A change meant to affect only
   performance, memory, or the build must leave the compressed bytes identical.
   The quickest check is to compress a field before and after and compare
   checksums; `fsz -z -i field.f32 -o out.fsz -eb rel 1e-3` then `md5sum`.
3. **Deliberate format changes are versioned.** The container carries a version
   field, and an existing `.fsz` file must keep decompressing. If it cannot,
   that is a major version.
4. **New behavior comes with a test.** Add the case to the suite that covers the
   affected interface, and prefer a case that fails before the change.
5. **Measurements are real.** If a change claims a speed or ratio improvement,
   include the numbers and the hardware they came from. Throughput figures are
   GPU compression and decompression rates on device-resident data.

## Style

- Follow the surrounding code: four-space indent, and the naming already in
  each file.
- Comments explain why something is done, not what the line does.
- Keep the public headers free of implementation detail; `src/fsz_internal.cuh`
  is where format constants live.

## License

Contributions are accepted under the BSD 3-Clause license that covers the
project. See `LICENSE`.
