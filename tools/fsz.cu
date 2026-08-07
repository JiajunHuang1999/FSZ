// fsz: command line front end for the FSZ GPU error-bounded lossy compressor.
// Run 'fsz --help' for usage.

#include "fsz/fsz.hpp"
#include "fsz_file_format.hpp"

#include <cuda_runtime.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <limits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr int RC_PASS  = 0;   // success, error bound respected
constexpr int RC_FAIL  = 1;   // bound violated or non-finite output
constexpr int RC_ERROR = 2;   // usage, file or device error

constexpr int WARMUP_ITERS = 3;
constexpr int TIMED_ITERS  = 5;

bool g_color = false;

const char* green() { return g_color ? "\033[0;32m" : ""; }
const char* red()   { return g_color ? "\033[0;31m" : ""; }
const char* reset() { return g_color ? "\033[0m"    : ""; }

std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "ERROR: %s\n", msg.c_str());
    std::exit(RC_ERROR);
}

[[noreturn]] void die_usage(const std::string& msg) {
    std::fprintf(stderr, "ERROR: %s\n", msg.c_str());
    std::fprintf(stderr, "Run 'fsz --help' for usage.\n");
    std::exit(RC_ERROR);
}

void check_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        die(fmt("CUDA failure at %s: %s", where, cudaGetErrorString(e)));
    }
}

enum class DType { F32, F64 };

template <class T> struct TypeTraits;
template <> struct TypeTraits<float> {
    static constexpr DType       dtype = DType::F32;
    static constexpr const char* name  = "float32";
};
template <> struct TypeTraits<double> {
    static constexpr DType       dtype = DType::F64;
    static constexpr const char* name  = "float64";
};

const char* dtype_name(DType t) { return (t == DType::F64) ? "float64" : "float32"; }
size_t      dtype_size(DType t) { return (t == DType::F64) ? sizeof(double) : sizeof(float); }

void usage(std::FILE* out) {
    std::fprintf(out,
        "FSZ %s  GPU error-bounded lossy compressor for float32 and float64 arrays\n"
        "\n"
        "Usage:\n"
        "  fsz -i data.f32 -eb rel 1e-3 [-d D1 [D2 [D3]]]\n"
        "        compress and decompress in memory, then report ratio,\n"
        "        throughput and reconstruction error (writes nothing)\n"
        "\n"
        "  fsz -z -i data.f32 -o data.fsz -eb rel 1e-3 [-d D1 [D2 [D3]]]\n"
        "        compress to a self-describing .fsz file\n"
        "\n"
        "  fsz -t f64 -z -i data.d64 -o data.fsz -eb rel 1e-3 [-d D1 [D2 [D3]]]\n"
        "        the same for a float64 input array\n"
        "\n"
        "  fsz -x -i data.fsz -o data.out.f32\n"
        "        decompress a .fsz file, no other arguments needed\n"
        "\n"
        "  fsz -x -i stream.bin -o out.f32 -n N -eb abs V --bare\n"
        "        decompress a bare bitstream written with --bare\n"
        "\n"
        "Options:\n"
        "  -z                  force compression\n"
        "  -x                  force decompression\n"
        "  -i <path>           input file\n"
        "  -o <path>           output file; without it -z and -x run fully\n"
        "                      but write nothing\n"
        "  -t f32|f64          element type of the raw array, float32 or\n"
        "                      float64 (default f32)\n"
        "  -eb abs <value>     absolute error bound\n"
        "  -eb rel <value>     relative error bound, a fraction of (max - min)\n"
        "                      of the input data\n"
        "  -d D1 [D2 [D3]]     dimensions in C order, D1 slowest-varying; the\n"
        "                      compact form -d D1xD2xD3 is also accepted\n"
        "  -n N                element count, for decompressing a bare bitstream\n"
        "  --bare              with -z, write the bitstream with no file header\n"
        "  --csv               also print one machine-readable csv line\n"
        "  --ssim              also compute the windowed structural similarity\n"
        "                      of the reconstruction (slower; uses -d shape)\n"
        "  -h, --help          this message\n"
        "\n"
        "Input is a raw float32 or float64 array of any dimensionality; up to\n"
        "three dimensions and the element type are recorded in the file header.\n"
        "Without -d the element count is the file size divided by the element\n"
        "size, 4 bytes for f32 and 8 for f64, and the header records a single\n"
        "axis. With -d the product of the dimensions must equal that count.\n"
        "\n"
        "Decompressing a container needs no -t: the element type comes from the\n"
        "header. A bare bitstream carries no header, so -t selects the type\n"
        "there as it does for compression.\n"
        "\n"
        "With neither -z nor -x the tool inspects the first four bytes of the\n"
        "input: an FSZ container is decompressed, anything else is reported.\n"
        "\n"
        "Exit status: 0 pass, 1 error bound violated, 2 usage or file error.\n",
        fsz::version());
}

bool parse_u64(const char* s, uint64_t& out) {
    if (!s || !*s) return false;
    errno = 0;
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return false;
    out = (uint64_t)v;
    return true;
}

std::string dims_string(const uint64_t dims[3], uint32_t ndims) {
    std::string s;
    for (uint32_t i = 0; i < ndims; ++i) {
        if (i) s += 'x';
        s += std::to_string((unsigned long long)dims[i]);
    }
    return s;
}

long file_size_or_die(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die("cannot open input " + path);
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fclose(f);
    if (bytes < 0) die("cannot determine the size of " + path);
    return bytes;
}

std::vector<unsigned char> read_all(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die("cannot open input " + path);
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes <= 0) { std::fclose(f); die("input " + path + " is empty"); }
    std::vector<unsigned char> v((size_t)bytes);
    if (std::fread(v.data(), 1, (size_t)bytes, f) != (size_t)bytes) {
        std::fclose(f);
        die("short read on " + path);
    }
    std::fclose(f);
    return v;
}

template <class T>
std::vector<T> read_values(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die("cannot open input " + path);
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes <= 0 || (bytes % (long)sizeof(T)) != 0) {
        std::fclose(f);
        die(fmt("%s is %ld bytes, which is not a positive multiple of %zu; "
                "the input must be a raw %s array", path.c_str(), bytes,
                sizeof(T), TypeTraits<T>::name));
    }
    std::vector<T> v((size_t)(bytes / (long)sizeof(T)));
    if (std::fread(v.data(), 1, (size_t)bytes, f) != (size_t)bytes) {
        std::fclose(f);
        die("short read on " + path);
    }
    std::fclose(f);
    return v;
}

void write_all(const std::string& path, const void* p, size_t n) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) die("cannot open output " + path);
    if (n && std::fwrite(p, 1, n, f) != n) {
        std::fclose(f);
        die("short write on " + path);
    }
    if (std::fclose(f) != 0) die("failed to close " + path);
}

float relative_to_absolute(const std::vector<float>& h, double eb_rel) {
    float lo = h[0], hi = h[0];
    for (size_t i = 0; i < h.size(); ++i) {
        const float v = h[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    const float range = hi - lo;
    if (!(range > 0.0f)) {
        die("the input is constant, so a relative error bound is undefined; "
            "use -eb abs instead");
    }
    return (float)((double)range * eb_rel);
}

double relative_to_absolute(const std::vector<double>& h, double eb_rel) {
    double lo = h[0], hi = h[0];
    for (size_t i = 0; i < h.size(); ++i) {
        const double v = h[i];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    const double range = hi - lo;
    if (!(range > 0.0)) {
        die("the input is constant, so a relative error bound is undefined; "
            "use -eb abs instead");
    }
    return range * eb_rel;
}

class EventTimer {
public:
    EventTimer()  { cudaEventCreate(&e0_); cudaEventCreate(&e1_); }
    ~EventTimer() { cudaEventDestroy(e0_); cudaEventDestroy(e1_); }
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    void  start() { cudaEventRecord(e0_); }
    float stop() {
        cudaEventRecord(e1_);
        cudaEventSynchronize(e1_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0_, e1_);
        return ms;
    }

private:
    cudaEvent_t e0_, e1_;
};

double gbps(size_t bytes, double ms) {
    return (ms > 0.0) ? (double)bytes / (ms * 1e6) : 0.0;
}

struct Options {
    std::string           in_path;
    std::string           out_path;
    std::string           eb_mode;          // "abs", "rel", or empty
    double                eb_value = 0.0;
    std::vector<uint64_t> dims;             // as supplied with -d
    uint64_t              n_arg      = 0;   // as supplied with -n
    bool                  have_n     = false;
    DType                 dtype      = DType::F32;  // as supplied with -t
    bool                  have_dtype = false;
    bool                  force_z    = false;
    bool                  force_x    = false;
    bool                  bare       = false;
    bool                  csv        = false;
    bool                  ssim       = false;
};

void parse_dim_list(int argc, char** argv, int& i, std::vector<uint64_t>& dims) {
    if (i + 1 >= argc) die_usage("-d needs at least one dimension");
    const std::string first = argv[++i];

    if (first.find('x') != std::string::npos || first.find('X') != std::string::npos) {
        std::string tok;
        for (size_t k = 0; k <= first.size(); ++k) {
            const char c = (k < first.size()) ? first[k] : 'x';
            if (c == 'x' || c == 'X') {
                uint64_t v = 0;
                if (!parse_u64(tok.c_str(), v) || v == 0) {
                    die_usage("bad dimension list '" + first +
                              "'; expected D1xD2xD3 with positive integers");
                }
                if (dims.size() == fsz_file::MAX_DIMS) {
                    die_usage("at most " + std::to_string(fsz_file::MAX_DIMS) +
                              " dimensions are supported, got '" + first + "'");
                }
                dims.push_back(v);
                tok.clear();
            } else {
                tok += c;
            }
        }
        return;
    }

    uint64_t v = 0;
    if (!parse_u64(first.c_str(), v) || v == 0) {
        die_usage("bad dimension '" + first + "'; expected a positive integer");
    }
    dims.push_back(v);
    while (dims.size() < fsz_file::MAX_DIMS && i + 1 < argc &&
           parse_u64(argv[i + 1], v)) {
        if (v == 0) die_usage("dimensions must be positive");
        dims.push_back(v);
        ++i;
    }
}

Options parse_args(int argc, char** argv) {
    Options o;
    if (argc == 1) { usage(stderr); std::exit(RC_ERROR); }

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
            usage(stdout);
            std::exit(RC_PASS);
        } else if (std::strcmp(a, "-z") == 0) {
            o.force_z = true;
        } else if (std::strcmp(a, "-x") == 0) {
            o.force_x = true;
        } else if (std::strcmp(a, "--bare") == 0) {
            o.bare = true;
        } else if (std::strcmp(a, "--csv") == 0) {
            o.csv = true;
        } else if (std::strcmp(a, "--ssim") == 0) {
            o.ssim = true;
        } else if (std::strcmp(a, "-i") == 0) {
            if (i + 1 >= argc) die_usage("-i needs a path");
            o.in_path = argv[++i];
        } else if (std::strcmp(a, "-o") == 0) {
            if (i + 1 >= argc) die_usage("-o needs a path");
            o.out_path = argv[++i];
        } else if (std::strcmp(a, "-eb") == 0) {
            if (i + 2 >= argc) die_usage("-eb needs a mode (abs or rel) and a value");
            o.eb_mode = argv[++i];
            if (o.eb_mode != "abs" && o.eb_mode != "rel") {
                die_usage("-eb mode must be 'abs' or 'rel', got '" + o.eb_mode + "'");
            }
            const char* vs = argv[++i];
            errno = 0;
            char* end = nullptr;
            o.eb_value = std::strtod(vs, &end);
            if (errno != 0 || end == vs || *end != '\0') {
                die_usage(fmt("bad error bound value '%s'", vs));
            }
            if (!(o.eb_value > 0.0)) die_usage("the error bound must be greater than 0");
        } else if (std::strcmp(a, "-t") == 0) {
            if (i + 1 >= argc) die_usage("-t needs an element type (f32 or f64)");
            const std::string t = argv[++i];
            if (t == "f32") {
                o.dtype = DType::F32;
            } else if (t == "f64") {
                o.dtype = DType::F64;
            } else {
                die_usage("-t must be 'f32' or 'f64', got '" + t + "'");
            }
            o.have_dtype = true;
        } else if (std::strcmp(a, "-d") == 0) {
            parse_dim_list(argc, argv, i, o.dims);
        } else if (std::strcmp(a, "-n") == 0) {
            if (i + 1 >= argc) die_usage("-n needs an element count");
            if (!parse_u64(argv[i + 1], o.n_arg) || o.n_arg == 0) {
                die_usage(fmt("bad element count '%s'", argv[i + 1]));
            }
            ++i;
            o.have_n = true;
        } else {
            die_usage(fmt("unrecognized option '%s'", a));
        }
    }

    if (o.in_path.empty()) die_usage("no input; pass -i <path>");
    if (o.force_z && o.force_x) die_usage("-z and -x are mutually exclusive");
    return o;
}

void resolve_shape(const Options& o, uint64_t n_file, const std::string& what,
                   DType dtype, uint64_t dims[3], uint32_t& ndims, uint64_t& n) {
    if (o.dims.empty()) {
        ndims   = 1;
        dims[0] = n_file;
        dims[1] = 1;
        dims[2] = 1;
        n       = n_file;
        return;
    }
    ndims = (uint32_t)o.dims.size();
    uint64_t prod = 1;
    for (uint32_t i = 0; i < 3; ++i) {
        dims[i] = (i < ndims) ? o.dims[i] : 1;
        prod   *= dims[i];
    }
    if (prod != n_file) {
        die(fmt("-d %s gives %llu elements but %s holds %llu %s values of %zu "
                "bytes each; the product of the dimensions must equal the "
                "element count",
                dims_string(dims, ndims).c_str(),
                (unsigned long long)prod, what.c_str(),
                (unsigned long long)n_file, dtype_name(dtype),
                dtype_size(dtype)));
    }
    n = prod;
}

template <class T>
double windowed_ssim(const std::vector<T>& ori, const std::vector<T>& dec,
                     const uint64_t dims[3], uint32_t ndims) {
    size_t s2 = 1, s1 = 1, s0 = 1;
    if (ndims >= 3)      { s2 = dims[0]; s1 = dims[1]; s0 = dims[2]; }
    else if (ndims == 2) { s1 = dims[0]; s0 = dims[1]; }
    else                 { s0 = dims[0]; }
    const size_t w2 = (s2 > 1) ? 7 : 1;
    const size_t w1 = (s1 > 1) ? 7 : 1;
    const size_t w0 = 7;
    if (w0 > s0 || w1 > s1 || w2 > s2)
        die("--ssim needs every recorded dimension to hold at least 7 values");
    double sum = 0.0;
    size_t nw = 0;
    const double np = (double)(w0 * w1 * w2);
    for (size_t o2 = 0; o2 + w2 <= s2; o2 += 2) {
        for (size_t o1 = 0; o1 + w1 <= s1; o1 += 2) {
            for (size_t o0 = 0; o0 + w0 <= s0; o0 += 2) {
                double xmin = (double)ori[o0 + s0 * (o1 + s1 * o2)];
                double xmax = xmin, xs = 0.0, ys = 0.0;
                for (size_t i2 = 0; i2 < w2; ++i2)
                    for (size_t i1 = 0; i1 < w1; ++i1)
                        for (size_t i0 = 0; i0 < w0; ++i0) {
                            const size_t idx =
                                (o0 + i0) + s0 * ((o1 + i1) + s1 * (o2 + i2));
                            const double x = (double)ori[idx];
                            const double y = (double)dec[idx];
                            if (x < xmin) xmin = x;
                            if (x > xmax) xmax = x;
                            xs += x; ys += y;
                        }
                const double xm = xs / np, ym = ys / np;
                double vx = 0.0, vy = 0.0, vxy = 0.0;
                for (size_t i2 = 0; i2 < w2; ++i2)
                    for (size_t i1 = 0; i1 < w1; ++i1)
                        for (size_t i0 = 0; i0 < w0; ++i0) {
                            const size_t idx =
                                (o0 + i0) + s0 * ((o1 + i1) + s1 * (o2 + i2));
                            const double dx = (double)ori[idx] - xm;
                            const double dy = (double)dec[idx] - ym;
                            vx += dx * dx; vy += dy * dy; vxy += dx * dy;
                        }
                vx /= np; vy /= np; vxy /= np;
                const double L  = xmax - xmin;
                const double c1 = (L == 0.0) ? 0.01 * 0.01 : 0.01 * 0.01 * L * L;
                const double c2 = (L == 0.0) ? 0.03 * 0.03 : 0.03 * 0.03 * L * L;
                sum += ((2.0 * xm * ym + c1) * (2.0 * vxy + c2))
                     / ((xm * xm + ym * ym + c1) * (vx + vy + c2));
                ++nw;
            }
        }
    }
    return sum / (double)nw;
}

template <class T>
int run_compress(const Options& o, bool write_output) {
    constexpr bool is_f64 = std::is_same<T, double>::value;

    if (o.eb_mode.empty()) {
        die_usage("no error bound; pass -eb abs <value> or -eb rel <value>");
    }
    const bool write_file = write_output && !o.out_path.empty();

    std::vector<T> h_in = read_values<T>(o.in_path);
    const uint64_t n_file = (uint64_t)h_in.size();

    uint64_t dims[3] = {0, 1, 1};
    uint32_t ndims   = 1;
    uint64_t n64     = 0;
    resolve_shape(o, n_file, o.in_path, TypeTraits<T>::dtype, dims, ndims, n64);
    const size_t n = (size_t)n64;

    const T eb_abs = (o.eb_mode == "rel")
                   ? relative_to_absolute(h_in, o.eb_value)
                   : (T)o.eb_value;
    if (!(eb_abs > (T)0)) {
        die("the resulting absolute error bound is not greater than 0");
    }

    T*             d_in  = nullptr;
    T*             d_out = nullptr;
    unsigned char* d_cmp = nullptr;
    const size_t cmp_alloc = fsz::max_compressed_bytes(n);
    check_cuda(cudaMalloc(&d_in,  n * sizeof(T)), "allocating the input buffer");
    check_cuda(cudaMalloc(&d_out, n * sizeof(T)), "allocating the output buffer");
    check_cuda(cudaMalloc(&d_cmp, cmp_alloc),     "allocating the compressed buffer");
    check_cuda(cudaMemcpy(d_in, h_in.data(), n * sizeof(T), cudaMemcpyHostToDevice),
               "copying the input to the device");

    fsz::Workspace ws(n);

    fsz_compress_result_t r{};
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        r = fsz::compress(d_in, d_cmp, n, eb_abs, ws);
        fsz::decompress(d_out, d_cmp, n, eb_abs, r, ws);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup");

    EventTimer t;
    double c_ms_sum = 0.0;
    for (int i = 0; i < TIMED_ITERS; ++i) {
        t.start();
        r = fsz::compress(d_in, d_cmp, n, eb_abs, ws);
        c_ms_sum += t.stop();
    }
    double d_ms_sum = 0.0;
    for (int i = 0; i < TIMED_ITERS; ++i) {
        t.start();
        fsz::decompress(d_out, d_cmp, n, eb_abs, r, ws);
        d_ms_sum += t.stop();
    }
    check_cuda(cudaGetLastError(), "the timed loop");

    const double c_ms = c_ms_sum / TIMED_ITERS;
    const double d_ms = d_ms_sum / TIMED_ITERS;

    std::vector<unsigned char> h_cmp;
    if (write_output) {
        h_cmp.resize(r.cmp_size);
        check_cuda(cudaMemcpy(h_cmp.data(), d_cmp, r.cmp_size, cudaMemcpyDeviceToHost),
                   "copying the bitstream to the host");
    }
    std::vector<T> h_out(n);
    check_cuda(cudaMemcpy(h_out.data(), d_out, n * sizeof(T), cudaMemcpyDeviceToHost),
               "copying the reconstruction to the host");

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_cmp);

    double max_err   = 0.0;
    size_t nonfinite = 0;
    double sum_sq    = 0.0;
    double omin      = (double)h_in[0];
    double omax      = (double)h_in[0];
    for (size_t i = 0; i < n; ++i) {
        const double x = (double)h_in[i];
        if (x < omin) omin = x;
        if (x > omax) omax = x;
        const T v = h_out[i];
        if (!std::isfinite(v)) { ++nonfinite; continue; }
        const double e = std::fabs(x - (double)v);
        if (e > max_err) max_err = e;
        sum_sq += e * e;
    }
    const bool ok = (nonfinite == 0) && (max_err <= 1.01 * (double)eb_abs);
    const double range = omax - omin;
    const double rmse  = std::sqrt(sum_sq / (double)n);
    const double psnr  = (range > 0.0 && rmse > 0.0)
                       ? 20.0 * std::log10(range / rmse)
                       : std::numeric_limits<double>::infinity();
    const double nrmse = (range > 0.0) ? rmse / range : 0.0;
    double ssim = -1.0;
    if (o.ssim) ssim = windowed_ssim(h_in, h_out, dims, ndims);

    uint64_t total_bytes = 0;
    if (write_file) {
        if (o.bare) {
            write_all(o.out_path, h_cmp.data(), h_cmp.size());
            total_bytes = (uint64_t)h_cmp.size();
        } else {
            const fsz_file::Header hdr =
                fsz_file::make_header(dims, ndims, (double)eb_abs, is_f64,
                                      (uint64_t)r.cmp_size);
            std::FILE* f = std::fopen(o.out_path.c_str(), "wb");
            if (!f) die("cannot open output " + o.out_path);
            if (std::fwrite(&hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
                std::fclose(f);
                die("short write on the header of " + o.out_path);
            }
            if (std::fwrite(h_cmp.data(), 1, h_cmp.size(), f) != h_cmp.size()) {
                std::fclose(f);
                die("short write on the bitstream of " + o.out_path);
            }
            if (std::fclose(f) != 0) die("failed to close " + o.out_path);
            total_bytes = (uint64_t)(sizeof(hdr) + h_cmp.size());
        }
    }

    const size_t in_bytes = n * sizeof(T);
    const double cr = (double)n * (double)sizeof(T) / (double)r.cmp_size;

    std::printf("FSZ %s\n", fsz::version());
    std::printf("  input      = %s (%.2f MB)\n",
                o.in_path.c_str(), (double)in_bytes / (1024.0 * 1024.0));
    std::printf("  dtype      = %s\n", TypeTraits<T>::name);
    std::printf("  elements   = %zu  (dims %s)\n", n, dims_string(dims, ndims).c_str());
    std::printf("  eb         = %s %g  ->  abs %.6e\n",
                o.eb_mode.c_str(), o.eb_value, (double)eb_abs);
    std::printf("  compressed = %zu bytes\n", (size_t)r.cmp_size);
    std::printf("  CR         = %.4f\n", cr);
    if (write_file) {
        if (o.bare) {
            std::printf("  output     = %s (%llu bytes, bare bitstream)\n",
                        o.out_path.c_str(), (unsigned long long)total_bytes);
        } else {
            std::printf("  output     = %s (%llu bytes total, %zu byte header)\n",
                        o.out_path.c_str(), (unsigned long long)total_bytes,
                        sizeof(fsz_file::Header));
        }
    }
    std::printf("  compress   = %.2f ms (%.2f GB/s)\n", c_ms, gbps(in_bytes, c_ms));
    std::printf("  decompress = %.2f ms (%.2f GB/s)\n", d_ms, gbps(in_bytes, d_ms));
    std::printf("  PSNR       = %.2f dB\n", psnr);
    std::printf("  NRMSE      = %.6e\n", nrmse);
    if (o.ssim)
        std::printf("  SSIM       = %.6f\n", ssim);
    std::printf("  max_err    = %.6e  (bound %.6e)\n", max_err, (double)eb_abs);
    if (ok) {
        std::printf("  %sPASS%s\n", green(), reset());
    } else if (nonfinite) {
        std::printf("  %sFAIL%s: %zu of %zu reconstructed values are not finite\n",
                    red(), reset(), nonfinite, n);
    } else {
        std::printf("  %sFAIL%s: max_err %.6e exceeds 1.01 x %.6e\n",
                    red(), reset(), max_err, (double)eb_abs);
    }

    if (o.csv) {
        std::printf("csv,%s,%zu,%s,%s,%g,%.6e,%zu,%.4f,%.4f,%.2f,%.4f,%.2f,%.6e,%s\n",
                    o.in_path.c_str(), n, dims_string(dims, ndims).c_str(),
                    o.eb_mode.c_str(), o.eb_value, (double)eb_abs,
                    (size_t)r.cmp_size, cr,
                    c_ms, gbps(in_bytes, c_ms), d_ms, gbps(in_bytes, d_ms),
                    max_err, ok ? "pass" : "fail");
    }

    return ok ? RC_PASS : RC_FAIL;
}

struct DecompressPlan {
    uint64_t             dims[3]      = {0, 1, 1};
    uint32_t             ndims        = 1;
    uint64_t             n            = 0;
    double               eb_abs       = 0.0;
    const unsigned char* payload      = nullptr;
    size_t               payload_size = 0;
    size_t               raw_size     = 0;
    bool                 from_header  = false;
};

template <class T>
int run_decompress_typed(const Options& o, const DecompressPlan& p) {
    const size_t n      = (size_t)p.n;
    const T      eb_abs = (T)p.eb_abs;

    fsz_compress_result_t r{};
    r.cmp_size    = p.payload_size;
    r.num_tiles   = fsz::num_tiles(n);
    r.data_offset = fsz::num_tiles(n) * 8;

    if (p.payload_size < r.data_offset) {
        die(fmt("the bitstream is %zu bytes but %zu elements need at least a "
                "%zu byte tile table; the element count or the file is wrong",
                p.payload_size, n, r.data_offset));
    }

    T*             d_out = nullptr;
    unsigned char* d_cmp = nullptr;
    size_t cmp_alloc = fsz::max_compressed_bytes(n);
    if (cmp_alloc < p.payload_size) cmp_alloc = p.payload_size;
    check_cuda(cudaMalloc(&d_out, n * sizeof(T)), "allocating the output buffer");
    check_cuda(cudaMalloc(&d_cmp, cmp_alloc),     "allocating the compressed buffer");
    check_cuda(cudaMemcpy(d_cmp, p.payload, p.payload_size, cudaMemcpyHostToDevice),
               "copying the bitstream to the device");

    fsz::Workspace ws(n);

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        fsz::decompress(d_out, d_cmp, n, eb_abs, r, ws);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup");

    EventTimer t;
    double d_ms_sum = 0.0;
    for (int i = 0; i < TIMED_ITERS; ++i) {
        t.start();
        fsz::decompress(d_out, d_cmp, n, eb_abs, r, ws);
        d_ms_sum += t.stop();
    }
    check_cuda(cudaGetLastError(), "the timed loop");
    const double d_ms = d_ms_sum / TIMED_ITERS;

    std::vector<T> h_out(n);
    check_cuda(cudaMemcpy(h_out.data(), d_out, n * sizeof(T), cudaMemcpyDeviceToHost),
               "copying the reconstruction to the host");

    cudaFree(d_out);
    cudaFree(d_cmp);

    if (!o.out_path.empty()) write_all(o.out_path, h_out.data(), n * sizeof(T));

    const size_t out_bytes = n * sizeof(T);
    const double cr = (double)n * (double)sizeof(T) / (double)p.payload_size;

    std::printf("FSZ %s\n", fsz::version());
    std::printf("  input      = %s (%zu bytes)\n", o.in_path.c_str(), p.raw_size);
    std::printf("  dtype      = %s%s\n", TypeTraits<T>::name,
                p.from_header ? "  (from the file header)" : "");
    std::printf("  elements   = %zu  (dims %s)\n", n,
                dims_string(p.dims, p.ndims).c_str());
    std::printf("  eb         = abs %.6e%s\n", (double)eb_abs,
                p.from_header ? "  (from the file header)" : "");
    std::printf("  compressed = %zu bytes\n", p.payload_size);
    std::printf("  CR         = %.4f\n", cr);
    if (!o.out_path.empty())
        std::printf("  output     = %s (%zu bytes)\n", o.out_path.c_str(), out_bytes);
    std::printf("  decompress = %.2f ms (%.2f GB/s)\n", d_ms, gbps(out_bytes, d_ms));

    if (o.csv) {
        std::printf("csv,%s,%zu,%s,abs,%g,%.6e,%zu,%.4f,,,%.4f,%.2f,,\n",
                    o.in_path.c_str(), n, dims_string(p.dims, p.ndims).c_str(),
                    (double)eb_abs, (double)eb_abs, p.payload_size, cr,
                    d_ms, gbps(out_bytes, d_ms));
    }

    return RC_PASS;
}

int run_decompress(const Options& o) {
    std::vector<unsigned char> raw = read_all(o.in_path);

    DecompressPlan p;
    p.raw_size = raw.size();

    uint64_t dims[3]  = {0, 1, 1};
    uint32_t ndims    = 1;
    uint64_t n64      = 0;
    double   eb_abs   = 0.0;
    DType    dtype    = o.dtype;
    size_t   payload_size = 0;
    const unsigned char* payload = nullptr;
    bool from_header = false;

    if (o.bare) {
        if (o.eb_mode != "abs") {
            die_usage("a bare bitstream carries no metadata; pass -eb abs <value>");
        }
        uint64_t n_arg = 0;
        if (o.have_n) n_arg = o.n_arg;
        if (!o.dims.empty()) {
            uint64_t prod = 1;
            for (size_t i = 0; i < o.dims.size(); ++i) prod *= o.dims[i];
            if (o.have_n && prod != n_arg) {
                die(fmt("-n %llu and -d %s disagree; -d gives %llu elements",
                        (unsigned long long)n_arg,
                        dims_string(o.dims.data(), (uint32_t)o.dims.size()).c_str(),
                        (unsigned long long)prod));
            }
            n_arg = prod;
            ndims = (uint32_t)o.dims.size();
            for (uint32_t i = 0; i < 3; ++i) dims[i] = (i < ndims) ? o.dims[i] : 1;
        } else {
            if (!o.have_n) {
                die_usage("a bare bitstream carries no element count; "
                          "pass -n <count> or -d <dimensions>");
            }
            ndims   = 1;
            dims[0] = n_arg;
            dims[1] = 1;
            dims[2] = 1;
        }
        n64          = n_arg;
        eb_abs       = (dtype == DType::F64) ? o.eb_value
                                             : (double)(float)o.eb_value;
        payload      = raw.data();
        payload_size = raw.size();
    } else {
        if (raw.size() < fsz_file::HEADER_SIZE) {
            die(fmt("%s is %zu bytes, too short to hold a %zu byte FSZ header; "
                    "if this is a bare bitstream, pass --bare with -n and -eb abs",
                    o.in_path.c_str(), raw.size(), fsz_file::HEADER_SIZE));
        }
        fsz_file::Header hdr{};
        std::memcpy(&hdr, raw.data(), sizeof(hdr));
        if (!fsz_file::has_magic(hdr.magic)) {
            die(o.in_path + " is not an FSZ container; if this is a bare bitstream, "
                            "pass --bare with -n and -eb abs");
        }
        if (hdr.version != fsz_file::FORMAT_VER) {
            die(fmt("%s uses container version %u, this build reads version %u",
                    o.in_path.c_str(), hdr.version, fsz_file::FORMAT_VER));
        }
        if (hdr.ndims < 1 || hdr.ndims > fsz_file::MAX_DIMS) {
            die(fmt("%s records %u dimensions, which is out of range",
                    o.in_path.c_str(), hdr.ndims));
        }
        const uint64_t have = (uint64_t)(raw.size() - fsz_file::HEADER_SIZE);
        if (hdr.payload_size > have) {
            die(fmt("%s claims a %llu byte bitstream but only %llu bytes follow "
                    "the header; the file looks truncated",
                    o.in_path.c_str(),
                    (unsigned long long)hdr.payload_size,
                    (unsigned long long)have));
        }
        dtype = fsz_file::header_is_f64(hdr) ? DType::F64 : DType::F32;
        if (o.have_dtype && o.dtype != dtype) {
            die(fmt("%s records %s data but -t asks for %s; the container "
                    "already carries its element type, so leave -t out",
                    o.in_path.c_str(), dtype_name(dtype), dtype_name(o.dtype)));
        }
        ndims  = hdr.ndims;
        dims[0] = hdr.dims[0];
        dims[1] = hdr.dims[1];
        dims[2] = hdr.dims[2];
        n64    = fsz_file::element_count(hdr);
        eb_abs = fsz_file::header_eb(hdr);
        payload      = raw.data() + fsz_file::HEADER_SIZE;
        payload_size = (size_t)hdr.payload_size;
        from_header  = true;
    }

    if (n64 == 0) die("the element count is 0");
    if (payload_size == 0) die("the bitstream is empty");
    if (!(eb_abs > 0.0)) die("the absolute error bound is not greater than 0");

    p.dims[0]      = dims[0];
    p.dims[1]      = dims[1];
    p.dims[2]      = dims[2];
    p.ndims        = ndims;
    p.n            = n64;
    p.eb_abs       = eb_abs;
    p.payload      = payload;
    p.payload_size = payload_size;
    p.from_header  = from_header;

    return (dtype == DType::F64) ? run_decompress_typed<double>(o, p)
                                 : run_decompress_typed<float>(o, p);
}

bool looks_like_container(const std::string& path) {
    const long bytes = file_size_or_die(path);
    if (bytes < (long)fsz_file::HEADER_SIZE) return false;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) die("cannot open input " + path);
    char magic[4] = {0, 0, 0, 0};
    const bool got = std::fread(magic, 1, 4, f) == 4;
    std::fclose(f);
    return got && fsz_file::has_magic(magic);
}

}  // namespace

int main(int argc, char** argv) {
    g_color = isatty(1) != 0;

    const Options o = parse_args(argc, argv);

    const bool f64 = (o.dtype == DType::F64);

    try {
        if (o.force_x) return run_decompress(o);
        if (o.force_z) {
            return f64 ? run_compress<double>(o, true) : run_compress<float>(o, true);
        }
        if (o.bare)    die_usage("--bare applies to -z and -x; pick one");
        if (looks_like_container(o.in_path)) return run_decompress(o);
        return f64 ? run_compress<double>(o, false) : run_compress<float>(o, false);
    } catch (const fsz::Error& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return RC_ERROR;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return RC_ERROR;
    }
}
