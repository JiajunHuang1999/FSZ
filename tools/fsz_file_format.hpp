#ifndef FSZ_TOOLS_FILE_FORMAT_HPP
#define FSZ_TOOLS_FILE_FORMAT_HPP

// On-disk container written by the fsz command line tool.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace fsz_file {

constexpr char     MAGIC[4]    = {'F', 'S', 'Z', '1'};
constexpr uint32_t FORMAT_VER  = 1;
constexpr size_t   HEADER_SIZE = 56;
constexpr uint32_t MAX_DIMS    = 3;

constexpr uint32_t FLAG_F64 = 0x1;

constexpr size_t EB_OFFSET = 40;

struct Header {
    char     magic[4];
    uint32_t version;
    uint32_t ndims;
    uint32_t flags;
    uint64_t dims[3];
    float    abs_eb;
    uint32_t reserved;
    uint64_t payload_size;
};

static_assert(sizeof(Header) == HEADER_SIZE, "FSZ file header must be 56 bytes");
static_assert(offsetof(Header, magic)        ==  0, "bad magic offset");
static_assert(offsetof(Header, version)      ==  4, "bad version offset");
static_assert(offsetof(Header, ndims)        ==  8, "bad ndims offset");
static_assert(offsetof(Header, flags)        == 12, "bad flags offset");
static_assert(offsetof(Header, dims)         == 16, "bad dims offset");
static_assert(offsetof(Header, abs_eb)       == 40, "bad abs_eb offset");
static_assert(offsetof(Header, reserved)     == 44, "bad reserved offset");
static_assert(offsetof(Header, payload_size) == 48, "bad payload_size offset");
static_assert(offsetof(Header, abs_eb) == EB_OFFSET, "bad error bound offset");

inline bool has_magic(const void* bytes) {
    return std::memcmp(bytes, MAGIC, 4) == 0;
}

inline bool header_is_f64(const Header& h) {
    return (h.flags & FLAG_F64) != 0;
}

inline void header_set_eb(Header& h, double abs_eb, bool f64) {
    unsigned char* p = reinterpret_cast<unsigned char*>(&h) + EB_OFFSET;
    if (f64) {
        h.flags |= FLAG_F64;
        std::memcpy(p, &abs_eb, sizeof(double));
    } else {
        const float    narrow = (float)abs_eb;
        const uint32_t zero   = 0;
        h.flags &= ~FLAG_F64;
        std::memcpy(p,     &narrow, sizeof(float));
        std::memcpy(p + 4, &zero,   sizeof(uint32_t));
    }
}

inline double header_eb(const Header& h) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&h) + EB_OFFSET;
    if (header_is_f64(h)) {
        double wide = 0.0;
        std::memcpy(&wide, p, sizeof(double));
        return wide;
    }
    float narrow = 0.0f;
    std::memcpy(&narrow, p, sizeof(float));
    return (double)narrow;
}

inline Header make_header(const uint64_t dims[3], uint32_t ndims,
                          double abs_eb, bool f64, uint64_t payload_size) {
    Header h{};
    std::memcpy(h.magic, MAGIC, 4);
    h.version      = FORMAT_VER;
    h.ndims        = ndims;
    h.flags        = 0;
    h.dims[0]      = dims[0];
    h.dims[1]      = dims[1];
    h.dims[2]      = dims[2];
    h.payload_size = payload_size;
    header_set_eb(h, abs_eb, f64);
    return h;
}

inline uint64_t element_count(const Header& h) {
    return h.dims[0] * h.dims[1] * h.dims[2];
}

}

#endif
