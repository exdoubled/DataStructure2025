// Header-only utilities for little-endian binary IO of vector datasets
// Format (little-endian):
// - magic[8] = "VECBIN1\0" (8 bytes, last is NUL)
// - uint32 dim
// - uint64 count (number of vectors)
// - uint32 dtype (1 = float32)
// - payload: count * dim float32 values in row-major order
//
// This header is header-only to avoid modifying build tasks.

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace binio {

static inline bool is_little_endian() {
    uint16_t x = 1;
    return *reinterpret_cast<uint8_t*>(&x) == 1;
}

static inline void to_little_endian_u32(uint32_t v, uint8_t out[4]) {
    if (is_little_endian()) {
        std::memcpy(out, &v, 4);
    } else {
        out[0] = static_cast<uint8_t>(v & 0xFF);
        out[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
        out[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
        out[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
    }
}

static inline void to_little_endian_u64(uint64_t v, uint8_t out[8]) {
    if (is_little_endian()) {
        std::memcpy(out, &v, 8);
    } else {
        for (int i = 0; i < 8; ++i) out[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
    }
}

static inline uint32_t from_le_u32(const uint8_t in[4]) {
    if (is_little_endian()) {
        uint32_t v; std::memcpy(&v, in, 4); return v;
    } else {
        uint32_t v = 0;
        v |= (uint32_t)in[0];
        v |= (uint32_t)in[1] << 8;
        v |= (uint32_t)in[2] << 16;
        v |= (uint32_t)in[3] << 24;
        return v;
    }
}

static inline uint64_t from_le_u64(const uint8_t in[8]) {
    if (is_little_endian()) {
        uint64_t v; std::memcpy(&v, in, 8); return v;
    } else {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= (uint64_t)in[i] << (8 * i);
        return v;
    }
}

constexpr uint32_t DTYPE_F32 = 1;

// Write flattened float32 vectors to a .bin file with the format above.
// flat.size() must be count * dim.
static inline bool write_vecbin(const std::string &path, int dim, const std::vector<float> &flat) {
    if (dim <= 0) return false;
    const uint64_t count = (flat.size() / (size_t)dim);
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return false;

    char magic[8] = { 'V','E','C','B','I','N','1','\0' };
    ofs.write(magic, 8);
    uint8_t buf4[4]; uint8_t buf8[8];
    to_little_endian_u32((uint32_t)dim, buf4); ofs.write(reinterpret_cast<const char*>(buf4), 4);
    to_little_endian_u64(count, buf8);          ofs.write(reinterpret_cast<const char*>(buf8), 8);
    to_little_endian_u32(DTYPE_F32, buf4);      ofs.write(reinterpret_cast<const char*>(buf4), 4);

    if (!flat.empty()) {
        if (is_little_endian()) {
            ofs.write(reinterpret_cast<const char*>(flat.data()), (std::streamsize)(flat.size() * sizeof(float)));
        } else {
            // slow path for big-endian: convert per float
            for (float f : flat) {
                uint32_t u; std::memcpy(&u, &f, 4);
                to_little_endian_u32(u, buf4);
                ofs.write(reinterpret_cast<const char*>(buf4), 4);
            }
        }
    }
    return ofs.good();
}

// Read .bin vectors into flat float32 array. Returns true on success.
static inline bool read_vecbin(const std::string &path, int &out_dim, size_t &out_count, std::vector<float> &out_flat) {
    out_dim = 0; out_count = 0; out_flat.clear();
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;

    char magic[8];
    ifs.read(magic, 8);
    if (!ifs || std::strncmp(magic, "VECBIN1\0", 8) != 0) return false;
    uint8_t buf4[4]; uint8_t buf8[8];
    ifs.read(reinterpret_cast<char*>(buf4), 4);
    ifs.read(reinterpret_cast<char*>(buf8), 8);
    ifs.read(reinterpret_cast<char*>(buf4), 4);
    if (!ifs) return false;
    const uint32_t dim = from_le_u32(buf4); // note: last read was dtype; we need both dim and dtype
    // Oops, we overwrote buf4; re-read properly by seeking back 16 bytes then reading all fields separately
    ifs.seekg(8, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(buf4), 4); // dim
    const uint32_t dim_le = from_le_u32(buf4);
    ifs.read(reinterpret_cast<char*>(buf8), 8); // count
    const uint64_t count_le = from_le_u64(buf8);
    ifs.read(reinterpret_cast<char*>(buf4), 4); // dtype
    const uint32_t dtype_le = from_le_u32(buf4);

    if (dtype_le != DTYPE_F32 || dim_le == 0) return false;
    out_dim = (int)dim_le;
    out_count = (size_t)count_le;
    const uint64_t total_vals = (uint64_t)out_dim * (uint64_t)out_count;
    if (total_vals > (uint64_t)(SIZE_MAX / sizeof(float))) return false; // overflow guard

    out_flat.resize((size_t)total_vals);
    if (is_little_endian()) {
        ifs.read(reinterpret_cast<char*>(out_flat.data()), (std::streamsize)(total_vals * sizeof(float)));
        if (!ifs) { out_flat.clear(); out_dim = 0; out_count = 0; return false; }
    } else {
        // big-endian host: read per-float and swap
        for (uint64_t i = 0; i < total_vals; ++i) {
            ifs.read(reinterpret_cast<char*>(buf4), 4);
            if (!ifs) { out_flat.clear(); out_dim = 0; out_count = 0; return false; }
            uint32_t u = from_le_u32(buf4);
            float f; std::memcpy(&f, &u, 4);
            out_flat[(size_t)i] = f;
        }
    }
    return true;
}

// Convenience: write queries (vector<vector<float>>) using the same format
static inline bool write_queries_vecbin(const std::string &path, int dim, const std::vector<std::vector<float>> &queries) {
    if (dim <= 0) return false;
    std::vector<float> flat; flat.reserve((size_t)dim * queries.size());
    for (const auto &q : queries) {
        if ((int)q.size() != dim) return false;
        flat.insert(flat.end(), q.begin(), q.end());
    }
    return write_vecbin(path, dim, flat);
}

// Convenience: read queries into vector<vector<float>>; if max_queries>0, clip to this count
static inline bool read_queries_vecbin(const std::string &path, int expected_dim,
                                       size_t max_queries, std::vector<std::vector<float>> &out_queries) {
    int dim = 0; size_t count = 0; std::vector<float> flat;
    if (!read_vecbin(path, dim, count, flat)) return false;
    if (expected_dim > 0 && dim != expected_dim) return false;
    if (max_queries > 0 && count > max_queries) count = max_queries;
    out_queries.assign(count, std::vector<float>(dim));
    for (size_t i = 0; i < count; ++i) {
        std::memcpy(out_queries[i].data(), flat.data() + i * dim, sizeof(float) * (size_t)dim);
    }
    return true;
}

} // namespace binio
