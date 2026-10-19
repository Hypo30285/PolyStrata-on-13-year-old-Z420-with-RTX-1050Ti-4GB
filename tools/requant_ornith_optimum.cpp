// requant_ornith_optimum.cpp
// Native C++17 multithreaded GGUF requantizer for Ornith 1.5 35B-A3B (APEX-I-QUALITY).
// Converts bloated Q6_K (210B/256), Q5_K (176B/256), and Q8_0 (34B/32) routed expert tensors
// (`blk.*.ffn_{gate,up,down}_exps.weight`) to unified IQ4_XS (type=23, 136B/256 = 4.25 bpw),
// while copying all existing IQ4_XS routed experts and all dense/shared/embedding tensors bit-for-bit.

#include "ggml.h"
#include "ggml-quants.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

static const int8_t kValuesIq4Nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static uint8_t g_iq4nl_lut[4097];

void init_iq4nl_lut() {
    for (int idx = 0; idx <= 4096; ++idx) {
        float x = (float)(idx - 2048) / 16.0f;
        int best = 0;
        float best_d = std::fabs(x - (float)kValuesIq4Nl[0]);
        for (int k = 1; k < 16; ++k) {
            float d = std::fabs(x - (float)kValuesIq4Nl[k]);
            if (d < best_d) {
                best_d = d;
                best = k;
            }
        }
        g_iq4nl_lut[idx] = (uint8_t)best;
    }
}

static inline int nearest_int_fast(float fval) {
    float val = fval + 12582912.f;
    int i;
    std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static inline uint8_t best_index_iq4nl_fast(float x) {
    int idx = nearest_int_fast(x * 16.0f) + 2048;
    if (idx < 0) idx = 0;
    else if (idx > 4096) idx = 4096;
    return g_iq4nl_lut[idx];
}

// Fast importance-weighted IQ4_XS super-block (256 elements) quantizer.
// Matches ggml's quantize_row_iq4_nl_impl(256, 32, ...) with sigma2 importance weighting
// and ntry=1 scale refinement + O(1) L1 LUT index lookup.
static void quantize_row_iq4_xs_fast(const float* __restrict x, block_iq4_xs* __restrict y, int64_t k) {
    const int64_t nblock = k / QK_K;
    float weight[32];
    float scales[QK_K / 32];
    uint8_t L[QK_K];

    for (int64_t ibl = 0; ibl < nblock; ++ibl) {
        const float* xbl = x + QK_K * ibl;
        block_iq4_xs& blk = y[ibl];

        float sigma2 = 0.0f;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j] * xbl[j];
        sigma2 *= 2.0f / (float)QK_K;

        std::memset(blk.qs, 0, QK_K / 2);
        blk.d = ggml_fp32_to_fp16(0.0f);
        blk.scales_h = 0;
        std::memset(blk.scales_l, 0, sizeof(blk.scales_l));

        float max_scale = 0.0f, amax_scale = 0.0f;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const float* xb = xbl + 32 * ib;
            float amax = 0.0f, max_val = 0.0f;
            for (int j = 0; j < 32; ++j) {
                float ax = std::fabs(xb[j]);
                if (ax > amax) {
                    amax = ax;
                    max_val = xb[j];
                }
                // Uniform imatrix=1.0 importance weighting: sqrt(sigma2 + x^2)
                weight[j] = std::sqrt(sigma2 + xb[j] * xb[j]);
            }
            if (amax < 1e-15f) {
                scales[ib] = 0.0f;
                continue;
            }
            float d = max_val / 127.0f; // -max_val / kValuesIq4Nl[0] where kValuesIq4Nl[0] = -127
            float id = 1.0f / d;
            float sumqx = 0.0f, sumq2 = 0.0f;
            for (int j = 0; j < 32; ++j) {
                uint8_t l = best_index_iq4nl_fast(id * xb[j]);
                float q = (float)kValuesIq4Nl[l];
                float w = weight[j];
                sumqx += w * q * xb[j];
                sumq2 += w * q * q;
            }
            d = sumq2 > 0.0f ? sumqx / sumq2 : 0.0f;
            float best = d * sumqx;

            // 7-step scale perturbation (-3..+3 around values[0] = -127, including itry = 0)
            for (int itry = -3; itry <= 3; ++itry) {
                id = ((float)itry - 127.0f) / max_val;
                sumqx = 0.0f;
                sumq2 = 0.0f;
                for (int j = 0; j < 32; ++j) {
                    uint8_t l = best_index_iq4nl_fast(id * xb[j]);
                    float q = (float)kValuesIq4Nl[l];
                    float w = weight[j];
                    sumqx += w * q * xb[j];
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2) {
                    d = sumqx / sumq2;
                    best = d * sumqx;
                }
            }
            scales[ib] = d;
            float abs_d = std::fabs(d);
            if (abs_d > amax_scale) {
                amax_scale = abs_d;
                max_scale = d;
            }
        }

        float d = -max_scale / 32.0f;
        blk.d = ggml_fp32_to_fp16(d);
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            int l = nearest_int_fast(id * scales[ib]);
            if (l < -32) l = -32;
            else if (l > 31) l = 31;
            float dl = d * (float)l;
            float idl = dl != 0.0f ? 1.0f / dl : 0.0f;
            uint8_t* Lb = L + 32 * ib;
            const float* xb = xbl + 32 * ib;
            for (int j = 0; j < 32; ++j) {
                Lb[j] = best_index_iq4nl_fast(idl * xb[j]);
            }
            int ls = l + 32;
            uint8_t l_l = (uint8_t)(ls & 0xf);
            uint8_t l_h = (uint8_t)(ls >> 4);
            if (ib % 2 == 0) blk.scales_l[ib / 2] = l_l;
            else blk.scales_l[ib / 2] |= (uint8_t)(l_l << 4);
            blk.scales_h |= (uint16_t)((uint16_t)l_h << (2 * ib));
        }

        for (int i = 0; i < QK_K / 32; ++i) {
            for (int j = 0; j < 16; ++j) {
                blk.qs[16 * i + j] = (uint8_t)(L[32 * i + j] | (L[32 * i + 16 + j] << 4));
            }
        }
    }
}

struct TensorDirEntry {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t orig_type = 0;
    uint64_t orig_offset = 0;
    size_t type_file_pos = 0;   // byte offset in GGUF header where `type` (uint32_t) lives
    size_t offset_file_pos = 0; // byte offset in GGUF header where `offset` (uint64_t) lives

    uint32_t new_type = 0;
    uint64_t new_offset = 0;
    uint64_t orig_bytes = 0;
    uint64_t new_bytes = 0;
    bool requant = false;

    uint64_t elements() const {
        uint64_t n = 1;
        for (uint64_t d : shape) n *= d;
        return n;
    }
};

class HeaderCursor {
public:
    HeaderCursor(const uint8_t* base, size_t size) : base_(base), size_(size) {}
    void need(size_t n) const {
        if (pos_ + n > size_) throw std::runtime_error("Unexpected EOF in GGUF header");
    }
    template <class T> T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = read<uint64_t>();
        need((size_t)n);
        std::string s(reinterpret_cast<const char*>(base_ + pos_), (size_t)n);
        pos_ += (size_t)n;
        return s;
    }
    void skip_value(uint32_t t, int depth = 0) {
        if (depth > 4) throw std::runtime_error("GGUF array nesting too deep");
        switch (t) {
            case 0: case 1: case 7: need(1); pos_ += 1; break;
            case 2: case 3: need(2); pos_ += 2; break;
            case 4: case 5: case 6: need(4); pos_ += 4; break;
            case 8: { uint64_t n = read<uint64_t>(); need((size_t)n); pos_ += (size_t)n; break; }
            case 10: case 11: case 12: need(8); pos_ += 8; break;
            case 9: {
                uint32_t et = read<uint32_t>();
                uint64_t cnt = read<uint64_t>();
                for (uint64_t i = 0; i < cnt; ++i) skip_value(et, depth + 1);
                break;
            }
            default: throw std::runtime_error("Unknown GGUF meta type " + std::to_string(t));
        }
    }
    size_t pos() const { return pos_; }
private:
    const uint8_t* base_;
    size_t size_;
    size_t pos_ = 0;
};

uint64_t tensor_bytes(uint32_t type, uint64_t nelements) {
    size_t bs = ggml_blck_size((ggml_type)type);
    size_t ts = ggml_type_size((ggml_type)type);
    if (bs == 0 || nelements % bs != 0) {
        throw std::runtime_error("Invalid block size for type " + std::to_string(type));
    }
    return (nelements / bs) * ts;
}

bool is_routed_expert_tensor(const std::string& name) {
    return name.find("_exps.weight") != std::string::npos;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <input.gguf> <output.gguf>\n", argv[0]);
        return 1;
    }
    const std::string in_path = argv[1];
    const std::string out_path = argv[2];

    init_iq4nl_lut();
    const auto t_start = std::chrono::steady_clock::now();

    int fd_in = ::open(in_path.c_str(), O_RDONLY);
    if (fd_in < 0) {
        std::perror("open input");
        return 1;
    }
    struct stat st{};
    if (fstat(fd_in, &st) != 0) {
        std::perror("fstat");
        return 1;
    }
    const size_t in_size = (size_t)st.st_size;
    const uint8_t* in_base = (const uint8_t*)mmap(nullptr, in_size, PROT_READ, MAP_PRIVATE, fd_in, 0);
    if (in_base == MAP_FAILED) {
        std::perror("mmap input");
        return 1;
    }

    HeaderCursor c(in_base, in_size);
    uint32_t magic = c.read<uint32_t>();
    uint32_t version = c.read<uint32_t>();
    if (magic != 0x46554747u || version != 3) {
        std::fprintf(stderr, "Invalid GGUF v3 header\n");
        return 1;
    }
    uint64_t n_tensors = c.read<uint64_t>();
    uint64_t n_kv = c.read<uint64_t>();

    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = c.str();
        uint32_t vtype = c.read<uint32_t>();
        if (key == "general.alignment" && vtype == 4) {
            alignment = c.read<uint32_t>();
        } else {
            c.skip_value(vtype);
        }
    }

    std::vector<TensorDirEntry> tensors(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorDirEntry& t = tensors[i];
        t.name = c.str();
        uint32_t nd = c.read<uint32_t>();
        t.shape.resize(nd);
        for (uint32_t d = 0; d < nd; ++d) t.shape[d] = c.read<uint64_t>();
        t.type_file_pos = c.pos();
        t.orig_type = c.read<uint32_t>();
        t.offset_file_pos = c.pos();
        t.orig_offset = c.read<uint64_t>();
    }

    const size_t header_end = c.pos();
    const size_t data_start = (header_end + alignment - 1) / alignment * alignment;

    auto align_up = [alignment](uint64_t off) -> uint64_t {
        return (off + alignment - 1) / alignment * alignment;
    };

    uint64_t cur_new_offset = 0;
    int n_requant = 0;
    int n_kept_iq4xs = 0;
    uint64_t orig_routed_bytes = 0;
    uint64_t new_routed_bytes = 0;

    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorDirEntry& t = tensors[i];
        uint64_t elems = t.elements();
        t.orig_bytes = tensor_bytes(t.orig_type, elems);

        if (is_routed_expert_tensor(t.name)) {
            orig_routed_bytes += t.orig_bytes;
            t.new_type = GGML_TYPE_IQ4_XS; // 23
            t.new_bytes = tensor_bytes(t.new_type, elems);
            new_routed_bytes += t.new_bytes;
            if (t.orig_type != GGML_TYPE_IQ4_XS) {
                t.requant = true;
                ++n_requant;
            } else {
                t.requant = false;
                ++n_kept_iq4xs;
            }
        } else {
            t.new_type = t.orig_type;
            t.new_bytes = t.orig_bytes;
            t.requant = false;
        }
        t.new_offset = cur_new_offset;
        cur_new_offset = align_up(cur_new_offset + t.new_bytes);
    }

    const uint64_t out_file_size = data_start + cur_new_offset;
    std::printf("=== Ornith 1.5 35B-A3B Optimum IQ4_XS Requantizer ===\n");
    std::printf("Input GGUF : %s (%.3f GiB)\n", in_path.c_str(), (double)in_size / (1024.0 * 1024.0 * 1024.0));
    std::printf("Output GGUF: %s (%.3f GiB)\n", out_path.c_str(), (double)out_file_size / (1024.0 * 1024.0 * 1024.0));
    std::printf("Routed experts: %d already IQ4_XS (copied bit-for-bit), %d to requantize (Q6_K/Q5_K/Q8_0 -> IQ4_XS)\n",
                n_kept_iq4xs, n_requant);
    std::printf("Routed expert footprint: %.3f GiB -> %.3f GiB (saved %.3f GiB)\n",
                (double)orig_routed_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)new_routed_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)(orig_routed_bytes - new_routed_bytes) / (1024.0 * 1024.0 * 1024.0));
    std::fflush(stdout);

    // Prepare patched header buffer
    std::vector<uint8_t> header_buf(data_start, 0);
    std::memcpy(header_buf.data(), in_base, header_end);
    for (const TensorDirEntry& t : tensors) {
        std::memcpy(header_buf.data() + t.type_file_pos, &t.new_type, sizeof(uint32_t));
        std::memcpy(header_buf.data() + t.offset_file_pos, &t.new_offset, sizeof(uint64_t));
    }

    FILE* fout = std::fopen(out_path.c_str(), "wb");
    if (!fout) {
        std::perror("fopen output");
        return 1;
    }
    // Use 16 MiB stdio buffer for high-speed NVMe sequential writes
    std::vector<char> io_buf(16 * 1024 * 1024);
    std::setvbuf(fout, io_buf.data(), _IOFBF, io_buf.size());

    if (std::fwrite(header_buf.data(), 1, data_start, fout) != data_start) {
        std::perror("fwrite header");
        return 1;
    }

    const int n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::printf("Using %d CPU threads for parallel IQ4_XS quantization...\n", n_threads);
    std::fflush(stdout);

    // Reusable buffer for requantized tensor output (max tensor is 136 MiB in IQ4_XS)
    std::vector<uint8_t> out_tensor_buf(136ULL * 1024ULL * 1024ULL);
    const std::vector<uint8_t> zero_pad(alignment, 0);

    int done_requant = 0;
    uint64_t file_pos = data_start;

    for (uint64_t i = 0; i < n_tensors; ++i) {
        const TensorDirEntry& t = tensors[i];
        const uint8_t* src_ptr = in_base + data_start + t.orig_offset;

        if (!t.requant) {
            if (std::fwrite(src_ptr, 1, t.new_bytes, fout) != t.new_bytes) {
                std::perror("fwrite copy tensor");
                return 1;
            }
        } else {
            ++done_requant;
            const auto t0 = std::chrono::steady_clock::now();
            if (out_tensor_buf.size() < t.new_bytes) out_tensor_buf.resize(t.new_bytes);

            const int64_t n_in = (int64_t)t.shape[0];
            const int64_t total_rows = (int64_t)(t.elements() / (uint64_t)n_in); // 512 * 256 = 131072 or 2048 * 256 = 524288
            const size_t src_row_bytes = (size_t)(t.orig_bytes / (uint64_t)total_rows);
            const size_t dst_row_bytes = (size_t)(t.new_bytes / (uint64_t)total_rows);

            std::atomic<int64_t> next_chunk{0};
            const int64_t chunk_rows = 256;

            auto worker = [&]() {
                std::vector<float> frow((size_t)chunk_rows * (size_t)n_in);
                for (;;) {
                    int64_t r0 = next_chunk.fetch_add(chunk_rows, std::memory_order_relaxed);
                    if (r0 >= total_rows) break;
                    int64_t r1 = std::min(total_rows, r0 + chunk_rows);
                    int64_t nrows = r1 - r0;
                    int64_t nelems = nrows * n_in;

                    const uint8_t* s = src_ptr + (size_t)r0 * src_row_bytes;
                    uint8_t* d = out_tensor_buf.data() + (size_t)r0 * dst_row_bytes;

                    if (t.orig_type == GGML_TYPE_Q6_K) {
                        dequantize_row_q6_K((const block_q6_K*)s, frow.data(), nelems);
                    } else if (t.orig_type == GGML_TYPE_Q5_K) {
                        dequantize_row_q5_K((const block_q5_K*)s, frow.data(), nelems);
                    } else if (t.orig_type == GGML_TYPE_Q8_0) {
                        dequantize_row_q8_0((const block_q8_0*)s, frow.data(), nelems);
                    } else {
                        std::fprintf(stderr, "Unsupported source type %u\n", t.orig_type);
                        std::abort();
                    }

                    quantize_row_iq4_xs_fast(frow.data(), (block_iq4_xs*)d, nelems);
                }
            };

            std::vector<std::thread> pool;
            pool.reserve(n_threads);
            for (int th = 0; th < n_threads; ++th) pool.emplace_back(worker);
            for (auto& th : pool) th.join();

            if (!ggml_validate_row_data(GGML_TYPE_IQ4_XS, out_tensor_buf.data(), t.new_bytes)) {
                std::fprintf(stderr, "Validation failed for %s\n", t.name.c_str());
                return 1;
            }

            if (std::fwrite(out_tensor_buf.data(), 1, t.new_bytes, fout) != t.new_bytes) {
                std::perror("fwrite requant tensor");
                return 1;
            }

            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (done_requant % 5 == 0 || done_requant == n_requant || done_requant == 1) {
                const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                std::printf("[%2d/%2d] %-30s (type %2u -> IQ4_XS, %6.1f MiB -> %5.1f MiB) in %.2fs [total %.1fs]\n",
                            done_requant, n_requant, t.name.c_str(), t.orig_type,
                            (double)t.orig_bytes / (1024.0 * 1024.0),
                            (double)t.new_bytes / (1024.0 * 1024.0),
                            dt, elapsed);
                std::fflush(stdout);
            }
        }

        file_pos += t.new_bytes;
        uint64_t padded = align_up(file_pos - data_start) + data_start;
        if (padded > file_pos) {
            size_t pad_len = (size_t)(padded - file_pos);
            if (std::fwrite(zero_pad.data(), 1, pad_len, fout) != pad_len) {
                std::perror("fwrite pad");
                return 1;
            }
            file_pos = padded;
        }
    }

    std::fclose(fout);
    munmap((void*)in_base, in_size);
    ::close(fd_in);

    const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::printf("SUCCESS: Wrote %s (%.3f GiB) in %.2f seconds.\n",
                out_path.c_str(), (double)file_pos / (1024.0 * 1024.0 * 1024.0), total_s);
    return 0;
}
