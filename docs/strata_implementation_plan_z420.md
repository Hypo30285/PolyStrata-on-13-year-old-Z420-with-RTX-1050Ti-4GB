# Strata Implementation Plan for HP Z420 ("The Z") — Deep Technical Review (Rev 2)

## 1. Executive Summary & Architectural Audit

Following a code-level audit of both upstream **`Niko1221/Strata`** and the community multi-architecture fork **`VecSzn/PolyStrata`** (`strata-poly`) against the HP Z420's exact hardware profile, **four critical engineering realities** emerged that reshape the implementation plan:

1. **Runtime Selection (`VecSzn/PolyStrata` vs. Upstream `Niko1221/Strata`):**
   * Upstream `Niko1221/Strata` (`strata` binary) is hardcoded exclusively for the **125B `Qwen3.8-Flash-Next`** (`qwen4exp`) architecture. Even its smallest `Coder IQ1_M` variant requires a **58.4 GB two-shard download** (`29.6 GB` Shard 1 + `~28.8 GB` Shard 2 Per-Layer Embedding n-gram table) and **~4.5–6.0 GB of minimum VRAM** just for dense weights + GDN state.
   * The community fork **`VecSzn/PolyStrata`** includes the upstream `strata` engine *plus* the **`strata-poly`** runtime (`src/poly/`), which maps standard **community GGUF models in-place (`mmap`)** with zero conversion overhead and natively supports **26B–35B MoE models** (`Qwen3.6-35B-A3B`, `Ornith 1.5 35B-A3B`, `Gemma 4 26B-A4B`) whose `Q4_K_M` / `Q5_K_M` quants land squarely inside your **27 GB to 35 GB physical RAM window**.

2. **The 4 GB VRAM Bottleneck on GTX 1050 Ti (Requires Runtime + Model Customization):**
   * Both `strata` and `strata-poly` upload **all non-routed ("dense") weights** (token embeddings, attention Q/K/V/O, GDN linear attention, router gates, shared experts, and LM head) into GPU VRAM.
   * Out of the box, 35B MoE models consume **4.0 GB to 5.5 GB of VRAM before caching a single routed expert**, which will overflow the GTX 1050 Ti's **~3,694 MiB of free VRAM** (`4,096 MiB` total minus `~402 MiB` desktop Xorg usage).
   * **Solution:** We will apply **surgical runtime patches** (keeping `token_embd.weight` in CPU RAM, disabling the GPU MTP draft block in favor of 0-VRAM CPU suffix speculation, shrinking staging/prompt-batch buffers) and **custom GGUF dense-tensor re-quantization** so non-expert VRAM drops from ~4.8 GB to **~1.9–2.3 GB**, leaving **~1.2–1.5 GB of VRAM** for hot GPU expert caching and workspace.

3. **Custom CUDA 6.1 (`sm_61`) Compilation on Ubuntu 26.04 (`GCC 15.2` + `CUDA 12.x`):**
   * Driver `580.178.04` reports CUDA 13.0 capability, but CUDA 13 removed Pascal `sm_61`. We must compile against **CUDA 12.8/12.9 (`nvcc`)** with `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61`.
   * Because Ubuntu 26.04 ships with **`gcc 15.2.0`**, CUDA 12's `nvcc` will abort with `#error -- unsupported GNU version!` unless we pair it with **`gcc-13`/`g++-13`** via `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13`.
   * Hardware `dp4a.s32.s32` (INT8 dot-product accumulate) and `STRATA_BF16_EMULATED=1` (FP32-backed BF16 storage for pre-Ampere GPUs) are automatically activated by Strata's CMake and `include/strata/kernels/dp4a.hpp` when `__CUDA_ARCH__ == 610`.

4. **Sandy Bridge Xeon E5-2690 (No AVX2 / No FMA / No F16C):**
   * Passing `-DSTRATA_ISA_FLOOR=avx` compiles the vendored `ggml-cpu` expert backend (`src/poly/cpu_experts.cpp`) with `GGML_AVX=ON`, `GGML_AVX2=OFF`, `GGML_FMA=OFF`, `GGML_F16C=OFF`, and compiles the AVX 1.0 router lookahead kernel (`router_lookahead_avx.cpp`).
   * Because Sandy Bridge lacks hardware `F16C` (`vcvtph2ps`) and 256-bit integer SIMD (`AVX2`), complex codebook quants (`IQ1_M`, `IQ2_XXS`) incur heavy CPU table-lookup overhead on Sandy Bridge. Using **4-bit/5-bit K-quants (`Q4_K_M`, `Q5_K_M`, `Q4_0`)** on a 35B-A3B model matches your **27–35 GB RAM target** while maximizing SSE4.1/AVX1 throughput across all 16 threads.

---

## 2. HP Z420 Hardware & Memory Budget Architecture

```
+---------------------------------------------------------------------------------------+
|                        HP Z420 Customized PolyStrata Architecture                     |
+---------------------------------------------------------------------------------------+
|                                                                                       |
|  [NVIDIA GTX 1050 Ti — 4,096 MiB GDDR5 (Pascal sm_61, DP4A INT8, BF16 Emulated)]      |
|   ├── Desktop / Xorg Overhead:         ~402 MiB                                       |
|   ├── Dense Weights (Quantized Q4_K):  ~1,350 MiB (Attn + GDN + Shared Exp + Output)  |
|   ├── GDN Recurrent State (30 layers): ~39 MiB                                        |
|   ├── Attention KV Cache (10 layers):  ~210–425 MiB (8K–16K ctx, dynamic kv_grow)     |
|   ├── Custom Stage & Scratch Buffers:  ~256 MiB (--prompt-batch 256 --stage-mib 256)  |
|   └── Hot GPU Expert Cache + Reserve:  ~1,200–1,400 MiB (DP4A MMVQ execution)         |
|                                         ▲                                             |
|                                         │ PCIe 3.0 x16 (~15.75 GB/s, Pinned Staging)  |
|                                         ▼                                             |
|  [System RAM — 64 GB ECC DDR3 Quad-Channel (~51.2 GB/s, Xeon E5-2690 8C/16T AVX1)]    |
|   ├── Routed Expert Arena (mmap):      24.0 GB – 31.0 GB (100% Pinned in DRAM)        |
|   ├── Token Embedding Table (CPU):     ~0.95 GB (Offloaded from 1050 Ti VRAM!)        |
|   ├── CPU Expert Threadpool (16T):     Executes cold routed experts via ggml-cpu AVX1 |
|   ├── CPU Suffix Speculative Drafter:  0 MiB VRAM, proposes 1–4 draft tokens on CPU   |
|   └── Free RAM Headroom:               ~22 GB – 30 GB (Guarantees 0 MB NVMe Swap)     |
|                                         ▲                                             |
|                                         │ Sequential Cold Boot Only (~12–18s)         |
|                                         ▼                                             |
|  [Fast NVMe Storage — 2 TB SSD (/mnt/nvme_storage)]                                   |
|   └── Custom GGUF Models (27–35 GB):   Zero swap activity (si=0, so=0) during decode  |
+---------------------------------------------------------------------------------------+
```

---

## 3. Deep Dive: Compiler & ISA Engineering for the Z

### 3.1 Custom CUDA 6.1 (`sm_61`) Build Pipeline
Inspecting `CMakeLists.txt` in `Niko1221/Strata` and `VecSzn/PolyStrata` reveals how Pascal `sm_61` is gated and compiled:

* **CMake Gate (`CMakeLists.txt` lines 48, 162–188):**
  By default, Strata rejects compute capabilities `< 7.0` unless `-DSTRATA_EXPERIMENTAL_SM60=ON` is explicitly passed.
* **Automatic BF16 Emulation (`STRATA_BF16_EMULATED=1`):**
  When `CMAKE_CUDA_ARCHITECTURES` contains `61`, `CMakeLists.txt` automatically sets `STRATA_NEED_BF16_EMULATED=ON`, which defines `-DSTRATA_BF16_EMULATED=1`. This replaces native `sm_80+` `__nv_bfloat16` arithmetic with bit-cast FP32 math while keeping memory storage at 16 bits per element.
* **Automatic Hardware DP4A Selection (`include/strata/kernels/dp4a.hpp`):**
  There is no separate CMake flag for DP4A; instead, `dp4a.hpp` checks `#if __CUDA_ARCH__ >= 610` inside device compilation and emits native `__dp4a(a, b, c)` instructions (`dp4a.s32.s32` in PTX) for all quantized `gemv_mmvq.cu` and `gemm_dp4a.cu` kernels.
* **Resolving the Ubuntu 26.04 GCC 15 + CUDA 12 Conflict:**
  * CUDA 13.0 dropped `sm_61` support (`nvcc fatal : Unsupported gpu architecture 'compute_61'`).
  * Therefore, we must install the **CUDA 12.8/12.9 compiler (`nvcc`)** alongside **`gcc-13` and `g++-13`** so CUDA 12's front-end parser does not choke on GCC 15.2's C++ headers.

### 3.2 Sandy Bridge No-AVX2 (`STRATA_ISA_FLOOR=avx`) Pipeline
Inspecting `CMakeLists.txt` (lines 51–54, 195–265) and `src/poly/cpu_experts.cpp`:

* **How CPU Experts Execute:**
  In `strata-poly`, any routed expert not currently cached in the GTX 1050 Ti's VRAM is computed on the CPU by `CpuExpertPool` (`src/poly/cpu_experts.cpp`), which calls `ggml_vec_dot_*` routines from the vendored `third_party/ggml-cpu` library across worker threads.
* **What `-DSTRATA_ISA_FLOOR=avx` Configures:**
  1. Sets `GGML_NATIVE=OFF`, `GGML_AVX=ON`, `GGML_AVX2=OFF`, `GGML_FMA=OFF`, `GGML_F16C=OFF`, `GGML_AVX512=OFF`.
  2. Compiles `src/kernels/router_lookahead_avx.cpp` with `-mavx` (providing 256-bit float SIMD for router lookahead without requiring AVX2).
  3. Compiles `router_lookahead_avx2.cpp` and `dequant_avx512.cpp` into separate translation units guarded by runtime `cpu_features()` CPUID checks so the Xeon E5-2690 never executes an illegal instruction (`SIGILL`).
* **Why K-Quants (`Q4_K_M` / `Q5_K_M`) Beat 1-Bit/2-Bit I-Quants (`IQ1_M`) on Sandy Bridge:**
  * On Sandy Bridge (`GGML_F16C=OFF`, `GGML_AVX2=OFF`), `ggml-cpu` executes quantized dot products using 128-bit **SSSE3/SSE4.1** integer instructions (`_mm_maddubs_epi16`) and a 64 KB L1/L2 lookup table for FP16 scale conversion.
  * `IQ1_M` and `IQ2_XXS` rely on non-linear importance grids that require extra bit-shifts and table lookups per weight, which are heavily penalized without `AVX2` (`vpshufb` / `vpgatherdd`).
  * By contrast, **`Q4_K_M` / `Q5_K_M` / `Q4_0`** unpack cleanly with simple nibble masks and shifts, allowing the 8-core / 16-thread Xeon E5-2690 to saturate its ~51 GB/s quad-channel DDR3 bus while giving dramatically higher model intelligence and coding accuracy!

---

## 4. Deep Dive: Overcoming the 4 GB VRAM Ceiling & Customizing Community Models

### 4.1 Why Stock Models Overflow 4 GB VRAM in `strata-poly`
In `src/poly/engine.cpp` (`Engine::load_model` and `Engine::upload`), the engine allocates GPU VRAM in this order before placing any routed experts in VRAM:
1. **Token Embedding Table (`token_embd.weight`):** Uploaded to VRAM (`d_tok_embd_`). For Qwen3.6 (`248,320 × 2,048`), if stored as `BF16` or dequantized due to an unsupported quant type, this single tensor takes **~970 MiB of VRAM**!
2. **Per-Layer Dense Weights (40 Layers):** Attention (`wq`, `wk`, `wv`, `wo`), Gated DeltaNet (`ssm_in`, `ssm_conv1d`, `ssm_out`), Router gates (`ffn_gate_inp`), Shared Experts (`ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`), and Output LM Head (`output.weight`, ~500–970 MiB).
   * **Crucial Code Discovery (`src/poly/engine.cpp:648`):**
     ```cpp
     if (t->type == gguf::GGML_TYPE_BF16) { ... }
     else if (kernels::native_mmvq_supported(t->type)) { ... /* stays quantized in VRAM! */ }
     else {
         // Dequantize on CPU to BF16 -> EXPANDS TO 16-BIT IN VRAM!
         dequant_Bytes_to_bf16(src, t->type, bf16.data(), n);
     }
     ```
     Whenever a community GGUF uses a quant type for dense/shared-expert tensors that is not in `native_mmvq_supported`, `Engine::upload` silently expands it to **16-bit BF16 in VRAM**!
3. **MTP Speculative Draft Layer (`nextn` / `mtp`):** Loads an entire extra 256-expert layer + projections (~500–800 MiB VRAM).
4. **Staging & Scratch Workspace (`src/poly/engine.hpp:51-55`):**
   * `prompt_batch` defaults to `1024` tokens (~350–600 MiB scratch VRAM).
   * `stage_mib` defaults to `2048` MiB.
   * `vram_reserve_mib` defaults to `600` MiB.

### 4.2 Our 4-Step Customization Pipeline for the GTX 1050 Ti (4 GB)

To make 27–35 GB community models run effortlessly inside the GTX 1050 Ti's **3,694 MiB free VRAM**, we will implement four targeted customizations:

#### Customization A: Runtime Patch — Pin `token_embd.weight` in CPU System RAM (`~500–970 MiB VRAM Saved`)
* **How `token_embd.weight` is used:** Unlike `output.weight` (which multiplies the hidden state against all 248,320 vocabulary rows on GPU), `token_embd.weight` is a pure **row lookup** (`1 row = 2,048 floats = 4 KB` per decoded token!).
* **The Patch (`src/poly/engine.cpp`):** Add an option (`--cpu-embd` / automatic when VRAM < 6 GB) that leaves `token_embd.weight` in `mmap`'d CPU DDR3 RAM, dequantizes the requested token row(s) on the CPU in `< 2 microseconds`, and copies just `batch * 4 KB` over PCIe into `d_x_`. This immediately frees **500–970 MiB of GPU VRAM** at virtually zero performance cost.

#### Customization B: Custom GGUF Dense-Tensor Surgery (`~800–1,200 MiB VRAM Saved`)
* We will build a lightweight Python/GGUF customization utility (`scripts/customize_gguf_for_z420.py` + `llama-quantize --tensor-type`) that inspects any community GGUF model and ensures:
  1. **All GPU-resident dense tensors** (`attn_q`, `attn_k`, `attn_v`, `attn_output`, `ssm_in`, `ssm_out`, `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`, and `output.weight`) are encoded in **native GPU DP4A MMVQ types (`Q4_K` or `Q5_K` or `Q8_0`)** so `Engine::upload` never expands them to 16-bit BF16 in VRAM.
  2. **All CPU-resident routed expert tensors** (`blk.*.ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight`) use **Sandy Bridge-friendly K-quants (`Q4_K` / `Q5_K`)** tuned to land the total GGUF size right at **27 GB to 33 GB**.

#### Customization C: Disable GPU MTP Draft Layer & Use Zero-VRAM CPU Suffix Speculation (`~600 MiB VRAM Saved`)
* Run with `--no-draft` (which skips uploading the `blk.40` / MTP weights to GPU VRAM) while keeping `SuffixTree` CPU prompt-lookup drafting active (or patch `engine.cpp` so `--cpu-suffix-only` runs the `SuffixTree` drafter without loading the GPU MTP block).

#### Customization D: Z420-Tuned Runtime Buffer & KV Flags (`~700 MiB VRAM Saved`)
* Configure default runtime flags for the Z420:
  * `--prompt-batch 256` (reduces prompt scratch buffer allocation by 75%)
  * `--stage-mib 256` (caps dynamic expert staging buffer to 256 MiB)
  * `--vram-reserve-mib 256` (tuned for Linux Xorg desktop overhead)
  * `--max-context 16384` (with `kv_grow=true`, allocates KV in 2K chunks; only 10 of 40 layers in Qwen3.6 use KV cache, costing just ~52 MiB per 2K tokens!).

---

## 5. Recommended Community Models (27 GB – 35 GB Target Window)

With **`VecSzn/PolyStrata`** (`strata-poly`) + our Z420 customization script, we can run standard community GGUF models from HuggingFace (`unsloth`, `bartowski`, `TicTac12x`) without any 28 GB Shard-2 PLE penalty:

| Priority | Community Model & Architecture | Quantization Variant | Disk / RAM Footprint | Est. VRAM (After Z420 Customizations) | Why It Fits "The Z" |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1 (Primary)** | **`Ornith 1.5 35B-A3B`** (`qwen35moe`, community coding/reasoning tune by `TicTac12x`) | `APEX-MTP-I-Quality` / Custom `Q4_K_M` (`z420-q4k`) | **~23.7 – 27.5 GB** | **~2.1 GB** (+ ~1.3 GB GPU expert cache) | **Best Coding & Reasoning 35B MoE**: Only 3B active parameters per token; 30 of 40 layers are linear GDN (tiny 39 MB state); fits 100% in 64 GB DDR3 with >30 GB free RAM. |
| **2 (General & Coding)** | **`Qwen3.6-35B-A3B`** (`qwen35moe`, official Alibaba base/instruct via `unsloth` / `bartowski`) | `UD-Q4_K_M` / `Q5_K_M` (with Z420 dense-Q4_K pass) | **~22.7 – 28.5 GB** | **~2.1 GB** (+ ~1.3 GB GPU expert cache) | **Gold-Standard 35B MoE**: `Q4_K_M` (22.7 GB) or `Q5_K_M` (~28 GB) hits the exact 27–35 GB sweet spot and uses fast SSE4.1/AVX1 K-quant unpacking on the E5-2690. |
| **3 (Alternative)** | **`Gemma 4 26B-A4B`** (`gemma4`, Google 26B MoE, 128 experts) | `Q5_K_M` / `Q6_K` (with Z420 dense-Q4_K pass) | **~22.0 – 27.0 GB** | **~2.5 GB** (+ ~0.9 GB GPU expert cache) | Strong alternative architecture supported natively by `strata-poly`. |
| **4 (Legacy Upstream)** | **`Qwen3.8-Flash-Next Coder`** (`qwen4exp`, 125B pruned to 256 experts) | `IQ1_M` (Shard 1 only, requires custom `dense.bin` re-quant) | **29.6 GB** (+ 28.8 GB Shard 2 if PLE used) | **~3.2 GB** (Tight on 4 GB 1050 Ti) | Supported by upstream `strata` binary, but `IQ1_M` is slower on AVX1-only CPUs and Shard 2 pushes total size to 58.4 GB. Kept as optional secondary target. |

---

## 6. Step-by-Step Implementation Phases

### Phase 1: Storage Layout & Toolchain Setup on `/mnt/nvme_storage`
1. Create project hierarchy on the 2 TB NVMe drive:
   * `/mnt/nvme_storage/ai_projects/polystrata` (Repository & custom runtime build)
   * `/mnt/nvme_storage/ai_projects/strata_models` (Raw and Z420-customized GGUF models)
2. Install the CUDA 12.x compiler toolkit (`nvidia-cuda-toolkit` / CUDA 12.8/12.9) and **GCC 13 host compiler** (`gcc-13`, `g++-13`) so `nvcc` can compile `sm_61` device code cleanly on Ubuntu 26.04.

### Phase 2: Source Customization & Compilation for `sm_61` + `AVX 1.0`
1. Clone `https://github.com/VecSzn/PolyStrata.git` into `/mnt/nvme_storage/ai_projects/polystrata`.
2. Apply the **Z420 4 GB VRAM Runtime Patch** to `src/poly/engine.hpp` and `src/poly/engine.cpp`:
   * Add `--cpu-embd` support so `token_embd.weight` remains in CPU DDR3 RAM and streams only the active token embeddings (`batch × 4 KB`) to the GPU per step.
   * Allow CPU `SuffixTree` speculative drafting to stay active even when `--no-draft` disables the GPU MTP layer.
   * Set Z420-safe default caps for 4 GB GPUs (`prompt_batch = 256`, `stage_mib = 256`, `vram_reserve_mib = 256`).
3. Configure and build `strata-poly` and `strata` with CMake + Ninja:
   ```bash
   cmake -B build -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DSTRATA_ENABLE_CUDA=ON \
     -DSTRATA_EXPERIMENTAL_SM60=ON \
     -DCMAKE_CUDA_ARCHITECTURES=61 \
     -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
     -DSTRATA_ISA_FLOOR=avx \
     -DSTRATA_PORTABLE=OFF \
     -DCMAKE_C_FLAGS="-march=sandybridge -mavx -mno-avx2 -mno-fma -mno-f16c" \
     -DCMAKE_CXX_FLAGS="-march=sandybridge -mavx -mno-avx2 -mno-fma -mno-f16c"
   cmake --build build --target strata-poly strata -j 16
   ```
4. **Binary ISA Audit:** Run `objdump -d build/strata-poly` check script to verify **zero** AVX2/FMA/F16C instructions exist in the unguarded code paths, and verify GPU detection (`compute capability 6.1, 4.0 GiB`).

### Phase 3: Community Model Customization Tooling (`customize_gguf_for_z420.py`)
1. Create `/mnt/nvme_storage/ai_projects/polystrata/scripts/customize_gguf_for_z420.py`:
   * Audits every tensor in a downloaded community GGUF (`Qwen3.6-35B-A3B` or `Ornith 1.5 35B-A3B`).
   * Calculates exact GPU VRAM vs. CPU DDR3 RAM footprint before running the model.
   * Re-quantizes any non-MMVQ dense/shared-expert/output tensors (`BF16`/`F16`) into DP4A-accelerated `Q4_K` / `Q8_0` so `Engine::upload` never expands them to 16-bit BF16 on the GTX 1050 Ti.
2. Download and customize the primary target model (**`Qwen3.6-35B-A3B` `Q4_K_M` / `Q5_K_M`** or **`Ornith 1.5 35B-A3B`**) into `/mnt/nvme_storage/ai_projects/strata_models/`.

### Phase 4: Zero-Swap Verification, Benchmarking & Service Launcher
1. Launch `strata-poly` with the Z420 profile:
   ```bash
   ./build/strata-poly \
     -m /mnt/nvme_storage/ai_projects/strata_models/model-z420.gguf \
     --cpu-embd \
     --no-draft \
     --max-context 16384 \
     --prompt-batch 256 \
     --stage-mib 256 \
     --vram-reserve-mib 256 \
     --threads 16 \
     --port 8080
   ```
2. Verify all 5 acceptance gates:
   * **Gate 1 (No `SIGILL`):** Clean startup and token generation on Xeon E5-2690.
   * **Gate 2 (VRAM $\le 3.6\text{ GiB}$):** `nvidia-smi` confirms dense weights + KV + GPU expert cache stay safely below 4,096 MiB without desktop artifacts or OOM.
   * **Gate 3 (Zero NVMe Swap):** `vmstat 1` and `free -h` confirm ~27–32 GB physical RAM usage with `si=0` and `so=0`.
   * **Gate 4 (DP4A + AVX1 Throughput):** Benchmark prompt prefill and token decode tok/s.
   * **Gate 5 (Documentation):** Create `/home/tom/.local/bin/strata-z420` launcher and update [software_and_models.md](file:///home/tom/.gemini/antigravity/scratch/software_and_models.md).

---

## 7. Open Decisions for Your Approval

1. **Primary 27–35 GB Model Selection:**
   * Would you like to start with **`Qwen3.6-35B-A3B` (`UD-Q4_K_M` / `Q5_K_M`, ~23–28 GB)** or the community coding/reasoning fine-tune **`Ornith 1.5 35B-A3B` (`APEX-MTP-I-Quality`, ~24–28 GB)**? (Both use the `qwen35moe` architecture and fit 100% in physical RAM with zero swap).
2. **Proceed to Phase 1 & 2 Execution:**
   * Shall we begin setting up the CUDA 12 + GCC 13 toolchain, cloning `PolyStrata` to `/mnt/nvme_storage/ai_projects/polystrata`, and applying the GTX 1050 Ti 4 GB VRAM patches?
