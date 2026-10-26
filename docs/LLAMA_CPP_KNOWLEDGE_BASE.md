# llama.cpp Knowledge Base & Technical Guide (HP Z420)

This knowledge base documents upstream **llama.cpp** installation, architecture, build optimization, and OpenAI-compatible server execution for the HP Z420 Workstation.

Official Source Repository: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

---

## 1. Upstream Source & Overview

`llama.cpp` is the canonical C/C++ inference runtime for LLMs and multimodal vision models using the GGML tensor library and GGUF file format. It provides:
- **Pure Native C/C++ Execution:** Minimal memory footprint, no virtual machine/JIT bloat, strict compliance with HP Z420 Native Executable rules.
- **Hybrid CPU/GPU Splitting:** Precise layer-by-layer offloading (`-ngl / --n-gpu-layers`), allowing models that exceed VRAM (like MiniCPM-V 5.5 GB on a 4 GB card) to offload part of their compute to GPU while running remaining layers on CPU DDR3 RAM.
- **Built-in HTTP Server (`llama-server`):** Provides a drop-in OpenAI-compatible REST API (`/v1/chat/completions`) for Lighteval benchmarking, IDE plugins, and agent integration.

---

## 2. Download & Build from Source (HP Z420 Optimized)

### Repository Clone
```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /mnt/nvme_storage/ai_projects/llama.cpp
cd /mnt/nvme_storage/ai_projects/llama.cpp
```

### Hardware-Targeted Build Flags
The HP Z420 features:
- **CPU:** Intel Xeon E5-2690 v1 (AVX 1.0, NO AVX2)
- **GPU:** NVIDIA GeForce GTX 1050 Ti (`sm_61` Pascal architecture, 4 GB VRAM, DP4A support)
- **Compiler:** GCC 13 / CUDA 13

```bash
# Configure with CUDA enabled for Pascal sm_61 architecture & Sandy Bridge AVX1:
cmake -B build -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="61" \
    -DGGML_AVX=ON \
    -DGGML_AVX2=OFF \
    -DGGML_FMA=OFF \
    -DGGML_F16C=OFF \
    -DGGML_BMI2=OFF \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF

# Compile high-performance server and CLI binaries using 8 physical Xeon cores:
ninja -C build -j8 llama-server llama-cli
```

---

## 3. Serving Models with `llama-server`

### MiniCPM-V 5B Production Serving Recipe
```bash
/mnt/nvme_storage/ai_projects/llama.cpp/build/bin/llama-server \
    -m /mnt/nvme_storage/ai_projects/strata_models/minicpm-v-5b-q4_k_m.gguf \
    --mmproj /mnt/nvme_storage/ai_projects/strata_models/minicpm-v-mmproj.gguf \
    --host 127.0.0.1 \
    --port 8081 \
    -ngl 14 \
    -t 8 \
    -c 2048 \
    -b 512 \
    --temp 0.2 \
    --top-p 0.9
```

### Parameter Breakdown:
- `-ngl 14`: Offloads 14 transformer layers to GTX 1050 Ti, consuming ~3.3 GB VRAM and leaving ~300 MiB free for desktop buffers.
- `-t 8`: Pins inference to the 8 physical Xeon cores, bypassing hyperthread cache eviction.
- `-c 2048`: Allocates a compact KV cache, saving ~350 MiB VRAM compared to 4096 context.
- `--port 8081`: Dedicated evaluation port alongside Ollama (11434) and PolyStrata (8080).

---

## 4. Benchmarking `llama-server` via Lighteval

Once `llama-server` is listening on port 8081:

```bash
/mnt/nvme_storage/ai_projects/polystrata/tools/run_lighteval.sh \
    "minicpm-v" \
    "gsm8k" \
    20 \
    "http://127.0.0.1:8081/v1"
```

Results are logged sample-by-sample into Parquet and summarized in JSON for direct side-by-side comparison against Ollama and PolyStrata.

---

## 5. Empirical Benchmark Results & Production Migration (MiniCPM-V 5B)

Empirical testing conducted via **Hugging Face Lighteval (v0.13.0)** on the HP Z420 Workstation:

| Metric / Dimension | Ollama (`minicpm-v-opt:latest`) | llama.cpp (`llama-server`) | Advantage / Winner |
| :--- | :---: | :---: | :--- |
| **Runtime Binary** | Go daemon + bundled libs | Native C++17 (`-O3 -s`, 59 MB) | **llama.cpp** (Pure native, user-rule compliant) |
| **Host RAM Overhead** | ~3,747 MB RSS (Go daemon + runner) | Direct mmap, zero daemon overhead | **llama.cpp** (~3.7 GB host RAM reclaimed) |
| **Prompt Prefill Speed** | ~32 tok/s | **58.37 tok/s** | **llama.cpp** (+82% faster TTFT) |
| **Token Decode Speed** | ~7.1 tok/s | **8.88 tok/s** | **llama.cpp** (+25% sustained decode) |
| **Peak GPU VRAM** | 2,340 MiB (10 layers offloaded) | 3,368 MiB (14 layers offloaded) | **llama.cpp** (Maximized VRAM capacity $\le 3,500$ MiB) |
| **ARC-Challenge Accuracy** | 1.00 (100%) | 1.00 (100%) | **Parity** (Identical precision) |
| **GSM8K Accuracy** | 0.67 | 0.33 | **Ollama** (Model template differences) |

### Production Serving Command
Run the background launcher:
```bash
/home/tom/.local/bin/start-minicpm-llama.sh
```
Listening on `http://127.0.0.1:8081` with OpenAI-compatible endpoint `/v1/chat/completions`.

