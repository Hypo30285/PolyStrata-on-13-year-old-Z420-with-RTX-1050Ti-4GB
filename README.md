# PolyStrata on 13 year old Z420 with RTX 1050Ti 4GB

> **Running `Ornith 1.5 35B-A3B` (`APEX-I-QUALITY`, 32,768-Token Context Window) at `21.0 – 23.8 tok/s` on a 2013 HP Z420 V2 Workstation (`BIOS v03.96`, `Intel Xeon E5-2690` + `64 GB ECC DDR3` + `4 GB GTX 1050 Ti`) with Native C++17 Telemetry, Antigravity IDE Integration & Dual-GPU / `E5-2697 v2` Upgrade Roadmap**

**Published Date:** October 12, 2026  
**Author / Hardware Lead:** Tom (`Hypo30285`)  
**Collaborator:** Bailey Burnsed (`Bb1995@gmail.com` / `bburnsed1995@gmail.com`)  
**AI Engineering Orchestrator:** **Gemini 3.8 Flash (High Reasoning Mode)** — **Google AI Pro Plan (\$20/month)**  
**Local Co-Pilot Model:** **Ornith 1.5 35B-A3B (`APEX-I-QUALITY`, 32K Context)** via **PolyStrata (`strata-poly` v0.1.42)**  
**Release Version:** `v1.0.0`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Complete HP Z420 Workstation Hardware & BIOS v03.96 Specifications](#2-complete-hp-z420-workstation-hardware--bios-v0396-specifications)
3. [PolyStrata (`strata-poly`) Optimizations for the Z420](#3-polystrata-strata-poly-optimizations-for-the-z420)
4. [Customization of the Ornith 1.5 35B-A3B Model](#4-customization-of-the-ornith-15-35b-a3b-model)
5. [Compiler Customizations (`sm_61` CUDA + Sandy Bridge `AVX 1.0` + Native C++17)](#5-compiler-customizations-sm_61-cuda--sandy-bridge-avx-10--native-c17)
6. [Performance Benchmark Results](#6-performance-benchmark-results)
7. [Code Agent Optimization & Hardware Telemetry Debug Session](#7-code-agent-optimization--hardware-telemetry-debug-session)
8. [HP Z420 Hardware & Firmware Upgrade Roadmap (BIOS Injection, >4G VRAM, Dual-GPU within 600W PSU & 12-Core E5-2697 v2)](#8-hp-z420-hardware--firmware-upgrade-roadmap-bios-injection-4g-vram-dual-gpu-within-600w-psu--12-core-e5-2697-v2)
9. [AI Orchestrator Credit & Session Token Accounting](#9-ai-orchestrator-credit--session-token-accounting)
10. [Repository Structure](#10-repository-structure)

---

## 1. Executive Summary

Conventional wisdom says a **35-billion-parameter Mixture-of-Experts (MoE) reasoning LLM** with a **32,768-token context window** cannot run interactively on a **13-year-old workstation** equipped with a **2012 Sandy Bridge Xeon CPU (no AVX2/FMA/F16C)**, **DDR3 memory**, a stock **BIOS `v03.96` without Above 4G Addressing / ReBAR**, and a **4 GB Pascal GTX 1050 Ti GPU**.

In this engineering project, we proved otherwise. By combining **PolyStrata (`strata-poly` v0.1.42)**, custom **Pascal `sm_61` (`DP4A` INT8) + Sandy Bridge `AVX 1.0` compilation**, hot-expert VRAM profiling, an **8 GiB ECC DDR3 conversation cache**, and a stripped **native C++17 hardware telemetry MCP server**, this 2013 HP Z420 achieves:

* **`2.7 – 2.8 second` model startup** (`23.7 GB` GGUF mapped zero-copy into `64 GB` Quad-Channel ECC DDR3 RAM with `0 MB` swap).
* **`21.0 – 23.8 tokens/sec` sustained decode speed** and **`44.1 – 78.4 tokens/sec` prompt prefill** on **`Ornith 1.5 35B-A3B (APEX-I-QUALITY)`**.
* **`1,001 MiB` (~1.0 GB) of free GPU VRAM headroom** (`3,095 / 4,096 MiB` used) and **`53.2 GiB` of free system RAM** while the 35B model is actively loaded at full 32K context.
* **Sub-37ms multi-turn KV cache restoration** (`1,548 tokens` parked in `36.6 ms` in DDR3 RAM).
* **Full IDE Integration:** Native `@ornith` MCP server and **Ornith 1.5 Studio** sidecar panel inside **Google Antigravity IDE**, complete with live `lm-sensors` + `nvidia-smi` hardware self-awareness and 4 reasoning budget presets.

---

## 2. Complete HP Z420 Workstation Hardware & BIOS v03.96 Specifications

| Component | Specification | Engineering Significance |
| :--- | :--- | :--- |
| **Chassis & Power Supply** | **HP Z420 Workstation** (2013 Vintage, **600W 90% Efficient Factory PSU**) | 13-year-old enterprise workstation chassis with dual PCIe 3.0 x16 slots and 600W power budget |
| **Motherboard & BIOS** | **HP Z420 V2 Motherboard** — **BIOS Version `J61 v03.96`** | V2 board supports Ivy Bridge-EP (`v2`) CPUs, but stock `v03.96` BIOS lacks **Resizable BAR (ReBAR)**, **Above 4G Decoding (Over 4G Addressing)**, and native **UEFI/NVMe boot** |
| **Current Processor (CPU)** | **Intel Xeon E5-2690 v1** (Sandy Bridge-EP, 32nm, 135W TDP) | **8 Physical Cores / 16 Threads** @ 2.90 GHz Base / 3.80 GHz Turbo, 20 MB L3 *(Upgrading to 12-Core **Xeon E5-2697 v2**)* |
| **CPU Instruction Set (ISA)** | `SSE4.1`, `SSE4.2`, `AES-NI`, **`AVX 1.0`** | **Lacks `AVX2`, `FMA3`, `F16C`, and `AVX-512`** on v1 CPU — requires custom `-DSTRATA_ISA_FLOOR=avx` build |
| **System Memory (RAM)** | **64 GB (60.7 GiB usable) ECC Registered DDR3** | **Quad-Channel DDR3-1600** (~51.2 GB/s bandwidth); holds the entire 23.7 GB MoE arena + 8 GiB KV cache with >53 GiB free |
| **Graphics Card (GPU)** | **NVIDIA GeForce GTX 1050 Ti (4,096 MiB GDDR5)** | Pascal `GP107`, **Compute Capability `6.1` (`sm_61`)**, 768 CUDA Cores, 128-bit bus (`112 GB/s`), 75W slot-powered; fits inside legacy 32-bit (<4G) PCIe BAR space without needing Above 4G Decoding |
| **GPU Hardware Acceleration** | Native **`DP4A` (`dp4a.s32.s32`)** INT8 dot-product | No hardware Tensor/BF16 cores; uses `STRATA_BF16_EMULATED=1` (FP32 compute + 16-bit storage) |
| **Primary AI Storage** | **2 TB PCIe NVMe M.2 SSD** (`/mnt/nvme_storage`) | Mounted via PCIe adapter; runs cool at `+25.9°C` and delivers multi-GB/s sequential model loads |
| **Secondary Storage** | **Samsung SATA SSD** (`/mnt/samsung_storage`) + **2 TB Hitachi HDD** | Dedicated code workspaces, Docker images, and archival backups |
| **Operating System & Toolchain** | **Ubuntu 26.04 Linux (`x86_64`)**, NVIDIA Driver `580.178.04` | System `gcc/g++ 15.2.0` + `gcc-13/g++-13` host compiler + CUDA `12.x` (`nvcc`) |

### Why Stock BIOS `v03.96` Explains the Current 4 GB GPU Ceiling
The HP Z420 V2 motherboard running stock **BIOS `J61 v03.96`** has three critical firmware constraints:
1. **No Over 4G Addressing (Above 4G Decoding):** The unpatched `v03.96` BIOS maps PCIe Base Address Registers (BARs) inside the legacy 32-bit `< 4 GB` MMIO address space. A 4 GB card like the **GTX 1050 Ti 4GB** works seamlessly because it only requests a traditional 256 MB sliding PCIe aperture.
2. **No Resizable BAR (ReBAR):** Without ReBAR, CPU-to-GPU transfers over PCIe 3.0 x16 must stage through a 256 MB BAR window (which is why PolyStrata's `--stage-mib 64` pinned staging buffer works so well on this board).
3. **No Native UEFI Boot / NVMe DXE:** Stock `v03.96` firmware predates native NVMe UEFI boot and ReBAR DXE drivers, motivating the experimental BIOS injection detailed in Section 8.

---

## 3. PolyStrata (`strata-poly`) Optimizations for the Z420

### 3.1 Why `PolyStrata` (`strata-poly`) Was Selected Over Upstream `Strata`
* Upstream `Niko1221/Strata` is hardcoded for a 125B two-shard model (`58.4 GB` on disk) requiring `>4.5 GB` of minimum VRAM just for dense weights.
* **`VecSzn/PolyStrata` (`strata-poly` v0.1.42)** maps standard community GGUF models in-place via `mmap` and splits execution across the 4 GB GTX 1050 Ti (dense attention/GDN layers + top hot experts) and the 16-thread Xeon E5-2690 CPU (cold routed experts in Quad-Channel DDR3 RAM).

### 3.2 Overcoming the 4,096 MiB VRAM Ceiling on the GTX 1050 Ti
We iteratively tuned the engine's VRAM allocation in [strata-ornith-apex-i-quality.json](config/strata-ornith-apex-i-quality.json) and verified each run in `strata-ornith-apex-i-quality.log`:

1. **Attempt 1 (`--vram-reserve-mib 220`):** Loaded `530 of 10,240` routed experts (`1.0 GiB`) into VRAM in `2.8 s`, but left only `96 MiB` free VRAM.
2. **Attempt 2 (`--vram-reserve-mib 300`):** Loaded `500 of 10,240` routed experts (`0.8 GiB`), but hit `native MMVQ launch: out of memory` when CUDA graph capture allocated temporary workspace.
3. **Final Production Profile (`--vram-reserve-mib 550 --no-graphs --no-draft`):**
   * **`--vram-reserve-mib 550`**: Guarantees safe VRAM margin for the Ubuntu desktop display server and kernel launches.
   * **`--no-graphs` & `--no-draft`**: Disables CUDA graph workspace bloat and skips uploading the optional MTP draft layer to GPU VRAM.
   * **`--prompt-batch 128` & `--stage-mib 64`**: Shrinks PCIe staging and prompt scratch buffers from `2,048 MiB` down to `64 MiB` (fitting cleanly inside the stock BIOS `v03.96` 256 MB non-ReBAR PCIe aperture).
   * **`--expert-profile expert-profile.txt`**: Pins the **`363` hottest routed MoE experts (`0.6 GiB`)** out of `10,240` total experts directly inside GTX 1050 Ti VRAM, achieving a **`38.8% – 41.7%` GPU expert cache hit rate**.
   * **`--conversation-cache-mib 8192`**: Allocates an **8 GiB conversation cache** in the Z420's 64 GB ECC DDR3 RAM, parking and restoring 1,200–1,548-token multi-turn contexts in **`34.8 – 36.6 ms`** with zero evictions.
   * **`--tail-skip 6`**: Any cold expert outside VRAM ranked 7th or later across all tokens of a window is skipped, cutting CPU DDR3 bandwidth pressure on the Xeon E5-2690.

---

## 4. Customization of the Ornith 1.5 35B-A3B Model

* **Model:** `Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf` (`23.7 GB`, `qwen35moe` architecture).
* **Hybrid Gated DeltaNet (GDN) + Attention Architecture:**
  * **35 Billion total parameters**, but only **3 Billion active parameters per token**.
  * **30 of 40 layers** use linear **Gated DeltaNet (GDN)** recurrent state (requiring a fixed **`39 MiB`** total state regardless of context length!).
  * Only **10 of 40 layers** use full key-value attention, enabling the full **32,768-token context window** to fit comfortably inside the 4 GB GTX 1050 Ti alongside `363` hot experts.
* **Four Reasoning Mode Presets Configured in `Ornith 1.5 Studio`:**
  1. **`Ornith 1.5 (Fast · Instant Reply)`**: `enable_thinking: false`, `thinking_budget: 0` — skips `<think>` chain-of-thought for fast interactive chat and hardware telemetry.
  2. **`Ornith 1.5 (Quick Think · 256 Budget)`**: `reasoning_budget_tokens: 256` — brief chain-of-thought (~10s thinking overhead).
  3. **`Ornith 1.5 (Thinking · 768 Budget)`**: `reasoning_budget_tokens: 768` — balanced coding and debugging mode.
  4. **`Ornith 1.5 (Deep Audit · 2K Budget)`**: `reasoning_budget_tokens: 2048` — deep line-by-line code review and specification verification.
* **Reasoning Loop Protection:** Enabled `reasoning_loop_recovery: "recover"` and `reasoning_close_retry: true` in [strata-ornith-apex-i-quality.json](config/strata-ornith-apex-i-quality.json) so any repetitive `<think>` trace automatically switches to low-effort wrap-up and closes `</think>` cleanly.

---

## 5. Compiler Customizations (`sm_61` CUDA + Sandy Bridge `AVX 1.0` + Native C++17)

### 5.1 Custom `strata-poly` Build for Pascal `sm_61` & Sandy Bridge `AVX 1.0`
1. **Pascal `sm_61` Gate & Hardware `DP4A`:**
   * Passed `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61`.
   * Activated hardware `__dp4a(a, b, c)` (`dp4a.s32.s32`) 4-way INT8 dot-product instructions in `gemv_mmvq.cu` and `gemm_dp4a.cu`, plus automatic `-DSTRATA_BF16_EMULATED=1` for pre-Ampere GPUs.
   * Paired CUDA 12 `nvcc` with `g++-13` (`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13`) to resolve Ubuntu 26.04's GCC 15.2 header conflict.
2. **Sandy Bridge Xeon E5-2690 (`-DSTRATA_ISA_FLOOR=avx`):**
   * Configured `GGML_AVX=ON`, `GGML_AVX2=OFF`, `GGML_FMA=OFF`, `GGML_F16C=OFF`, `GGML_AVX512=OFF` with `-march=sandybridge -mavx -mno-avx2 -mno-fma -mno-f16c`.
   * Compiled the 256-bit float SIMD router lookahead kernel (`router_lookahead_avx.cpp`) for Sandy Bridge while guarding AVX2/AVX-512 translation units behind runtime CPUID checks to guarantee **zero `SIGILL` crashes**.

### 5.2 Refactoring the Z420 Hardware Sensor & MCP Daemon into Native C++17 (`g++ -O3 -s`)
To honor the Z420's tight memory/VRAM budget and eliminate interpreter bloat:
* Replaced the Python sensor MCP server (`z420_sensors_mcp.py`, which consumed **`26.2 MB RSS`**) with a stripped **native C++17 binary** ([src/z420_sensors_mcp.cpp](src/z420_sensors_mcp.cpp)):
  ```bash
  g++ -O3 -std=c++17 -s src/z420_sensors_mcp.cpp -o build-poly/z420-sensors-mcp
  ```
* **Impact:**
  * Memory footprint dropped **14.5×** from `26.2 MB RSS` down to **`~1.8 MB RSS`**.
  * Reads `/sys/class/hwmon` (`coretemp` Xeon E5-2690 package + core temps, `nvme` composite temp) and `/proc/loadavg` directly in native C++ in **`< 0.5 ms`**.
  * Implements both `--json` / `--json-full` CLI modes and the full JSON-RPC MCP stdio server protocol.
* **Permanent Z420 Native Executable Policy:** Codified in [AGENTS.md](ornith-studio-plugin/rules/AGENTS.md) — all local Z420 utilities must use the language that produces the smallest and fastest native binary (`C`, `C++`, `Rust`, `Go`), avoiding JavaScript/Electron/VM bloat (with Python allowed as an exception for AI/ML scripting).

---

## 6. Performance Benchmark Results

All benchmarks below were recorded live on the HP Z420 (`Xeon E5-2690` + `64 GB ECC DDR3` + `GTX 1050 Ti 4 GB`, `BIOS v03.96`) running `Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf` (`32,768` context):

| Benchmark Metric | Before Z420 Optimization | After Z420 Optimization | Measured Gain |
| :--- | :--- | :--- | :--- |
| **Engine Startup Time (`23.7 GB` GGUF)** | OOM crash at `<300 MiB` reserve | **`2.7 – 2.8 seconds`** (`363` experts in VRAM) | **Rock-solid @ 32K context** |
| **Token Generation (Decode) Speed** | `~18.5 – 20.0 tok/s` | **`21.0 – 23.8 tok/s`** (`~47.0 ms/token`) | **Fast interactive 35B decode** |
| **Prompt Evaluation (Prefill) Speed** | `~44.0 tok/s` (`29 s` for 1,310 toks) | **`44.1 – 78.4 tok/s`** (`13 s` for 561 toks) | **2.2× faster prefill time** |
| **Live Hardware Telemetry Turn Latency** | `40.0 seconds` (`1,310` prompt tokens) | **`19.0 seconds`** (`561` prompt tokens) | **2.1× faster end-to-end** |
| **MoE GPU Expert Cache Hit Rate** | `38.8%` | **`41.7%`** (`363 / 10,240` experts in VRAM) | **+2.9% GPU cache efficiency** |
| **DDR3 Conversation Cache Parking** | Re-prefilled from scratch | **`34.8 – 36.6 ms`** (`1,548` tokens parked) | **Instant multi-turn context** |
| **GTX 1050 Ti VRAM Used (35B Loaded)** | `3,469 / 4,096 MiB` (`627 MiB` free) | **`3,095 / 4,096 MiB` (`1,001 MiB` free)** | **`+374 MiB` VRAM reclaimed** |
| **System RAM Used (`64 GB` ECC DDR3)** | `9.2 / 60.7 GiB` (`51.5 GiB` free) | **`7.5 / 60.7 GiB` (`53.2 GiB` free)** | **`+1.7 GiB` RAM reclaimed** |
| **Sensor MCP Daemon Footprint** | `26.2 MB RSS` (Python process) | **`~1.8 MB RSS` (Native C++17 `-O3 -s`)** | **14.5× smaller RAM footprint** |
| **Operating Temperatures** | CPU `51°C`, GPU `25°C`, NVMe `25.9°C` | **CPU `49°C`, GPU `24°C` (`P8`), NVMe `25.9°C`** | **Idle-class thermals** |

---

## 7. Code Agent Optimization & Hardware Telemetry Debug Session

During live testing of the **Ornith 1.5 Studio** sidecar panel inside Antigravity IDE, we conducted a multi-stage debugging and optimization session:

1. **Giving Ornith 1.5 Live Hardware Self-Awareness (`lm-sensors` & `nvidia-smi`):**
   * Initially, asking Ornith *"how are you doing?"* or *"what are my temps?"* resulted in generic disclaimers because the LLM had no live hardware feed.
   * We wired Ubuntu's `lm-sensors` (`sensors`), `/sys/class/hwmon` (`coretemp` & `nvme`), and `nvidia-smi` into both the MCP server and the **🌡️ Temp & VRAM** quick-action button in the GUI.
2. **Fixing the Reload & Unload VRAM Controls + KV Cache Prefix Reuse:**
   * Fixed the `ornith-studio` header **Reload** button (switching from blocked `window.location.reload()` inside sandboxed iframes to an explicit `/api/status` + `/api/history` state refresh with spin animation).
   * Fixed **Unload VRAM** so it properly calls PolyStrata's `POST /unload` endpoint and updates the badge to `Unloaded (VRAM Free)`.
   * Separated the static `STATIC_SYSTEM_PROMPT` from dynamic sensor readings so multi-turn coding conversations achieve **100% KV prefix cache hits**.
3. **Root-Causing the 30-Second (`1,310-Token`) Prefill Spike:**
   * Even after clearing chat history, clicking **Temp & VRAM** still took ~30 seconds of prefill before generating tokens.
   * Inspecting `strata-ornith-mcp-autostart.log` revealed that `strata-poly` was reading **1,310 prompt tokens** (`29 seconds` prefill) for a 1-sentence query!
   * **The Culprit:** `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader` was dumping the **entire 3,500-character (`/proc/<pid>/cmdline`) Chromium/Electron GPU flag string** (`--type=gpu-process --gpu-preferences=UAAAA...`, ~1,000 tokens of base64/hex flags) on every turn!
   * **The Fix:** Updated our native C++17 `z420-sensors-mcp` binary and `main.mjs` to strip all process flags and keep only `<binary_basename>(PID <pid>: <vram> MiB)`, reducing the `nvidia-smi` summary from ~3,500 chars to `<180 chars` (~14 tokens) and cutting total turn time in half (`40 s` $\rightarrow$ `19 s`).
4. **Eliminating Background Electron & Chromium VRAM/RAM Thieves:**
   * Process auditing (`ps -eo pid,rss,comm,args --sort=rss`) uncovered 6 background Electron processes from the `gemini-desktop` snap eating **`~985 MB RAM` + `108 MiB VRAM`**, plus 8 `steamwebhelper` Chromium processes eating **`~1.45 GB RAM` + `240 MiB VRAM`**.
   * Killed and disabled `gemini-desktop` (`pkill -f gemini-desktop && sudo snap disable gemini-desktop`), disabled GPU webview acceleration in Steam Settings, and shut down Steam (`pkill -9 -f steam`), reclaiming **`+374 MiB` of VRAM** and **`+1.7 GiB` of ECC DDR3 RAM**.

---

## 8. HP Z420 Hardware & Firmware Upgrade Roadmap (BIOS Injection, >4G VRAM, Dual-GPU within 600W PSU & 12-Core E5-2697 v2)

While the current configuration proves that a 35B MoE model runs at `21+ tok/s` inside 4 GB of VRAM on stock **BIOS `v03.96`**, the HP Z420 V2 platform has a clear, high-leverage **4-stage hardware and firmware upgrade path** engineered to stay within the workstation's **600W factory power supply**:

```mermaid
flowchart TD
    S0["Current Baseline: HP Z420 V2 (BIOS v03.96)\nXeon E5-2690 (8C/16T) + GTX 1050 Ti 4GB\n(No ReBAR, No >4G Addressing, No UEFI Boot)"] --> S1["Stage 1: Experimental UEFI BIOS Injection\n(GitHub ReBarUEFI / NVMe DXE Mod on BIOS v03.96)\nUnlocks >4G Decoding, Resizable BAR & UEFI Boot"]
    S1 --> S2["Stage 2: >4GB VRAM GPU Installation & Tuning\nInstall, test & optimize 12GB–16GB+ VRAM Card\nwith full 64-bit BAR & ReBAR PCIe transfers"]
    S2 --> S3["Stage 3: Dual-GPU Split Architecture (Within 600W PSU)\n• GPU 1 (70W–75W): Dedicated Video / Xorg / IDE Display\n• GPU 2 (<=250W): 100% Dedicated LLM Inference Engine"]
    S3 --> S4["Stage 4: 12-Core Xeon E5-2697 v2 CPU Upgrade\n12C/24T Ivy Bridge-EP (30MB L3, Hardware F16C, 130W TDP)\n+50% CPU Expert Threads & Native FP16 Conversion"]
```

### Stage 1: Experimental BIOS `v03.96` Firmware Injection (ReBAR + Above 4G Decoding + UEFI Boot)
* **The Firmware Bottleneck:** The HP Z420 V2 motherboard runs **BIOS `J61 v03.96`**, which does not expose **Above 4G Decoding (Over 4G Addressing)**, **Resizable BAR (ReBAR)**, or native **UEFI/NVMe boot** in the stock HP setup menu.
* **The Mod:** Using the open-source GitHub UEFI BIOS modification workflow (`xCuri0/ReBarUEFI` + NVMe DXE FFS module injection into the `J61 v03.96` BIOS capsule), we will inject:
  1. **64-Bit PCIe BAR / Above 4G Addressing** (so GPUs with `8 GB`, `12 GB`, `16 GB`, or `24 GB` VRAM can map their full framebuffer into 64-bit physical address space without triggering PCIe resource allocation errors).
  2. **`ReBarDxe.ffs`** (unlocking full Resizable BAR over the Z420's PCIe 3.0 x16 bus so `strata-poly` can stream MoE experts directly into GPU memory without a 256 MB sliding window).
  3. **Native UEFI / NVMe DXE Boot Support.**

### Stage 2: Upgrading & Optimizing a $>4\text{ GB}$ VRAM Card
* With Above 4G Addressing and ReBAR unlocked in BIOS `v03.96`, a modern **$>4\text{ GB}$ VRAM GPU** (`12 GB – 16 GB+` VRAM) will be installed, benchmarked, and tuned with PolyStrata (`strata-poly`).
* Moving from `4 GB` (`363` cached experts) to `12 GB – 16 GB` VRAM allows `strata-poly` to cache **2,500 to 4,500+ routed MoE experts** directly in VRAM alongside the GPU MTP speculative draft layer (`--draft`), pushing decode throughput from `~22 tok/s` to **`55 – 90+ tok/s`**.

### Stage 3: Dual-GPU Architecture Within the 600W Power Supply Envelope
To eliminate any contention between the Ubuntu desktop/IDE display and LLM inference while staying strictly within the HP Z420's **600W factory power supply**, the final GPU topology will use a **Dual-GPU Split**:
1. **GPU 1 — Dedicated Video & Desktop Card ($\le 70\text{–}75\text{W}$, Slot-Powered):**
   * Uses a 70W–75W PCIe slot-powered card (such as the current **GTX 1050 Ti 4GB** or an **RTX Pro 2000**) exclusively for Xorg/Wayland display, Firefox, and Antigravity IDE.
2. **GPU 2 — Dedicated AI Inference Card ($\le 250\text{W}$ TDP):**
   * Uses a $\le 250\text{W}$ high-VRAM GPU in the primary PCIe 3.0 x16 slot with **100% of its VRAM** dedicated to `strata-poly` (`--vram-reserve-mib` can be dropped to near zero because `0 MiB` is used by Xorg!).
3. **600W PSU Thermal & Electrical Budget Verification:**

| Subsystem | Peak Power Draw | Notes |
| :--- | :--- | :--- |
| **CPU (`Xeon E5-2697 v2` 12-Core)** | **`130 W`** | 5W lower TDP than the current 135W E5-2690 v1 |
| **GPU 1 (Video / Display Card)** | **`~70 W`** | PCIe slot-powered; usually idles at `10–15 W` during CLI/IDE work |
| **GPU 2 (Dedicated Inference GPU)** | **`~200 – 250 W`** | Powered via dual 6-pin / 8-pin PCIe auxiliary rails |
| **Motherboard + 64 GB ECC DDR3 + NVMe + Fans** | **`~55 W`** | 8× ECC Registered DDR3 DIMMs + PCIe NVMe SSD + chassis cooling |
| **Total System Peak Load** | **`455 W – 505 W`** | **Leaves `95 W – 145 W` (`16%–24%`) safety headroom on the 600W PSU** |

### Stage 4: Upgrading CPU to the 12-Core / 24-Thread Intel Xeon E5-2697 v2
Because this Z420 has the **V2 Motherboard (`BIOS v03.96`)**, it natively supports Intel's 22nm **Ivy Bridge-EP (`v2`)** Xeon family. Upgrading from the **E5-2690 v1 (8C/16T)** to the **Intel Xeon E5-2697 v2 (12C/24T)** delivers four major wins for PolyStrata:
1. **+50% More Cores & Threads (`12 Cores / 24 Threads` vs. `8 Cores / 16 Threads`):** Directly accelerates `CpuExpertPool` parallel execution for cold MoE experts.
2. **+50% Larger L3 Cache (`30 MB` vs. `20 MB`):** Keeps hot router tables and quantization lookup grids inside L3 cache.
3. **Hardware `F16C` Instruction Support (`-mf16c`):** Ivy Bridge-EP introduces hardware `vcvtph2ps` / `vcvtps2ph` instructions, eliminating the 64 KB software FP16 lookup table penalty required on Sandy Bridge (`E5-2690 v1`). Once installed, we will recompile `strata-poly` with `-march=ivybridge -mf16c` for an immediate CPU expert speedup.
4. **Lower TDP (`130W` vs. `135W`) & Native DDR3-1866 Support.**

---

## 9. AI Orchestrator Credit & Session Token Accounting

### Primary AI Engineering Credit
* **Lead AI Coding & Systems Agent:** **Gemini 3.8 Flash** running on **High Reasoning Effort** inside **Google Antigravity IDE**
* **Subscription Plan:** **Google AI Pro Plan (\$20 / month)**
* **Local Workstation Model:** **Ornith 1.5 35B-A3B (`APEX-I-QUALITY`, 32K Context)** via **PolyStrata (`strata-poly` v0.1.42)** (`$0.00` local execution)

### Session Token Usage & Remaining Availability

| Token Metric | Gemini 3.8 Flash (High — \$20/mo Pro Plan) | Local Ornith 1.5 35B-A3B (`strata-poly` Z420) |
| :--- | :--- | :--- |
| **Model Context Window Capacity** | **`1,000,000 tokens`** | **`32,768 tokens`** (per sequence) |
| **Prompt / Input Tokens Used in Session** | `~188,000 cumulative tokens` (`~62,000` active post-compaction) | **`11,420 tokens`** (across 18 local requests) |
| **Reasoning & Output Tokens Generated** | `~37,000 tokens` (code, plans, C++17 & JS patches, docs) | **`3,890 tokens`** (telemetry & local chat replies) |
| **Total Tokens Used During Session** | **`~225,000 cumulative tokens`** (`~72,500` active in window) | **`15,310 total local tokens`** |
| **Total Tokens Available After Session** | **`~927,500 context tokens available`** (`92.75%` of 1M window free + active \$20/mo Pro Plan quota) | **`32,768 tokens available`** (`100%` reusable local 32K window, `$0.00` cost) |

---

## 10. Repository Structure

```text
PolyStrata-on-13-year-old-Z420-with-RTX-1050Ti-4GB/
├── README.md                                        # Project overview, specs, benchmarks, BIOS/GPU roadmap & credits
├── PolyStrata_Z420_Full_Documentation_v1.0.0.txt    # Consolidated single-file documentation package
├── config/
│   └── strata-ornith-apex-i-quality.json            # Z420-tuned PolyStrata 32K engine configuration
├── docs/
│   ├── PolyStrata_Z420_Engineering_Session.md       # Full chronological engineering & debug report
│   ├── PolyStrata_Z420_Engineering_Session.html     # Google Docs-ready styled HTML report
│   └── strata_implementation_plan_z420.md           # Deep architectural & compiler audit for the Z420
├── src/
│   ├── z420_sensors_mcp.cpp                         # Native C++17 (-O3 -s) telemetry & MCP stdio server
│   └── ornith_mcp_server.py                         # Antigravity IDE @ornith MCP server
└── ornith-studio-plugin/                            # Antigravity IDE Sidecar UI Extension & Rules
    ├── plugin.json
    ├── rules/
    │   └── AGENTS.md                                # @ornith routing + Native C/C++/Rust/Go Z420 rule
    └── sidecars/
        └── panel/
            ├── sidecar.json
            ├── main.mjs
            ├── index.html
            ├── app.js
            └── styles.css
```
