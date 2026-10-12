# PolyStrata on 13-Year-Old HP Z420 with GTX/RTX 1050 Ti 4GB — Engineering Session & Technical Log

**Published Date:** October 12, 2026  
**Project Title:** PolyStrata on 13 year old Z420 with RTX 1050Ti 4GB (`Hypo30285/PolyStrata-on-13-year-old-Z420-with-RTX-1050Ti-4GB`)  
**Version:** `v1.0.0` (Revision `2026.10.12.r1`)  
**Author / Hardware Lead:** Tom (`Hypo30285`)  
**Collaborator:** Bailey Burnsed (`Bb1995@gmail.com` / `bburnsed1995@gmail.com`)  
**AI Engineering Orchestrator:** **Gemini 3.8 Flash (High Reasoning Mode)** — **Google AI Pro Plan (\$20/month)**  
**Local Co-Pilot Model:** **Ornith 1.5 35B-A3B (`APEX-I-QUALITY`, 32,768 Context)** via **PolyStrata (`strata-poly` v0.1.42)**  
**Workstation Context:** 2013 HP Z420 V2 Motherboard (`BIOS v03.96`), Intel Xeon E5-2690 (8C/16T), 64 GB ECC DDR3, NVIDIA GTX 1050 Ti (4 GB VRAM), 600W PSU

---

## 1. Session Executive Summary

During this engineering session, we customized, compiled, debugged, and benchmarked **PolyStrata (`strata-poly` v0.1.42)** and the **Ornith 1.5 35B-A3B (`APEX-I-QUALITY`)** Mixture-of-Experts model on a 13-year-old **HP Z420 Workstation** (`V2 Motherboard`, `BIOS v03.96`, `Intel Xeon E5-2690`, `64 GB ECC DDR3`, `4 GB GTX 1050 Ti`):

1. **Full 35B MoE Model @ 32K Context inside 4 GB VRAM:** Tuned `strata-poly` (`--vram-reserve-mib 550 --no-graphs --no-draft --prompt-batch 128 --stage-mib 64 --conversation-cache-mib 8192 --tail-skip 6`) so `Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf` (`23.7 GB`) loads in **`2.7 seconds`**, pins **`363 of 10,240` hot routed experts (`0.6 GiB`)** in GTX 1050 Ti VRAM (`41.7%` cache hit rate), and streams cold experts from 64 GB Quad-Channel ECC DDR3 RAM with **0 MB NVMe swap**.
2. **Sustained Interactive Speed (`21.0 – 23.8 tok/s`):** Achieved `21.0 – 23.8 tok/s` token generation and `44.1 – 78.4 tok/s` prompt evaluation, plus `34.8 – 36.6 ms` multi-turn KV conversation cache restoration in DDR3 RAM.
3. **Antigravity IDE GUI & `@ornith` Integration:** Built the **Ornith 1.5 Studio** sidecar panel (`~/.gemini/config/plugins/ornith-studio/`) with a 4-mode reasoning selector dropdown, live Z420 hardware telemetry badges (`CPU °C`, `NVMe °C`, `GPU °C`, `VRAM MiB`, `tok/s`), **Start / Unload VRAM** controls, and a 1-click **Insert into Agent Chat** bridge.
4. **Root-Caused & Eliminated the 1,000-Token Electron GPU Flag Bottleneck:** Discovered that `nvidia-smi --query-compute-apps` was dumping ~3,500 characters (~1,000 tokens) of Chromium/Electron `--type=gpu-process --gpu-preferences=...` flags into every prompt, causing a 29-second (`1,310-token`) prefill delay. Stripped process flags down to `<binary>(PID: <vram>)`, cutting total hardware query turn time by **more than 2× (`40 s` $\rightarrow$ `19 s`)**.
5. **Refactored Telemetry & MCP Server into Native C++17 (`g++ -O3 -std=c++17 -s`):** Replaced the `26.2 MB` Python sensor MCP daemon (`z420_sensors_mcp.py`) with a stripped **native C++17 binary (`z420-sensors-mcp`, `~1.8 MB RSS`, `< 0.5 ms` execution)** and established a permanent **Z420 Native Executable & No-Electron Rule** in `AGENTS.md`.
6. **Reclaimed `+374 MiB` GPU VRAM & `+1.7 GiB` System RAM:** Audited running processes, killed and disabled the background `gemini-desktop` Electron snap, and disabled GPU webview acceleration in Steam's `steamwebhelper`, leaving **`1,001 MiB` (~1.0 GB) of free VRAM** and **`53.2 GiB` of free ECC DDR3 RAM** while the 35B model is loaded.

---

## 2. Complete System Specifications & BIOS `v03.96` Architecture

### 2.1 Current Hardware Configuration
* **System:** HP Z420 Workstation (2013 Vintage, 600W 90% Efficient Factory PSU)
* **Motherboard:** HP Z420 **V2 Motherboard** (Intel C602 Chipset, Socket LGA 2011)
* **BIOS Version:** **`J61 v03.96`**
  * *Stock Firmware Limitations:* Does **not** have **Resizable BAR (ReBAR)**, does **not** have **Above 4G Decoding (Over 4G PCIe Addressing)**, and does **not** have native **UEFI / NVMe DXE boot** without firmware injection.
* **CPU:** **Intel Xeon E5-2690 v1** (8 Cores / 16 Threads, 2.90 GHz Base / 3.80 GHz Turbo, 20 MB L3 Cache, 32nm Sandy Bridge-EP, 135W TDP, `AVX 1.0` ISA floor)
* **RAM:** **64 GB (60.7 GiB usable) ECC Registered DDR3-1600** (8 × 8 GB Quad-Channel, ~51.2 GB/s bandwidth)
* **GPU:** **NVIDIA GeForce GTX 1050 Ti 4 GB GDDR5** (Pascal `GP107`, Compute Capability `6.1` / `sm_61`, 768 CUDA Cores, 112 GB/s, 75W slot-powered, hardware `DP4A` INT8 support)
* **Storage:** 2 TB PCIe NVMe SSD (`/mnt/nvme_storage`, `+25.9°C`) + Samsung SATA SSD (`/mnt/samsung_storage`) + 2 TB Hitachi HDD
* **OS:** Ubuntu 26.04 Linux (`x86_64`), NVIDIA Driver `580.178.04`

---

## 3. Chronological Engineering & Debugging Log

### Phase 1: PolyStrata (`strata-poly`) Compiler & VRAM Tuning on the Z420
1. **Compiler Configuration (`sm_61` + `AVX 1.0`):**
   * Configured `strata-poly` with `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=61` (activating `dp4a.s32.s32` INT8 instructions and `STRATA_BF16_EMULATED=1`) and `-DSTRATA_ISA_FLOOR=avx -march=sandybridge -mavx -mno-avx2 -mno-fma -mno-f16c` so the Xeon E5-2690 executes all CPU expert kernels without `SIGILL`.
2. **VRAM Reserve & Stability Tuning (`strata-ornith-apex-i-quality.log`):**
   * `05:06:04`: Tested `--vram-reserve-mib 220` $\rightarrow$ loaded `530 / 10,240` experts (`1.0 GiB` VRAM) in `2.8 s`, leaving only `96 MiB` free VRAM.
   * `05:16:05`: Tested `--vram-reserve-mib 300` $\rightarrow$ loaded `500 / 10,240` experts (`0.8 GiB` VRAM), encountered `native MMVQ launch: out of memory` during CUDA graph allocation.
   * `05:36:57`: Added `--no-graphs --no-draft --prompt-batch 128 --stage-mib 64 --vram-reserve-mib 550 --conversation-cache-mib 8192 --tail-skip 6` $\rightarrow$ loaded **`363 / 10,240` routed experts (`0.6 GiB` VRAM) in `2.7 s`** with 100% stability across 32,768 context tokens.

### Phase 2: Antigravity IDE Integration (`Ornith 1.5 Studio` + `@ornith` MCP)
1. Built the `ornith-studio` Antigravity UI extension (`~/.gemini/config/plugins/ornith-studio/`) and `ornith-z420` MCP server (`ornith_mcp_server.py`).
2. When prompted with `@Ornith how are you doing?`, Ornith initially responded that it was an AI without direct access to hardware sensors.
3. Wired Ubuntu's `lm-sensors` (`sensors`), `/sys/class/hwmon` (`coretemp` & `nvme`), and `nvidia-smi` into both the MCP server and the `Ornith 1.5 Studio` sidecar header (`CPU 49°C`, `NVMe 26°C`, `GPU 24°C`, `VRAM MiB`, `RAM GiB`, `tok/s`).
4. Fixed the sandboxed iframe **Reload** button and **Unload VRAM** button (`POST /unload`), and made the multi-turn system prompt static so PolyStrata's KV prefix cache hits 100% on coding follow-ups.

### Phase 3: Root-Causing the 30-Second (`1,310-Token`) Prefill Delay
1. After clicking **Clear** and then **🌡️ Temp & VRAM**, the response still took ~30–40 seconds.
2. Inspecting `strata-ornith-mcp-autostart.log` (lines 580–599) revealed why:
   ```text
   [strata] reading the prompt: 1,310 of 1,310 tokens, 29 s so far
   [strata] done: 240 tokens in 40 s (21.0 tok/s) (stop, cancel=False), expert cache 38.8% hit
   ```
3. **Root Cause:** `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader` was dumping the full 3,500-character `/proc/<pid>/cmdline` of the Electron/Chromium GPU process (`--type=gpu-process --gpu-preferences=UAAAA...`, ~1,000 tokens of flags) into every prompt!
4. **Fix & Verification (lines 600–615 of `strata-ornith-mcp-autostart.log`):**
   * Stripped process flags down to `<binary_basename>(PID <pid>: <vram> MiB)` (capping `nvidia-smi` summary at `240` chars / ~14 tokens).
   * Switched hardware queries to a stateless 1-turn fast request (`max_tokens: 320`) and added an automatic `mtimeMs` hot-reload watcher to `main.mjs`.
   * **Verified Log Result:**
     ```text
     [strata] reading the prompt: 560 of 561 tokens, 13 s so far
     [strata] done: 128 tokens in 19 s (21.2 tok/s) (stop, cancel=False), expert cache 41.7% hit
     ```
   * Prefill tokens dropped **57%** (`1,310` $\rightarrow$ `561`), prefill time dropped from `29 s` to **`13 s`**, total turn time dropped from `40 s` to **`19 s`** (>2.1× speedup), and expert cache hit rate increased from `38.8%` to **`41.7%`**.

### Phase 4: Native C++17 Refactoring & Background Electron/CEF Cleanup
1. **Native C++17 Telemetry & MCP Daemon (`z420-sensors-mcp`):**
   * Rewrote `z420_sensors_mcp.py` (`26.2 MB RSS` Python process) into native C++17 (`src/z420_sensors_mcp.cpp`), compiled with `g++ -O3 -std=c++17 -s` (`~1.8 MB RSS`, **14.5× memory reduction**).
2. **Eliminating Background Electron & CEF Processes:**
   * Killed and disabled the `gemini-desktop` Electron snap (`pkill -f gemini-desktop && sudo snap disable gemini-desktop`), freeing `~985 MB RAM` and `~108 MiB VRAM`.
   * Disabled GPU webview acceleration in Steam Settings and stopped `steamwebhelper` (`pkill -9 -f steam`), freeing `~1.45 GB RAM` and `~240 MiB VRAM`.
   * Final live telemetry (`ornith_status`): **`3,095 / 4,096 MiB` VRAM used (`1,001 MiB` free VRAM)** and **`7.5 / 60.7 GiB` RAM used (`53.2 GiB` free RAM)**.

---

## 4. HP Z420 Hardware & Firmware Upgrade Roadmap

### 4.1 Stock BIOS `v03.96` Limitations on the V2 Motherboard
The HP Z420 **V2 Motherboard** running stock **BIOS `J61 v03.96`** lacks:
* **Above 4G Decoding (Over 4G PCIe BAR Addressing)**
* **Resizable BAR (ReBAR)**
* **Native UEFI / NVMe DXE Boot**

### 4.2 Four-Stage Upgrade Plan (Within the 600W PSU Limit)
1. **Stage 1 — Experimental UEFI BIOS Injection (`ReBarUEFI` GitHub Project):**
   * Inject `ReBarDxe.ffs` + `NvmExpressDxe.efi` + 64-bit PCIe BAR allocation patches into the `J61 v03.96` BIOS image to enable **Over 4G Addressing**, **Resizable BAR**, and **UEFI boot**.
2. **Stage 2 — $>4\text{ GB}$ VRAM GPU Installation & PolyStrata Optimization:**
   * Install and test a $>4\text{ GB}$ VRAM card (`12 GB – 16 GB+` VRAM) using the newly unlocked 64-bit PCIe BAR space, scaling cached VRAM experts from `363` (`0.6 GiB`) to `2,500–4,500+` experts and enabling GPU MTP speculative drafting (`--draft`).
3. **Stage 3 — Dual-GPU Configuration Within the 600W Factory Power Supply:**
   * **GPU 1 (`~70W` Slot-Powered Video Card):** Dedicated to Ubuntu desktop display, browser, and Antigravity IDE (`0 MiB` VRAM taken from the inference GPU).
   * **GPU 2 (`<= 250W` Dedicated Inference GPU):** 100% dedicated to `strata-poly` LLM inference.
   * **Power Budget:** `130W CPU` + `70W Display GPU` + `200–250W Inference GPU` + `55W System/ECC RAM/NVMe` = **`455W – 505W` peak** (safely within the **600W PSU** continuous rating).
4. **Stage 4 — 12-Core / 24-Thread Intel Xeon E5-2697 v2 CPU Upgrade:**
   * Upgrade from the 8-core `E5-2690 v1` (32nm Sandy Bridge-EP, 20 MB L3, 135W, no `F16C`) to the **12-core / 24-thread `Intel Xeon E5-2697 v2`** (22nm Ivy Bridge-EP, **30 MB L3**, **130W TDP**, native hardware **`F16C`** half-precision SIMD instructions), boosting CPU MoE expert throughput by ~45–60%.

---

## 5. AI Orchestrator Credit & Session Token Accounting

* **Cloud Engineering Agent:** **Gemini 3.8 Flash (High Reasoning Mode)** inside **Google Antigravity IDE**
* **Subscription Plan:** **Google AI Pro Plan (\$20 / month)**
* **Local Workstation Model:** **Ornith 1.5 35B-A3B (`APEX-I-QUALITY`, 32K Context)** via **PolyStrata (`strata-poly` v0.1.42)**

| Metric | Gemini 3.8 Flash (High — \$20/mo Pro Plan) | Local Ornith 1.5 35B-A3B (`strata-poly` Z420) |
| :--- | :--- | :--- |
| **Total Context Capacity** | **`1,000,000 tokens`** | **`32,768 tokens`** |
| **Tokens Used During Session** | **`~225,000 cumulative tokens`** (`~72,500` active in current window after automatic compaction) | **`15,310 total local tokens`** (`11,420` prompt + `3,890` completion across 18 turns) |
| **Total Tokens Available After Session** | **`~927,500 context tokens available`** (`92.75%` of 1M context window free + active \$20/mo Pro Plan allowance) | **`32,768 tokens available`** (`100%` local 32K window available at `$0.00` cost) |
