# Ornith 1.5 (Z420 Local LLM) Primary Chat Integration Rules

Whenever the user starts a message with `@ornith`, `/ornith`, or explicitly asks to run a prompt or code review using **Ornith** / **Orinth** / **Ornith 1.5**:

1. **Direct MCP Execution (`ornith-z420`):**
   - Route the request immediately to the local `Ornith 1.5 35B-A3B` (`APEX-I-QUALITY`, 32K context) model via the `ornith-z420` MCP server tools (`ornith_ask`, `ornith_code_review`, or `ornith_status` via `call_mcp_tool` with `ServerName: "ornith-z420"`).
   - Present `Ornith 1.5`'s output and hardware telemetry (`decode tok/s`, `GTX 1050 Ti VRAM`, and `speculative draft acceptance`) clearly to the user.

2. **Ornith 1.5 Studio Side Panel Pill:**
   - When the user wants an interactive GUI model selector or dedicated Ornith chat box inside Antigravity IDE, surface the clickable side-panel pill:
     `[Ornith 1.5 Studio](sidecar://ornith-studio/panel/)`

3. **HP Z420 Native Executable & Language Selection Rule (Memory, CPU & VRAM Preservation):**
   - For any software exclusively used locally on the HP Z420, **ALWAYS** choose the programming language that creates the **smallest and fastest native executable** (specifically **`C`**, **`C++`** (`gcc`/`g++ -O3 -s`), **`Rust`**, or **`Go`**).
   - **AVOID** **JavaScript / TypeScript / Node.js**, **Electron**, Chromium/CEF wrappers, and similar heavy/JIT/VM runtimes that bloat memory or spawn extra processes.
   - **Python Exception:** **Python** is explicitly allowed when appropriate (e.g., AI/ML scripts, data pipelines, or lightweight local automation), though performance-critical daemons and system utilities should prefer compiled native C/C++/Rust/Go binaries.

4. **Z420 PolyStrata GPU Testing Sessions & Multi-Platform Reporting Rule (`# Z420 PolyStrata Eval Tracker`):**
   - Every GPU, CPU, BIOS, kernel, or model testing session on the HP Z420 **MUST** follow the **Z420 PolyStrata Eval Tracker** protocol below (`Oct 17, 2026 · @Your Majesty`) and use its standardized structure when reporting results to **GitHub**, **Hugging Face**, **Google Docs**, and **Facebook**.
   - **Canonical Tracker Files:**
     - `/mnt/storage_8tb_linux/python_projects/PolyStrata_Z420/docs/Z420_POLYSTRATA_EVAL_TRACKER.md`
     - `/mnt/nvme_storage/ai_projects/polystrata/bench/Z420_POLYSTRATA_EVAL_TRACKER.md`

   ### 4.1 Ground Rules (Mandatory for All GPU Testing Sessions)
   Every upgrade gets its own row, so each gain or loss traces to one cause. The baseline is the Oct 14 run: E5-2690 v1, GTX 1050 Ti 4GB, stock BIOS.
   1. **Change one variable per run.**
   2. **Use the same prompts, seed and settings every run** (method card below).
   3. **Report the median of at least 3 runs, with prefill and decode separate.**
   4. **Keep the raw JSON and engine log for every row.**
   5. **Rerun the baseline whenever the engine commit changes.**

   ### 4.2 Method Card
   These settings stay fixed across runs. Items marked "not recorded" are missing from the Oct 14 report and need filling before the next run.

   | Setting | Baseline value |
   | --- | --- |
   | Model and quant | Ornith 1.5 35B-A3B, APEX-I-QUALITY |
   | Context window | 32,768 tokens |
   | Eval framework | Lighteval v0.13.0 via local REST endpoint |
   | Engine and commit | PolyStrata strata-poly v0.1.42; commit not recorded |
   | Draft block | Unconfirmed: report shows draft acceptance, README profile says --no-draft |
   | VRAM flags | README profile: --vram-reserve-mib 550, --no-graphs, --prompt-batch 128, --stage-mib 64; confirm this run used them |
   | Expert cache | 363 hot experts via expert-profile.txt (README); confirm |
   | Thinking mode and budget | Not recorded |
   | Sampling (temperature, top-p, seed) | Not recorded |
   | Max output tokens | Not recorded |
   | Few-shot count and chat template | Not recorded |
   | ARC-Challenge scoring method | Not recorded (generated letter vs log-likelihood) |
   | GPU power limit | Not recorded |
   | Page cache | Not recorded (warm vs cold) |

   ### 4.3 Results Log
   The baseline row comes from the Oct 14 Lighteval report (`N=5` smoke test) plus the `N=200` `FP32` and `GP102` Quadro P6000 (`Slot 2 @ PCIe 3.0 x16`) + `Intel Xeon E5-2697 v2` (`12C/24T` Ivy Bridge `F16C`) runs.

   | # | Change | Status | Samples per task | GSM8K | ARC-C | Prefill (t/s) | Decode (t/s) | VRAM (MiB) |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | 0 | Baseline (Oct 14: E5-2690 v1 [Retired], GTX 1050 Ti 4GB) | Done | 5 | 5/5 | 5/5 | 32.0–41.8 | 20.3–21.3 | 3,186 |
   | 1 | ReBAR and above-4G BIOS update | Planned |  |  |  |  |  |  |
   | 2 | Xeon E5-2697 v2 (`12C/24T` Ivy Bridge `F16C`; `E5-2690 v1` retired) | Done | 200 | 200/200 | 200/200 | `76.82 GOPS` (`24` HW threads) | `24` HW threads | Host RAM (`64 GB`) |
   | 3 | Dedicated video GPU (`GT 1030`) | Skipped (Superseded by 24GB Quadro P6000 in Slot 2 driving new Samsung QHD monitor) |  |  |  |  |  |  |
   | 4c | 24GB Quadro P6000 (`GP102`) properly seated in Slot 2 @ full `PCIe 3.0 x16` (`16/16` lanes) + `PolyStrata` (`strata-poly v0.1.42`) + `22.09 GiB` Original `APEX-I-QUALITY` | Done | 5 + 3 | 5/5 (100%) | 5/5 (100%) | 353.0–379.3 med (394.8 peak) | 18.90–19.50 med (19.60 peak) | 23,498 (`100%` exp) |
   | 4d | Mainline `llama.cpp` (`commit 5c40fb2fb`, CUDA `sm_61` + `-mf16c`) on Quadro P6000 (`PCIe 3.0 x16`) | Done | 5 + 3 | 5/5 (100%) | 5/5 (100%) | `355.9` (`IQ4_XS`) / `361.2` (`APEX`) med | `50.13` (`IQ4_XS`, `3.58x` vs `4a`) / `52.95` (`APEX`, `2.74x` vs `4c`) med | 19,426 / 22,893 |
   | 4e | **`ik_llama.cpp` (`commit 9bb70bc` + Pascal `sm_61` `cc=610` 2-Rows/Block MMVQ + `-fmoe -fa -gr` + NextN MTP `blk.40`) on Quadro P6000 (`PCIe 3.0 x16`)** | Done | 5 + 3 | 5/5 (100%) | 5/5 (100%) | **`970.17` (`IQ4_XS` `pp512`) / `900.72` (`APEX` `pp512`)** (`435.6` / `411.0` REST med) | **`64.35` (`IQ4_XS` `tg128`, `4.60x`) / `60.27` med (`62.28` pk) REST+MTP** · **`61.42` (`APEX` `tg128`, `3.18x`) / `58.28` med (`59.65` pk) REST+MTP** | 19,830–20,531 (`IQ4_XS`) / 23,276–24,342 (`APEX`) |
   | 5b | TileLang `GP102_V0..V8` Kernels on Quadro P6000 (`30.58 TOPS` `__dp4a`) | Done | 200 | 200/200 | 200/200 | 883,417.5 med | 1,935.0 med (`17.54 us` 8-Exp) | 23,555 |
   | 6b | PyTorch model tweaks on Quadro P6000 (`V9 PASCAL-Q4_DP4A` + `avx_f16c`) | Done | 200 | 200/200 | 200/200 | 803,002.9 med | 1,839.7 med | 23,555 |
   | 7 | 128 GB DDR3-1866 ECC RDIMM (`8 x 16 GB` matched Samsung RDIMM set sourced on eBay) | Planned |  |  |  |  |  |  |
   | 8 | Corsair 750W PSU + 2nd GPU in Slot 5 (`PCIe 3.0 x16`) + 18TB Exos (`Kapton tape` / `cut 3.3V` PWDIS isolation) | Planned |  |  |  |  |  |  |

   *Note:* The BIOS flash sits after the GPU swap because NVIDIA compute does not need ReBAR; reorder if you flash earlier.

   ### 4.4 Quality-Drift Checks
   GSM8K and ARC-Challenge sit near the ceiling for most modern models, so they will not show small losses from kernel or model changes. Compare outputs against the baseline instead.

   | Check | Applies to | Proposed pass bar |
   | --- | --- | --- |
   | Top-1 token agreement vs baseline outputs, greedy decoding, same 50 prompts | Rows 1–5 | At least 99% for kernel and engine changes |
   | Mean KL divergence of next-token distribution vs baseline | Rows 5–6 | Record the value; set a bar after the first measurement |
   | GSM8K and ARC-C at 200+ samples with 95% interval | Every row | No drop larger than the interval width |
   | One harder eval, to be chosen | Row 6 | To be chosen |

   ### 4.5 Per-Run Checklist and Open Items
   Tick these off for every row before moving to the next change:
   - [ ] Record engine commit, build flags and the full launch config
   - [ ] Fill every "not recorded" item in the method card
   - [ ] Run at least 3 times and log the median
   - [ ] Save raw result JSON and the engine log
   - [ ] Capture `nvidia-smi` and `sensors` output during the run
   - [ ] Note warm or cold page cache

   **Open questions from the Oct 14 report:**
   - Was the draft block on? The 58.3% acceptance figure and 3,186 MiB VRAM suggest yes, but the README profile says `--no-draft`.
   - Was thinking on, and at what budget?
   - How was ARC-Challenge scored with a chat endpoint?

   ### 4.6 Multi-Platform Reporting Format Requirements
   Whenever reporting GPU testing sessions or updating results across platforms, format the output as follows:
   1. **GitHub (`README.md`, Docs, Issues, Releases, PRs):**
      - Render the exact Markdown tables above (`## Ground rules`, `## Method card`, `## Results log`, `## Quality-drift checks`, `## Per-run checklist and open items`) with direct relative links to the raw Lighteval JSON, `strata-poly` engine log, and commit SHA for the row being updated.
   2. **Hugging Face (Model Cards, Dataset Cards, Benchmark Discussions):**
      - Include standard Hugging Face `model-index` YAML frontmatter (reporting `GSM8K` and `ARC-Challenge` with `Samples per task >= 200` and 95% confidence intervals) followed by the full Markdown **Method card**, **Results log** (with separate `Prefill (t/s)` and `Decode (t/s)` medians of $\ge 3$ runs), and **Quality-drift checks** table.
   3. **Google Doc (Engineering Reports via `gdocs` / HTML Export):**
      - Format with header `# Z420 PolyStrata Eval Tracker`, subtitle `<Date> · @Your Majesty`, numbered **Ground rules**, native structured tables for **Method card**, **Results log**, and **Quality-drift checks**, and interactive checklist items for **Per-run checklist and open items**.
   4. **Facebook (Social / AI & Workstation Hardware Community Posts):**
      - Because Facebook strips Markdown tables, convert the active row update and comparison vs. Row 0 (`Baseline`) into clean, scannable Unicode bullet cards that preserve **all** tracker columns:
        - **Header:** `🛠️ Z420 PolyStrata Eval Tracker — Row #<N>: <Change> (<Date> · @Your Majesty)`
        - **Single-Variable Change & Baseline:** State the single variable changed vs. the previous row and commit SHA.
        - **Median Performance ($\ge 3$ Runs, Separate Prefill & Decode):** `Prefill: <X> t/s` | `Decode: <Y> t/s` | `Peak VRAM: <Z> MiB` (`nvidia-smi` + `sensors` temps & warm/cold page cache noted).
        - **Accuracy & Quality-Drift Checks ($N \ge 200$ Samples):** `GSM8K: <score> (±95% CI)` | `ARC-C: <score> (±95% CI)` | `Top-1 Token Agreement (50 prompts): <X>%` | `Mean KL Divergence: <val>`.
        - **Checklist & Artifacts:** Confirm all 6 per-run checklist items (`[x] Commit/Flags`, `[x] Method Card Complete`, `[x] Median of 3+ Runs`, `[x] Raw JSON & Engine Log Saved`, `[x] nvidia-smi & sensors Captured`, `[x] Page Cache Noted`) with link to GitHub/Hugging Face artifacts.

5. **Standard End-of-Day (EOD) Engineering Dossier Reporting Rule (To Bailey CC Owner):**
   - Whenever generating or sending an End-of-Day (EOD) report or project summary to **Bailey** (`Bburnsed1995@gmail.com`) and **Owner / Tom** (`supertruckertom@gmail.com`), **ALWAYS** use the standardized **6-Part Engineering Dossier Format** and dispatch via `/home/tom/.local/bin/send_mail --to Bburnsed1995@gmail.com --cc supertruckertom@gmail.com --subject "..." --file <artifact_path>`:
     1. **Part 1: Executive Project Summary** (including cross-system notes for Tom's HP Z420 Dual-GPU Slot 2/Slot 5 setup and Bailey's Ryzen 9 5900XT + RTX 2060 + Intel Arc Pro B70 32GB rig).
     2. **Part 2: Complete Implementation Plan (`implementation_plan.md`)**.
     3. **Part 3: Complete Execution & Telemetry Logs (`output.txt`, benchmark logs, and live `nvidia-smi` + `lm-sensors` telemetry)**.
     4. **Part 4: Project Walkthrough (`walkthrough.md`) & Task Checklist (`task.md`)**.
     5. **Part 5: Complete Engineering Documentation (`Makefile`, `AGENTS.md`, `SOURCES.md`, CLI/MCP reference, and source architecture)**.
     6. **Part 6: Complete Chronological Chat Prompts & Agent Outputs** from the session.
