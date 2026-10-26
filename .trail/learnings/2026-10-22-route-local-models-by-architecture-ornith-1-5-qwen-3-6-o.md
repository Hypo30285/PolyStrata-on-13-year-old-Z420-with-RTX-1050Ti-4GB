---
title: Route local models by architecture — Ornith 1.5 & Qwen 3.6 on strata-poly, Gemma 4 26B-A4B on single 24GB
kind: decision
author: Tom
date: 2026-10-22
touches:
  - src/kernel_router.hpp
  - tools/lighteval_engine.py
---
## Decided
- Use VecSzn/PolyStrata (`strata-poly v0.1.42+`) with zero-copy DDR3 `mmap` and hot MoE expert pinning in VRAM (rather than upstream `Niko1221/Strata`, which is hardcoded for a 125B two-shard model, and noting `strata-poly` rejects dense `qwen2` models like MiniCPM-V) to run Ornith 1.5 35B-A3B (verified 20.3-23.8 decode tok/s, 32.0-78.4 prefill tok/s, 58.3% 7/12 MTP draft acceptance, and 100% 5/5 GSM8K & ARC-Challenge accuracy on Z420) and Qwen 3.6 35B-A3B / Tiel Coder 35B-A3B for unattended overnight Hermes Agent coding.
- For single-24GB GPU operation without DDR3 spillover, route interactive IDE and multimodal RAG tasks to Gemma 4 26B-A4B MoE (~15.8 GB at `Q4_K_M`, 3.8B active parameters, native vision), and use Ornith 1.5 9B Dense (~5.6 GB at `Q4_K_M`, 70.6% SWE-bench) on 8 GB-12 GB tiers.
- Disqualify `EXL2`/`GPTQ`/`AWQ`/`vLLM` on Pascal `sm_61` and avoid `TQ3_4S` (tensor type ID 46) on mainline `llama.cpp`.
- Keep Lighteval v0.13.0 patched for `huggingface_hub` 2.x (`api.list_models(search=...)` in `metrics_sample.py`) and `inspect_ai` keyword alignment (`main_inspect.py`).
