---
title: Two-tier vision pipeline — native llama-server (:8081), FP16 mmproj, 135W P6000 away-sweep, and burst grouping
kind: decision
author: Tom
date: 2026-10-22
touches:
  - src/vision_minicpm.hpp
  - src/db.hpp
  - /home/tom/.local/bin/start-minicpm-llama.sh
---
## Decided
- Retire the Ollama Go daemon for vision workloads and serve MiniCPM-V 5B / 4.6 via custom-compiled C++17 `llama-server` on `127.0.0.1:8081` (`start-minicpm-llama.sh`), which achieved 58.37 tok/s prefill (+82%), 8.88 tok/s decode (+25%), and reclaimed ~3,747 MB of host DDR3 RAM on the Z420.
- Track `vision_tier`, `vision_model`, and `power_limit_w` in `photo_catalog` (`src/db.hpp`):
  - Tier 1 (`fast_448`): Slot 2 GTX 1050 Ti (~75W) running `MiniCPM-V-4_6-Q4_K_M.gguf` at `448x448` in `/dev/shm` RAM disk.
  - Tier 2 (`deep_1344` 5-Day Away Sweep): Slot 5 Quadro P6000 (24GB locked at `135W` via `DEVVAULT_AWAY_WATTS=135`) running `Qwen2.5-VL-32B-Instruct-Q4_K_M.gguf`, `Gemma-3-27B-IT-Q4_K_M.gguf`, or `MiniCPM-V-4.7-35B-A3B` via `llama-mtmd-cli` at `1344x1344` UHD multi-tile in `/dev/shm`.
- Always pair quantized GGUF vision backbones with an unquantized FP16/FP32 `mmproj` vision projector (`mmproj-model-f16.gguf`) so OCR and fine visual details do not degrade, and group sequential smartphone burst shots (`< 1.0s` delta + visual hash inheritance) to prevent category flip-flopping.
