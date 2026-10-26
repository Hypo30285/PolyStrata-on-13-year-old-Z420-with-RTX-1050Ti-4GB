---
title: Enforce 96 MiB headless CUDA reserve, 3,368 MiB display-shared cap on 1050 Ti, and Dual P6000 VRAM modes
kind: decision
author: Tom
date: 2026-10-22
touches:
  - src/subsystems.hpp
  - src/vision_minicpm.hpp
  - tune_dual_pascal.py
---
## Decided
- Reserve at least 96 MiB (480-576 MiB on 24 GB cards) on headless Pascal GPUs for `cuCtxCreate` and CUBIN module overhead (capping a headless 4,096 MiB GTX 1050 Ti at 4,000 MiB and a headless 24,576 MiB Quadro P6000 at 24,000-24,480 MiB, with Xorg isolated via `/etc/X11/xorg.conf.d/10-nvidia-display.conf` `AutoAddGPU=false` / `ProbeAllGpus=false`).
- When the GTX 1050 Ti 4GB drives Xorg (`/dev/fb0`) and Antigravity IDE in Slot 2, cap `llama-server` at `-ngl 14` (3,368 MiB VRAM, safely below the 3,500 MiB display-shared ceiling) or `--vram-reserve-mib 550` (3,095-3,186 MiB used, 363 hot experts pinned) under `strata-poly`.
- In `DualP6000Governor` (`src/subsystems.hpp`), dynamically switch between:
  - `coding` (`AI_DAILY_CODING`): 4 GB video (up to 4 monitors on Slot 2 GPU 0) + 44 GB AI (20 GB GPU 0 + 24 GB Slot 5 GPU 1, `--tensor-split 20,24`).
  - `gaming` (`GAMING_INVERSE`): 40 GB video/gaming (24 GB GPU 0 + 16 GB GPU 1 + SLI) + 8 GB background AI (GPU 1).
  - `away` (`away-sweep`): 48 GB headless AI for 5-day `1344px` deep photo sweeps at 125W-135W per card.
