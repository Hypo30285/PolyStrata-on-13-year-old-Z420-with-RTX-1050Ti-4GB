---
title: quadro-p6000-slot2-pcie-gen3-x16-reseat-12gbps-dma-and-371tps-prefill
kind: decision
author: Tom & Antigravity
date: 2026-10-26
touches:
  - /mnt/nvme_storage/ai_projects/polystrata/bench/Z420_POLYSTRATA_EVAL_TRACKER.md
  - /home/tom/projects/Tile Lang for Pascal/tilelang/tilelang/sli/__init__.py
---
## Decided
- After cleaning HP Z420 Slot 2 (`PCIe3 x16`) and reseating the NVIDIA Quadro P6000 (`GP102`, `0000:05:00.0`), the PCIe link trained at full `8.0 GT/s PCIe x16` (`16 / 16` lanes, verified by `probe_pcie_aperture()`). Pinned DMA bandwidth doubled from `6.06 GB/s H2D / 6.57 GB/s D2H` (`PCIe Gen3 x8`) to `12.11 GB/s H2D / 13.09 GB/s D2H` (`2.00x` speedup). On `Row #4c` (`22.09 GiB` original un-requantized `Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf` with `10,240 / 10,240` experts in VRAM), Thinking Prefill Median rose to `353.0 tok/s` (`371.5 tok/s` peak — `6.24x` faster than GTX 1050 Ti `59.5 tok/s` and `9.57x` faster than Oct 14 baseline `36.9 tok/s`), Non-Thinking Prefill Median rose to `327.7 tok/s` (`335.8 tok/s` peak, `+7.1%` vs `x8`), Non-Thinking Decode Median rose to `19.50 tok/s` (`19.60 tok/s` peak, `51.09 ms/tok`), and Hugging Face Lighteval v0.13.0 completed GSM8K (`5/5 = 100%`) in `19.48s` and ARC-Challenge (`5/5 = 100%`) in `16.15s`.

## Tried and ruled out
- Operating the Quadro P6000 in Slot 2 at degraded `PCIe Gen3 x8` (`6.06 GB/s` H2D), which was caused by physical contact seating on lanes 8-15 rather than software/driver flags and was completely resolved by cleaning Slot 2 and reseating the card (`12.11 GB/s` H2D / `13.09 GB/s` D2H).
