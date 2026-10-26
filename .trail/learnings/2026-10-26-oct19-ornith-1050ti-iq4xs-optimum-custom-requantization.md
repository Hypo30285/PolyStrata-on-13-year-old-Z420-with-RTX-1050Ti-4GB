---
title: oct19-ornith-1050ti-iq4xs-optimum-custom-requantization
kind: decision
author: Tom & Antigravity
date: 2026-10-26
touches:
  - /mnt/nvme_storage/ai_projects/polystrata/strata-ornith-iq4xs-optimum.json
  - /mnt/nvme_storage/ai_projects/polystrata/strata-ornith-apex-i-quality.json
---
## Decided
- Backfilled the pre-Trail Oct 19 custom 1050 Ti requantization (`Ornith-1.5-35B-A3B-APEX-MTP-IQ4_XS-Optimum.gguf`, 18.35 GiB vs 22.09 GiB original `Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf`) created via `requant_ornith_optimum.cpp`. That tool converted 30x Q6_K (`6.56 bpw`) and 30x Q5_K (`5.50 bpw`) routed expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` across 40 MoE layers) down to uniform `IQ4_XS` (`4.25 bpw`, `kvalues_iq4nl` LUT) while preserving all 365 attention/Gated DeltaNet (`ssm_*`)/shared-expert/router/embedding tensors bit-for-bit. On the 4 GB GTX 1050 Ti (`Row #0c`), this shrank cold-expert PCIe transfers by -19.0% (`17.92 GiB` active), increased hot VRAM experts from 633 -u003e 778, and boosted prefill from 36.9 -u003e 59.5 tok/s (`+61.2%`).

## Tried and ruled out
- Using `Ornith-1.5-35B-A3B-APEX-MTP-IQ4_XS-Optimum.gguf` as the default checkpoint on 24 GB VRAM GPUs (`Quadro P6000`), because when 100% of all 10,240 experts fit in VRAM (`19.3 GiB`), non-linear `IQ4_XS` table lookups are slower than arithmetic `Q6_K`/`Q5_K` K-quant `dp4a` kernels on Pascal `sm_61`.
