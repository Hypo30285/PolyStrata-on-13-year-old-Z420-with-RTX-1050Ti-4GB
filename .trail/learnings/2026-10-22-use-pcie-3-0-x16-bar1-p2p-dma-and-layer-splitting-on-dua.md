---
title: Use PCIe 3.0 x16 BAR1 P2P DMA and Layer Splitting on Dual P6000s — never SLI or mixed sm_60/sm_61 dies
kind: guardrail
author: Tom
date: 2026-10-22
touches:
  - src/subsystems.hpp
  - src/kernel_router.hpp
  - /etc/default/grub
---
## Guardrail
- Do not attempt to pass CUDA/LLM compute tensors over the Quadro P6000 HB SLI bridge: on GP102 silicon, SLI fingers connect only to the MIO display scanout/ROP pipeline (~2-3 GB/s) and compute microcode is cryptographically locked by the on-die Falcon ACR processor.
- Instead, use the Quadro P6000 unlocked GPUDirect PCIe 3.0 x16 BAR1 Peer-to-Peer (P2P) DMA (~12-14 GB/s) across Slot 2 and Slot 5 (wired to CPU0 Root Complex, `PHB`/`PIX` in `nvidia-smi topo -m`) with `iommu=pt` in `/etc/default/grub` and Layer Splitting (`--split-mode layer`) so only a single compact activation vector crosses PCIe once per token.
- Never mix GP100 (`sm_60`, Tesla P100 / Quadro GP100) and GP102 (`sm_61`, Quadro P6000): `sm_60` has 2:1 FP16 but lacks `__dp4a`, whereas `sm_61` has `__dp4a` (~48 TOPS) but 1:64 crippled FP16.
