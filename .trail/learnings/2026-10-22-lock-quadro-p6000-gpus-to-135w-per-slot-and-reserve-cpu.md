---
title: Lock Quadro P6000 GPUs to 135W per slot and reserve CPU threads for embeddings
kind: guardrail
author: Tom
date: 2026-10-22
touches:
  - src/subsystems.hpp
  - src/vision_minicpm.hpp
  - devvault-gpu-governor.service
---
## Guardrail
- Enforce a `135W` power limit per Quadro P6000 (`nvidia-smi -pm 1` and `nvidia-smi -i <id> -pl 135`) on the HP Z420 PSU.
- Keep `bge-small-en-v1.5` embeddings on host CPU AVX1 threads (`0 MB` GPU VRAM) so GPU VRAM remains dedicated to MiniCPM-V 4.6 vision classification and coding/agent models.
