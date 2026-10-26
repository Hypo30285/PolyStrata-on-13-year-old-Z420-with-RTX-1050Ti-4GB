---
title: Pin CUDA 12.6/12.8 and NVIDIA Driver 580 LTS for Pascal sm_61
kind: guardrail
author: Tom
date: 2026-10-22
touches:
  - Makefile
  - /etc/default/grub
---
## Guardrail
- Pin the HP Z420 OS to Ubuntu 26.04 LTS, NVIDIA Production Driver 580 LTS (closed-source `nvidia.ko`, which also supports coexisting Ada Lovelace cards like the 70W RTX 2000 Ada whereas `nvidia-open` drops Pascal), and CUDA 12.6 / 12.8 (`-DCMAKE_CUDA_ARCHITECTURES="61"` / `-arch=sm_61` with `rdblacklist=nouveau nouveau.modeset=0` in `/etc/default/grub`), because CUDA 13.0+ permanently dropped Compute Capability 6.1 (Pascal) support.
