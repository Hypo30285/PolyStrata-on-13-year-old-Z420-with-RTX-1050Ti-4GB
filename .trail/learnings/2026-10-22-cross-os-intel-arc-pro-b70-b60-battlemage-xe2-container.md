---
title: Cross-OS Intel Arc Pro B70/B60 (Battlemage Xe2) container setup and multi-GPU port/SSH routing
kind: skill
author: Tom & Bailey
date: 2026-10-22
touches:
  - src/kernel_router.hpp
  - docker-compose.yml
---
## Skill
- Intel Arc Pro B70 32GB / B60 24GB (Battlemage Xe2) Setup (Ubuntu & Arch Linux): Require Linux kernel `6.11+` (`xe` KMD), BIOS Resizable BAR + Above 4G Decoding enabled, `intel-compute-runtime` and `level-zero-loader`, user in `video,render,docker` groups, `vm.max_map_count=262144` in `/etc/sysctl.d/99-vllm.conf`, and `intel/llm-scaler-vllm:latest` with `--device=/dev/dri --ipc=host --shm-size="32g"`.
- Multi-GPU Port Isolation & Remote Offload: Pin asymmetric GPUs explicitly via `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` across dedicated ports (`:8080` primary coding/MoE engine, `:8081` vision/auxiliary engine, `:8088` CPU `bge-small-en-v1.5` embeddings, `:8089` DevVault MCP HTTP gateway) and use SSH port forwarding (`ssh -L`) or 10GbE LAN routing so laptops and mobile sessions offload heavy inference to the Z420 and shared Big Box server.
