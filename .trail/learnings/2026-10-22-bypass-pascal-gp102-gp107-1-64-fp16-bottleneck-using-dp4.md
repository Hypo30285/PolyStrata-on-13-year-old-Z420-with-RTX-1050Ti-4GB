---
title: Bypass Pascal GP102/GP107 1:64 FP16 bottleneck using __dp4a INT8 kernels and GGML_CUDA_FORCE_MMQ
kind: decision
author: Tom
date: 2026-10-22
touches:
  - src/kernel_router.hpp
  - kernels/p6000_tilelang_dp4a.py
---
## Decided
- Never dispatch native FP16 (`hfma2` / cuBLAS FP16 GEMM) on Pascal GP107 (GTX 1050 Ti, 6 SMs), GP106 (Quadro P2000/P2200, 8/10 SMs), GP104 (Quadro P4000/P5000, 14/20 SMs), or GP102 (Quadro P6000, 30 SMs), which execute FP16 at a crippled 1:64 rate (~0.19 TFLOPS on P6000 vs 12.6 TFLOPS FP32).
- Compile `llama.cpp` with `-DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FA_ALL_QUANTS=ON` and run `q8_0` KV caches (`-ctk q8_0 -ctv q8_0`) so GGUF K-quants (`Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`) execute on hardware 4-way INT8 dot-product units (`__dp4a`, ~48 TOPS INT8 per P6000).
- In TileLang (`tile-ai/tilelang` v0.1.9+), pass explicit `target="cuda -arch=sm_61"`, use `T.CUDASourceCodeKernel` / `T.dp4a` accumulating `int8` tiles into `float32` registers, constrain thread-block shared memory to `<= 48 KB` (maintaining 2-block occupancy inside `sm_61` 96 KB SM pool) and `<= 32-64` registers/thread, enforce wave-quantization alignment (`Grid % SM_Count \== 0` for 6 SMs on GP107 and 30 SMs on GP102), and bound L2 swizzle panel widths (`T.use_swizzle`) below 75% of physical L2 cache (768 KB on GP107, 2,304 KB on GP102).
