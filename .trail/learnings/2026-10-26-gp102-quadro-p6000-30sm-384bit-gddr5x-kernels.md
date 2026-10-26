---
title: gp102-quadro-p6000-30sm-384bit-gddr5x-kernels
kind: decision
author: Tom & Antigravity
date: 2026-10-26
touches:
  - Makefile
  - test_pascal_sm61_and_sli.cu
  - test_pascal_sm61_and_sli.py
  - tilelang/tilelang/carver/arch/cuda.py
  - tilelang/tilelang/sli/__init__.py
---
## Decided
- Implemented and benchmarked the 9-stage Quadro P6000 (GP102, sm_61, 30 SMs, 3,840 CUDA cores, 384-bit GDDR5X @ 432.8 GB/s, 3.0 MB L2, 24 GB VRAM) TileLang kernel suite (GP102_V0..GP102_V8) alongside Intel Xeon E5-2697 v2 (12C/24T Ivy Bridge-EP F16C, -march=ivybridge -mtune=ivybridge -mavx -msse4.2 -mf16c -mno-avx2 -mno-fma -fopenmp): (1) GP102_V1 128x128x32 symmetric 8-warp (256-thread) wide CTA with GROUP_M=6 L2 swizzle reaches 30.15-31.54 TOPS (1.91x-1.99x over V0) on 30-SM wave-aligned shapes (1920x1536x1024); (2) GP102_V3 64x128x32 4-warp 8x8 register-tiled double-buffered SMEM with exact-divisor L2 swizzle (GROUP_M=4 when (M/64)%4==0 u0026u0026 (M/64)%6!=0, else GROUP_M=6) reaches 29.02-31.21 TOPS on 1024x1024x1024; (3) GP102_V4 30-SM wave-adaptive dispatcher routes u003c120 CTAs (e.g., 256x1024x1024 MoE prefill) to 2-warp 64x64x32 double-buffered CTAs (64 CTAs across 30 SMs instead of 32 CTAs with a 2-CTA wave tail), reaching 14.41-17.58 TOPS; (4) GP102_V5 30-SM 64x64x32 (4x8 register tile, __launch_bounds__(128, 8)) PASCAL-Q4_DP4A MoE prefill reaches 0.036-0.038 ms (14.03-14.74 TOPS vs 0.052 ms / 10.24 TOPS on GP107_V9, 1.37x-1.42x speedup); (5) GP102_V7 30-SM register-cached activation (r_va0, r_va1 loaded once per lane for 16x register reuse across 8 experts x 2 columns, zero SMEM) grouped 8-expert decode reaches 17.54-17.74 us (238.6-241.3 GB/s GDDR5X bandwidth, 2.94x faster than 8 separate launches); (6) GP102_V8 30-SM 2-col/warp SMEM-cached M=8 MTP speculative tree verification reaches 9.52-10.26 us; and (7) probe_pcie_aperture() now detects and diagnoses PCIe link-width downgrades (e.g., 8.0 GT/s PCIe x8 vs Max x16 on 0000:05:00.0).

## Tried and ruled out
- (1) Forcing __launch_bounds__(128, 3) (80 regs/thread cap) on 8x8 register-tiled (64 int32 accumulators) GP102 GEMM kernels caused 68-80 bytes of local memory register spills per thread, dropping throughput from 29.1 TOPS to 19.6 TOPS; (2) Launching 64x128x32 (32 CTAs) on 256x1024x1024 MoE prefill on GP102's 30 SMs leaves a 2-CTA second wave where 28 of 30 SMs sit idle (solved by 64x64x32 64-CTA wave-adaptive tiling in GP102_V4 and GP102_V5); (3) Calling prebuilt PyTorch CUDA kernels (torch.zeros/repeat on device='cuda:0') on sm_61 raises cudaErrorNoKernelImageForDevice (solved by allocating with torch.empty on CUDA and copying from CPU tensors).
