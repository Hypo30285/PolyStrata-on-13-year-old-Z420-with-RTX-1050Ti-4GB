---
title: parallel-24t-compile-and-devvault-ops
kind: decision
author: Tom & Antigravity
date: 2026-10-25
touches:
  - Makefile
  - src/parallel_pool.hpp
  - src/scrubber.cpp
  - src/embed_avx.cpp
  - src/db.cpp
  - src/trail_native.cpp
  - src/jev_kb_gate.cpp
  - src/subsystems.cpp
  - src/vision_minicpm.cpp
  - src/mcp_server.cpp
---
## Decided
- Split DevVault v2.0 into 9 translation units compiled in parallel via make ivybridge -j 24 (668% CPU, 8.82s clean / 4.88s incremental vs 28.14s single-TU) and parallelized chunk embedding, 798,512-vector WAL MMAP AVX1 cosine search, ContextGem paragraph extraction, Jev batch evaluation, and SHA-256 array verification across 24 Xeon E5-2697 v2 threads via ParallelPool.

## Tried and ruled out
- Single-translation-unit header-only compilation with <regex> included in headers (peg 1 thread for 28.14s while 23 threads idle) and single-threaded sequential chunk_vectors scans.
