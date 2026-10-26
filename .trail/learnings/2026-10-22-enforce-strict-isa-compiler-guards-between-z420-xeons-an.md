---
title: Enforce strict ISA compiler guards between Z420 Xeons and Bailey's Ryzen/Server rigs
kind: guardrail
author: Tom & Bailey
date: 2026-10-22
touches:
  - Makefile
  - src/embed_avx.hpp
---
## Guardrail
- Never emit unguarded `-mavx2` or `-mfma` instructions in shared default build targets; the HP Z420 Intel Xeon E5-2690 v1 (Sandy Bridge) and E5-2697 v2 (Ivy Bridge) will crash with `SIGILL`.
- Use `make sandybridge` prior to the Z420 BIOS flash, `make ivybridge` (adds `-mf16c`) after installing the Xeon E5-2697 v2, and isolate AVX2/FMA/XMX optimizations under `make server` for Bailey's Ryzen 9 5900XT + Intel Arc Pro B70 32GB rig and the Late-2027 Server Build.
