---
date: 2026-10-25
author: Tom & Antigravity
touches:
  - Makefile
  - src/db.hpp
  - src/embed_avx.hpp
  - src/subsystems.hpp
  - src/trail_native.hpp
  - src/jev_kb_gate.hpp
  - src/main.cpp
  - src/mcp_server.hpp
---
## Decided
- Cut over HP Z420 compilation target from `make sandybridge` (`E5-2690 v1`) to `make ivybridge` (`Intel Xeon E5-2697 v2` 12C/24T: `-march=ivybridge -mavx -msse4.2 -mf16c -mno-avx2 -mno-fma -j 24`), enabling hardware `F16C` (`vcvtph2ps` / `vcvtps2ph`) while strictly blocking `AVX2`/`FMA` (`SIGILL` prevention).
- Removed the 24-chunk (~29 KB), 24,000-char, and 48,000-char limits across `db.hpp`, `embed_avx.hpp`, and `subsystems.hpp` (`DEVVAULT_MAX_CHUNKS=20000`, ~21 MB per file), with `bge-small-en-v1.5` (`127.0.0.1:8088`) HTTP embeddings computed outside `BEGIN IMMEDIATE TRANSACTION`.
- Added `src/jev_kb_gate.hpp` combining TypeSafe `/v1/systemone` probability math (`choice`, `score`, `noul`), `browser-use/jev-ultrafast` & `fast-jev-compaction` supersession (`is_superseded=1` skipped by default in RAG search, 3-tier non-lossy compaction pointers `[kb://ptr/item_<id>]`), and `shcherbak-ai/contextgem` paragraph/sentence-grounded `Aspects`, `Concepts`, and pre-turn multi-label Agent Tool Group selection (`devvault_select_tools` at 0.30 threshold).
- Linked `.trail/learnings/` bidirectionally with Google Drive Doc `02_SHARED_REPO_LEARNINGS` (`1KChNqGiNuYN-HndzkMDLhDZnffQtrs1hQFuyQ23I7VY`) for Antigravity IDE, Browser Gemini, and Mobile Gemini synchronization.

## Tried and ruled out
- Keeping `-march=sandybridge -mno-f16c` as default after installing the Xeon E5-2697 v2 (leaves hardware FP16->FP32 conversion instructions unused on 24 threads).
- Lossy LLM text summarization during context compaction (destroys exact compiler flags, file paths, and hex offsets; replaced by Jev 3-tier non-lossy pointer compaction).
- Returning superseded or obsolete hardware epoch learnings (`is_superseded=1`) in default RAG queries (causes recursive self-improvement loops to regress to older configs).
- Truncating ingested source/markdown/transcript files at 24 chunks (~28.8 KB) or 24,000/48,000 chars.
