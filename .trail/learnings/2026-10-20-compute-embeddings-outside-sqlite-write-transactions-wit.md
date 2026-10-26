---
title: Compute embeddings outside SQLite write transactions with 30s circuit breaker
kind: decision
author: Tom
date: 2026-10-20
touches:
  - src/db.hpp
  - src/embed_avx.hpp
---
## Decided
- Raise `DEVVAULT_MAX_CHUNKS` to `20000` so large transcripts and repo files are never truncated at 24 chunks (~29 KB).
- Compute HTTP embeddings (`127.0.0.1:8088` `bge-small-en-v1.5-q8_0.gguf`) before `BEGIN IMMEDIATE TRANSACTION` so network/CPU latency never holds a SQLite write lock.
- Tag every vector row with `embed_model` (`bge-small-en-v1.5` vs `lexical-v1`) so `./bin/devvault reembed` can upgrade fallback vectors later.
