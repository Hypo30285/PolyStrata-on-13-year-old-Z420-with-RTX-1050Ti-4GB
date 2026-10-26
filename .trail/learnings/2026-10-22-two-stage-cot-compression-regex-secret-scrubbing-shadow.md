---
title: Two-stage CoT compression, regex secret scrubbing, shadow Git snapshots, and SHA-256 drive retirement
kind: skill
author: Tom & Bailey
date: 2026-10-22
touches:
  - src/db.hpp
  - src/scrubber.hpp
  - src/sha256.hpp
  - src/subsystems.hpp
  - src/trail_native.hpp
---
## Skill
- Secret Scrubber Guardrail (`src/scrubber.hpp`): Run regex + entropy token masking (`AIza...`, `sk-...`, tokens) on all scraped prompts, terminal logs, and PR reviews before writing to SQLite (`devvault.db`) or syncing to Google Drive (`export_devvault_to_gdrive.py`).
- Two-Stage Reasoning Trace Storage: Compress raw `<thought>` traces via `zlib` in SQLite while vectorizing only the Goal Hypothesis and Conclusion segments to preserve vector recall and NVMe space.
- Shadow Git Snapshots & Lock Safety: Capture uncommitted working-tree state via `git write-tree` and coordinate background cron sweeps with manual CLI runs via `fcntl.flock` on `/tmp/devvault.lock`.
- SHA-256 Drive Retirement (`devvault retire-drive`): Verify every evacuated file with `Sha256::hash_file()`, log moves in `file_move_ledger`, and update `knowledge_items` and `photo_catalog` URIs in-place via `relocate_uri_prefix()`.
