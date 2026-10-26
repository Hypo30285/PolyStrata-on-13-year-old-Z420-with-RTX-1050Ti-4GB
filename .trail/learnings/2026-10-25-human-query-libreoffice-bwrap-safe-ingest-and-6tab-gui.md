---
title: human-query-libreoffice-bwrap-safe-ingest-and-6tab-gui
kind: decision
author: Tom & Antigravity
date: 2026-10-25
touches:
  - src/safe_ingest.hpp
  - src/safe_ingest.cpp
  - devvault_gui.py
  - scripts/setup_libreoffice_base.sh
---
## Decided
- We decided to expose 6 human-readable SQLite Views (v_human_knowledge_catalog, v_active_decisions_and_guardrails, v_aspects_and_concepts, v_vault_file_ledger, v_security_audit_log, v_category_summary) with ReadOnly=1 ODBC (~/.odbc.ini [DevVault_Z420]) and 1-click LibreOffice Calc export, expand devvault_gui.py to a 6-tab GTK3 human command and Local AI RAG Q&A studio, and enforce SafeIngestGate (./bin/devvault ingest) which classifies safe web pages/documents directly into DevVault while routing all executables and .zip/.tgz/.7z archives through an Ubuntu Bubblewrap (bwrap --unshare-all --dev /dev --proc /proc) security sandbox.

## Tried and ruled out
- We tried and ruled out exposing raw chunk_vectors float32 BLOBs or FTS5 shadow tables directly to LibreOffice Base, and ruled out importing downloaded archives or binaries without magic-byte (/usr/bin/file --mime-type) and bwrap sandbox inspection.
