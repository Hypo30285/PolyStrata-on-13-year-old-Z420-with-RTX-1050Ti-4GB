---
title: Native C++17 / Rust binaries only — no Node.js or Electron runtimes
kind: preference
author: Tom & Bailey
date: 2026-10-22
touches:
  - src/main.cpp
  - src/trail_native.hpp
  - src/mcp_server.hpp
---
## Preference
- Implement all CLI tools, Trail-Code equivalents (`skeleton`, `note`, `check`), and stdio JSON-RPC 2.0 MCP servers in native compiled C++17 (`./bin/devvault`) with zero Node.js, TypeScript, or Electron runtime overhead.
