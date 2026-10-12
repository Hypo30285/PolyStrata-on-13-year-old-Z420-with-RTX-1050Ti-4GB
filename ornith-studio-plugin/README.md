# Ornith 1.5 Studio (HP Z420 Local LLM)

Integrates **Ornith 1.5 35B-A3B (`APEX-I-QUALITY`, 32K context window)** running on the local **PolyStrata (`strata-poly`)** Mixture-of-Experts engine directly into Antigravity IDE.

## Features

1. **Ornith 1.5 Studio Side Panel (`sidecar://ornith-studio/panel/`)**
   - Dedicated **Model / Reasoning Mode Selector Dropdown**:
     - `Ornith 1.5 35B-A3B — Thinking (2K Budget)`
     - `Ornith 1.5 35B-A3B — Fast (No Thinking)`
     - `Ornith 1.5 35B-A3B — Deep Audit (4K Budget)`
     - `Ornith 1.5 35B-A3B — Quick Think (512 Budget)`
   - **Live Z420 Hardware Monitor**: Real-time GTX 1050 Ti VRAM (`MiB`), GPU temperature, System RAM, decode `tok/s`, and CPU Suffix-Drafter speculative acceptance rate.
   - **One-Click Start / Unload VRAM Toggle**: Start PolyStrata on port `8080` or unload the model from GPU VRAM on demand.
   - **Send to Main Agent**: Hand any Ornith response or code snippet directly to the active Antigravity conversation with one click.

2. **Primary Chat Box `@ornith` Pass-Through**
   - Mention `@ornith` in the main Antigravity chat box to route prompts or code reviews directly to local `Ornith 1.5 35B-A3B` via the `ornith-z420` MCP server.
