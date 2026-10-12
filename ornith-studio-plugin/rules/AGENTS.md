# Ornith 1.5 (Z420 Local LLM) Primary Chat Integration Rules

Whenever the user starts a message with `@ornith`, `/ornith`, or explicitly asks to run a prompt or code review using **Ornith** / **Orinth** / **Ornith 1.5**:

1. **Direct MCP Execution (`ornith-z420`):**
   - Route the request immediately to the local `Ornith 1.5 35B-A3B` (`APEX-I-QUALITY`, 32K context) model via the `ornith-z420` MCP server tools (`ornith_ask`, `ornith_code_review`, or `ornith_status` via `call_mcp_tool` with `ServerName: "ornith-z420"`).
   - Present `Ornith 1.5`'s output and hardware telemetry (`decode tok/s`, `GTX 1050 Ti VRAM`, and `speculative draft acceptance`) clearly to the user.

2. **Ornith 1.5 Studio Side Panel Pill:**
   - When the user wants an interactive GUI model selector or dedicated Ornith chat box inside Antigravity IDE, surface the clickable side-panel pill:
     `[Ornith 1.5 Studio](sidecar://ornith-studio/panel/)`

3. **HP Z420 Native Executable & Language Selection Rule (Memory, CPU & VRAM Preservation):**
   - For any software exclusively used locally on the HP Z420, **ALWAYS** choose the programming language that creates the **smallest and fastest native executable** (specifically **`C`**, **`C++`** (`gcc`/`g++ -O3 -s`), **`Rust`**, or **`Go`**).
   - **AVOID** **JavaScript / TypeScript / Node.js**, **Electron**, Chromium/CEF wrappers, and similar heavy/JIT/VM runtimes that bloat memory or spawn extra processes.
   - **Python Exception:** **Python** is explicitly allowed when appropriate (e.g., AI/ML scripts, data pipelines, or lightweight local automation), though performance-critical daemons and system utilities should prefer compiled native C/C++/Rust/Go binaries.

