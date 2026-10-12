#!/usr/bin/env python3
"""Model Context Protocol (MCP) Stdio Server for local Ornith 1.5 35B-A3B (APEX-I-QUALITY) on HP Z420.

Exposes three tools to Antigravity IDE:
  - ornith_ask: Query Ornith 1.5 35B-A3B (32K context) locally with optional <think> reasoning and budget.
  - ornith_code_review: Perform a deep bug-hunt / code review on a file or snippet using Ornith 1.5.
  - ornith_status: Check live Z420 GPU VRAM, RAM, tok/s, and expert cache status (or start/unload the engine).
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

POLYSTRATA_ROOT = Path("/mnt/nvme_storage/ai_projects/polystrata")
PYTHON_EXE = POLYSTRATA_ROOT / ".venv" / "bin" / "python"
SERVER_PY = POLYSTRATA_ROOT / "serve" / "server.py"
CONFIG_JSON = POLYSTRATA_ROOT / "strata-ornith-apex-i-quality.json"
BASE_URL = os.environ.get("ORNITH_BASE_URL", "http://127.0.0.1:8080")
MODEL_NAME = "ornith-1.5-35b-a3b-apex-i-quality"

DEFAULT_SYSTEM = (
    "You are Ornith 1.5 35B-A3B (APEX-I-QUALITY), a 35B-parameter Mixture-of-Experts model "
    "(3B active parameters per token, 32K context window) running 100% locally via PolyStrata "
    "on Tom's HP Z420 Workstation (Intel Xeon E5-2690 + 64 GB ECC DDR3 RAM + 4 GB GTX 1050 Ti)."
)


def _run_cmd(args: list[str], timeout: float = 2.5) -> str:
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return (res.stdout or "").strip()
    except Exception:
        return ""


def _reload_sidecar_if_stale() -> bool:
    """If the ornith-studio sidecar `node main.mjs` process was started before `main.mjs` was modified,
    send SIGTERM so Antigravity's supervisor (restart_policy: always) respawns it with the latest code."""
    import signal
    main_mjs = Path("/home/tom/.gemini/config/plugins/ornith-studio/sidecars/panel/main.mjs")
    if not main_mjs.is_file():
        return False
    mtime = main_mjs.stat().st_mtime
    restarted = False
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
            if "node" in cmdline and "main.mjs" in cmdline:
                cwd = os.readlink(proc_dir / "cwd")
                if "ornith-studio" in cwd and proc_dir.stat().st_mtime < mtime - 2.0:
                    os.kill(int(proc_dir.name), signal.SIGTERM)
                    restarted = True
        except Exception:
            continue
    return restarted


NATIVE_SENSORS_BIN = POLYSTRATA_ROOT / "build-poly" / "z420-sensors-mcp"


def _read_z420_sensors() -> dict:
    """Read live CPU package/core temps, NVMe temp, compact `sensors` (lm-sensors), and compact `nvidia-smi` via native C++17 binary."""
    _reload_sidecar_if_stale()
    try:
        raw = _run_cmd([str(NATIVE_SENSORS_BIN), "--json-full"], timeout=2.0)
        if raw:
            data = json.loads(raw)
            data["lm_sensors"] = data.get("lm_sensors_raw", "")
            data["nvidia_smi"] = data.get("nvidia_smi_raw", "")
            return data
    except Exception:
        pass
    return {
        "cpu_pkg_c": None,
        "cpu_max_core_c": None,
        "nvme_c": None,
        "load_1m": round(os.getloadavg()[0], 2),
        "lm_sensors": _run_cmd(["sensors"]),
        "nvidia_smi": "",
    }


def _build_dynamic_system_prompt() -> str:
    sensors = _read_z420_sensors()
    gpu_info = "GTX 1050 Ti (4 GB)"
    ram_info = "64 GB ECC DDR3"
    speed_info = "ready"
    try:
        st = _http_json("GET", "/v1/status", timeout=2.0)
        gpu = (st.get("machine") or {}).get("gpu") or {}
        ram = (st.get("machine") or {}).get("ram") or {}
        lt = st.get("last_timings") or {}
        if gpu.get("used_mib") is not None:
            used = int(gpu["used_mib"])
            tot = int(gpu.get("total_mib", 4096))
            gpu_info = f"VRAM = {used}/{tot} MiB ({tot - used} MiB free), GPU = {gpu.get('temp_c', '?')}°C, Util = {gpu.get('util_pct', 0)}%"
        if ram.get("used_gib") is not None:
            ram_info = f"{ram['used_gib']}/{ram.get('total_gib', 60.7)} GiB ECC DDR3"
        if lt.get("predicted_per_second"):
            speed_info = f"{lt['predicted_per_second']} tok/s decode"
    except Exception:
        pass
    cpu_temp = f"{sensors['cpu_pkg_c']}°C (max core {sensors['cpu_max_core_c']}°C)" if sensors.get("cpu_pkg_c") is not None else "N/A"
    nvme_temp = f"{sensors['nvme_c']}°C" if sensors.get("nvme_c") is not None else "N/A"
    lm_out = sensors.get("lm_sensors") or ""
    smi_out = sensors.get("nvidia_smi") or ""
    return (
        "You are Ornith 1.5 (35B-A3B MoE, APEX-I-QUALITY, 32K context), running locally on Tom's HP Z420 Workstation "
        "(Intel Xeon E5-2690 8C/16T + 64 GB ECC DDR3 RAM + NVIDIA GTX 1050 Ti 4 GB VRAM, Ubuntu Linux) via PolyStrata.\n"
        f"[LIVE Z420 SENSORS: CPU = {cpu_temp} | Load = {sensors.get('load_1m')} | NVMe = {nvme_temp} | {gpu_info} | RAM = {ram_info} | Speed = {speed_info} "
        f"| lm-sensors: {lm_out} | nvidia-smi: {smi_out}]"
    )


def _http_json(method: str, path: str, payload: dict | None = None, timeout: float = 600.0) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _ensure_server_running(auto_start: bool = True) -> dict:
    try:
        return _http_json("GET", "/v1/status", timeout=3.0)
    except Exception:
        if not auto_start:
            return {"loaded": False, "service": "offline"}
    # Start PolyStrata server in background if not running
    log_path = POLYSTRATA_ROOT / "strata-ornith-mcp-autostart.log"
    with open(log_path, "ab") as logf:
        subprocess.Popen(
            [
                str(PYTHON_EXE),
                str(SERVER_PY),
                "--engine",
                "strata",
                "--config",
                str(CONFIG_JSON),
                "--port",
                "8080",
            ],
            cwd=str(POLYSTRATA_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=logf,
            start_new_session=True,
        )
    deadline = time.time() + 45.0
    while time.time() < deadline:
        time.sleep(1.0)
        try:
            st = _http_json("GET", "/v1/status", timeout=3.0)
            if st.get("loaded"):
                return st
        except Exception:
            pass
    return _http_json("GET", "/v1/status", timeout=5.0)


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lm_sensors",
            "description": "Run the Ubuntu `sensors` (lm-sensors) CLI utility and return live CPU package, per-core Xeon E5-2690, and NVMe temperatures.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nvidia_smi",
            "description": "Run `nvidia-smi` and return live NVIDIA GTX 1050 Ti GPU temperature, VRAM usage, utilization, and active processes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _chat_completion(
    messages: list[dict],
    max_tokens: int = 4096,
    enable_thinking: bool = True,
    thinking_budget: int = 2048,
    include_thinking_in_output: bool = False,
) -> str:
    _ensure_server_running(auto_start=True)

    def _make_payload(msgs: list[dict]) -> dict:
        p: dict = {
            "model": MODEL_NAME,
            "messages": msgs,
            "tools": OPENAI_TOOLS,
            "strata_mcp": True,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if enable_thinking:
            p["reasoning_budget_tokens"] = thinking_budget
        else:
            p["reasoning_effort"] = "none"
            p["chat_template_kwargs"] = {"enable_thinking": False}
        return p

    t0 = time.time()
    resp = _http_json("POST", "/v1/chat/completions", payload=_make_payload(messages), timeout=600.0)
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    if msg.get("tool_calls"):
        followup = list(messages) + [msg]
        s_data = _read_z420_sensors()
        for tc in msg["tool_calls"]:
            fn_name = (tc.get("function") or {}).get("name", "lm_sensors")
            tool_out = s_data.get("nvidia_smi") if fn_name == "nvidia_smi" else s_data.get("lm_sensors")
            followup.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", "call_1"),
                    "name": fn_name,
                    "content": tool_out or "No output",
                }
            )
        resp = _http_json("POST", "/v1/chat/completions", payload=_make_payload(followup), timeout=600.0)
        msg = (resp.get("choices") or [{}])[0].get("message") or {}

    elapsed = time.time() - t0
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = resp.get("usage") or {}
    comp_toks = usage.get("completion_tokens", 0)
    prompt_toks = usage.get("prompt_tokens", 0)
    tok_s = comp_toks / max(0.1, elapsed)

    out_parts = []
    if include_thinking_in_output and reasoning:
        out_parts.append(f"<details><summary>Ornith 1.5 Thinking ({thinking_budget} token budget)</summary>\n\n{reasoning}\n\n</details>\n")
    out_parts.append(content)
    out_parts.append(
        f"\n\n---\n*Ornith 1.5 35B-A3B (Z420 Local): {comp_toks} tokens generated in {elapsed:.1f}s "
        f"(~{tok_s:.1f} tok/s, prompt: {prompt_toks} tokens)*"
    )
    return "\n".join(out_parts)


TOOLS = [
    {
        "name": "ornith_ask",
        "description": (
            "Query the local Ornith 1.5 35B-A3B (APEX-I-QUALITY, 32K context) Mixture-of-Experts model "
            "running on the HP Z420 Workstation via PolyStrata (~20-24 tok/s, zero cloud cost). "
            "Use this when the user asks to consult Ornith, generate code with Ornith, or run local LLM reasoning."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt or question to send to Ornith 1.5.",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional custom system prompt.",
                },
                "enable_thinking": {
                    "type": "boolean",
                    "description": "Enable chain-of-thought <think> reasoning before answering (default: true). Recommended for coding and math.",
                    "default": True,
                },
                "thinking_budget": {
                    "type": "integer",
                    "description": "Maximum tokens Ornith may spend in <think> before writing its answer (default: 2048).",
                    "default": 2048,
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum total tokens (thinking + answer) to generate (default: 4096).",
                    "default": 4096,
                },
                "include_thinking": {
                    "type": "boolean",
                    "description": "Whether to include Ornith's <think> trace in the returned output (default: false).",
                    "default": False,
                },
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "ornith_code_review",
        "description": (
            "Have the local Ornith 1.5 35B-A3B reasoning model review a file or code snippet for runtime exceptions, "
            "logic bugs, edge cases, and specification violations, returning a bug list and corrected code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Optional absolute path to a file on disk to review.",
                },
                "code": {
                    "type": "string",
                    "description": "Optional inline code snippet to review (if file_path is not provided).",
                },
                "focus": {
                    "type": "string",
                    "description": "Optional instructions on what to check (e.g. 'Check for GGUF v3 spec bugs and Python runtime exceptions').",
                },
                "thinking_budget": {
                    "type": "integer",
                    "description": "Thinking token budget (default: 2048).",
                    "default": 2048,
                },
            },
        },
    },
    {
        "name": "ornith_status",
        "description": (
            "Check the live status and hardware telemetry of the local PolyStrata Ornith 1.5 35B-A3B engine on the HP Z420 "
            "(loaded state, 32K context window, GTX 1050 Ti VRAM usage, 64 GB DDR3 RAM usage, last request tok/s), "
            "or start/unload the model."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "start", "unload"],
                    "description": "'status' (default) checks live status without starting if offline; 'start' ensures the engine is loaded; 'unload' frees GPU VRAM and RAM.",
                    "default": "status",
                }
            },
        },
    },
]


def _handle_tool_call(name: str, args: dict) -> str:
    if name == "ornith_status":
        action = args.get("action", "status")
        if action == "start":
            st = _ensure_server_running(auto_start=True)
            _http_json("POST", "/load", payload={})
            st = _http_json("GET", "/v1/status", timeout=5.0)
            st["z420_sensors"] = _read_z420_sensors()
            return json.dumps(st, indent=2)
        elif action == "unload":
            res = _http_json("POST", "/unload", payload={})
            res["z420_sensors"] = _read_z420_sensors()
            return json.dumps(res, indent=2)
        else:
            st = _ensure_server_running(auto_start=False)
            st["z420_sensors"] = _read_z420_sensors()
            return json.dumps(st, indent=2)

    if name == "ornith_ask":
        prompt = args.get("prompt", "")
        sys_prompt = args.get("system_prompt") or _build_dynamic_system_prompt()
        enable_thinking = bool(args.get("enable_thinking", True))
        thinking_budget = int(args.get("thinking_budget", 2048))
        max_tokens = int(args.get("max_tokens", 4096))
        include_thinking = bool(args.get("include_thinking", False))
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        return _chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            include_thinking_in_output=include_thinking,
        )

    if name == "ornith_code_review":
        file_path = args.get("file_path")
        code = args.get("code") or ""
        focus = args.get("focus") or "runtime exceptions, logic bugs, edge cases, and specification issues"
        thinking_budget = int(args.get("thinking_budget", 2048))
        if file_path:
            p = Path(file_path)
            if not p.is_file():
                raise ValueError(f"File not found: {file_path}")
            code = p.read_text(encoding="utf-8", errors="replace")
        if not code.strip():
            raise ValueError("Provide either 'file_path' or 'code' to review.")
        prompt = (
            f"Carefully review the following code for {focus}. "
            "Trace execution line by line, list every bug you find with its severity, and provide the corrected code:\n\n"
            f"```\n{code}\n```"
        )
        messages = [
            {"role": "system", "content": _build_dynamic_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        return _chat_completion(
            messages=messages,
            max_tokens=6144,
            enable_thinking=True,
            thinking_budget=thinking_budget,
            include_thinking_in_output=True,
        )

    raise ValueError(f"Unknown tool: {name}")


def _read_message() -> tuple[dict | None, bool]:
    """Read a JSON-RPC message from stdin. Supports both newline-delimited JSON and Content-Length headers."""
    line = sys.stdin.buffer.readline()
    if not line:
        return None, False
    stripped = line.strip()
    while not stripped:
        line = sys.stdin.buffer.readline()
        if not line:
            return None, False
        stripped = line.strip()

    if stripped.lower().startswith(b"content-length:"):
        length = int(stripped.split(b":", 1)[1].strip())
        while True:
            hdr = sys.stdin.buffer.readline()
            if not hdr or hdr.strip() == b"":
                break
        body = sys.stdin.buffer.read(length)
        return json.loads(body.decode("utf-8")), True

    return json.loads(stripped.decode("utf-8")), False


def _write_message(msg: dict, use_headers: bool) -> None:
    raw = json.dumps(msg).encode("utf-8")
    if use_headers:
        sys.stdout.buffer.write(f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii") + raw)
    else:
        sys.stdout.buffer.write(raw + b"\n")
    sys.stdout.buffer.flush()


def main() -> None:
    while True:
        try:
            req, use_headers = _read_message()
        except Exception:
            break
        if req is None:
            break

        method = req.get("method")
        msg_id = req.get("id")
        params = req.get("params") or {}

        # Notifications have no id and expect no response
        if msg_id is None:
            continue

        if method == "initialize":
            client_proto = params.get("protocolVersion", "2024-11-05")
            _write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": client_proto,
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "ornith-z420",
                            "version": "1.0.0",
                        },
                    },
                },
                use_headers,
            )
        elif method == "ping":
            _write_message({"jsonrpc": "2.0", "id": msg_id, "result": {}}, use_headers)
        elif method == "tools/list":
            _write_message({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}, use_headers)
        elif method == "resources/list":
            _write_message({"jsonrpc": "2.0", "id": msg_id, "result": {"resources": []}}, use_headers)
        elif method == "prompts/list":
            _write_message({"jsonrpc": "2.0", "id": msg_id, "result": {"prompts": []}}, use_headers)
        elif method == "tools/call":
            tname = params.get("name", "")
            targs = params.get("arguments") or {}
            try:
                text = _handle_tool_call(tname, targs)
                _write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [{"type": "text", "text": text}],
                            "isError": False,
                        },
                    },
                    use_headers,
                )
            except Exception as e:
                _write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [{"type": "text", "text": f"Error calling {tname}: {e}"}],
                            "isError": True,
                        },
                    },
                    use_headers,
                )
        else:
            _write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                },
                use_headers,
            )


if __name__ == "__main__":
    main()
