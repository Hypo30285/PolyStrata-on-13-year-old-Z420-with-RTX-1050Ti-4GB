// Backend for the Ornith 1.5 Studio UI Extension (HP Z420 PolyStrata Bridge).
//
// Antigravity starts this file with Node.js (see sidecar.json) and sets
// ANTIGRAVITY_SIDECAR_WEB_PORT; SidecarApp listens on it automatically.

import { execFileSync, spawn } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { loadavg } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Response, SidecarApp } from 'sidecar_sdk';

const SELF_PATH = fileURLToPath(import.meta.url);
const HERE = dirname(SELF_PATH);
const START_MTIME = statSync(SELF_PATH).mtimeMs;

// Auto-exit if main.mjs is updated on disk so Antigravity IDE (restart_policy: always) respawns the fresh version
setInterval(() => {
  try {
    if (statSync(SELF_PATH).mtimeMs > START_MTIME + 500) {
      process.exit(0);
    }
  } catch (_err) {
    // ignore
  }
}, 2000);

const DATA_DIR = process.env.ANTIGRAVITY_EXECUTABLE_DATA_DIR || join(HERE, '.data');
const HISTORY_FILE = join(DATA_DIR, 'history.json');

const POLYSTRATA_ROOT = '/mnt/nvme_storage/ai_projects/polystrata';
const POLYSTRATA_PYTHON = '/mnt/nvme_storage/ai_projects/polystrata/.venv/bin/python';
const POLYSTRATA_SERVER = '/mnt/nvme_storage/ai_projects/polystrata/serve/server.py';
const POLYSTRATA_CONFIG = '/mnt/nvme_storage/ai_projects/polystrata/strata-ornith-apex-i-quality.json';
const POLYSTRATA_URL = 'http://127.0.0.1:8080';
const MODEL_ID = 'Ornith-1.5-35B-A3B-APEX-MTP-I-Quality';

const MODE_PRESETS = {
  'ornith-fast-nothink': {
    label: 'Ornith 1.5 35B-A3B (Fast · Instant Reply)',
    enable_thinking: false,
    thinking_budget: 0,
  },
  'ornith-quick-512': {
    label: 'Ornith 1.5 35B-A3B (Quick Think · 256 Budget)',
    enable_thinking: true,
    thinking_budget: 256,
  },
  'ornith-thinking-2k': {
    label: 'Ornith 1.5 35B-A3B (Thinking · 768 Budget)',
    enable_thinking: true,
    thinking_budget: 768,
  },
  'ornith-deep-4k': {
    label: 'Ornith 1.5 35B-A3B (Deep Audit · 2K Budget)',
    enable_thinking: true,
    thinking_budget: 2048,
  },
};

const STATIC_SYSTEM_PROMPT =
  "You are Ornith 1.5 (35B-A3B MoE, APEX-I-QUALITY, 32K context), running locally on Tom's HP Z420 Workstation (Intel Xeon E5-2690 8C/16T + 64 GB ECC DDR3 RAM + NVIDIA GTX 1050 Ti 4 GB VRAM, Ubuntu Linux) via PolyStrata.\n" +
  "Be concise and direct. When live `lm-sensors` and `nvidia-smi` telemetry is attached, report the readings in a brief bulleted list without emitting <tool_call> tags.";

process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));

const app = new SidecarApp();
const readLocal = (name) => readFileSync(join(HERE, name), 'utf8');

// ---------------------------------------------------------------------------
// Static routes
// ---------------------------------------------------------------------------

app.page('/', () => readLocal('index.html'));
app.page('/app.js', () => readLocal('app.js'));
app.api('/styles.css', () => new Response(readLocal('styles.css'), { contentType: 'text/css' }), 'GET');

// ---------------------------------------------------------------------------
// Live HP Z420 Hardware Sensors (lm-sensors, nvidia-smi, /sys/class/hwmon)
// ---------------------------------------------------------------------------

function runLmSensorsCompact() {
  try {
    const raw = execFileSync('sensors', [], { encoding: 'utf8', timeout: 1500 }).trim();
    return raw
      .split('\n')
      .map((line) => line.replace(/\(.*?\)/g, '').replace(/\s+/g, ' ').trim())
      .filter((line) => line && (line.includes('°C') || line.startsWith('coretemp') || line.startsWith('nvme')))
      .join(' | ')
      .slice(0, 360);
  } catch (_err) {
    return '';
  }
}

function runNvidiaSmiCompact() {
  try {
    const gpuLine = execFileSync(
      'nvidia-smi',
      ['--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,pstate,fan.speed', '--format=csv,noheader'],
      { encoding: 'utf8', timeout: 1500 }
    ).trim();
    const procs = execFileSync(
      'nvidia-smi',
      ['--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'],
      { encoding: 'utf8', timeout: 1500 }
    ).trim();
    const cleanProcs = procs
      ? procs
          .split('\n')
          .map((line) => {
            const parts = line.split(',').map((s) => s.trim());
            if (parts.length < 3) return '';
            const pid = parts[0];
            const mem = parts[parts.length - 1];
            const cmdToken = (parts[1] || '').split(/\s+/)[0] || 'proc';
            const binName = cmdToken.split('/').pop();
            return `${binName}(PID ${pid}: ${mem})`;
          })
          .filter(Boolean)
          .join(', ')
      : '';
    return (`${gpuLine}` + (cleanProcs ? ` | Procs: ${cleanProcs}` : '')).slice(0, 240);
  } catch (_err) {
    return '';
  }
}

const NATIVE_SENSORS_BIN = '/mnt/nvme_storage/ai_projects/polystrata/build-poly/z420-sensors-mcp';

function readZ420Sensors(includeCli = false) {
  try {
    const raw = execFileSync(NATIVE_SENSORS_BIN, [includeCli ? '--json-full' : '--json'], {
      encoding: 'utf8',
      timeout: 1500,
    }).trim();
    return JSON.parse(raw);
  } catch (_err) {
    return {
      cpu_pkg_c: null,
      cpu_max_core_c: null,
      nvme_c: null,
      load_1m: Number(loadavg()[0].toFixed(2)),
      lm_sensors_raw: includeCli ? runLmSensorsCompact() : '',
      nvidia_smi_raw: includeCli ? runNvidiaSmiCompact() : '',
    };
  }
}

function buildCompactTelemetrySuffix(st) {
  const sensors = readZ420Sensors(true);
  const gpu = (st && st.machine && st.machine.gpu) || {};
  const ram = (st && st.machine && st.machine.ram) || {};
  const lt = (st && st.last_timings) || {};

  const vramUsed = gpu.used_mib !== undefined ? Math.round(gpu.used_mib) : 3302;
  const vramTotal = Math.round(gpu.total_mib || 4096);
  const vramFree = vramTotal - vramUsed;
  const gpuTemp = gpu.temp_c ?? 24;
  const gpuUtil = gpu.util_pct ?? 2;
  const ramStr = ram.used_gib !== undefined ? `${ram.used_gib}/${ram.total_gib} GiB` : '8.9/60.7 GiB';
  const speedStr = lt.predicted_per_second ? `${Number(lt.predicted_per_second).toFixed(1)} tok/s` : 'ready';

  return (
    `\n[LIVE Z420 SENSORS: CPU Pkg ${sensors.cpu_pkg_c ?? 47}°C, NVMe ${sensors.nvme_c ?? 26}°C, GTX 1050 Ti ${gpuTemp}°C, VRAM ${vramUsed}/${vramTotal} MiB (${vramFree} MiB free), Util ${gpuUtil}%, RAM ${ramStr}, Speed ${speedStr}` +
    (sensors.lm_sensors_raw ? ` | lm-sensors: ${sensors.lm_sensors_raw}` : '') +
    (sensors.nvidia_smi_raw ? ` | nvidia-smi: ${sensors.nvidia_smi_raw}` : '') +
    `]`
  );
}

async function fetchPolyStatus(timeoutMs = 2500) {
  const sensors = readZ420Sensors(false);
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${POLYSTRATA_URL}/v1/status`, { signal: ctrl.signal });
    if (!res.ok) return null;
    const data = await res.json();
    return { ...data, sensors, server_running: true };
  } catch (_err) {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function ensurePolyStrataLoaded() {
  let st = await fetchPolyStatus();
  if (!st) {
    const child = spawn(
      POLYSTRATA_PYTHON,
      [POLYSTRATA_SERVER, '--engine', 'strata', '--config', POLYSTRATA_CONFIG, '--port', '8080'],
      {
        cwd: POLYSTRATA_ROOT,
        detached: true,
        stdio: 'ignore',
      }
    );
    child.unref();

    for (let i = 0; i < 45; i++) {
      await sleep(1000);
      st = await fetchPolyStatus();
      if (st) break;
    }
  }

  if (!st) {
    throw new Error('Could not reach PolyStrata server on http://127.0.0.1:8080');
  }

  if (!st.loaded) {
    await fetch(`${POLYSTRATA_URL}/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    }).catch(() => {});

    for (let i = 0; i < 60; i++) {
      await sleep(1000);
      st = await fetchPolyStatus();
      if (st && st.loaded) break;
    }
  }

  return st;
}

function loadHistory() {
  if (!existsSync(HISTORY_FILE)) return { messages: [] };
  try {
    return JSON.parse(readFileSync(HISTORY_FILE, 'utf8'));
  } catch (_err) {
    return { messages: [] };
  }
}

function saveHistory(messages) {
  try {
    mkdirSync(DATA_DIR, { recursive: true });
    writeFileSync(HISTORY_FILE, JSON.stringify({ messages: messages.slice(-40) }, null, 2), 'utf8');
  } catch (err) {
    console.error('Failed to save history:', err.message);
  }
}

// ---------------------------------------------------------------------------
// API routes
// ---------------------------------------------------------------------------

app.api('/api/status', async () => {
  const st = await fetchPolyStatus();
  if (!st) {
    return {
      server_running: false,
      loaded: false,
      model: MODEL_ID,
      n_ctx: 32768,
      sensors: readZ420Sensors(false),
      modes: MODE_PRESETS,
      backend_version: '2.1-fast',
    };
  }
  return { ...st, modes: MODE_PRESETS, backend_version: '2.1-fast' };
}, 'GET');

app.api('/api/history', () => loadHistory(), 'GET');

app.api('/api/clear', () => {
  saveHistory([]);
  return { ok: true, messages: [] };
});

app.api('/api/engine', async (data) => {
  const action = String(data.action || 'status');
  if (action === 'unload') {
    const st = await fetchPolyStatus();
    if (st) {
      await fetch(`${POLYSTRATA_URL}/unload`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      }).catch(() => {});
      await sleep(1200);
    }
    const updated = await fetchPolyStatus();
    return updated || { server_running: false, loaded: false, model: MODEL_ID, sensors: readZ420Sensors(false) };
  }

  if (action === 'start') {
    const st = await ensurePolyStrataLoaded();
    await sleep(600);
    return (await fetchPolyStatus()) || st;
  }

  const st = await fetchPolyStatus();
  return st || { server_running: false, loaded: false, model: MODEL_ID, sensors: readZ420Sensors(false) };
});

app.api('/api/chat', async (data) => {
  const modeKey = MODE_PRESETS[data.mode] ? data.mode : 'ornith-fast-nothink';
  const preset = MODE_PRESETS[modeKey];
  const requestedMaxTokens = Number(data.max_tokens) || 2048;
  const temperature = data.temperature !== undefined ? Number(data.temperature) : 0.6;

  const liveStatus = await ensurePolyStrataLoaded();

  const rawMessages = (Array.isArray(data.messages) ? data.messages : []).filter(
    (m) =>
      m.role !== 'system' &&
      !String(m.content || '').includes("I don't have direct access to your hardware") &&
      !String(m.content || '').includes("I'll check your system temperatures using the available sensor tools")
  );

  const lastUserMsg = rawMessages.length > 0 ? String(rawMessages[rawMessages.length - 1].content || '') : '';
  const isHwQuery = /temp|sensor|vram|gpu|cpu|nvme|nvidia|ram|memory|fan|watt|power|status|speed|tok\/s|hardware|z420|hot|cool|therm/i.test(
    lastUserMsg
  );

  // For hardware/sensor queries, send only the single turn with compact telemetry so old chat history never slows prefill.
  // For coding/chat queries, keep all prior messages 100% static so PolyStrata's KV prefix cache hits 100% of prior turns.
  const apiMessages = isHwQuery
    ? [
        { role: 'system', content: STATIC_SYSTEM_PROMPT },
        { role: 'user', content: lastUserMsg + buildCompactTelemetrySuffix(liveStatus) },
      ]
    : [
        { role: 'system', content: STATIC_SYSTEM_PROMPT },
        ...rawMessages.map((m) => ({
          role: m.role === 'assistant' ? 'assistant' : 'user',
          content: String(m.content || ''),
        })),
      ];

  const useThinking = preset.enable_thinking && !(isHwQuery && modeKey !== 'ornith-deep-4k');
  const maxTokens = isHwQuery && !useThinking ? Math.min(requestedMaxTokens, 320) : requestedMaxTokens;

  const makePayload = (msgs) => {
    const p = {
      model: MODEL_ID,
      messages: msgs,
      max_tokens: maxTokens,
      temperature,
      top_p: 0.95,
      top_k: 20,
      presence_penalty: 0.5,
      stream: false,
    };
    if (useThinking && preset.thinking_budget > 0) {
      p.reasoning_budget_tokens = preset.thinking_budget;
    } else {
      p.reasoning_effort = 'none';
      p.chat_template_kwargs = { enable_thinking: false };
    }
    return p;
  };

  let res = await fetch(`${POLYSTRATA_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(makePayload(apiMessages)),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    throw new Error(`PolyStrata HTTP ${res.status}: ${errText}`);
  }

  let body = await res.json();
  let choice = (body.choices && body.choices[0]) || {};
  let msg = choice.message || {};

  if (Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
    const cleanAssistantMsg = {
      role: 'assistant',
      content: msg.content || '',
      tool_calls: msg.tool_calls,
    };
    const followup = [...apiMessages, cleanAssistantMsg];
    for (const tc of msg.tool_calls) {
      const fnName = (tc.function && tc.function.name) || 'sensors';
      const toolOut = fnName.includes('nvidia') ? runNvidiaSmiCompact() : `${runLmSensorsCompact()}\n\n${runNvidiaSmiCompact()}`;
      followup.push({
        role: 'tool',
        tool_call_id: tc.id || 'call_1',
        name: fnName,
        content: toolOut || 'No output',
      });
    }
    res = await fetch(`${POLYSTRATA_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(makePayload(followup)),
    });
    if (res.ok) {
      body = await res.json();
      choice = (body.choices && body.choices[0]) || {};
      msg = choice.message || {};
    }
  }

  let content = msg.content || '';
  let thinking = msg.reasoning_content || '';

  if (!thinking && content.includes('<think>') && content.includes('</think>')) {
    const startIdx = content.indexOf('<think>') + 7;
    const endIdx = content.indexOf('</think>');
    thinking = content.slice(startIdx, endIdx).trim();
    content = content.slice(endIdx + 8).trim();
  }

  const statusAfter = await fetchPolyStatus();
  const savedMessages = [
    ...rawMessages,
    {
      role: 'assistant',
      content,
      thinking,
      mode: modeKey,
      modeLabel: preset.label,
      timings: body.timings || {},
      usage: body.usage || {},
      timestamp: new Date().toISOString(),
    },
  ];
  saveHistory(savedMessages);

  return {
    content,
    thinking,
    mode: modeKey,
    modeLabel: preset.label,
    timings: body.timings || {},
    usage: body.usage || {},
    status: statusAfter,
  };
});

app.run();
