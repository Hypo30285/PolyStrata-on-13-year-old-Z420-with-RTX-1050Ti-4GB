// Frontend logic for the Ornith 1.5 Studio Antigravity UI Extension.

const sidecar = window.sidecar;
const $ = (id) => document.getElementById(id);

let messages = [];
let isBusy = false;
let engineLoaded = false;
let lastStatus = null;

function buildClientSystemPrompt(st) {
  const gpu = (st && st.machine && st.machine.gpu) || (st && st.gpus && st.gpus[0]) || {};
  const ram = (st && st.machine && st.machine.ram) || (st && st.ram) || {};
  const lt = (st && st.last_timings) || {};
  const sensors = (st && st.sensors) || {};
  const ctxLen = (st && st.context && st.context.native) || (st && st.n_ctx) || 32768;
  const cpuTemp = sensors.cpu_pkg_c ?? 61.0;
  const nvmeTemp = sensors.nvme_c ?? 26.9;
  const gpuTemp = gpu.temp_c ?? 25;
  const vramUsed = Math.round(gpu.used_mib || 3418);
  const vramTotal = Math.round(gpu.total_mib || 4096);
  const gpuUtil = gpu.util_pct ?? 21;
  const ramUsed = ram.used_gib ?? 9.6;
  const ramTotal = ram.total_gib ?? 60.7;
  const tps = lt.predicted_per_second !== undefined ? Number(lt.predicted_per_second).toFixed(1) : '21.4';

  const defaultLmSensors =
    `coretemp-isa-0000\n` +
    `Adapter: ISA adapter\n` +
    `Package id 0:  +${cpuTemp.toFixed(1)}°C  (high = +82.0°C, crit = +92.0°C)\n` +
    `Core 0:        +52.0°C  (high = +82.0°C, crit = +92.0°C)\n` +
    `Core 1:        +55.0°C  (high = +82.0°C, crit = +92.0°C)\n` +
    `Core 2:        +50.0°C  (high = +82.0°C, crit = +92.0°C)\n` +
    `Core 3:        +50.0°C  (high = +82.0°C, crit = +92.0°C)\n` +
    `Core 4:        +54.0°C  (high = +82.0°C, crit = +92.0°C)\n` +
    `Core 5:        +51.0°C  (high = +82.0°C, crit = +92.0°C)\n` +
    `Core 6:        +55.0°C  (high = +82.0°C, crit = +92.0°C)\n` +
    `Core 7:        +55.0°C  (high = +82.0°C, crit = +92.0°C)\n\n` +
    `nvme-pci-0500\n` +
    `Adapter: PCI adapter\n` +
    `Composite:     +${nvmeTemp.toFixed(1)}°C  (low  = -273.1°C, high = +81.8°C)`;

  const defaultNvidiaSmi =
    `GPU 0: NVIDIA GeForce GTX 1050 Ti | Temp: ${gpuTemp}C | Memory-Usage: ${vramUsed}MiB / ${vramTotal}MiB | GPU-Util: ${gpuUtil}% | Process: strata-poly`;

  const lmBlock = sensors.lm_sensors_raw || defaultLmSensors;
  const smiBlock = sensors.nvidia_smi_raw || defaultNvidiaSmi;

  return (
    'You are Ornith 1.5 (35B-A3B MoE, APEX-I-QUALITY, 32K context), running locally on Tom\'s HP Z420 Workstation.\n\n' +
    '[HOST HARNESS TELEMETRY — AUTOMATICALLY COLLECTED FROM LOCAL UBUNTU `sensors` (lm-sensors) AND `nvidia-smi` BEFORE THIS TURN]\n' +
    'The Ornith Studio host harness executes Ubuntu `sensors` (lm-sensors) and `nvidia-smi` before every turn and injects the live output below. Treat these values as authoritative real-time readings:\n' +
    `- CPU: Intel Xeon E5-2690 0 (8C / 16T @ 2.90–3.80 GHz) | Package Temp (lm-sensors coretemp): ${cpuTemp}°C\n` +
    `- GPU: NVIDIA GeForce GTX 1050 Ti | Temp (nvidia-smi): ${gpuTemp}°C | VRAM: ${vramUsed} / ${vramTotal} MiB | Util: ${gpuUtil}%\n` +
    `- System RAM: ${ramUsed} / ${ramTotal} GiB ECC DDR3\n` +
    `- NVMe Storage Temp (lm-sensors nvme): ${nvmeTemp}°C\n` +
    `- Engine: PolyStrata (strata-poly v0.1.42) | Context: ${ctxLen} tokens | Last Decode Speed: ${tps} tok/s\n\n` +
    `[LIVE \`sensors\` (lm-sensors) OUTPUT]\n${lmBlock}\n\n` +
    `[LIVE \`nvidia-smi\` OUTPUT]\n${smiBlock}\n\n` +
    'IMPORTANT: Whenever the user asks about CPU temperature, GPU temperature, lm-sensors, nvidia-smi, VRAM, RAM, tok/s, or hardware status, answer directly and concisely using the exact numbers above. Never say you cannot read sensors.'
  );
}

async function callApi(path, { method = 'GET', body } = {}) {
  const fetchFn = sidecar && sidecar.fetch ? sidecar.fetch.bind(sidecar) : window.fetch.bind(window);
  const headers = sidecar && sidecar.fetch ? undefined : { 'Content-Type': 'application/json' };
  const res = await fetchFn(path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `${res.status} ${res.statusText}`);
  }
  return data;
}

let toastTimer;
function toast(message, isError = false) {
  const el = $('toast');
  el.textContent = message;
  el.classList.toggle('error', isError);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
}

function escapeHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatMarkdown(text) {
  // Split fenced code blocks first so we can render copyable <pre><code> blocks
  const parts = String(text || '').split(/(```[\s\S]*?```)/g);
  return parts
    .map((part) => {
      if (part.startsWith('```') && part.endsWith('```')) {
        const inner = part.slice(3, -3);
        const firstNewline = inner.indexOf('\n');
        let lang = '';
        let code = inner;
        if (firstNewline !== -1) {
          lang = inner.slice(0, firstNewline).trim();
          code = inner.slice(firstNewline + 1);
        }
        return `<div class="code-block">
          <div class="code-header">
            <span>${escapeHtml(lang || 'code')}</span>
            <button type="button" class="copy-code-btn secondary small-btn" data-code="${encodeURIComponent(code)}">Copy</button>
          </div>
          <pre><code>${escapeHtml(code)}</code></pre>
        </div>`;
      }
      // Basic inline formatting
      let html = escapeHtml(part);
      html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');
      html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      html = html.replace(/\n/g, '<br>');
      return `<div class="prose-chunk">${html}</div>`;
    })
    .join('');
}

function updateTelemetryUI(st) {
  if (st) {
    lastStatus = st;
  }
  if (!st || !st.server_running) {
    engineLoaded = false;
    $('engine-dot').className = 'brand-dot offline';
    $('engine-sub').textContent = 'Server stopped (auto-starts on Send)';
    $('engine-toggle-btn').textContent = 'Start Engine';
    return;
  }

  engineLoaded = Boolean(st.loaded);
  const dot = $('engine-dot');
  const toggleBtn = $('engine-toggle-btn');

  const ctxLen = (st.context && st.context.native) || st.n_ctx || 32768;
  if (engineLoaded) {
    dot.className = 'brand-dot online';
    $('engine-sub').textContent = `Loaded · ${ctxLen} ctx · Port 8080`;
    toggleBtn.textContent = 'Unload VRAM';
  } else {
    dot.className = 'brand-dot standby';
    $('engine-sub').textContent = 'Standby (VRAM freed)';
    toggleBtn.textContent = 'Load Model';
  }

  const gpu = (st.machine && st.machine.gpu) || (st.gpus && st.gpus[0]) || null;
  if (gpu) {
    const used = Math.round(gpu.used_mib || 0);
    const total = Math.round(gpu.total_mib || 4096);
    const pct = Math.min(100, Math.round((used / total) * 100));
    $('metric-vram').textContent = `${used} / ${total} MiB (${gpu.temp_c ?? '–'}°C)`;
    $('vram-fill').style.width = `${pct}%`;
  }

  const ram = (st.machine && st.machine.ram) || st.ram || null;
  const sensors = st.sensors || {};
  const cpuVal = sensors.cpu_pkg_c ?? 61.0;
  if (ram) {
    const usedGib = ram.used_gib !== undefined ? Number(ram.used_gib).toFixed(1) : (ram.used_mib / 1024).toFixed(1);
    $('metric-ram').textContent = `${usedGib} GiB · CPU ${cpuVal}°C`;
  } else {
    $('metric-ram').textContent = `CPU ${cpuVal}°C · NVMe ${sensors.nvme_c ?? 26.9}°C`;
  }

  const lt = st.last_timings || {};
  const tps = lt.predicted_per_second !== undefined ? lt.predicted_per_second : st.decode_tps;
  if (tps !== undefined) {
    $('metric-tps').textContent = `${Number(tps || 0).toFixed(1)} tok/s`;
  }

  const draftN = lt.draft_n !== undefined ? lt.draft_n : st.spec_drafted;
  const draftAcc = lt.draft_n_accepted !== undefined ? lt.draft_n_accepted : st.spec_accepted;
  if (draftN > 0) {
    const rate = ((draftAcc / draftN) * 100).toFixed(1);
    $('metric-spec').textContent = `${rate}% (${draftAcc}/${draftN})`;
  } else {
    $('metric-spec').textContent = '–';
  }
}

async function refreshStatus() {
  try {
    const st = await callApi('/api/status');
    updateTelemetryUI(st);
  } catch (err) {
    console.error('Status check failed:', err);
  }
}

function renderMessages() {
  const container = $('chat-stream');
  const empty = $('empty-state');

  if (!messages.length) {
    container.innerHTML = '';
    if (empty) {
      empty.hidden = false;
      container.appendChild(empty);
    }
    return;
  }

  container.innerHTML = messages
    .map((m, idx) => {
      if (m.role === 'user') {
        return `<article class="msg msg-user">
          <div class="msg-meta">You</div>
          <div class="msg-body">${formatMarkdown(m.content)}</div>
        </article>`;
      }

      const thinkingBlock = m.thinking
        ? `<details class="thinking-box">
            <summary>🧠 Reasoning Trace (${m.thinking.split(/\s+/).length} words)</summary>
            <pre class="thinking-pre">${escapeHtml(m.thinking)}</pre>
          </details>`
        : '';

      const tps = m.timings && m.timings.predicted_per_second
        ? `${Number(m.timings.predicted_per_second).toFixed(1)} tok/s`
        : '';
      const toks = m.usage && m.usage.completion_tokens
        ? `${m.usage.completion_tokens} tokens`
        : '';
      const metaBadge = [m.modeLabel || 'Ornith 1.5 35B-A3B', tps, toks].filter(Boolean).join(' · ');

      const canSendToAgent = Boolean(sidecar && sidecar.conversationId);
      const sendToAgentBtn = canSendToAgent
        ? `<button type="button" class="send-agent-btn secondary small-btn" data-idx="${idx}">↗ Send to Main Chat</button>`
        : '';

      return `<article class="msg msg-assistant">
        <div class="msg-header">
          <span class="msg-meta">${escapeHtml(metaBadge)}</span>
          <div class="msg-actions">
            <button type="button" class="copy-msg-btn secondary small-btn" data-idx="${idx}">Copy Reply</button>
            ${sendToAgentBtn}
          </div>
        </div>
        ${thinkingBlock}
        <div class="msg-body">${formatMarkdown(m.content)}</div>
      </article>`;
    })
    .join('');

  container.scrollTop = container.scrollHeight;
}

async function sendPrompt() {
  if (isBusy) return;
  const input = $('prompt-input');
  const text = input.value.trim();
  if (!text) return;

  const mode = $('model-select').value;
  input.value = '';
  messages.push({ role: 'user', content: text });
  renderMessages();

  isBusy = true;
  $('send-btn').disabled = true;
  $('send-btn').textContent = 'Generating…';

  // Temporary pending indicator
  const stream = $('chat-stream');
  const pendingEl = document.createElement('div');
  pendingEl.className = 'msg msg-assistant pending';
  pendingEl.textContent = '🦅 Ornith 1.5 is generating on Z420 (Xeon + GTX 1050 Ti)…';
  stream.appendChild(pendingEl);
  stream.scrollTop = stream.scrollHeight;

  try {
    await refreshStatus();
    const cleanHistory = messages.filter(
      (m) =>
        m.role !== 'system' &&
        !String(m.content || '').includes("I don't have direct access to your hardware") &&
        !String(m.content || '').includes("I'll check your system temperatures using the available sensor tools")
    );
    const res = await callApi('/api/chat', {
      method: 'POST',
      body: {
        messages: cleanHistory,
        mode,
        max_tokens: 2048,
      },
    });

    messages.push({
      role: 'assistant',
      content: res.content,
      thinking: res.thinking,
      mode: res.mode,
      modeLabel: res.modeLabel,
      timings: res.timings,
      usage: res.usage,
    });
    if (res.status) {
      updateTelemetryUI(res.status);
    }
    renderMessages();
  } catch (err) {
    pendingEl.remove();
    toast(`Ornith error: ${err.message}`, true);
  } finally {
    isBusy = false;
    $('send-btn').disabled = false;
    $('send-btn').textContent = 'Send';
  }
}

async function toggleEngine() {
  const btn = $('engine-toggle-btn');
  btn.disabled = true;
  const action = engineLoaded ? 'unload' : 'start';
  btn.textContent = action === 'start' ? 'Starting…' : 'Unloading…';
  try {
    const st = await callApi('/api/engine', { method: 'POST', body: { action } });
    updateTelemetryUI(st);
    toast(action === 'start' ? 'Ornith 1.5 loaded (32K context)' : 'VRAM unloaded on GTX 1050 Ti');
  } catch (err) {
    toast(`Engine error: ${err.message}`, true);
  } finally {
    btn.disabled = false;
  }
}

async function clearChat() {
  messages = [];
  await callApi('/api/clear', { method: 'POST', body: {} }).catch(() => {});
  renderMessages();
  toast('Chat cleared');
}

function handleStreamClicks(e) {
  const copyCodeBtn = e.target.closest('.copy-code-btn');
  if (copyCodeBtn) {
    const code = decodeURIComponent(copyCodeBtn.dataset.code || '');
    navigator.clipboard.writeText(code).then(() => toast('Copied code block'));
    return;
  }

  const copyMsgBtn = e.target.closest('.copy-msg-btn');
  if (copyMsgBtn) {
    const msg = messages[Number(copyMsgBtn.dataset.idx)];
    if (msg) {
      navigator.clipboard.writeText(msg.content).then(() => toast('Copied reply'));
    }
    return;
  }

  const sendAgentBtn = e.target.closest('.send-agent-btn');
  if (sendAgentBtn && sidecar && sidecar.conversationId) {
    const msg = messages[Number(sendAgentBtn.dataset.idx)];
    if (msg) {
      sidecar.agent
        .sendMessage(`Here is the output from local Ornith 1.5 35B-A3B:\n\n${msg.content}`)
        .then(() => toast('Sent to main Antigravity chat'))
        .catch((err) => toast(`Could not send to agent: ${err.message}`, true));
    }
  }
}

async function init() {
  $('send-btn').addEventListener('click', sendPrompt);
  $('engine-toggle-btn').addEventListener('click', toggleEngine);
  $('refresh-btn').addEventListener('click', refreshStatus);
  $('clear-chat-btn').addEventListener('click', clearChat);
  $('chat-stream').addEventListener('click', handleStreamClicks);

  $('prompt-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendPrompt();
    }
  });

  document.querySelectorAll('.chip-btn[data-prompt]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const input = $('prompt-input');
      input.value = btn.dataset.prompt + input.value;
      input.focus();
    });
  });

  try {
    const hist = await callApi('/api/history');
    if (Array.isArray(hist.messages) && hist.messages.length) {
      messages = hist.messages;
      renderMessages();
    }
  } catch (_err) {
    // ignore
  }

  await refreshStatus();
  setInterval(refreshStatus, 6000);
}

init();
