#!/usr/bin/env python3
"""Benchmark live Ornith 1.5 35B-A3B (strata-poly REST endpoint) output tok/s against Oct 14 Baseline."""
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8080"
MODEL_NAME = "ornith-1.5-35b-a3b-apex-i-quality"

# Exact prompts from the Oct 14 baseline (bench/results/lighteval_ornith/logs.json)
GSM8K_PROMPTS = [
    (
        'Solve the following math problem step by step. The last line of your\n'
        'response should be of the form "ANSWER: $ANSWER" (without quotes)\n'
        'where $ANSWER is the answer to the problem.\n\n'
        'Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers\' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers\' market?\n\n'
        'Remember to put your answer on its own line at the end in the form\n'
        '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to\n'
        'the problem, and you do not need to use a \\boxed command.\n\n'
        'Reasoning:'
    ),
    (
        'Solve the following math problem step by step. The last line of your\n'
        'response should be of the form "ANSWER: $ANSWER" (without quotes)\n'
        'where $ANSWER is the answer to the problem.\n\n'
        'A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\n\n'
        'Remember to put your answer on its own line at the end in the form\n'
        '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to\n'
        'the problem, and you do not need to use a \\boxed command.\n\n'
        'Reasoning:'
    ),
    (
        'Solve the following math problem step by step. The last line of your\n'
        'response should be of the form "ANSWER: $ANSWER" (without quotes)\n'
        'where $ANSWER is the answer to the problem.\n\n'
        'Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?\n\n'
        'Remember to put your answer on its own line at the end in the form\n'
        '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to\n'
        'the problem, and you do not need to use a \\boxed command.\n\n'
        'Reasoning:'
    ),
    (
        'Solve the following math problem step by step. The last line of your\n'
        'response should be of the form "ANSWER: $ANSWER" (without quotes)\n'
        'where $ANSWER is the answer to the problem.\n\n'
        'James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?\n\n'
        'Remember to put your answer on its own line at the end in the form\n'
        '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to\n'
        'the problem, and you do not need to use a \\boxed command.\n\n'
        'Reasoning:'
    ),
    (
        'Solve the following math problem step by step. The last line of your\n'
        'response should be of the form "ANSWER: $ANSWER" (without quotes)\n'
        'where $ANSWER is the answer to the problem.\n\n'
        'Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi\'s flock is 20 chickens?\n\n'
        'Remember to put your answer on its own line at the end in the form\n'
        '"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to\n'
        'the problem, and you do not need to use a \\boxed command.\n\n'
        'Reasoning:'
    ),
]


def http_json(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def run_single(prompt: str, run_idx: int, enable_thinking: bool = True) -> dict:
    # Add a unique nonce prefix so prompt cache_n=0 (measures true prefill + true decode every run)
    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": f"[Run #{run_idx}] {prompt}"}],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.5,
        "seed": 2026 + run_idx,
        "max_tokens": 512,
        "stream": False,
    }
    if not enable_thinking:
        body["reasoning_effort"] = "none"
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["reasoning_budget_tokens"] = 2048

    resp = http_json("POST", "/v1/chat/completions", body)
    st = http_json("GET", "/v1/status")
    lt = st.get("last_timings") or resp.get("timings") or {}
    gpu = (st.get("machine") or {}).get("gpu") or {}
    return {
        "prompt_n": lt.get("prompt_n", 0),
        "cache_n": lt.get("cache_n", 0),
        "prompt_ms": lt.get("prompt_ms", 0.0),
        "prefill_tps": lt.get("prompt_per_second", 0.0),
        "predicted_n": lt.get("predicted_n", 0),
        "predicted_ms": lt.get("predicted_ms", 0.0),
        "decode_tps": lt.get("predicted_per_second", 0.0),
        "decode_ms_per_tok": lt.get("predicted_per_token_ms", 0.0),
        "draft_n": lt.get("draft_n", 0),
        "draft_accepted": lt.get("draft_n_accepted", 0),
        "vram_mib": gpu.get("used_mib", 0),
        "gpu_temp_c": gpu.get("temp_c", 0),
        "answer_tail": ((resp.get("choices") or [{}])[0].get("message") or {}).get("content", "")[-60:].replace("\n", " "),
    }


def main():
    print("====================================================================================")
    print("Ornith 1.5 35B-A3B (18.35 GiB IQ4_XS Optimum) Live Output Tok/s vs. Oct 14 Baseline")
    print("====================================================================================")
    # Hot-swap to ornith-1.5-35b-a3b-iq4xs-optimum and then back to ornith-1.5-35b-a3b-apex-i-quality
    # so server.py reloads the updated strata-ornith-apex-i-quality.json pointing to the 18.35 GiB GGUF
    print("Loading 18.35 GiB Ornith-1.5-35B-A3B-APEX-MTP-IQ4_XS-Optimum.gguf into strata-poly...")
    http_json("POST", "/api/models/load", {"model": "ornith-1.5-35b-a3b-iq4xs-optimum"})
    http_json("POST", "/api/models/load", {"model": "ornith-1.5-35b-a3b-apex-i-quality"})
    st0 = http_json("GET", "/v1/status")
    print(f"Engine Version: {st0.get('engine')} | Model: {st0.get('model')}")
    gpu0 = (st0.get("machine") or {}).get("gpu") or {}
    print(f"Initial GPU State: {gpu0.get('name')} | VRAM: {gpu0.get('used_mib')}/{gpu0.get('total_mib')} MiB | Temp: {gpu0.get('temp_c')}°C")
    pcie_speed = Path("/sys/bus/pci/devices/0000:05:00.0/current_link_speed").read_text().strip()
    print(f"Idle PCIe Link Speed: {pcie_speed}")
    log_tail = Path("/mnt/nvme_storage/ai_projects/polystrata/strata-ornith-apex-i-quality.log").read_text().splitlines()[-4:]
    for line in log_tail:
        print(f"  [log] {line}")
    print("------------------------------------------------------------------------------------")

    # Mode A: Thinking Enabled (exact Oct 14 Method Card settings, 5 GSM8K samples)
    print("\n[A] Mode 1: Thinking Enabled (Oct 14 Method Card Settings, 5 GSM8K Samples):")
    think_runs = []
    for i, p in enumerate(GSM8K_PROMPTS, 1):
        r = run_single(p, i, enable_thinking=True)
        think_runs.append(r)
        print(
            f"  Sample {i}/5: Prefill = {r['prefill_tps']:6.1f} t/s ({r['prompt_n']} tok in {r['prompt_ms']:.0f} ms) | "
            f"Decode = {r['decode_tps']:5.2f} t/s ({r['predicted_n']} tok, {r['decode_ms_per_tok']:.2f} ms/tok) | "
            f"Draft = {r['draft_accepted']}/{r['draft_n']} | VRAM = {r['vram_mib']:.0f} MiB | Tail: {r['answer_tail']}"
        )

    med_prefill_think = statistics.median(r["prefill_tps"] for r in think_runs)
    med_decode_think = statistics.median(r["decode_tps"] for r in think_runs)
    max_prefill_think = max(r["prefill_tps"] for r in think_runs)
    max_decode_think = max(r["decode_tps"] for r in think_runs)

    # Mode B: Non-Thinking Direct Answer (3 Runs)
    print("\n[B] Mode 2: Non-Thinking Fast Answer (`enable_thinking=False`, 3 Runs):")
    fast_runs = []
    for i in range(1, 4):
        r = run_single(GSM8K_PROMPTS[i - 1], 100 + i, enable_thinking=False)
        fast_runs.append(r)
        print(
            f"  Run {i}/3:    Prefill = {r['prefill_tps']:6.1f} t/s ({r['prompt_n']} tok in {r['prompt_ms']:.0f} ms) | "
            f"Decode = {r['decode_tps']:5.2f} t/s ({r['predicted_n']} tok, {r['decode_ms_per_tok']:.2f} ms/tok) | "
            f"Draft = {r['draft_accepted']}/{r['draft_n']} | VRAM = {r['vram_mib']:.0f} MiB | Tail: {r['answer_tail']}"
        )

    med_prefill_fast = statistics.median(r["prefill_tps"] for r in fast_runs)
    med_decode_fast = statistics.median(r["decode_tps"] for r in fast_runs)

    print("\n====================================================================================")
    print("SUMMARY vs. OCT 14 BASELINE (Row #0: Prefill 32.0–41.8 t/s | Decode 20.3–21.3 t/s)")
    print("====================================================================================")
    print(f"  Thinking Mode (5-Run Median):     Prefill = {med_prefill_think:6.1f} t/s (Peak {max_prefill_think:6.1f} t/s) | Decode = {med_decode_think:5.2f} t/s (Peak {max_decode_think:5.2f} t/s)")
    print(f"  Non-Thinking Mode (3-Run Median): Prefill = {med_prefill_fast:6.1f} t/s | Decode = {med_decode_fast:5.2f} t/s")
    print("====================================================================================")


if __name__ == "__main__":
    main()
