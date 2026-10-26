---
title: quadro-p6000-24gb-lighteval-original-22gib-vs-iq4xs-optimum
kind: decision
author: Tom & Antigravity
date: 2026-10-26
touches:
  - /mnt/nvme_storage/ai_projects/polystrata/strata-ornith-apex-i-quality.json
  - /mnt/nvme_storage/ai_projects/polystrata/serve/server.py
  - /mnt/nvme_storage/ai_projects/polystrata/bench/Z420_POLYSTRATA_EVAL_TRACKER.md
---
## Decided
- On the 24 GB NVIDIA Quadro P6000 (`GP102`, `sm_61`, 30 SMs, 3,840 CUDA cores, 384-bit GDDR5X @ 432.8 GB/s) + Intel Xeon E5-2697 v2 (`12C/24T` Ivy Bridge-EP `F16C`), evict background Ollama models (`qwen2.5vl:7b` and `minicpm-v4.6:latest`, freeing 8,814 MiB -u003e 23,847 MiB free VRAM) and serve the original un-requantized `22.09 GiB` `Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf` (`30x Q6_K + 30x Q5_K + 60x IQ4_XS` experts) with `--no-draft --prompt-batch 512 --stage-mib 64 --vram-reserve-mib 500 --conversation-cache-mib 8192`, CUDA Graphs enabled, and `0` tail-skipped experts (`all 8/8` active experts computed per layer). All `10,240 / 10,240` (`100%`) routed experts (`19.3 GiB`) fit in Quadro P6000 VRAM (`23,630 / 24,576 MiB`). Live 35B REST u0026 Lighteval v0.13.0 results (`Row #4b`): GSM8K = `5/5 (100%)` (`84.84s`), ARC-Challenge = `5/5 (100%)` (`73.47s`), Thinking Prefill Median = `339.8 tok/s` (`360.1 tok/s` peak — `6.05x` faster than 1050 Ti `59.5 tok/s` and `9.21x` faster than Oct 14 baseline `36.9 tok/s`), and Thinking Decode Median = `19.30 tok/s` (`19.60 tok/s` peak, `51.03 ms/tok` — `+40.0%` faster decode than the 18.35 GiB `IQ4_XS-Optimum.gguf` `13.80 tok/s` median on the same P6000). On the `N=200` `PascalPyTorchLightevalModel` 3-run suite (`Rows #0d, #5b, #6b`), Quadro P6000 achieves `883,417.5 tok/s` median prefill (`892,967.8` peak, `7.74x` vs P6000 FP32 `114,192.7 tok/s`, `2.67x` vs 1050 Ti) and `1,935.0 tok/s` median decode on `PASCAL-Q8_DP4A` (`200/200 = 100%` GSM8K u0026 ARC-C, `99.50%` Top-1 agreement, `0.000060` Mean KL) and `803,002.9 tok/s` median prefill / `1,839.7 tok/s` median decode on `PASCAL-Q4_DP4A` (`200/200 = 100%`, `98.50%` Top-1, `0.000172` Mean KL).

## Tried and ruled out
- 1. Leaving `--tail-skip 6` or `--no-graphs` enabled on the 24 GB Quadro P6000 (`0` tail-skipped experts computes all 8/8 routed experts per layer at 19.30-19.60 tok/s directly from GDDR5X VRAM). 2. Removing `--no-draft` on `qwen35moe` checkpoints where `blk.40` is a GDN block (`Qwen35::check_draft_block()` throws `draft block 40 is a GDN block; --no-draft loads the model without it`). 3. Prepending a duplicate `system` message in `serve/server.py` `render_prompt()` when the client (`lighteval` / `inspect_ai`) already supplies a leading `system` turn (fixed by merging into `messages[0]['content']`).
