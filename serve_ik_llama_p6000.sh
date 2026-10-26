#!/usr/bin/env bash
# Launch ik_llama.cpp (Pascal sm_61 / GP102 + Ivy Bridge E5-2697 v2 optimized) on HP Z420 Quadro P6000 24GB
# Usage:
#   ./serve_ik_llama_p6000.sh [apex|iq4] [port] [--mtp]
set -euo pipefail

MODEL_VARIANT="${1:-apex}"
PORT="${2:-8080}"
MTP_FLAG="${3:-}"

IK_SERVER="/mnt/nvme_storage/ai_projects/ik_llama.cpp/build/bin/llama-server"

if [[ "$MODEL_VARIANT" == "iq4" ]]; then
    MODEL_PATH="/mnt/nvme_storage/ai_projects/strata_models/ornith-APEX-I-QUALITY/Ornith-1.5-35B-A3B-APEX-MTP-IQ4_XS-Optimum.gguf"
    ALIAS="ornith-1.5-35b-a3b-iq4-xs-optimum"
else
    MODEL_PATH="/mnt/nvme_storage/ai_projects/strata_models/ornith-APEX-I-QUALITY/Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf"
    ALIAS="ornith-1.5-35b-a3b-apex-i-quality"
fi

EXTRA_ARGS=()
if [[ "$MTP_FLAG" == "--mtp" ]]; then
    EXTRA_ARGS+=(--spec-type "mtp:n_max=1,p_min=0.0")
fi

exec "$IK_SERVER" \
    -m "$MODEL_PATH" \
    --alias "$ALIAS" \
    --host 127.0.0.1 \
    --port "$PORT" \
    -ngl 99 \
    -c 32768 \
    -np 1 \
    -fmoe \
    -fa \
    -gr \
    -ctk q8_0 \
    -ctv q8_0 \
    --no-mmap \
    --jinja \
    "${EXTRA_ARGS[@]}"
