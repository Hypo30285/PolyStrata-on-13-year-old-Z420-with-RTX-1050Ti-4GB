# Hugging Face Lighteval Knowledge Base & Technical Guide

Lighteval is an evaluation toolkit developed by Hugging Face's Leaderboard and Evals team for evaluating Large Language Models (LLMs) and inference engines across diverse environments and backends.

Source References:
- [Hugging Face Lighteval Documentation](https://huggingface.co/docs/lighteval/en/index)
- [Official llama.cpp Repository (ggml-org)](https://github.com/ggml-org/llama.cpp)

---

## 1. Architectural Overview & Backends

Lighteval decouples the **evaluation task logic** (datasets, prompt templates, few-shot samplers, metrics) from the **inference runtime**. This allows evaluating the exact same benchmark across different execution environments:

| Backend | Identifier | Description | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **Inspect AI** | `eval` | Uses UK AISI `inspect-ai` as execution engine | Standard recommended evaluation runner |
| **Transformers / Accelerate** | `accelerate` / `transformers` | Direct PyTorch model execution via Hugging Face Transformers | Local checkpoint weights evaluation on CPU/GPU |
| **OpenAI / LiteLLM API** | `endpoint` / `litellm` | Queries OpenAI-compatible HTTP endpoints (`/v1/chat/completions`) | Testing local inference engines (**PolyStrata**, **Ollama**, **vLLM**, **TGI**, **SGLang**) |
| **Custom Model Class** | `custom` | Subclass `LightevalModel` with custom Python logic | Testing bespoke runtimes, quantizers, or non-standard architectures |
| **VLLM** | `vllm` | High-throughput batched vLLM engine | High-speed local GPU benchmarking |
| **SGLang** | `sglang` | RadixAttention-accelerated inference backend | High-throughput structured generation evaluation |
| **TGI** | `tgi` | Hugging Face Text Generation Inference container | Enterprise self-hosted cluster benchmarks |
| **Inference Providers** | `inference-providers` | Hugging Face serverless provider router (Together, Fireworks, etc.) | Cloud model validation |

---

## 2. Testing Local LLMs & Inference Engines (e.g., PolyStrata / Z420)

When benchmarking local inference engines on workstations like the HP Z420, models are typically exposed via an **OpenAI-compatible HTTP REST API** (e.g. PolyStrata `strata-poly` serving Ornith 1.5 at `http://127.0.0.1:8080/v1` or Ollama at `http://127.0.0.1:11434/v1`).

### Running Endpoint Evaluations via CLI

```bash
# 1. Set the target inference engine URL
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="dummy-key-or-local"

# 2. Run lighteval against the endpoint
lighteval eval \
    "openai/ornith-1.5-35b-a3b" \
    gsm8k \
    --max-samples 50 \
    --output-dir results/lighteval \
    --save-details
```

### Using YAML Configuration for Endpoint Generation

You can create an endpoint profile (e.g., `polystrata_endpoint.yaml`):

```yaml
model_parameters:
  model_name: "openai/strata-ornith-1.5"
  base_url: "http://127.0.0.1:8080/v1"
  api_key: "polystrata-local"
generation_parameters:
  temperature: 0.0
  max_new_tokens: 1024
  top_p: 0.95
```

Execute:
```bash
lighteval endpoint litellm polystrata_endpoint.yaml gsm8k --output-dir results/
```

---

## 3. Core Task Library & Benchmark Formats

Lighteval provides hundreds of built-in evaluation tasks. Tasks are referenced either by their shorthand or full canonical identifier: `{suite}|{task}|{num_few_shot}|{truncate}`.

### Popular Benchmark Suites
- **MMLU / MMLU-Pro**: Multidisciplinary knowledge and academic reasoning (`mmlu`, `mmlu:pro`).
- **GSM8K**: Grade-school math multi-step reasoning (`gsm8k`).
- **HumanEval / MBPP**: Python coding synthesis and verification (`humaneval`).
- **MATH**: Challenging high-school competition mathematics (`math`).
- **IFEval**: Instruction following compliance assessment (`ifeval`).
- **GPQA**: Expert graduate-level Google-proof Q&A (`gpqa:diamond`).
- **TruthfulQA / HellaSwag / ARC**: Multiple choice commonsense & scientific reasoning.

### Listing Available Tasks
```bash
lighteval tasks list
lighteval tasks inspect gsm8k
```

---

## 4. Custom Model Implementation (`LightevalModel`)

When an engine has a direct Python binding or custom C++ shared library wrapper, Lighteval allows evaluating it via a custom model class without needing an HTTP server.

```python
from typing import List
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached

class CustomEngineModel(LightevalModel):
    def __init__(self, config):
        super().__init__(config)
        self._cache = SampleCache(config)
        # Load local engine / model weights here

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: List[Doc]) -> List[ModelResponse]:
        """Generates completions until a stop token or max length."""
        responses = []
        for doc in docs:
            prompt = doc.query
            max_tokens = doc.generation_size or 512
            # Call inference engine
            generated_text = self.my_engine_generate(prompt, max_tokens, doc.stop_sequence)
            responses.append(ModelResponse(result=generated_text))
        return responses

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: List[Doc]) -> List[ModelResponse]:
        """Calculates continuation log-probabilities for multiple-choice tasks."""
        responses = []
        for doc in docs:
            logprob = self.my_engine_logprob(doc.query, doc.choices[doc.gold_index])
            responses.append(ModelResponse(result=logprob))
        return responses

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: List[Doc]) -> List[ModelResponse]:
        """Calculates rolling perplexity for language modeling tests."""
        # Rolling loglikelihood implementation
        pass
```

Run custom model:
```bash
lighteval custom "my-engine" "path/to/custom_model.py" "gsm8k" --max-samples 10
```

---

## 5. Adding Custom Benchmark Tasks

Custom tasks are defined using `LightevalTaskConfig` and `Doc` prompt mapping:

```python
from lighteval.tasks.requests import Doc
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.metrics import Metrics

def custom_prompt_fn(line: dict, task_name: str) -> Doc:
    return Doc(
        task_name=task_name,
        query=f"Instruction: {line['instruction']}\nInput: {line['input']}\nOutput:",
        choices=[line["target"]],
        gold_index=0,
    )

custom_task = LightevalTaskConfig(
    name="local_engine_eval",
    prompt_function=custom_prompt_fn,
    hf_repo="local/json_or_hub_dataset",
    hf_subset="default",
    metrics=[Metrics.EXACT_MATCH, Metrics.ROUGE_L],
    generation_size=512,
    stop_sequence=["\nInstruction:"],
)

TASKS_TABLE = [custom_task]
```

---

## 6. Performance Best Practices on HP Z420 Workstation

1. **Concurrency Control (`--max-connections`)**:
   - For local single-GPU or CPU-bound inference (such as `strata-poly` on Xeon E5-2690 + GTX 1050 Ti), keep `--max-connections 1` or `2` to prevent memory contention and context thrashing.
2. **Timeout Buffers (`--timeout`)**:
   - For complex tasks with large reasoning outputs, set `--timeout 300` or `--timeout 600` so slow generation loops do not drop requests.
3. **Sample Slicing (`--max-samples`)**:
   - When verifying inference engine parity or debugging engine changes, test with `--max-samples 10` or `--max-samples 25` before launching full-scale benchmark suites.
4. **Detailed Artifacts (`--save-details`)**:
   - Outputs sample-by-sample prompt-response-score tables in Parquet format, essential for diagnosing edge-case regressions or tokenization mismatches.

---

## 7. Empirical Benchmark Case Study: MiniCPM-V 5B Optimization

Empirical testing conducted on October 14, 2026 across Ollama, PolyStrata, and llama.cpp on the HP Z420 Workstation:

### Key Architectural Discoveries:
1. **PolyStrata (`strata-poly`) Specialization:**
   - `strata-poly` is strictly a specialized Mixture-of-Experts (MoE) engine optimized for sparse expert routing (e.g. Ornith 1.5 35B-A3B). When tested against MiniCPM-V 5B GGUF weights, it explicitly exited with:
     `strata-poly: minicpm-v-5b-q4_k_m.gguf is a 'qwen2' model, which this engine does not run`
   - Therefore, for dense multimodal vision architectures, native `llama.cpp` (`llama-server`) serves as the designated native runtime.
2. **Sandy Bridge Instruction Set Tuning:**
   - Intel Xeon E5-2690 v1 supports AVX 1.0 but lacks F16C and BMI2. Default upstream CMake builds generated F16C conversion instructions causing `SIGILL` (code 132).
   - Building with `-DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF` resolved all illegal instruction faults while preserving full vector performance.
3. **Comparative Evaluation Scorecard:**

| Metric / Dimension | Ollama (`minicpm-v-opt:latest`) | llama.cpp (`llama-server`) | Winning Engine |
| :--- | :---: | :---: | :--- |
| **Engine Architecture** | Go daemon + bundled runtime | Pure Native C++17 binary (`-O3 -s`) | **llama.cpp** |
| **Host System RAM Overhead** | ~3,747 MB RSS | Pure direct mmap, zero daemon bloat | **llama.cpp** (~3.7 GB RAM reclaimed) |
| **Prompt Prefill Speed** | ~32 tok/s | **58.37 tok/s** | **llama.cpp** (+82% speedup) |
| **Token Decode Speed** | ~7.1 tok/s | **8.88 tok/s** | **llama.cpp** (+25% speedup) |
| **Peak GPU VRAM** | 2,340 MiB (10 layers) | 3,368 MiB (14 layers) | **llama.cpp** (Optimized layer offload) |
| **ARC-Challenge Accuracy** | 1.00 (100%) | 1.00 (100%) | **Parity** |
| **GSM8K Accuracy** | 0.67 | 0.33 | **Ollama** |



---

## 8. Official Lighteval Benchmark: Ornith 1.5 35B-A3B (APEX-I-QUALITY)

Evaluation conducted on October 14, 2026 using Hugging Face Lighteval (v0.13.0) via PolyStrata (`http://127.0.0.1:8080/v1`):

| Evaluation Task | Samples | Accuracy | Decode Speed | Prompt Prefill Speed | Speculative Acceptance | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** (Math Reasoning) | 5 | **1.000 (100%)** | 20.3 tok/s | 41.8 tok/s | 58.3% (7/12 drafts) | **Passed** |
| **ARC-Challenge** (Science Q&A) | 5 | **1.000 (100%)** | 21.3 tok/s | 32.0 tok/s | 50.0% (1/2 drafts) | **Passed** |

### Hardware Telemetry During Evaluation:
- **Engine Process:** `strata-poly` (PID 57039)
- **GPU VRAM:** 3,186 MiB / 4,096 MiB on GeForce GTX 1050 Ti (~647 MiB safe headroom)
- **Host ECC RAM:** 7.5 GiB / 60.7 GiB
- **Thermals:** GPU 38°C, CPU Package 59.0°C
- **Evaluation Logs:** `/mnt/nvme_storage/ai_projects/polystrata/bench/results/lighteval_ornith/`
