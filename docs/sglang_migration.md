# SGLang Migration Guide

**When to read this**: When the Ollama bottleneck (single-stream, ~90 tok/s on RTX 5080) becomes the
experiment wall-clock limiter and you want to scale to larger Nim/Reversi sweeps or run multiple
concurrent models.

---

## 1. Why migrate?

| Backend | Throughput | Concurrency | Constrained decoding | Notes |
|---|---|---|---|---|
| Ollama (current) | ~90 tok/s single-stream | 1 request | No | Simple, zero config |
| SGLang | ~300–500 tok/s | N async requests | Yes (regex/grammar) | More setup, much faster |
| vLLM | ~250–400 tok/s | N async requests | Yes (outlines) | Alternative; see §6 |

For a 255-game T1 sweep the savings are substantial:

- Ollama @ 90 tok/s → **~2.5 h**
- SGLang @ 350 tok/s → **~38 min** (4–5× speedup)
- SGLang @ 350 tok/s + async batching (N=4) → **~10 min** (15× speedup)

Constrained decoding is a bonus: it eliminates `parse_failed` entirely by forcing the model to emit a
syntactically valid `MOVE: pile=X take=N` tag, removing a key confound from the data.

---

## 2. Current architecture (Ollama)

```
SweepConfig
    └── build_client(cfg.model)
            └── OllamaClient          ← src/cot_knob/llm/ollama_client.py
                    .generate(prompt, max_tokens, ...)
                    .aclose()
```

`OllamaClient` wraps the Ollama REST API (`/api/generate` or `/api/chat`). It is synchronous
internally but awaited via `asyncio`. No batching — one request at a time.

---

## 3. SGLang server setup

### 3a. Install SGLang

```bash
# In the project virtualenv:
pip install "sglang[all]>=0.3" flashinfer-python

# Or if using uv:
uv pip install "sglang[all]" flashinfer-python
```

> **RTX 5080 / Blackwell (sm_120) note**: SGLang's FlashInfer and Triton kernels need to support
> sm_120. As of May 2026, FlashInfer ≥ 0.1.9 and SGLang ≥ 0.3.0 include sm_120 support.
> Verify with:
> ```bash
> python -c "import flashinfer; print(flashinfer.__version__)"
> python -c "import torch; print(torch.cuda.get_device_capability())"  # should be (12, 0)
> ```
> If the installed wheel doesn't have sm_120 kernels, fall back to eager mode:
> `--disable-cuda-graph --attention-backend triton` in the launch command.

### 3b. Launch the server

```bash
# Llama 3.1 8B Q4_K_M (GGUF — same quant as Ollama)
python -m sglang.launch_server \
    --model-path ~/.ollama/models/blobs/<gguf-sha>  \
    --dtype float16 \
    --port 30000 \
    --context-length 8192

# Or using a HF model (float16, faster than Q4 GGUF on GPU):
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --port 30000 \
    --context-length 8192 \
    --tensor-parallel-size 1
```

Wait for `Server is ready` in the log (can take 30–60 s on first load).

### 3c. Smoke-test

```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "prompt": "Hello", "max_tokens": 10}'
```

---

## 4. Required code changes

### 4a. New client: `SGLangClient`

Create `src/cot_knob/llm/sglang_client.py`:

```python
"""SGLang inference client — OpenAI-compatible /v1/completions endpoint."""
from __future__ import annotations
import time
import httpx
from cot_knob.llm.base import LLMClient, Completion

class SGLangClient(LLMClient):
    kind = "sglang"

    def __init__(self, base_url: str = "http://localhost:30000", **kwargs):
        self._url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=300)

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        seed: int | None = None,
        system: str | None = None,
        regex: str | None = None,  # constrained decoding pattern
        **_,
    ) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict = {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            body["seed"] = seed
        if regex is not None:
            body["regex"] = regex  # SGLang-specific constrained decoding

        t0 = time.monotonic()
        resp = await self._client.post(f"{self._url}/v1/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.monotonic() - t0) * 1000

        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data.get("usage", {})
        return Completion(
            text=text,
            n_input_tokens=usage.get("prompt_tokens", 0),
            n_output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            latency_ms=latency_ms,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
```

### 4b. Wire into `build_client`

In `src/cot_knob/llm/factory.py`, add a branch:

```python
from cot_knob.llm.sglang_client import SGLangClient

def build_client(cfg: dict) -> LLMClient:
    backend = cfg.get("backend", "ollama")
    if backend == "sglang":
        return SGLangClient(base_url=cfg.get("base_url", "http://localhost:30000"))
    if backend == "ollama":
        ...  # existing
```

### 4c. Update experiment configs

In any YAML config, change:

```yaml
model:
  backend: sglang          # was: ollama
  name: llama3.1:8b        # ignored by SGLang (server has one model); kept for logging
  base_url: http://localhost:30000
  temperature: 0.0
```

### 4d. Enable constrained decoding (optional but recommended)

Once `SGLangClient` accepts a `regex` kwarg, thread it through `LLMAgent._choose_nim`:

```python
# In LLMAgent._choose_nim — after rendering the prompt
MOVE_REGEX = r"MOVE: pile=[A-Ca-c] take=[1-9][0-9]*"  # adjust pile letters to actual game

comp = await self._client.generate(
    reason_prompt,
    max_tokens=self.budget,
    temperature=self.temperature,
    seed=seed,
    system=system_prompt,
    regex=MOVE_REGEX,  # forces MOVE tag at end
)
```

With this enabled, `parse_failed` becomes structurally impossible and the
`pass2_finish_reason` will always be `"stop"`.

> **Caution**: constrained decoding changes the model's token distribution slightly
> (tokens outside the regex are masked). This is a different measurement condition
> from unconstrained Ollama. Keep it as a separate sweep variant, not a replacement.

---

## 5. Enabling async batching (parallelism)

The sweep currently runs trials sequentially in `run_sweep`. To run N trials concurrently:

```python
# In sweep.py, replace the sequential loop with asyncio.gather:
import asyncio

async def _run_cell(idx, B, seed, llm_side, ...):
    memory = LastMoveMemory()
    llm_agent = LLMAgent(client, memory, budget=B, ...)
    return await play_match(...)

# Run up to CONCURRENCY games in parallel
CONCURRENCY = 4  # tune to GPU VRAM + server throughput
semaphore = asyncio.Semaphore(CONCURRENCY)

async def _bounded(coro):
    async with semaphore:
        return await coro

results = await asyncio.gather(*[
    _bounded(_run_cell(idx, B, seed, llm_side, ...))
    for idx, B, seed, llm_side in cells
])
```

With SGLang's continuous batching, 4 concurrent requests saturate the GPU efficiently
and reduce end-to-end sweep time by ~3–4× beyond single-stream gains.

---

## 6. vLLM as an alternative

If SGLang's sm_120 support is incomplete, vLLM is a drop-in alternative with the same
OpenAI-compatible `/v1/chat/completions` endpoint:

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 --port 30000
```

The `SGLangClient` code above works unchanged for vLLM (it's OpenAI-compatible).
Constrained decoding in vLLM uses `guided_regex` in the request body instead of `regex`.

---

## 7. Migration checklist

- [ ] Confirm sm_120 support: `python -c "import torch; print(torch.cuda.get_device_capability())"`
- [ ] Install SGLang + FlashInfer; verify CUDA kernels compile without fallback
- [ ] Launch server and smoke-test with `curl`
- [ ] Implement `SGLangClient` in `src/cot_knob/llm/sglang_client.py`
- [ ] Wire into `build_client` factory
- [ ] Run existing pytest suite with `backend: sglang` mock (add `SGLangClient` to `mock_client.py` test coverage)
- [ ] Run `smoke_n3.yaml` with `backend: sglang` and verify results match Ollama baseline (within noise)
- [ ] Benchmark: measure actual tok/s via `estimate_sweep_time.py --tps <measured>`
- [ ] (Optional) Add constrained decoding and run as a new sweep variant
- [ ] (Optional) Enable async batching in `sweep.py` with `CONCURRENCY=4`

---

## 8. What NOT to change

- `runner.py`, `config.py`, `store.py` — no changes needed
- JSONL schema — no changes; SGLang output is recorded identically
- Analysis scripts — no changes; data shape is the same
- `nim_b0_anchor` / B=0 cells — SGLang not invoked (budget=0 path), still instant

---

## 9. Decision criteria

| Condition | Recommendation |
|---|---|
| Sweeps finish in < 3 h | Stay on Ollama; simplicity wins |
| Need > 500 trials or T3–T6 | Migrate to SGLang |
| Reversi experiments (two-pass) | SGLang required for throughput |
| Need constrained decoding data | SGLang required |
| sm_120 kernels unstable | Stay on Ollama until patch release |
