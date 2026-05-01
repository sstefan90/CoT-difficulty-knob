# Run log

One short paragraph per experimental run. Newest first. Mirror the SQLite
`runs.run_id` so we can cross-reference.

## Template

```
### YYYY-MM-DD — <run_id> — <config name>
- Backend: ollama / sglang / mock
- Model: deepseek-r1:7b
- N per cell: 3
- Wallclock: HH:MM
- Outcome: 1-2 sentences. Win rates by B. Anything weird.
- Followups: ...
```

---

### 2026-04-30 — run_3b5e0e753407 — smoke_n3_mock (harness validation)

- Backend: mock
- Model: mock-r1
- N per cell: 3
- Wallclock: 0:00:48
- Outcome: 12/12 games completed end-to-end (sweep → DB → JSONL → plots).
  As expected for the deterministic mock, it loses every game to UCT-50
  (win rate 0% at every B). Pass-1 token counts track B closely
  (0 / 64.0 / 247.7 / 1001.5 mean). Confirms the harness, schema,
  analytics, and plots all work without an LLM serving.
- Followups: run real Ollama sweep (`configs/smoke_n3.yaml`) once
  `bootstrap_mac.sh` has installed openjdk@17 + Ollama + deepseek-r1:7b.
