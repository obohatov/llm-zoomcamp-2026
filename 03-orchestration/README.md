# Module 3: AI Orchestration with Kestra

Homework solution for [LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp) — Module 3.

Unlike previous modules, this one has no Python code: everything runs in [Kestra](https://kestra.io/) (Docker + YAML flows + UI). Answers to Q3–Q5 are based on actual execution logs from the `log_token_usage` task.

## Setup

Environment: GitHub Codespaces (Docker pre-installed).

Download `docker-compose.yml` and the example flows from the course repo:

```bash
mkdir 03-orchestration && cd 03-orchestration

curl -sO https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-orchestration/docker-compose.yml

mkdir flows && cd flows
for f in 1_chat_without_rag 2_chat_with_rag 3_rag_with_websearch 4_simple_agent 5_web_research_agent 6_multi_agent_research; do
  curl -sO https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-orchestration/flows/$f.yaml
done
cd ..
```

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey) (free tier is enough), then export it and start Kestra. Note the `echo -n` — without it, base64 encodes a trailing newline and Kestra receives an invalid key:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export SECRET_GEMINI_API_KEY=$(echo -n "$GEMINI_API_KEY" | base64)

docker compose up -d
```

Verify the key reached the container:

```bash
docker compose exec kestra env | grep GEMINI
```

Import all flows:

```bash
for f in flows/*.yaml; do
  curl -X POST -u 'admin@kestra.io:Admin1234!' \
    http://localhost:8080/api/v1/flows/import -F fileUpload=@$f
done
```

Open the Kestra UI at `http://localhost:8080` (in Codespaces: **Ports** tab → port 8080). Login: `admin@kestra.io` / `Admin1234!`. All flows live in the `zoomcamp` namespace.

When done:

```bash
docker compose down
```

## Answers

### Q1: Context Engineering

**Answer: AI Copilot has access to current Kestra plugin documentation.**

The same prompt ("Create a Kestra flow that loads NYC taxi data from CSV to BigQuery") produces hallucinated/outdated plugin syntax in ChatGPT, but valid, executable YAML in Kestra's AI Copilot. The difference is not the model, token count, or internet access — it's that Copilot is grounded in the plugin documentation of the running Kestra version.

### Q2: RAG vs No RAG

Ran `1_chat_without_rag` and `2_chat_with_rag` and compared the execution logs.

**Answer: Vague, generic, or fabricated — the model guesses from training data.**

Without RAG, the model confidently listed features that are not part of the Kestra 1.1 release (theme customization, plugin management UI, GCP Secret Manager integration). With RAG (release blog post ingested into the KV Store), the response matched the actual release notes: redesigned filters, no-code dashboard editor, multi-agent AI systems, Fix with AI, Human Task, air-gapped support.

### Q3: Token usage — short summary

Ran `4_simple_agent` with `summary_length = short`. From the `log_token_usage` task:

```
Multilingual Agent:
- Input tokens: 282
- Output tokens: 98
- Total tokens: 380
```

**Answer: 60–100 tokens** (measured: 98).

### Q4: Token usage — long summary

Ran `4_simple_agent` with `summary_length = long`:

```
Multilingual Agent:
- Input tokens: 282
- Output tokens: 191
- Total tokens: 473
```

191 / 98 ≈ 2x. **Answer: 2–5x more.**

Side observation: the multilingual agent's input tokens are identical across runs (282) since only one word in the system message changes, while the `english_brevity` agent's input grows with summary length (113 → 206) because it consumes the first agent's output — costs cascade in agent chains.

### Q5: Modifying a flow

Changed the `english_brevity` prompt from "exactly 1 sentence" to "exactly 3 sentences" (see [`flows/4_simple_agent_3sentences.yaml`](flows/4_simple_agent_3sentences.yaml)) and ran with `summary_length = long`:

| Prompt version | English Brevity output tokens |
|---|---|
| exactly 1 sentence | 47 |
| exactly 3 sentences | 94 |

94 / 47 = 2x. **Answer: 2–4x more.**

### Q6: Best Practices

**Answer: Use traditional task-based workflows for predictability and auditability.**

For deterministic, repeatable processes with strict compliance requirements (e.g., financial reporting), agents are the wrong tool by definition: their execution path is non-deterministic and not exactly reproducible. Traditional task-based flows are predictable, auditable, and cheaper.

## Notes

- The `docker-compose.yml` expects two env vars: `GEMINI_API_KEY` (used by the AI Copilot configuration) and `SECRET_GEMINI_API_KEY` (base64-encoded, used by `{{ secret('GEMINI_API_KEY') }}` in flows).
- API keys are never committed — they only exist as environment variables in the Codespaces session.
- Gemini free tier rate limits are low; on `429 Resource Exhausted`, wait a minute and retry.
- LLM outputs are non-deterministic, so token counts vary slightly between runs; the homework says to pick the closest option.
