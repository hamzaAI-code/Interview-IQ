# Technical AI Interview Agent

A technical-only interview agent. Candidate uploads **Resume + JD** → the system builds a **Neo4j knowledge graph** for each, ranks topics by overlap and JD priority, then asks technical questions using **Gemini**. Asks follow-ups when the answer is shallow or contradicts the resume; advances when current topic coverage ≥ 70%; emits a structured final evaluation.

## Stack

- Python 3.11+
- Gemini (`google-genai`) — `gemini-2.5-flash` for fast path, `gemini-2.5-pro` for extraction + evaluation
- Neo4j 5 (async driver)
- Streamlit UI

## Quick start

```bash
# 1. Neo4j
docker compose up -d neo4j

# 2. Python env
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Env vars
cp .env.example .env
# edit .env and set GOOGLE_API_KEY

# 4. Run
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 (UI) and http://localhost:7474 (Neo4j Browser, user `neo4j` / pw from `.env`).

## Flow

1. Upload Resume (PDF/DOCX/TXT) + paste JD.
2. App extracts entities concurrently (Gemini structured output) → ingests into Neo4j → derives ranked `:Topic` nodes from JD requirements ∩ resume claims.
3. Interview begins. For each topic: question → answer → analyze (score, depth, contradictions) → follow-up if needed → advance when coverage ≥ 0.70.
4. After all `must_have` topics covered (or session ended), final evaluation report is generated and shown.

## Project layout

```
app/
  streamlit_app.py        # UI entry
  config.py               # env-driven settings
  core/                   # interview state machine + context manager
  graph/                  # Neo4j client, schema, builder, matcher
  llm/                    # Gemini client + extractors + question/analysis/eval
  services/               # resume + JD parsers
  prompts/                # versioned prompt strings
  utils/                  # logger, parallel helpers
```

## Inspect the graph

In Neo4j Browser:

```cypher
// All entities for a session
MATCH (n) WHERE n.session_id = $sid RETURN n;

// Ranked topics
MATCH (s:Session {id:$sid})-[:HAS_TOPIC]->(t:Topic)
RETURN t.name, t.importance ORDER BY t.importance DESC;

// Interview audit trail
MATCH (t:Topic {session_id:$sid})-[r:ASKED]->(:Session {id:$sid})
RETURN t.name, r.question, r.answer, r.score ORDER BY r.ts;
```
