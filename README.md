# StoreOps Triage API

A production-structured **Python + FastAPI** service that triages retail store operations incident reports using structured LLM classification, ambiguity/human-review routing, and parameterized SQLite persistence.

## Architecture

- **FastAPI (`app/main.py`)**: Async HTTP endpoints with dependency injection (`Depends`) for the classifier and database path.
- **Pydantic v2 (`app/schemas.py`)**: Strict request validation (`IncidentCreate`) and typed classification contracts (`ClassificationResult`, `IncidentTriageResponse`).
- **Classifier Layer (`app/classifier.py`)**:
  - `OpenAICompatibleClassifier`: Async `httpx` client supporting any OpenAI-compatible LLM endpoint with strict JSON schema validation, timeout handling (`504`), malformed-output protection (`502`), and confidence-threshold human-review routing.
  - `RuleBasedClassifier`: Deterministic offline classifier used when `LLM_API_KEY` is unset and in automated tests.
- **SQLite Persistence (`app/db.py`)**: Parameterized SQL queries (`INSERT` / `SELECT`) ensuring failed classifications never persist partial records.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/incidents/triage` | Validate, classify, and persist an incident report (`201 Created`) |
| `GET` | `/incidents/{id}` | Retrieve a stored incident by ID (`404` if missing) |
| `GET` | `/incidents?needs_review=true` | List stored incidents with optional human-review filter |

## Quickstart

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run API server
uvicorn app.main:app --reload
```

Open interactive Swagger docs at `http://127.0.0.1:8000/docs`.

### Optional Live LLM Configuration

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"
export LLM_TIMEOUT_S="10.0"
```

## Running Tests & Evaluation

```bash
pytest -v
python -m eval.run_eval
```

## Sample Request

```bash
curl -X POST http://127.0.0.1:8000/incidents/triage \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": "DOM-DEL-1042",
    "title": "POS billing terminals down",
    "description": "All 3 POS counters stopped accepting UPI payments during dinner rush."
  }'
```
