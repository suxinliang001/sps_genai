# sps_genai

A FastAPI app with a bigram text generation endpoint and a spaCy word embedding endpoint.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check, returns `{"Hello": "World"}` |
| POST | `/generate` | Generates text from a start word using a bigram model |
| POST | `/embedding` | Returns the spaCy (`en_core_web_md`, 300-dim) embedding for a word |

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

The API is then available at http://127.0.0.1:8000 and the interactive docs at http://127.0.0.1:8000/docs.

## Example requests

Bigram text generation:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_word": "bigram", "length": 5}'
```

Word embedding:

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "king"}'
```

Response (embedding truncated):

```json
{"word": "king", "dimension": 300, "embedding": [-0.606, -0.512, ...]}
```

## Run locally without Docker

```bash
uv sync
uv run fastapi dev app/main.py
```
