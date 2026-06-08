# anthropic-flask-middleware

A small Flask middleware wrapper for building secure, validated Anthropic API endpoints.

This project gives developers a practical starting point for adding Anthropic-backed streaming and structured JSON extraction to a Flask service without hardcoded secrets, paid infrastructure, or live API calls in tests.

## What It Solves

Many prototypes call the Anthropic API directly from route handlers and leave validation, retry behavior, authentication, and error formatting as afterthoughts. This repository wraps those basics into a focused Flask service:

- Local API key protection with `X-API-Key`
- Environment-only Anthropic credentials
- Pydantic request and response validation
- Streaming text responses
- Structured task extraction with automatic correction attempts
- Tenacity retry handling around Anthropic message calls
- Optional MySQL token usage logging through DBUtils connection pooling
- Tests that use mocks instead of spending API credits

## Architecture Overview

`anthropic_service.py` contains the Flask app factory, route definitions, Pydantic models, Anthropic wrapper, retry logic, structured JSON parsing, and consistent API error responses.

The `/stream` endpoint accepts a prompt and streams text from Anthropic. The `/structured` endpoint asks Anthropic for JSON matching `TaskExtractionResult`, validates it with Pydantic, and retries correction up to two times if the model returns invalid JSON or invalid fields.

If MySQL environment variables are configured, the service initializes a `llm_usage_log` table and records token usage, latency, estimated cost, and correction attempts. If MySQL is not configured, logging is skipped and the API still runs normally.

## Tech Stack

- Python
- Flask
- Anthropic Python SDK
- Pydantic
- Tenacity
- python-dotenv
- PyMySQL
- DBUtils
- pytest

## Repository Structure

```text
anthropic-flask-middleware/
|-- anthropic_service.py
|-- README.md
|-- requirements.txt
|-- .env.example
|-- .gitignore
`-- tests/
    `-- test_structured.py
```

## Features

- `GET /health` public health check
- `POST /stream` protected streaming endpoint
- `POST /structured` protected structured extraction endpoint
- `APP_API_KEY` based local authentication
- `ANTHROPIC_API_KEY` loaded only from environment variables
- Clean JSON error responses
- Exponential backoff retry logic for Anthropic message calls
- Optional MySQL usage logging for live calls
- Mocked pytest coverage for local development

## Setup

Create and activate a virtual environment.

For Git Bash on Windows:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

For PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Set these values:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
APP_API_KEY=your_local_app_api_key_here
ANTHROPIC_MODEL=claude-3-5-haiku-latest
FLASK_ENV=development
PORT=5000
LOG_LEVEL=INFO
MYSQL_HOST=
MYSQL_PORT=3306
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_DATABASE=
```

`.env` is ignored by Git. Do not commit real secrets.

The MySQL variables are optional. Leave them blank for a zero-infrastructure local setup.

## Run Locally

```bash
python anthropic_service.py
```

The service defaults to port `5000`.

## Test

```bash
pytest
```

The tests do not call the real Anthropic API. They use mocked services and mocked Anthropic message responses, so local validation and endpoint behavior can be tested without spending credits.

Only live `/stream` and `/structured` manual testing requires a valid `ANTHROPIC_API_KEY`.

## API Examples

### Health

```bash
curl http://localhost:5000/health
```

Response:

```json
{
  "status": "ok",
  "service": "anthropic-flask-middleware"
}
```

### Stream

```bash
curl -N -X POST http://localhost:5000/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_local_app_api_key_here" \
  -d '{"prompt":"Write a short product description"}'
```

### Structured

```bash
curl -X POST http://localhost:5000/structured \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_local_app_api_key_here" \
  -d '{"prompt":"Create a high priority task to ship API docs by Friday"}'
```

Example response:

```json
{
  "result": {
    "title": "Ship API docs",
    "priority": "high",
    "due_date": "Friday",
    "tags": ["api", "docs"],
    "summary": "Create and ship API documentation by Friday."
  },
  "meta": {
    "session_id": "generated-session-id",
    "input_tokens": 123,
    "output_tokens": 45,
    "latency_ms": 850,
    "correction_attempts": 0
  }
}
```

## Error Format

All API errors use this shape:

```json
{
  "error": true,
  "message": "Human-readable message.",
  "code": "machine_readable_code",
  "details": {}
}
```

Handled cases include missing API keys, invalid API keys, invalid JSON bodies, missing prompts, Anthropic API failures, structured validation failures, and unknown server errors.

## Security Notes

- `ANTHROPIC_API_KEY` must come from the environment.
- `APP_API_KEY` must come from the environment.
- `/health` is public.
- `/stream` and `/structured` require `X-API-Key`.
- No real secrets are included in this repository.
- `.env` is intentionally ignored by Git.

## Free-Tier Testing Notes

This project is designed for a $0 local development workflow. Unit tests do not call Anthropic and do not require paid infrastructure. You can verify Flask routing, API-key behavior, Pydantic validation, and structured correction logic locally with `pytest`.

Live Anthropic calls are only needed when manually testing `/stream` or `/structured` with a real `ANTHROPIC_API_KEY`.

## Technical Merit Inbound Strategy

This repository is intentionally small, inspectable, and useful. It demonstrates practical backend engineering choices: clear validation boundaries, environment-based configuration, safe error responses, retry handling, testability without external spend, and a simple API surface that other Flask developers can adapt.

The project should earn attention through code quality and usefulness rather than paid promotion, fake reviews, or platform dependency.

## GitHub Push Commands

Preferred repository URL:

```text
https://github.com/gharisj3/anthropic-flask-middleware.git
```

Manual Git setup:

```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/gharisj3/anthropic-flask-middleware.git
git push -u origin main
```

If GitHub CLI is installed and authenticated:

```bash
gh repo create gharisj3/anthropic-flask-middleware --public --source=. --remote=origin --push
```

If authentication is required, complete `gh auth login` or create the repository manually on GitHub before pushing.

## License

Add the license that matches your intended reuse policy before publishing. MIT is a common choice for small developer utilities.
