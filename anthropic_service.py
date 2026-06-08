import json
import logging
import os
import time
import uuid
from typing import Any, Generator, Optional, Type, TypeVar

import anthropic
import pymysql
from dbutils.pooled_db import PooledDB
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, stream_with_context
from pydantic import BaseModel, Field, ValidationError, field_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)


load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

SERVICE_NAME = "anthropic-flask-middleware"
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
MAX_CORRECTION_ATTEMPTS = 2


class APIError(Exception):
    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system: Optional[str] = None
    session_id: Optional[str] = None

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt must not be blank")
        return stripped


class TaskExtractionResult(BaseModel):
    title: str = Field(..., min_length=1)
    priority: str = Field(..., pattern="^(low|medium|high)$")
    due_date: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    summary: str = Field(..., min_length=1)


class UsageLogger:
    def __init__(self) -> None:
        self.pool: Optional[PooledDB] = None

    def enabled(self) -> bool:
        required = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
        return all(os.getenv(name) for name in required)

    def get_pool(self) -> Optional[PooledDB]:
        if not self.enabled():
            return None
        if self.pool is None:
            self.pool = PooledDB(
                creator=pymysql,
                maxconnections=10,
                mincached=1,
                host=os.environ["MYSQL_HOST"],
                port=int(os.environ.get("MYSQL_PORT", "3306")),
                user=os.environ["MYSQL_USER"],
                password=os.environ["MYSQL_PASSWORD"],
                database=os.environ["MYSQL_DATABASE"],
                charset="utf8mb4",
                autocommit=True,
            )
        return self.pool

    def init_db(self) -> None:
        pool = self.get_pool()
        if pool is None:
            logger.info("MySQL usage logging is disabled because MySQL env vars are not configured.")
            return

        ddl = """
        CREATE TABLE IF NOT EXISTS llm_usage_log (
            id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            session_id CHAR(36) NOT NULL,
            model VARCHAR(64) NOT NULL,
            input_tokens INT UNSIGNED NOT NULL DEFAULT 0,
            output_tokens INT UNSIGNED NOT NULL DEFAULT 0,
            cost_usd DECIMAL(10, 6) NOT NULL DEFAULT 0.000000,
            latency_ms INT UNSIGNED NOT NULL DEFAULT 0,
            correction_attempts TINYINT UNSIGNED NOT NULL DEFAULT 0,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session (session_id),
            INDEX idx_created (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        conn = pool.connection()
        try:
            with conn.cursor() as cur:
                cur.execute(ddl)
        finally:
            conn.close()

    def log_usage(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        correction_attempts: int = 0,
        cost_per_million_input: float = 3.0,
        cost_per_million_output: float = 15.0,
    ) -> None:
        pool = self.get_pool()
        if pool is None:
            return

        cost = (
            input_tokens / 1_000_000 * cost_per_million_input
            + output_tokens / 1_000_000 * cost_per_million_output
        )
        sql = (
            "INSERT INTO llm_usage_log "
            "(session_id, model, input_tokens, output_tokens, cost_usd, latency_ms, correction_attempts) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        conn = pool.connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        session_id,
                        model,
                        input_tokens,
                        output_tokens,
                        cost,
                        latency_ms,
                        correction_attempts,
                    ),
                )
        except Exception as exc:
            logger.error("Usage log write failed: %s", exc)
        finally:
            conn.close()


usage_logger = UsageLogger()


def error_response(
    message: str,
    code: str,
    status_code: int,
    details: Optional[dict[str, Any]] = None,
) -> tuple[Response, int]:
    return (
        jsonify(
            {
                "error": True,
                "message": message,
                "code": code,
                "details": details or {},
            }
        ),
        status_code,
    )


def validate_json_body() -> PromptRequest:
    body = request.get_json(silent=True)
    if body is None:
        raise APIError("Request body must be valid JSON.", "invalid_json", 400)
    try:
        return PromptRequest.model_validate(body)
    except ValidationError as exc:
        raise APIError(
            "Request body must include a non-empty prompt.",
            "invalid_request",
            400,
            {"errors": exc.errors()},
        ) from exc


def require_api_key() -> None:
    expected_key = os.getenv("APP_API_KEY")
    provided_key = request.headers.get("X-API-Key")
    if not provided_key:
        raise APIError("Missing X-API-Key header.", "missing_api_key", 401)
    if not expected_key:
        raise APIError("APP_API_KEY is not configured.", "app_api_key_not_configured", 500)
    if provided_key != expected_key:
        raise APIError("Invalid X-API-Key header.", "invalid_api_key", 403)


class AnthropicService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[anthropic.Anthropic] = None,
        logger_backend: UsageLogger = usage_logger,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or DEFAULT_MODEL
        self.client = client
        self.usage_logger = logger_backend

    def get_client(self) -> anthropic.Anthropic:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise APIError(
                "ANTHROPIC_API_KEY is not configured.",
                "anthropic_api_key_not_configured",
                500,
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        return self.client

    @retry(
        retry=retry_if_exception_type(
            (
                anthropic.RateLimitError,
                anthropic.APIStatusError,
                anthropic.APIConnectionError,
            )
        ),
        wait=wait_exponential_jitter(initial=1, max=16, jitter=1),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def call_claude(
        self,
        messages: list[dict[str, str]],
        system: str,
        max_tokens: int = 1024,
    ) -> anthropic.types.Message:
        return self.get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )

    def call_structured(
        self,
        prompt: str,
        schema: Type[T],
        system: str = "",
        max_correction_attempts: int = MAX_CORRECTION_ATTEMPTS,
        session_id: Optional[str] = None,
    ) -> tuple[T, dict[str, Any]]:
        if session_id is None:
            session_id = str(uuid.uuid4())

        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        base_system = (
            f"{system}\n\n"
            "You MUST respond with valid JSON only. No markdown fences, no explanation. "
            f"The JSON must conform exactly to this schema:\n{schema_json}"
        )

        messages = [{"role": "user", "content": prompt}]
        total_input = 0
        total_output = 0
        correction_attempts = 0
        started_at = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt in range(max_correction_attempts + 1):
            response = self.call_claude(messages, system=base_system)
            raw_text = response.content[0].text.strip()
            usage = getattr(response, "usage", None)
            total_input += int(getattr(usage, "input_tokens", 0) or 0)
            total_output += int(getattr(usage, "output_tokens", 0) or 0)

            try:
                parsed = schema.model_validate_json(raw_text)
                latency_ms = int((time.monotonic() - started_at) * 1000)
                self.usage_logger.log_usage(
                    session_id,
                    self.model,
                    total_input,
                    total_output,
                    latency_ms,
                    correction_attempts,
                )
                return parsed, {
                    "session_id": session_id,
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "latency_ms": latency_ms,
                    "correction_attempts": correction_attempts,
                }
            except (ValidationError, ValueError) as exc:
                last_error = exc
                if attempt >= max_correction_attempts:
                    break

                correction_attempts += 1
                if isinstance(exc, ValidationError):
                    error_summary = "; ".join(
                        f"{error['loc']}: {error['msg']}" for error in exc.errors()
                    )
                else:
                    error_summary = str(exc)

                messages += [
                    {"role": "assistant", "content": raw_text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response failed schema validation with these errors: "
                            f"{error_summary}. Correct your JSON output to fix all errors. "
                            "Return only the corrected JSON."
                        ),
                    },
                ]
                logger.warning(
                    "Correction attempt %d/%d for session %s: %s",
                    correction_attempts,
                    max_correction_attempts,
                    session_id,
                    error_summary,
                )

        raise APIError(
            "Anthropic response could not be converted into the required structured schema.",
            "structured_validation_failed",
            422,
            {"validation_error": str(last_error) if last_error else "unknown"},
        )

    def stream_claude(
        self,
        messages: list[dict[str, str]],
        system: str,
    ) -> Generator[str, None, None]:
        started_at = time.monotonic()
        session_id = str(uuid.uuid4())

        try:
            with self.get_client().messages.stream(
                model=self.model,
                max_tokens=2048,
                system=system,
                messages=messages,
            ) as stream:
                for text_chunk in stream.text_stream:
                    yield f"data: {json.dumps({'delta': text_chunk})}\n\n"

                final = stream.get_final_message()
                usage = getattr(final, "usage", None)
                input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                latency_ms = int((time.monotonic() - started_at) * 1000)
                self.usage_logger.log_usage(
                    session_id,
                    self.model,
                    input_tokens,
                    output_tokens,
                    latency_ms,
                )
                done_payload = {
                    "event": "done",
                    "usage": {
                        "session_id": session_id,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency_ms": latency_ms,
                    },
                }
                yield f"data: {json.dumps(done_payload)}\n\n"
        except APIError:
            raise
        except Exception as exc:
            logger.error("Anthropic streaming request failed: %s", exc)
            payload = {
                "error": True,
                "message": "Anthropic streaming request failed.",
                "code": "anthropic_stream_error",
                "details": {},
            }
            yield f"data: {json.dumps(payload)}\n\n"


def create_app(service: Optional[AnthropicService] = None) -> Flask:
    app = Flask(__name__)
    app.config["SERVICE"] = service or AnthropicService()

    @app.errorhandler(APIError)
    def handle_api_error(exc: APIError) -> tuple[Response, int]:
        return error_response(exc.message, exc.code, exc.status_code, exc.details)

    @app.errorhandler(Exception)
    def handle_unknown_error(exc: Exception) -> tuple[Response, int]:
        logger.exception("Unhandled server error: %s", exc)
        return error_response("Unexpected server error.", "server_error", 500)

    @app.get("/health")
    def health() -> Response:
        return jsonify({"status": "ok", "service": SERVICE_NAME})

    @app.post("/stream")
    def stream_endpoint() -> Response:
        require_api_key()
        body = validate_json_body()
        system = body.system or "You are a helpful assistant."
        messages = [{"role": "user", "content": body.prompt}]
        service_instance: AnthropicService = app.config["SERVICE"]
        return Response(
            stream_with_context(service_instance.stream_claude(messages, system)),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/structured")
    def structured_endpoint() -> Response:
        require_api_key()
        body = validate_json_body()
        service_instance: AnthropicService = app.config["SERVICE"]
        result, meta = service_instance.call_structured(
            prompt=body.prompt,
            schema=TaskExtractionResult,
            system=body.system or "Extract a task from the user's prompt.",
            session_id=body.session_id,
        )
        return jsonify({"result": result.model_dump(), "meta": meta})

    return app


app = create_app()


if __name__ == "__main__":
    usage_logger.init_db()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
