# syntax=docker/dockerfile:1

# ---- builder -----------------------------------------------------------------
# Dependencies are installed in a layer keyed on uv.lock alone, so editing source
# does not re-download torch.
FROM python:3.13-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_FROZEN=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ---- runtime -----------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

# Non-root. The process only ever reads its own code and talks to Postgres.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Keep the sentence-transformers cache inside the writable app dir; the
    # default location is unwritable for a non-root user.
    HF_HOME=/app/.cache/huggingface

RUN mkdir -p /app/.cache/huggingface && chown -R app:app /app/.cache

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "kontrakt_guard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
