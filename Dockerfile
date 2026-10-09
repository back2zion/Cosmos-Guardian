FROM node:22-bookworm-slim AS dashboard
WORKDIR /build
COPY dashboard/package*.json ./
RUN npm ci
COPY dashboard/ ./
ENV VITE_API_BASE=/api
RUN npm run build

FROM python:3.10-slim-bookworm AS dependencies
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN uv sync --no-cache --frozen --no-dev --no-install-project
RUN groupadd --gid 10001 cosmos && useradd --uid 10001 --gid cosmos --create-home cosmos
RUN mkdir -p data uploads audit && chown -R cosmos:cosmos /app/data /app/uploads /app/audit

FROM dependencies AS inference-dependencies
RUN uv sync --no-cache --frozen --no-dev --no-install-project --extra inference

FROM dependencies AS api
COPY core/ core/
COPY prompts/ prompts/
COPY server.py product_server.py worker.py ./
COPY scripts/product_backup.py scripts/product_backup.py
COPY --from=dashboard /build/dist ./dashboard/dist
USER cosmos
EXPOSE 8888
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8888/api/health',timeout=2)"
CMD ["uvicorn", "product_server:app", "--host", "0.0.0.0", "--port", "8888"]

FROM dependencies AS worker
# Keep the installed runtime, excluding the multi-gigabyte installer cache.
COPY --from=inference-dependencies /app/.venv /app/.venv
COPY core/ core/
COPY prompts/ prompts/
COPY worker.py ./
ENV HF_HUB_CACHE=/app/data/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
USER cosmos
CMD ["python", "worker.py"]
