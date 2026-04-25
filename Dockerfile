ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

WORKDIR /app
RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock* ./
COPY models.py client.py __init__.py openenv.yaml ./
COPY server/ ./server/


RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# HF Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
