FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml ./
COPY aws_cost_env ./aws_cost_env
COPY server ./server
COPY inference.py ./

RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=7860
ENV WORKERS=2
ENV MAX_CONCURRENT_ENVS=100

EXPOSE 7860

CMD ["python", "-m", "server.run"]
