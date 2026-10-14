# Web UI + API for captchasolver (FastAPI + headless Chromium for the test runner).
# Built by `wrangler deploy` for Cloudflare Containers (see cloudflare/), or locally:
#   docker build -t captchasolver .  &&  docker run --rm -p 8000:8000 captchasolver
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    CAPTCHASOLVER_CONFIG=/tmp/model_config.json

WORKDIR /app

# The browser comes first: it is the big layer and almost never changes, so redeploying
# new app code does not download Chromium again. The later `pip install .` keeps this
# Playwright version (it already satisfies the requirement), so browser and library match.
RUN pip install playwright \
 && playwright install --with-deps chromium \
 && rm -rf /var/lib/apt/lists/* \
 && chmod -R a+rX /ms-playwright

# A real init as PID 1: Chromium leaves orphaned helper processes, and uvicorn does not reap them, so they
# would pile up as zombies (each keeps a slot in the process table) over many runs and restarts.
RUN apt-get update && apt-get install -y --no-install-recommends tini && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[server]"

# Nothing needs root at runtime, and nothing is written outside /tmp.
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "captchasolver.server:app", "--host", "0.0.0.0", "--port", "8000"]
