"""Minimal async client for the Cloudflare Workers AI REST API."""
from __future__ import annotations

import asyncio
import json as _json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

API = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"
RETRY_STATUS = {429, 500, 502, 503, 504}


class WorkersAIError(RuntimeError):
    pass


def describe_exc(e: BaseException, limit: int = 160) -> str:
    """One-line 'Type: message' that never fails, even for exceptions with an empty message
    (httpx.ReadTimeout and friends), which used to crash the error handler itself."""
    msg = (str(e).strip().splitlines() or [""])[0][:limit]
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


class WorkersAI:
    """Reads CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN unless given explicitly."""

    def __init__(self, account_id: str | None = None, api_token: str | None = None,
                 timeout: float = 90.0, retries: int = 2, concurrency: int = 8,
                 backoff: float = 0.5, transport: httpx.AsyncBaseTransport | None = None):
        self.account_id = account_id or os.environ["CLOUDFLARE_ACCOUNT_ID"]
        token = api_token or os.environ["CLOUDFLARE_API_TOKEN"]
        self.retries = retries
        self.backoff = backoff
        self._sem = asyncio.Semaphore(concurrency)
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"}, timeout=timeout, transport=transport
        )

    async def run(self, model: str, *, json: dict | None = None,
                  image: bytes | None = None) -> Any:
        """POST to a model. `json` for LLMs, `image` (raw bytes) for vision models."""
        async with self._sem:
            return await self._run(model, json, image)

    async def _run(self, model: str, json: dict | None, image: bytes | None) -> Any:
        url = API.format(account=self.account_id, model=model)
        for attempt in range(self.retries + 1):
            try:
                if image is not None:
                    r = await self._http.post(
                        url, content=image,
                        headers={"Content-Type": "application/octet-stream"})
                else:
                    r = await self._http.post(url, json=json)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                # slow or dropped connection: retry, then fail with a message that says what happened
                if attempt < self.retries:
                    await asyncio.sleep(self.backoff * 2**attempt)
                    continue
                raise WorkersAIError(
                    f"{model}: no response after {attempt + 1} attempts ({describe_exc(e)})") from e
            if r.status_code in RETRY_STATUS and attempt < self.retries:
                await asyncio.sleep(self.backoff * 2**attempt)
                continue
            break
        body = r.json()
        if r.status_code >= 400 or not body.get("success", False):
            raise WorkersAIError(f"{model}: {r.status_code} {body.get('errors')}")
        if not body.get("result"):
            raise WorkersAIError(f"{model}: empty result, full response body: {r.text[:800]}")
        return body["result"]

    async def raw(self, model: str, payload: dict) -> tuple[int, str]:
        """One un-retried POST; returns (status, body text). For diagnostics."""
        r = await self._http.post(API.format(account=self.account_id, model=model),
                                  json=payload)
        return r.status_code, r.text

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "WorkersAI":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()


def data_uri(image: bytes) -> str:
    import base64
    mime = "image/jpeg" if image[:3] == b"\xff\xd8\xff" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(image).decode()}"


def find_answer(result: Any) -> str:
    """Pull the text answer out of a Moondream result, whatever wrapper it comes in.

    Cloudflare's docs don't show an example payload, so accept a bare string,
    {"answer": ..}, a task-keyed wrapper like {"query": {"answer": ..}}, or an
    OpenAI-style {"response": ..} / choices list. Fail loudly with the raw payload.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("answer", "response", "caption", "text"):
            if isinstance(result.get(key), str):
                return result[key]
        for v in result.values():
            if isinstance(v, (dict, list)):
                try:
                    return find_answer(v)
                except ValueError:
                    pass
    if isinstance(result, list):
        for v in result:
            try:
                return find_answer(v)
            except ValueError:
                pass
        if result and isinstance(result[0], dict) and "message" in result[0]:
            content = result[0]["message"].get("content")
            if isinstance(content, str):
                return content
    raise ValueError(f"no text answer in Workers AI result: {result!r}")


VLM_MODEL = "@cf/google/gemma-4-26b-a4b-it"

# Models that "think" before answering: a tiny max_tokens would be eaten by the reasoning
# and leave an empty answer, so give them room.
REASONING_HINTS = ("gemma-4", "kimi", "glm", "qwen")
REASONING_MIN_TOKENS = 1024


@dataclass
class ModelConfig:
    """Per-model request quirks, discovered by `captchasolver compare`'s preflight."""
    extra: dict = field(default_factory=dict)   # merged into the request body
    send_temperature: bool = True               # some models reject temperature


MODEL_CONFIG: dict[str, ModelConfig] = {}

# What `captchasolver compare`/preflight learned about each model survives between runs here.
CONFIG_FILE = Path(os.environ.get("CAPTCHASOLVER_CONFIG",
                                  Path.home() / ".cache" / "captchasolver" / "model_config.json"))


def load_model_config(path: Path | None = None) -> None:
    try:
        raw = _json.loads((path or CONFIG_FILE).read_text())
        MODEL_CONFIG.update({m: ModelConfig(c.get("extra", {}), c.get("send_temperature", True))
                             for m, c in raw.items()})
    except (OSError, ValueError, AttributeError):
        pass  # no cache yet, or unreadable: fall back to plain requests


def save_model_config(path: Path | None = None) -> Path:
    p = path or CONFIG_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps({m: {"extra": c.extra, "send_temperature": c.send_temperature}
                              for m, c in MODEL_CONFIG.items()}, indent=2) + "\n")
    return p


load_model_config()


def chat_payload(model: str, image: bytes, question: str, *, max_tokens: int,
                 temperature: float) -> dict:
    cfg = MODEL_CONFIG.get(model, ModelConfig())
    if any(h in model for h in REASONING_HINTS):
        max_tokens = max(max_tokens, REASONING_MIN_TOKENS)
    payload: dict = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": data_uri(image)}},
    ]}], "max_tokens": max_tokens}
    if cfg.send_temperature:
        payload["temperature"] = temperature
    payload.update(cfg.extra)
    return payload


async def vlm_query(ai: WorkersAI, image: bytes, question: str, *,
                    model: str = VLM_MODEL, max_tokens: int = 32,
                    temperature: float = 0) -> str:
    """Ask a vision-language model one question about one image; return its text.

    Moondream uses its own {task, image, question} schema; every other model is
    assumed to speak chat messages with an image_url part.
    """
    if "moondream" in model:
        payload = {"task": "query", "image": data_uri(image), "question": question,
                   "reasoning": False, "temperature": temperature,
                   "max_tokens": max_tokens}
    else:
        payload = chat_payload(model, image, question, max_tokens=max_tokens,
                               temperature=temperature)
    return find_answer(await ai.run(model, json=payload))
