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
    """A failed Workers AI call. Keeps Cloudflare's own status and error list so callers can tell a bad
    token from a missing licence from a used-up allowance (see `explain_error`)."""

    def __init__(self, message: str, status: int | None = None, errors: list | None = None):
        super().__init__(message)
        self.status = status
        self.errors = errors or []


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
            raise WorkersAIError(f"{model}: {r.status_code} {body.get('errors')}", r.status_code, body.get("errors"))
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


def explain_error(e: BaseException) -> tuple[str, str]:
    """(kind, plain-English explanation) for a Workers AI failure.

    kind: auth | forbidden | licence | quota | other. 401 and 403 are NOT the same thing: Cloudflare also
    answers 403 for a model whose licence you have not accepted, and a used-up allowance is not a
    credentials problem at all."""
    status = getattr(e, "status", None)
    errors = [x for x in (getattr(e, "errors", None) or []) if isinstance(x, dict)]
    codes = {x.get("code") for x in errors}
    said = "; ".join(str(x.get("message", "")) for x in errors) or str(e)
    low = said.lower()
    if 5016 in codes or "model agreement" in low:
        return "licence", ("This model needs its licence accepted once on your Cloudflare account, or pick a model "
                           f"that does not. Cloudflare says: {said[:300]}")
    if status == 429 or "daily free allocation" in low or "neurons" in low:
        return "quota", ("Cloudflare is rate limiting you or your Workers AI allowance is used up. This is not a "
                         f"credentials problem; wait or raise the plan's limit. Cloudflare says: {said[:300]}")
    if status == 401 or 10000 in codes or "authentication error" in low:
        return "auth", "Cloudflare did not accept the API token for this account."
    if status == 403:
        return "forbidden", ("The token is recognised but not allowed to do this: it needs the Workers AI permission, "
                             f"and the account ID must be the one it belongs to. Cloudflare says: {said[:300]}")
    return "other", describe_exc(e, 300)


async def token_status(ai: "WorkersAI") -> str:
    """Ask Cloudflare about the token itself: active | expired | disabled | unknown (not recognised).

    Tries the user-token and the account-token check, since either kind may be in use."""
    try:
        for path in ("/user/tokens/verify", f"/accounts/{ai.account_id}/tokens/verify"):
            try:
                r = await ai._http.get(f"https://api.cloudflare.com/client/v4{path}")
                body = r.json()
            except Exception:  # noqa: BLE001 - this is a best-effort extra opinion
                continue
            if body.get("success") and isinstance(body.get("result"), dict):
                return str(body["result"].get("status", "unknown"))
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


async def explain_auth_failure(ai: "WorkersAI", e: BaseException) -> tuple[str, str]:
    """`explain_error`, and for a rejected token also ask Cloudflare WHY (expired? revoked? wrong account?)."""
    kind, text = explain_error(e)
    if kind != "auth":
        return kind, text
    status = await token_status(ai)
    if status == "expired":
        return "auth", "The API token has expired. Create a new one in the Cloudflare dashboard and save it in Settings."
    if status == "disabled":
        return "auth", "The API token is disabled or revoked. Create a new one and save it in Settings."
    if status == "active":
        return "auth", ("The token itself is valid, but Cloudflare did not accept it for THIS account. Check that the "
                        "account ID in Settings is the account the token was created for, and that the token has the "
                        "Workers AI permission.")
    return "auth", ("Cloudflare does not recognise this token (a typo, a different Cloudflare account's token, or it "
                    "was deleted/rolled). Create a new one and save it in Settings.")
