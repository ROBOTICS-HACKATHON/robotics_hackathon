import httpx
import pytest

from captchasolver.client import WorkersAI, WorkersAIError, describe_exc

OK = {"success": True, "result": {"response": "yes"}, "errors": [], "messages": []}


def make_ai(handler, **kw):
    return WorkersAI("acct", "tok", transport=httpx.MockTransport(handler), backoff=0, **kw)


def test_describe_exc_survives_empty_messages():
    # these have str(e) == "" and used to crash `str(e).splitlines()[0]`
    assert describe_exc(httpx.ReadTimeout("")) == "ReadTimeout"
    assert describe_exc(httpx.ReadTimeout("")) and describe_exc(RuntimeError()) == "RuntimeError"
    assert describe_exc(ValueError("first line\nsecond")) == "ValueError: first line"
    assert len(describe_exc(ValueError("x" * 500), limit=50)) <= len("ValueError: ") + 50


async def test_timeouts_are_retried_then_succeed():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ReadTimeout("", request=request)
        return httpx.Response(200, json=OK)
    ai = make_ai(handler)
    assert await ai.run("@cf/x/y", json={}) == {"response": "yes"}
    assert len(calls) == 3
    await ai.aclose()


async def test_persistent_timeout_raises_a_readable_error_not_a_bare_one():
    def handler(request):
        raise httpx.ReadTimeout("", request=request)
    ai = make_ai(handler)
    with pytest.raises(WorkersAIError, match=r"no response after 3 attempts \(ReadTimeout\)"):
        await ai.run("@cf/x/y", json={})
    await ai.aclose()


async def test_connection_errors_also_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("", request=request)
        return httpx.Response(200, json=OK)
    ai = make_ai(handler)
    assert (await ai.run("@cf/x/y", json={}))["response"] == "yes"
    await ai.aclose()


async def test_5xx_still_retried_and_4xx_not():
    seen = []

    def handler(request):
        seen.append(1)
        return httpx.Response(503 if len(seen) == 1 else 200, json=OK)
    ai = make_ai(handler)
    await ai.run("@cf/x/y", json={})
    assert len(seen) == 2
    seen.clear()
    ai2 = make_ai(lambda r: (seen.append(1), httpx.Response(403, json={"success": False, "errors": ["license"]}))[1])
    with pytest.raises(WorkersAIError, match="403"):
        await ai2.run("@cf/x/y", json={})
    assert len(seen) == 1
    await ai.aclose(); await ai2.aclose()
