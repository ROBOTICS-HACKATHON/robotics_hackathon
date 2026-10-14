// What we pass on to the container. Pure, so it can be tested without the Workers runtime.

// Never forward Access material or cookies to the app; it only needs the Cloudflare credentials
// headers (X-CF-Account-Id / X-CF-Api-Token) the user's browser or script sends.
const STRIP = ["cookie", "cf-access-jwt-assertion", "cf-access-client-id", "cf-access-client-secret", "x-auth-identity", "x-client-ip"];

export function forwardHeaders(incoming: Headers, identity: string): Headers {
  const out = new Headers(incoming);
  for (const name of STRIP) out.delete(name); // also removes a spoofed X-Auth-Identity
  out.set("x-auth-identity", identity);
  // The app rate limits per visitor. Only the edge-set header can be trusted, never a client-sent one.
  const ip = incoming.get("cf-connecting-ip");
  if (ip) out.set("x-client-ip", ip);
  return out;
}

export function jsonError(status: number, detail: string): Response {
  return new Response(JSON.stringify({ detail }), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}

const STARTING_TEXT =
  "The solver is starting up. The first start after a deploy or a long idle period can take a few minutes.";

function startingPage(message: string): Response {
  const esc = message.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
  const html = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="10"><title>Starting… · Captcha Solver</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:34rem;margin:15vh auto;padding:0 1rem;color:#1b1f27}
@media(prefers-color-scheme:dark){body{background:#12151a;color:#e8eaed}}</style>
<h1>Starting the solver…</h1><p>${esc}</p><p>This page refreshes by itself every 10 seconds.</p>`;
  return new Response(html, { status: 503, headers: { "content-type": "text/html; charset=utf-8", "retry-after": "10", "cache-control": "no-store" } });
}

/**
 * @cloudflare/containers does not throw when it cannot start the container: it *returns* a plain-text
 * 503/429/500 Response. Turn those into something a person (or a script) can act on. Responses from
 * the app itself (JSON, HTML, FastAPI's own errors) return null and are passed through untouched.
 */
export async function friendlyContainerError(resp: Response, accept: string): Promise<Response | null> {
  if (![429, 500, 503].includes(resp.status)) return null;
  if (!(resp.headers.get("content-type") ?? "").startsWith("text/plain")) return null;
  const text = (await resp.clone().text()).trim();
  let status: number, message: string;
  if (text.startsWith("There is no Container instance available")) {
    status = 503; message = STARTING_TEXT;
  } else if (text.startsWith("Failed to start container:")) {
    status = 503; message = `${STARTING_TEXT} (${text.slice(0, 300)})`;
  } else if (resp.status === 429 && text.length > 0 && text.length < 300 && /rate/i.test(text)) {
    status = 429; message = "Cloudflare is rate limiting container starts. Try again shortly.";
  } else {
    return null;
  }
  if (accept.includes("text/html")) return startingPage(message);
  const r = jsonError(status, message);
  r.headers.set("retry-after", "10");
  return r;
}

/** Environment for the container, from the Worker's vars. Rate limits are on unless explicitly "off". */
export function containerEnv(vars: { ALLOWED_TEST_HOSTS?: string; RATE_LIMITS?: string }): Record<string, string> {
  return {
    CAPTCHASOLVER_ALLOWED_HOSTS: vars.ALLOWED_TEST_HOSTS?.trim() || "skibidisigma67rizz.pages.dev",
    CAPTCHASOLVER_RATE_LIMITS: vars.RATE_LIMITS?.trim().toLowerCase() === "off" ? "off" : "on",
  };
}
