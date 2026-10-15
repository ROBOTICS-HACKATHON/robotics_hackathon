import assert from "node:assert/strict";
import { test } from "node:test";
import { SignJWT, generateKeyPair, type JWTVerifyGetKey } from "jose";
import { checkAccess } from "./access.ts";
import { containerEnv, forwardHeaders, friendlyContainerError } from "./forward.ts";

const TEAM = "myteam.cloudflareaccess.com";
const AUD = "a".repeat(64);
const env = { ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD };

const { privateKey, publicKey } = await generateKeyPair("RS256");
const other = await generateKeyPair("RS256");
const getKey: JWTVerifyGetKey = async () => publicKey;

async function jwt(claims: Record<string, unknown> = {}, opts: { iss?: string; aud?: string; exp?: string; key?: CryptoKey } = {}) {
  return new SignJWT({ email: "alvaro@example.com", ...claims })
    .setProtectedHeader({ alg: "RS256" })
    .setIssuer(opts.iss ?? `https://${TEAM}`)
    .setAudience(opts.aud ?? AUD)
    .setIssuedAt()
    .setExpirationTime(opts.exp ?? "5m")
    .sign(opts.key ?? privateKey);
}
const req = (headers: Record<string, string> = {}) => new Request("https://app.example.com/api/health", { headers });

test("a valid Access token is accepted and the email becomes the identity", async () => {
  const r = await checkAccess(req({ "Cf-Access-Jwt-Assertion": await jwt() }), env, getKey);
  assert.deepEqual(r, { ok: true, identity: "alvaro@example.com" });
});

test("the CF_Authorization cookie works too (browsers)", async () => {
  const r = await checkAccess(req({ Cookie: `x=1; CF_Authorization=${await jwt()}; y=2` }), env, getKey);
  assert.equal(r.ok, true);
});

test("a service token (no email) is identified by its common_name", async () => {
  const token = await new SignJWT({ common_name: "robot.access" })
    .setProtectedHeader({ alg: "RS256" }).setIssuer(`https://${TEAM}`).setAudience(AUD).setExpirationTime("5m").sign(privateKey);
  const r = await checkAccess(req({ "Cf-Access-Jwt-Assertion": token }), env, getKey);
  assert.deepEqual(r, { ok: true, identity: "robot.access" });
});

test("missing token -> 401", async () => {
  const r = await checkAccess(req(), env, getKey);
  assert.equal(r.ok === false && r.status, 401);
});

for (const [name, token] of [
  ["wrong audience", () => jwt({}, { aud: "b".repeat(64) })],
  ["wrong issuer", () => jwt({}, { iss: "https://evil.cloudflareaccess.com" })],
  ["expired", () => jwt({}, { exp: "-1m" })],
  ["signed with another key", () => jwt({}, { key: other.privateKey })],
  ["garbage", async () => "not.a.jwt"],
] as const) {
  test(`rejects a token with ${name} -> 403`, async () => {
    const r = await checkAccess(req({ "Cf-Access-Jwt-Assertion": await token() }), env, getKey);
    assert.equal(r.ok === false && r.status, 403);
  });
}

test("an unsigned (alg none) token is rejected", async () => {
  const b64 = (o: object) => Buffer.from(JSON.stringify(o)).toString("base64url");
  const none = `${b64({ alg: "none", typ: "JWT" })}.${b64({ iss: `https://${TEAM}`, aud: AUD, email: "x@y.z", exp: 4102444800 })}.`;
  const r = await checkAccess(req({ "Cf-Access-Jwt-Assertion": none }), env, getKey);
  assert.equal(r.ok === false && r.status, 403);
});

test("fails closed when Access is not configured", async () => {
  for (const e of [{}, { ACCESS_TEAM_DOMAIN: TEAM }, { ACCESS_AUD: AUD }, { ACCESS_TEAM_DOMAIN: "", ACCESS_AUD: "" }]) {
    const r = await checkAccess(req({ "Cf-Access-Jwt-Assertion": await jwt() }), e, getKey);
    assert.equal(r.ok === false && r.status, 503);
  }
});

test("rejects a team domain that is not a cloudflareaccess.com host (misconfiguration guard)", async () => {
  for (const team of ["evil.com", "myteam.cloudflareaccess.com.evil.com", "http://x.cloudflareaccess.com", "a b.cloudflareaccess.com"]) {
    const r = await checkAccess(req({ "Cf-Access-Jwt-Assertion": await jwt() }), { ACCESS_TEAM_DOMAIN: team, ACCESS_AUD: AUD }, getKey);
    assert.equal(r.ok === false && r.status, 503, team);
  }
});

test("the dev flag bypasses Access only when it is exactly 'true'", async () => {
  assert.equal((await checkAccess(req(), { ALLOW_UNAUTHENTICATED_DEV: "true" })).ok, true);
  for (const v of ["false", "1", "TRUE", "yes", ""]) {
    const r = await checkAccess(req(), { ALLOW_UNAUTHENTICATED_DEV: v });
    assert.equal(r.ok, false, v);
  }
});

test("forwarded headers drop cookies and Access secrets, keep the Cloudflare credentials, and cannot spoof identity", () => {
  const h = forwardHeaders(new Headers({
    Cookie: "CF_Authorization=secret", "Cf-Access-Jwt-Assertion": "jwt", "CF-Access-Client-Secret": "s",
    "CF-Access-Client-Id": "i", "X-Auth-Identity": "admin@spoofed", "X-CF-Account-Id": "acct", "X-CF-Api-Token": "tok",
    "Content-Type": "multipart/form-data",
  }), "alvaro@example.com");
  assert.equal(h.get("cookie"), null);
  assert.equal(h.get("cf-access-jwt-assertion"), null);
  assert.equal(h.get("cf-access-client-secret"), null);
  assert.equal(h.get("cf-access-client-id"), null);
  assert.equal(h.get("x-auth-identity"), "alvaro@example.com");
  assert.equal(h.get("x-cf-account-id"), "acct");
  assert.equal(h.get("x-cf-api-token"), "tok");
  assert.equal(h.get("content-type"), "multipart/form-data");
});

test("public mode lets anyone in, but only when AUTH_MODE is exactly 'public'", async () => {
  assert.deepEqual(await checkAccess(req(), { AUTH_MODE: "public" }), { ok: true, identity: "public" });
  for (const v of ["Public", "PUBLIC", "true", "open", "", "public "]) {
    const r = await checkAccess(req(), { AUTH_MODE: v });
    assert.equal(r.ok === false && r.status, 503, `AUTH_MODE=${JSON.stringify(v)}`);   // falls back to needing Access
  }
});

test("a configured Access deployment still enforces tokens when AUTH_MODE is not public", async () => {
  const r = await checkAccess(req(), { ...env, AUTH_MODE: "access" }, getKey);
  assert.equal(r.ok === false && r.status, 401);
});

test("the visitor IP comes from the edge header only; a client-sent X-Client-Ip is discarded", () => {
  const spoofed = forwardHeaders(new Headers({ "X-Client-Ip": "6.6.6.6" }), "public");
  assert.equal(spoofed.get("x-client-ip"), null);                                    // no edge header -> none
  const real = forwardHeaders(new Headers({ "X-Client-Ip": "6.6.6.6", "CF-Connecting-IP": "203.0.113.9" }), "public");
  assert.equal(real.get("x-client-ip"), "203.0.113.9");
});

const NO_INSTANCE = "There is no Container instance available at this time.\nThis is likely because you have reached your max concurrent instance count (set in wrangler config) or are you currently provisioning the Container.";
const plain = (text: string, status: number) => new Response(text, { status, headers: { "content-type": "text/plain;charset=UTF-8" } });

test("'no container instance' becomes a friendly JSON 503 with Retry-After for API clients", async () => {
  const r = await friendlyContainerError(plain(NO_INSTANCE, 503), "application/json");
  assert.ok(r);
  assert.equal(r.status, 503);
  assert.equal(r.headers.get("retry-after"), "10");
  const body = await r.json() as { detail: string };
  assert.match(body.detail, /starting up/);
  assert.doesNotMatch(body.detail, /max concurrent instance/);       // no internal jargon
});

test("browsers get a self-refreshing page instead", async () => {
  const r = await friendlyContainerError(plain(NO_INSTANCE, 503), "text/html,application/xhtml+xml");
  assert.ok(r);
  assert.match(r.headers.get("content-type") ?? "", /text\/html/);
  const html = await r.text();
  assert.match(html, /http-equiv="refresh" content="10"/);
  assert.match(html, /Starting the solver/);
});

test("a failed container start is reported, with its reason escaped in the page", async () => {
  const resp = plain("Failed to start container: <script>alert(1)</script>", 500);
  const json = await friendlyContainerError(resp, "application/json");
  assert.ok(json);
  assert.equal(json.status, 503);
  assert.match((await json.json() as { detail: string }).detail, /Failed to start container/);
  const page = await (await friendlyContainerError(resp, "text/html"))!.text();
  assert.doesNotMatch(page, /<script>alert/);                         // escaped, not injected
  assert.match(page, /&lt;script&gt;/);
});

test("the app's own responses pass through untouched (including FastAPI's plain-text 500)", async () => {
  assert.equal(await friendlyContainerError(plain("Internal Server Error", 500), "text/html"), null);
  assert.equal(await friendlyContainerError(new Response('{"detail":"x"}', { status: 503, headers: { "content-type": "application/json" } }), ""), null);
  assert.equal(await friendlyContainerError(new Response("ok", { status: 200, headers: { "content-type": "text/plain" } }), ""), null);
  const original = plain("Internal Server Error", 500);
  await friendlyContainerError(original, "");
  assert.equal(await original.text(), "Internal Server Error");         // body still readable afterwards
});

test("rate limits are only switched off by an explicit RATE_LIMITS=off", () => {
  assert.equal(containerEnv({ RATE_LIMITS: "off" }).CAPTCHASOLVER_RATE_LIMITS, "off");
  assert.equal(containerEnv({ RATE_LIMITS: " OFF " }).CAPTCHASOLVER_RATE_LIMITS, "off");
  for (const v of [undefined, "", "on", "false", "0", "no"]) {
    assert.equal(containerEnv({ RATE_LIMITS: v }).CAPTCHASOLVER_RATE_LIMITS, "on", String(v));
  }
  assert.equal(containerEnv({}).CAPTCHASOLVER_ALLOWED_HOSTS, "robot-captcha-demo.pages.dev");
  assert.equal(containerEnv({ ALLOWED_TEST_HOSTS: "a.example,b.example" }).CAPTCHASOLVER_ALLOWED_HOSTS, "a.example,b.example");
});
