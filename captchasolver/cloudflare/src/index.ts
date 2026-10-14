import { Container, getContainer } from "@cloudflare/containers";
import { checkAccess, type AccessEnv } from "./access.ts";
import { containerEnv, forwardHeaders, friendlyContainerError, jsonError } from "./forward.ts";

interface Env extends AccessEnv {
  CAPTCHA_APP: DurableObjectNamespace<CaptchaApp>;
  ALLOWED_TEST_HOSTS?: string;
  RATE_LIMITS?: string;
}

// The Python app (FastAPI + headless Chromium). One named instance serves everyone because test
// runs live in its memory. Results are not stored: when it sleeps, they are gone.
export class CaptchaApp extends Container<Env> {
  defaultPort = 8000;
  sleepAfter = "30m";

  constructor(ctx: ConstructorParameters<typeof Container<Env>>[0], env: Env) {
    super(ctx, env);
    this.envVars = containerEnv(env);
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const access = await checkAccess(request, env);
    if (!access.ok) return jsonError(access.status, access.message);

    const forwarded = new Request(request, { headers: forwardHeaders(request.headers, access.identity) });
    try {
      const resp = await getContainer(env.CAPTCHA_APP, "main").fetch(forwarded);
      return (await friendlyContainerError(resp, request.headers.get("accept") ?? "")) ?? resp;
    } catch (e) {
      console.error("container fetch failed", e instanceof Error ? e.message : String(e));
      return jsonError(503, "The solver is starting up. Try again in a few seconds.");
    }
  },
} satisfies ExportedHandler<Env>;
