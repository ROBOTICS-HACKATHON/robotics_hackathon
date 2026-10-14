// Decides who may use the app: either "public" (AUTH_MODE=public) or only people holding a valid
// Cloudflare Access JWT (defence in depth: Access already gates the hostname at the edge). Pure module:
// it imports only `jose`, so it can be unit-tested in plain Node.
import { createRemoteJWKSet, jwtVerify, type JWTVerifyGetKey } from "jose";

export interface AccessEnv {
  AUTH_MODE?: string; // "public" = no login; anything else (or unset) = Cloudflare Access required
  ACCESS_TEAM_DOMAIN?: string;
  ACCESS_AUD?: string;
  ALLOW_UNAUTHENTICATED_DEV?: string;
}

export type AccessResult =
  | { ok: true; identity: string }
  | { ok: false; status: number; message: string };

const TEAM_DOMAIN = /^[a-z0-9][a-z0-9-]*\.cloudflareaccess\.com$/;
const jwksByTeam = new Map<string, JWTVerifyGetKey>();

function tokenFrom(request: Request): string | null {
  const header = request.headers.get("Cf-Access-Jwt-Assertion");
  if (header) return header;
  const cookie = request.headers.get("Cookie") ?? "";
  const m = /(?:^|;\s*)CF_Authorization=([^;]+)/.exec(cookie);
  return m ? m[1] : null;
}

export async function checkAccess(
  request: Request,
  env: AccessEnv,
  getKey?: JWTVerifyGetKey, // tests inject a local key; production uses Cloudflare's JWKS
): Promise<AccessResult> {
  // Public deployment: anyone may use it, but only with their own Cloudflare credentials. It must be
  // switched on explicitly and exactly; any other value falls through to the Access checks below.
  if (env.AUTH_MODE === "public") return { ok: true, identity: "public" };
  if (env.ALLOW_UNAUTHENTICATED_DEV === "true") return { ok: true, identity: "dev (no Access)" };

  const team = (env.ACCESS_TEAM_DOMAIN ?? "").trim().toLowerCase();
  const aud = (env.ACCESS_AUD ?? "").trim();
  if (!TEAM_DOMAIN.test(team) || !aud) {
    return { ok: false, status: 503, message: "Cloudflare Access is not configured for this deployment (set ACCESS_TEAM_DOMAIN and ACCESS_AUD)." };
  }

  const token = tokenFrom(request);
  if (!token) return { ok: false, status: 401, message: "Sign in through Cloudflare Access." };

  let key = getKey ?? jwksByTeam.get(team);
  if (!key) {
    key = createRemoteJWKSet(new URL(`https://${team}/cdn-cgi/access/certs`));
    jwksByTeam.set(team, key);
  }
  try {
    const { payload } = await jwtVerify(token, key, {
      issuer: `https://${team}`,
      audience: aud,
      algorithms: ["RS256"],
    });
    // people have an email; Access service tokens (scripts, the robot) have a common_name
    const identity = String(payload.email ?? payload.common_name ?? payload.sub ?? "unknown");
    return { ok: true, identity };
  } catch {
    return { ok: false, status: 403, message: "Invalid or expired Access token." };
  }
}
