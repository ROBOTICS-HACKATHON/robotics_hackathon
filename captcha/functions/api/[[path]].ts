// Robot link API (Cloudflare Pages Function), served next to the site on the same domain.
//
//   POST /api/touches   the page reports a touch or a challenge result   (needs the robot key)
//   GET  /api/touches   the robot reads them: ?after=<id>                (needs the robot key)
//   GET  /api/state     the practice page reads the robot's current target / message
//   POST /api/state     the robot sets it                                (needs the robot key)
//
// Setup (once): a D1 database bound as DB, and a secret ROBOT_KEY. See captcha/README.md "Robot link".
// Without them the site works as before and these endpoints answer 503.

interface Env {
  DB?: any; // D1Database
  ROBOT_KEY?: string;
}

const SCHEMA = [
  `CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, type TEXT NOT NULL,
     u REAL, v REAL, w REAL, h REAL, x REAL, y REAL, pointer TEXT, el TEXT, page TEXT, phase TEXT,
     state TEXT, challenge TEXT)`,
  `CREATE TABLE IF NOT EXISTS robot_state (k TEXT PRIMARY KEY, v TEXT NOT NULL, ts REAL NOT NULL)`,
];
let ready = false;

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' } });

const num = (x: unknown) => (typeof x === 'number' && Number.isFinite(x) ? x : null);
const str = (x: unknown, n = 120) => (typeof x === 'string' ? x.slice(0, n) : null);

export const onRequest = async ({ request, env, params }: { request: Request; env: Env; params: { path?: string[] } }) => {
  const route = (params.path ?? []).join('/');
  if (!env.DB || !env.ROBOT_KEY) return json({ error: 'robot link not set up (D1 binding DB and secret ROBOT_KEY)' }, 503);
  if (!ready) {
    for (const sql of SCHEMA) await env.DB.prepare(sql).run();
    ready = true;
  }
  const authed = request.headers.get('x-robot-key') === env.ROBOT_KEY;
  const now = Date.now() / 1000;

  if (route === 'touches' && request.method === 'POST') {
    if (!authed) return json({ error: 'bad key' }, 401);
    const d: any = await request.json().catch(() => null);
    if (!d || (d.type !== 'touch' && d.type !== 'result')) return json({ error: 'bad body' }, 400);
    const r = await env.DB.prepare(
      `INSERT INTO events (ts, type, u, v, w, h, x, y, pointer, el, page, phase, state, challenge)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).bind(now, d.type, num(d.u), num(d.v), num(d.w), num(d.h), num(d.x), num(d.y), str(d.pointer, 12),
      str(d.el), str(d.page), str(d.phase, 20), str(d.state, 20), str(d.challenge)).run();
    if (Math.random() < 0.02) await env.DB.prepare('DELETE FROM events WHERE ts < ?').bind(now - 3 * 86400).run();
    return json({ ok: true, id: r.meta?.last_row_id ?? null });
  }

  if (route === 'touches' && request.method === 'GET') {
    if (!authed) return json({ error: 'bad key' }, 401);
    const after = Number(new URL(request.url).searchParams.get('after') ?? '-1');
    const latest = await env.DB.prepare('SELECT COALESCE(MAX(id), 0) AS id FROM events').first();
    if (!(after >= 0)) return json({ latest: latest?.id ?? 0, events: [], now });
    const rows = await env.DB.prepare('SELECT * FROM events WHERE id > ? ORDER BY id LIMIT 200').bind(after).all();
    return json({ latest: latest?.id ?? 0, events: rows.results ?? [], now });
  }

  if (route === 'state' && request.method === 'GET') {
    const row = await env.DB.prepare("SELECT v, ts FROM robot_state WHERE k = 'state'").first();
    return json({ ...(row ? JSON.parse(row.v) : {}), age: row ? now - row.ts : null });
  }

  if (route === 'state' && request.method === 'POST') {
    if (!authed) return json({ error: 'bad key' }, 401);
    const d = await request.text();
    if (d.length > 4000) return json({ error: 'too big' }, 413);
    await env.DB.prepare("INSERT INTO robot_state (k, v, ts) VALUES ('state', ?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v, ts = excluded.ts")
      .bind(d || '{}', now).run();
    return json({ ok: true });
  }

  return json({ error: 'not found' }, 404);
};
