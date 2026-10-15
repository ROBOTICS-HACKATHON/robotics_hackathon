// Robot link: report every touch (and each challenge's result) to /api/touches, so the robot learns
// where its taps really landed. Off unless this browser has the robot key: open any page once with
// ?robot=<ROBOT_KEY> (stored in this browser, removed from the address bar). ?robot= (empty) turns it off.
// Coordinates: u, v from the rectangle between the camera markers' outer corners (18 CSS px inset),
// exactly like touch/calib.html, so they match the arm's calibration when the page is full-screen.
(() => {
  const INSET = 18;
  let key = null;
  try {
    const q = new URLSearchParams(location.search);
    if (q.has('robot')) {
      const k = q.get('robot');
      if (k) localStorage.setItem('robot-key', k); else localStorage.removeItem('robot-key');
      q.delete('robot');
      history.replaceState(history.state, '', location.pathname + (q.toString() ? '?' + q : '') + location.hash);
    }
    key = localStorage.getItem('robot-key');
  } catch { /* storage blocked: link stays off */ }
  window.robotLink = { on: !!key, sent: 0, failed: 0 };
  if (!key) return;
  document.documentElement.dataset.robotLink = 'on';
  // A stylus press that lasts a moment is "press and hold" on a Windows touch screen: it selects the text it is
  // on (the checkbox label) or opens the context menu, and the click never happens. Neither is wanted here.
  const noHold = document.createElement('style');
  noHold.textContent = 'html[data-robot-link="on"], html[data-robot-link="on"] * { -webkit-user-select: none !important; user-select: none !important; -webkit-touch-callout: none !important; }';
  document.head.appendChild(noHold);
  document.addEventListener('contextmenu', (e) => e.preventDefault(), true);
  document.addEventListener('selectstart', (e) => e.preventDefault(), true);

  const send = (body) => fetch('/api/touches', {
    method: 'POST', keepalive: true,
    headers: { 'content-type': 'application/json', 'x-robot-key': key },
    body: JSON.stringify(body),
  }).then((r) => { if (r.ok) window.robotLink.sent++; else window.robotLink.failed++; })
    .catch(() => { window.robotLink.failed++; });

  const describe = (t) => {
    const e = t instanceof Element ? t : null;
    if (!e) return '';
    const tile = e.closest('[data-tile-index]');
    if (tile) return 'tile:' + tile.getAttribute('data-tile-index');
    const c = e.closest('button, a, input, label, select, [id]');
    if (!c) return e.tagName.toLowerCase();
    return c.id ? '#' + c.id : c.tagName.toLowerCase() + (typeof c.className === 'string' && c.className ? '.' + c.className.split(' ')[0] : '');
  };
  // F11 (browser full-screen) does not set document.fullscreenElement: compare the page with the screen instead
  const isFull = () => !!document.fullscreenElement
    || (Math.abs(innerWidth - screen.width) <= 2 && Math.abs(innerHeight - screen.height) <= 2);
  const challenge = () => document.querySelector('#session-progress-label')?.textContent?.trim() || '';
  const phase = () => document.querySelector('dialog[open]')?.getAttribute('data-phase') || '';

  document.addEventListener('pointerdown', (e) => {
    if (!e.isPrimary) return;
    const w = innerWidth - 2 * INSET, h = innerHeight - 2 * INSET;
    const d = { type: 'touch', u: (e.clientX - INSET) / w, v: (e.clientY - INSET) / h, w, h, x: e.clientX, y: e.clientY,
      pointer: e.pointerType, el: describe(e.target), page: location.pathname, phase: phase(), challenge: challenge(),
      state: isFull() ? 'full' : 'window' };
    document.dispatchEvent(new CustomEvent('robot-touch', { detail: d }));
    send(d);
  }, { capture: true, passive: true });

  // Which screen is showing, and where its buttons are (u, v centres): the robot taps these, so nobody has
  // to teach it the layout. Sent when something changes, and every 10 s so the robot knows it is fresh.
  const centre = (sel) => {
    const el = document.querySelector(sel);
    if (!el || el.closest('[hidden]')) return null;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return null;
    const w = innerWidth - 2 * INSET, h = innerHeight - 2 * INSET;
    return [+((r.left + r.width / 2 - INSET) / w).toFixed(4), +((r.top + r.height / 2 - INSET) / h).toFixed(4),
      +(r.width / w).toFixed(4), +(r.height / h).toFixed(4)];
  };
  const screen = () => {
    const dlg = document.querySelector('dialog[open]');
    const path = location.pathname;
    const t = {};
    let kind = 'other';
    if (path.startsWith('/verified')) { kind = 'verified'; t.restart = centre('#restart'); }
    else if (dlg && dlg.dataset.phase === 'checkbox') {
      kind = 'checkbox'; t.checkbox = centre('#human-checkbox');
      if (!t.checkbox || document.querySelector('#human-checkbox')?.disabled) kind = 'loading';
    } else if (dlg && dlg.dataset.phase === 'challenge') {
      kind = dlg.dataset.kind === 'slider' ? 'slider' : 'grid'; t.verify = centre('.verify-button');
    } else if (path.startsWith('/training')) {
      kind = 'choice'; t.start = centre('#start-session'); t.card = centre('.choice[data-challenge="image"]');
    }
    return { kind, path, targets: t, w: innerWidth, h: innerHeight, sw: screen.width, sh: screen.height, dpr: devicePixelRatio, fullscreen: isFull(),
      markers: !!document.querySelector('#screen-markers:popover-open'), challenge: challenge() };
  };
  let lastScreen = '', lastSent = 0;
  const report = () => {
    if (document.visibilityState !== 'visible') return;
    const s = screen();
    const j = JSON.stringify(s);
    if (j === lastScreen && Date.now() - lastSent < 10000) return;
    lastScreen = j; lastSent = Date.now();
    fetch('/api/page', { method: 'POST', keepalive: true, headers: { 'content-type': 'application/json', 'x-robot-key': key },
      body: j }).catch(() => {});
  };
  setInterval(report, 400);
  addEventListener('load', report);

  // pass / fail of each challenge, straight from the page (no camera needed to know the result)
  document.addEventListener('challenge-state', (e) => {
    const s = e.detail && e.detail.state;
    if (s !== 'success' && s !== 'error') return;
    send({ type: 'result', state: s === 'success' ? 'pass' : 'fail', page: location.pathname, challenge: challenge(),
      el: e.detail.kind || '' });
  });
})();
