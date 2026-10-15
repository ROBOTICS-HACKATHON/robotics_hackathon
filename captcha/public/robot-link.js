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
  const challenge = () => document.querySelector('#session-progress-label')?.textContent?.trim() || '';
  const phase = () => document.querySelector('dialog[open]')?.getAttribute('data-phase') || '';

  document.addEventListener('pointerdown', (e) => {
    if (!e.isPrimary) return;
    const w = innerWidth - 2 * INSET, h = innerHeight - 2 * INSET;
    const d = { type: 'touch', u: (e.clientX - INSET) / w, v: (e.clientY - INSET) / h, w, h, x: e.clientX, y: e.clientY,
      pointer: e.pointerType, el: describe(e.target), page: location.pathname, phase: phase(), challenge: challenge() };
    document.dispatchEvent(new CustomEvent('robot-touch', { detail: d }));
    send(d);
  }, { capture: true, passive: true });

  // pass / fail of each challenge, straight from the page (no camera needed to know the result)
  document.addEventListener('challenge-state', (e) => {
    const s = e.detail && e.detail.state;
    if (s !== 'success' && s !== 'error') return;
    send({ type: 'result', state: s === 'success' ? 'pass' : 'fail', page: location.pathname, challenge: challenge(),
      el: e.detail.kind || '' });
  });
})();
