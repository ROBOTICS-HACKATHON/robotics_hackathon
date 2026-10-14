import { readStorage, writeStorage } from './browser-state';

type Detail = 'all' | 'markers' | 'coordinates' | 'distances';
type Focus = 'all' | 'checkbox' | 'verify' | 'slider' | `tile-${number}`;
type Target = { label: string; focus: Focus; element: HTMLElement; bounds: DOMRect };
type Point = { x: number; y: number };

const preference = 'captcha-training-overlay';

function center(bounds: DOMRect): Point {
  return { x: bounds.left + bounds.width / 2, y: bounds.top + bounds.height / 2 };
}

function crosshair({ x, y }: Point, size = 7): string {
  return `<path d="M${x - size} ${y}H${x + size}M${x} ${y - size}V${y + size}" />`;
}

function stored(key: string): string | null {
  return readStorage('local', key).unwrapOr(null);
}

function persist(key: string, value: string): void {
  writeStorage('local', key, value).match(() => {}, () => {
    const message = 'Overlay settings could not be saved. Enable site storage to keep them after reloading.';
    const status = document.querySelector('#fullscreen-status');
    const browserStatus = document.querySelector('#browser-status');
    if (status) status.textContent = message;
    if (browserStatus) browserStatus.textContent = message;
    const feedback = document.querySelector<HTMLElement>('#browser-feedback');
    if (feedback) feedback.hidden = false;
  });
}

function isDetail(value: string | null): value is Detail {
  return ['all', 'markers', 'coordinates', 'distances'].some((mode) => mode === value);
}

function isFocus(value: string | null): value is Focus {
  return value !== null && (['all', 'checkbox', 'verify', 'slider'].includes(value) || /^tile-[1-9]$/.test(value));
}

function panel(lines: readonly string[], x: number, y: number, maxWidth = 220): string {
  const width = Math.min(maxWidth, Math.max(...lines.map((line) => line.length)) * 7 + 10);
  const height = lines.length * 15 + 8;
  const left = Math.max(0, Math.min(x, window.innerWidth - width));
  const top = Math.max(0, Math.min(y, window.innerHeight - height));
  return `<svg x="${left}" y="${top}" width="${width}" height="${height}" overflow="hidden"><rect width="${width}" height="${height}" fill="#000" stroke="#fff" stroke-width="1" /><text x="5" y="16" fill="#fff" font-family="monospace" font-size="11" font-weight="700">${lines.map((line, index) => `<tspan x="5" dy="${index === 0 ? 0 : 15}">${line}</tspan>`).join('')}</text></svg>`;
}

function box(bounds: DOMRect, target?: string): string {
  return `<rect${target ? ` data-target="${target}"` : ''} x="${bounds.left}" y="${bounds.top}" width="${bounds.width}" height="${bounds.height}" />`;
}

function pointLabel(point: Point, width: number, height: number): string {
  return `${Math.round(point.x)},${Math.round(point.y)} (${(point.x / width).toFixed(3)},${(point.y / height).toFixed(3)})`;
}

/** Draw viewport coordinates for the currently visible challenge. */
export function initializeTrainingOverlay(): void {
  const overlayElement = document.querySelector<HTMLElement>('#training-overlay');
  const dialogElement = document.querySelector<HTMLDialogElement>('#challenge-dialog');
  const toggle = document.querySelector<HTMLInputElement>('#training-overlay-toggle');
  const svgElement = overlayElement?.querySelector<SVGSVGElement>('svg');
  const readoutElement = overlayElement?.querySelector<HTMLElement>('.training-readout');
  if (!overlayElement || !dialogElement || !toggle || !svgElement || !readoutElement) return;
  const overlay = overlayElement;
  const dialog = dialogElement;
  const svg = svgElement;
  const readout = readoutElement;
  const detailSelect = document.querySelector<HTMLSelectElement>('#training-overlay-detail');
  const focusSelect = document.querySelector<HTMLSelectElement>('#training-overlay-focus');

  let enabled = stored(preference) === 'on';
  const savedDetail = stored(`${preference}-detail`);
  const savedFocus = stored(`${preference}-focus`);
  let detail: Detail = isDetail(savedDetail) ? savedDetail : 'all';
  let focus: Focus = isFocus(savedFocus) ? savedFocus : 'all';
  toggle.checked = enabled;
  if (detailSelect) detailSelect.value = detail;
  if (focusSelect) focusSelect.value = focus;
  let pointer: Point | undefined;
  let frame = 0;

  function visible(element: HTMLElement): boolean {
    return element.checkVisibility() && element.getClientRects().length > 0;
  }

  function targets(): Target[] {
    const candidates: { label: string; focus: Focus; element: HTMLElement | null }[] = Array.from(
      dialog.querySelectorAll<HTMLButtonElement>('#image-challenge .image-tile'),
      (element, index) => ({ label: `Tile ${index + 1}`, focus: `tile-${index + 1}`, element }),
    );
    for (const [selector, name, targetFocus] of [
      ['#image-challenge .verify-button', 'Verify', 'verify'],
      ['#image-challenge .reset-button', 'Reset', 'all'],
      ['.checkbox-box', 'Checkbox', 'checkbox'],
      ['#puzzle-slider', 'Slider track', 'slider'],
      ['.reset-slider', 'Reset', 'slider'],
    ] as const) candidates.push({ label: name, focus: targetFocus, element: dialog.querySelector<HTMLElement>(selector) });
    return candidates.flatMap(({ label, focus: targetFocus, element }) => element && visible(element)
      && !(targetFocus.startsWith('tile-') && element instanceof HTMLButtonElement && element.disabled)
      && (focus === 'all' || targetFocus === focus)
      ? [{ label, focus: targetFocus, element, bounds: element.getBoundingClientRect() }] : []);
  }

  function render(): void {
    frame = 0;
    if (!enabled || !dialog.open) return;
    const width = window.innerWidth;
    const height = window.innerHeight;
    const activeTargets = targets();
    const coordinates = detail === 'all' || detail === 'coordinates';
    const distances = detail === 'all' || detail === 'distances';
    const showSlider = focus === 'all' || focus === 'slider';
    const shapes: string[] = [];
    const captions: string[] = [];
    const lines = [`Viewport ${width} × ${height} CSS px · DPR ${window.devicePixelRatio.toFixed(2)}`];
    if (coordinates) lines.push('Origin (0,0) top-left · x,y (x/W,y/H)');
    for (const [index, point] of [
      { x: 12, y: 12 }, { x: width - 12, y: 12 },
      { x: 12, y: height - 12 }, { x: width - 12, y: height - 12 },
    ].entries()) {
      shapes.push(crosshair(point, 10));
      captions.push(panel([['TL', 'TR', 'BL', 'BR'][index]], index % 2 ? width - 48 : 24, point.y - 11));
    }

    for (const target of activeTargets) {
      const point = center(target.bounds);
      shapes.push(box(target.bounds, target.label), crosshair(point));
      if (target.focus.startsWith('tile-')) {
        captions.push(panel(coordinates ? [
          target.label, `${Math.round(point.x)},${Math.round(point.y)} px`,
          `(${(point.x / width).toFixed(3)},${(point.y / height).toFixed(3)})`,
        ] : [target.label], target.bounds.left + 3, target.bounds.top + 3, target.bounds.width - 6));
      } else {
        captions.push(panel([target.label], target.bounds.left + 3, target.bounds.top - 25));
        if (coordinates) lines.push(`${target.label}: ${pointLabel(point, width, height)}`);
      }
    }
    const slider = dialog.querySelector<HTMLInputElement>('#puzzle-slider');
    if (showSlider && slider && visible(slider)) {
      const bounds = slider.getBoundingClientRect();
      const thumbWidth = Number.parseFloat(getComputedStyle(slider).getPropertyValue('--slider-thumb-width'));
      const min = Number(slider.min);
      const max = Number(slider.max);
      if (Number.isFinite(thumbWidth) && thumbWidth > 0 && max > min) {
        const progress = (slider.valueAsNumber - min) / (max - min);
        const handle = {
          x: bounds.left + thumbWidth / 2 + progress * (bounds.width - thumbWidth),
          y: bounds.top + bounds.height / 2,
        };
        shapes.push(crosshair(handle, 10));
        captions.push(panel(['Handle'], handle.x + 5, handle.y - 32));
        if (coordinates) lines.push(`Slider handle: ${pointLabel(handle, width, height)}`);
      }
    }
    const piece = dialog.querySelector<HTMLElement>('.puzzle-piece');
    const gap = dialog.querySelector<HTMLElement>('.puzzle-gap');
    if (showSlider && piece && gap && visible(piece) && visible(gap)) {
      const pieceBounds = piece.getBoundingClientRect();
      const gapBounds = gap.getBoundingClientRect();
      const pieceCenter = center(pieceBounds);
      const gapCenter = center(gapBounds);
      shapes.push(box(pieceBounds, 'Piece'), box(gapBounds, 'Gap'), crosshair(pieceCenter), crosshair(gapCenter));
      captions.push(panel(['Piece'], pieceBounds.left, pieceBounds.bottom + 3));
      captions.push(panel(['Gap'], gapBounds.left, gapBounds.top - 25));
      if (coordinates) lines.push(`Piece: ${pointLabel(pieceCenter, width, height)}`, `Gap: ${pointLabel(gapCenter, width, height)}`);
      if (distances) {
        shapes.push(`<path d="M${pieceCenter.x} ${pieceCenter.y}L${gapCenter.x} ${gapCenter.y}" />`);
        lines.push(`Piece → gap dx ${(gapBounds.left - pieceBounds.left).toFixed(1)} px · distance ${Math.hypot(gapCenter.x - pieceCenter.x, gapCenter.y - pieceCenter.y).toFixed(1)} px`);
      }
    }
    if (pointer) {
      const currentPointer = pointer;
      shapes.push(crosshair(currentPointer, 12));
      const hit = document.elementFromPoint(pointer.x, pointer.y);
      const hovered = activeTargets.find((target) => target.element === hit || (hit && target.element.contains(hit)));
      const nearest = hovered ?? activeTargets.reduce<Target | undefined>((best, target) => {
        if (!best) return target;
        const a = center(target.bounds);
        const b = center(best.bounds);
        return Math.hypot(a.x - currentPointer.x, a.y - currentPointer.y) < Math.hypot(b.x - currentPointer.x, b.y - currentPointer.y) ? target : best;
      }, undefined);
      if (coordinates) lines.push(`Pointer ${pointLabel(pointer, width, height)}`);
      if (distances && nearest) {
        const point = center(nearest.bounds);
        const distance = Math.hypot(point.x - pointer.x, point.y - pointer.y);
        shapes.push(`<path d="M${pointer.x} ${pointer.y}L${point.x} ${point.y}" />`);
        lines.push(`${hovered ? 'Hovered' : 'Nearest'} ${nearest.label} · center distance ${distance.toFixed(1)} px`);
      }
    } else if (distances) lines.push('Move pointer to measure target distance');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.innerHTML = `<g fill="none" stroke="#000" stroke-width="7">${shapes.join('')}</g><g fill="none" stroke="#ffe600" stroke-width="3">${shapes.join('')}</g>${captions.join('')}`;
    readout.textContent = lines.join('\n');
  }

  function schedule(): void {
    if (enabled && dialog.open && !frame) frame = requestAnimationFrame(render);
  }

  function sync(): void {
    overlay.hidden = !enabled || !dialog.open;
    if (overlay.hidden && frame) {
      cancelAnimationFrame(frame);
      frame = 0;
    } else schedule();
  }

  toggle.addEventListener('change', () => {
    enabled = toggle.checked;
    persist(preference, enabled ? 'on' : 'off');
    sync();
  });
  detailSelect?.addEventListener('change', () => {
    if (!isDetail(detailSelect.value)) return;
    detail = detailSelect.value;
    persist(`${preference}-detail`, detail);
    schedule();
  });
  focusSelect?.addEventListener('change', () => {
    if (!isFocus(focusSelect.value)) return;
    focus = focusSelect.value;
    persist(`${preference}-focus`, focus);
    schedule();
  });
  document.addEventListener('pointermove', (event) => {
    if (!enabled || !dialog.open) return;
    pointer = { x: event.clientX, y: event.clientY };
    schedule();
  });
  dialog.addEventListener('input', schedule);
  dialog.addEventListener('close', sync);
  window.addEventListener('resize', schedule);
  window.addEventListener('scroll', schedule, { capture: true, passive: true });
  document.addEventListener('fullscreenchange', schedule);
  const mutations = new MutationObserver((records) => {
    if (records.some((record) => !overlay.contains(record.target))) sync();
  });
  mutations.observe(dialog, {
    attributes: true, subtree: true, childList: true,
    attributeFilter: ['open', 'hidden', 'style', 'class', 'src', 'data-phase', 'data-kind', 'data-state'],
  });
  const resize = new ResizeObserver(schedule);
  resize.observe(dialog);
  const canvas = dialog.querySelector('.puzzle-canvas');
  if (canvas) resize.observe(canvas);
  sync();
}
