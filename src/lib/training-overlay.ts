type Target = { label: string; element: HTMLElement; bounds: DOMRect };
type Point = { x: number; y: number };

const preference = 'captcha-training-overlay';

function center(bounds: DOMRect): Point {
  return { x: bounds.left + bounds.width / 2, y: bounds.top + bounds.height / 2 };
}

function crosshair({ x, y }: Point, size = 7): string {
  return `<path d="M${x - size} ${y}H${x + size}M${x} ${y - size}V${y + size}" />`;
}

function label(text: string, x: number, y: number): string {
  return `<text x="${x}" y="${y}">${text}</text>`;
}

function labelLines(lines: readonly string[], x: number, y: number): string {
  return `<text x="${x}" y="${y}">${lines.map((line, index) => `<tspan x="${x}" dy="${index === 0 ? 0 : 12}">${line}</tspan>`).join('')}</text>`;
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

  let enabled = false;
  try { enabled = localStorage.getItem(preference) === 'on'; } catch { /* Storage may be unavailable. */ }
  toggle.checked = enabled;
  let pointer: Point | undefined;
  let frame = 0;

  function visible(element: HTMLElement): boolean {
    return element.checkVisibility() && element.getClientRects().length > 0;
  }

  function targets(): Target[] {
    const candidates: { label: string; element: HTMLElement | null }[] = Array.from(
      dialog.querySelectorAll<HTMLButtonElement>('#image-challenge .image-tile'),
      (element, index) => ({ label: `Tile ${index + 1}`, element }),
    );
    for (const [selector, name] of [
      ['#image-challenge .verify-button', 'Verify'],
      ['#image-challenge .reset-button', 'Reset'],
      ['.checkbox-box', 'Checkbox'],
      ['#puzzle-slider', 'Slider track'],
      ['.reset-slider', 'Reset'],
    ]) candidates.push({ label: name, element: dialog.querySelector<HTMLElement>(selector) });
    return candidates.flatMap(({ label, element }) => element && visible(element)
      ? [{ label, element, bounds: element.getBoundingClientRect() }] : []);
  }

  function render(): void {
    frame = 0;
    if (!enabled || !dialog.open) return;
    const width = window.innerWidth;
    const height = window.innerHeight;
    const activeTargets = targets();
    const shapes: string[] = [];
    const captions: string[] = [];
    const lines = [`Viewport ${width} × ${height} CSS px · DPR ${window.devicePixelRatio.toFixed(2)}`, 'Origin (0,0) top-left · labels x,y (x/W,y/H)'];
    for (const [index, point] of [
      { x: 12, y: 12 }, { x: width - 12, y: 12 },
      { x: 12, y: height - 12 }, { x: width - 12, y: height - 12 },
    ].entries()) {
      shapes.push(crosshair(point, 10));
      captions.push(label(['TL', 'TR', 'BL', 'BR'][index], index % 2 ? width - 40 : 24, point.y + 4));
    }

    for (const target of activeTargets) {
      const point = center(target.bounds);
      shapes.push(box(target.bounds, target.label), crosshair(point));
      if (target.label.startsWith('Tile ')) {
        captions.push(labelLines([
          target.label,
          `${Math.round(point.x)},${Math.round(point.y)} px`,
          `(${(point.x / width).toFixed(3)},${(point.y / height).toFixed(3)})`,
        ], target.bounds.left + 3, target.bounds.top + 12));
      } else {
        captions.push(label(target.label, target.bounds.left + 3, target.bounds.top - 4));
        lines.push(`${target.label}: ${pointLabel(point, width, height)}`);
      }
    }
    const slider = dialog.querySelector<HTMLInputElement>('#puzzle-slider');
    if (slider && visible(slider)) {
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
        captions.push(label('Handle', handle.x + 5, handle.y - 12));
        lines.push(`Slider handle: ${pointLabel(handle, width, height)}`);
      }
    }
    const piece = dialog.querySelector<HTMLElement>('.puzzle-piece');
    const gap = dialog.querySelector<HTMLElement>('.puzzle-gap');
    if (piece && gap && visible(piece) && visible(gap)) {
      const pieceBounds = piece.getBoundingClientRect();
      const gapBounds = gap.getBoundingClientRect();
      const pieceCenter = center(pieceBounds);
      const gapCenter = center(gapBounds);
      shapes.push(box(pieceBounds, 'Piece'), box(gapBounds, 'Gap'), crosshair(pieceCenter), crosshair(gapCenter));
      shapes.push(`<path d="M${pieceCenter.x} ${pieceCenter.y}L${gapCenter.x} ${gapCenter.y}" />`);
      captions.push(label(`Piece ${pointLabel(pieceCenter, width, height)}`, pieceBounds.left, pieceBounds.bottom + 12));
      captions.push(label(`Gap ${pointLabel(gapCenter, width, height)}`, gapBounds.left, gapBounds.top - 5));
      lines.push(`Piece → gap dx ${(gapBounds.left - pieceBounds.left).toFixed(1)} px · distance ${Math.hypot(gapCenter.x - pieceCenter.x, gapCenter.y - pieceCenter.y).toFixed(1)} px`);
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
      lines.push(`Pointer ${pointLabel(pointer, width, height)}`);
      if (nearest) {
        const point = center(nearest.bounds);
        const distance = Math.hypot(point.x - pointer.x, point.y - pointer.y);
        shapes.push(`<path d="M${pointer.x} ${pointer.y}L${point.x} ${point.y}" />`);
        lines.push(`${hovered ? 'Hovered' : 'Nearest'} ${nearest.label} · center distance ${distance.toFixed(1)} px`);
      }
    } else lines.push('Pointer: move over the viewport to measure');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.innerHTML = `<g fill="none" stroke="#000" stroke-width="3">${shapes.join('')}</g><g fill="none" stroke="#fff" stroke-width="1">${shapes.join('')}</g><g fill="#fff" stroke="#000" stroke-width="3" paint-order="stroke" font-family="monospace" font-size="10">${captions.join('')}</g>`;
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
    try { localStorage.setItem(preference, enabled ? 'on' : 'off'); } catch { /* The toggle still works without storage. */ }
    sync();
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
