import { sliderChallenges, type SliderChallenge } from '../data/slider-challenges';

type ChallengeState = 'idle' | 'success' | 'error';

export function initializeSliderChallenge(): void {
  const root = document.querySelector<HTMLElement>('#slider-challenge');
  if (!root) return;

  const canvas = root.querySelector<HTMLElement>('.puzzle-canvas');
  const piece = root.querySelector<HTMLElement>('.puzzle-piece');
  const slider = root.querySelector<HTMLInputElement>('#puzzle-slider');
  const status = root.querySelector<HTMLElement>('#slider-status');
  const resetButton = root.querySelector<HTMLButtonElement>('.reset-slider');
  const image = root.querySelector<HTMLImageElement>('.puzzle-canvas img');
  const description = root.querySelector<HTMLElement>('.puzzle-description');
  if (!canvas || !piece || !slider || !status || !resetButton || !image || !description) return;

  bindSlider(root, canvas, piece, slider, status, resetButton, image, description);
}

function bindSlider(
  root: HTMLElement,
  canvas: HTMLElement,
  piece: HTMLElement,
  slider: HTMLInputElement,
  status: HTMLElement,
  resetButton: HTMLButtonElement,
  image: HTMLImageElement,
  description: HTMLElement,
): void {
  let challenge: SliderChallenge = sliderChallenges[0];

  function setState(state: ChallengeState, message: string): void {
    root.dataset.state = state;
    status.textContent = message;
    root.dispatchEvent(new CustomEvent('challenge-state', {
      bubbles: true,
      detail: { kind: 'slider', state },
    }));
  }

  function paint(): void {
    const { targetX, targetY, pieceSize } = challenge;
    const travel = 1 - pieceSize;
    const position = Number(slider.value) / Number(slider.max) * travel;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    root.style.setProperty('--piece-x', `${position * 100}%`);
    piece.style.backgroundSize = `${width}px ${height}px`;
    piece.style.backgroundPosition = `${-targetX * width}px ${-targetY * height}px`;
    slider.setAttribute('aria-valuetext', `Piece ${Math.round(position / travel * 100)} percent across`);
  }

  function reset(): void {
    slider.value = '0';
    slider.disabled = false;
    paint();
    setState('idle', 'Waiting for alignment.');
  }

  slider.addEventListener('input', () => {
    paint();
    if (root.dataset.state === 'error') setState('idle', 'Adjust the piece and release to verify.');
  });
  slider.addEventListener('change', () => {
    const { targetX, pieceSize, tolerance } = challenge;
    const travel = 1 - pieceSize;
    const position = Number(slider.value) / Number(slider.max) * travel;
    if (Math.abs(position - targetX) <= tolerance) {
      slider.value = String(targetX / travel * Number(slider.max));
      paint();
      slider.disabled = true;
      setState('success', 'Verified. The piece fits.');
    } else {
      setState('error', 'Not aligned yet. Slide again or reset.');
    }
  });
  resetButton.addEventListener('click', reset);
  document.addEventListener('challenge-reset', reset);
  document.addEventListener('challenge-select', (event) => {
    if (!(event instanceof CustomEvent)) return;
    const detail: unknown = event.detail;
    if (!detail || typeof detail !== 'object' || !('kind' in detail) || detail.kind !== 'slider' || !('index' in detail)) return;
    if (typeof detail.index !== 'number' || !Number.isInteger(detail.index)) return;
    const selected = sliderChallenges[detail.index];
    if (!selected) return;
    challenge = selected;
    const { src, alt, targetX, targetY, pieceSize, tolerance } = challenge;
    image.src = src;
    description.textContent = `Puzzle image: ${alt}`;
    root.dataset.challengeId = challenge.id;
    root.dataset.targetX = String(targetX);
    root.dataset.targetY = String(targetY);
    root.dataset.pieceSize = String(pieceSize);
    root.dataset.tolerance = String(tolerance);
    root.style.setProperty('--puzzle-image', `url('${src}')`);
    root.style.setProperty('--target-x', `${targetX * 100}%`);
    root.style.setProperty('--target-y', `${targetY * 100}%`);
    root.style.setProperty('--piece-size', `${pieceSize * 100}%`);
    reset();
  });
  new ResizeObserver(paint).observe(canvas);
  reset();
}
