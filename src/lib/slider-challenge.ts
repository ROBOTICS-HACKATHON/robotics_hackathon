type ChallengeState = 'idle' | 'success' | 'error';

export function initializeSliderChallenge(): void {
  const root = document.querySelector<HTMLElement>('#slider-challenge');
  if (!root) return;

  const canvas = root.querySelector<HTMLElement>('.puzzle-canvas');
  const piece = root.querySelector<HTMLElement>('.puzzle-piece');
  const slider = root.querySelector<HTMLInputElement>('#puzzle-slider');
  const status = root.querySelector<HTMLElement>('#slider-status');
  const resetButton = root.querySelector<HTMLButtonElement>('.reset-slider');
  if (!canvas || !piece || !slider || !status || !resetButton) return;

  bindSlider(root, canvas, piece, slider, status, resetButton);
}

function bindSlider(
  root: HTMLElement,
  canvas: HTMLElement,
  piece: HTMLElement,
  slider: HTMLInputElement,
  status: HTMLElement,
  resetButton: HTMLButtonElement,
): void {
  const targetX = Number(root.dataset.targetX);
  const targetY = Number(root.dataset.targetY);
  const pieceSize = Number(root.dataset.pieceSize);
  const tolerance = Number(root.dataset.tolerance);
  const travel = 1 - pieceSize;

  function setState(state: ChallengeState, message: string): void {
    root.dataset.state = state;
    status.textContent = message;
    root.dispatchEvent(new CustomEvent('challenge-state', {
      bubbles: true,
      detail: { kind: 'slider', state },
    }));
  }

  function paint(): void {
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
  new ResizeObserver(paint).observe(canvas);
  reset();
}
