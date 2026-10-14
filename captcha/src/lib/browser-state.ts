import { Result, ResultAsync } from 'neverthrow';

export type BrowserFailure = { operation: string; cause: unknown };
export type StorageKind = 'local' | 'session';

function storage(kind: StorageKind): Storage {
  return kind === 'local' ? window.localStorage : window.sessionStorage;
}

/** Read browser storage without leaking security or quota exceptions. */
export function readStorage(kind: StorageKind, key: string): Result<string | null, BrowserFailure> {
  return Result.fromThrowable(() => storage(kind).getItem(key), (cause): BrowserFailure => ({ operation: `${kind} storage read`, cause }))();
}

export function writeStorage(kind: StorageKind, key: string, value: string): Result<void, BrowserFailure> {
  return Result.fromThrowable(() => storage(kind).setItem(key, value), (cause): BrowserFailure => ({ operation: `${kind} storage write`, cause }))();
}

export function removeStorage(kind: StorageKind, key: string): Result<void, BrowserFailure> {
  return Result.fromThrowable(() => storage(kind).removeItem(key), (cause): BrowserFailure => ({ operation: `${kind} storage remove`, cause }))();
}

export function requestFullscreen(): ResultAsync<void, BrowserFailure> {
  return ResultAsync.fromThrowable(async () => { await document.documentElement.requestFullscreen(); }, (cause): BrowserFailure => ({ operation: 'fullscreen request', cause }))();
}

export function exitFullscreen(): ResultAsync<void, BrowserFailure> {
  return ResultAsync.fromThrowable(async () => { await document.exitFullscreen(); }, (cause): BrowserFailure => ({ operation: 'fullscreen exit', cause }))();
}

export function openDialog(dialog: HTMLDialogElement): Result<void, BrowserFailure> {
  return Result.fromThrowable(() => dialog.showModal(), (cause): BrowserFailure => ({ operation: 'dialog open', cause }))();
}

export function prepareImages(images: readonly HTMLImageElement[]): ResultAsync<void, BrowserFailure> {
  return ResultAsync.fromThrowable(async () => {
    for (const image of images) image.loading = 'eager';
    await Promise.all(images.map((image) => image.decode()));
  }, (cause): BrowserFailure => ({ operation: 'challenge image decode', cause }))();
}

export function navigate(path: '/' | '/verified/', replace = false): Result<void, BrowserFailure> {
  return Result.fromThrowable(() => {
    if (replace) window.location.replace(path);
    else window.location.assign(path);
  }, (cause): BrowserFailure => ({ operation: 'page navigation', cause }))();
}
