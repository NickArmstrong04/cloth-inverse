/** Presentation mode for the demo video (plan 2026-11-03): `?present=1` replays static JSON
 *  (public/present/entry.json = one JobResult, public/present/camera.json = one recorded
 *  camera-path job) with no backend, and exposes `window.__present` + key bindings so a
 *  capture script (or a hand on the keyboard) can drive every state change on cue. */

import type { CameraJob, CamStatus, JobResult } from './api';
import type { CamState } from './StepStrip';

export const PRESENT = new URLSearchParams(window.location.search).get('present') === '1';

export interface PresentData {
  entry: JobResult;
  camera: { job: CameraJob; timeline: { t: number; status: string }[] } | null;
}

export async function loadPresent(): Promise<PresentData> {
  const entry = (await (await fetch('/present/entry.json')).json()) as JobResult;
  let camera: PresentData['camera'] = null;
  try {
    const r = await fetch('/present/camera.json');
    if (r.ok) camera = (await r.json()) as PresentData['camera'];
  } catch { /* strip stays hidden */ }
  return { entry, camera };
}

/** Camera-path statuses in backend order (camera.py / app._run_camera). */
const ORDER: CamStatus[] = ['simulating', 'render', 'sam', 'vggt', 'decode', 'grid', 'predict', 'narrate', 'done'];
/** Stage key that lands when the status moves on to the next entry of ORDER. */
const LANDS: Record<string, string> = {
  render: 'simulating', sam: 'render', vggt: 'sam', decode: 'vggt', grid: 'decode', predict: 'grid', narrate: 'predict', done: 'narrate',
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Feed a recorded job to the strip one status at a time: the tile of the running step shows
 *  "working…" with its own elapsed counter, then lands with the real artifact. `stepS` per step
 *  (a number, or one per status); the result (used by the predict tile) appears at 'narrate',
 *  as the backend does. Returns when 'done' is set. */
export async function replayCamera(
  job: CameraJob, stepS: number | Partial<Record<CamStatus, number>>, setCam: (c: CamState | null) => void,
): Promise<void> {
  const stages = job.stages ?? {};
  const t0 = Date.now();
  for (const status of ORDER) {
    const done: Record<string, unknown> = {};
    for (const k of ORDER.slice(0, ORDER.indexOf(status))) if (k in stages) done[k] = (stages as Record<string, unknown>)[k];
    if (status === 'done') for (const k of Object.keys(stages)) done[k] = (stages as Record<string, unknown>)[k];
    const withResult = status === 'narrate' || status === 'done';
    setCam({
      id: 'present',
      startedAt: t0,
      job: { status, progress: (ORDER.indexOf(status) + 1) / ORDER.length, stages: done as CameraJob['stages'],
        ...(withResult && job.result ? { result: job.result } : {}) },
    });
    if (status === 'done') return;
    const s = typeof stepS === 'number' ? stepS : (stepS[status] ?? 1.5);
    await sleep(s * 1000);
  }
}

export interface PresentApi {
  setFrame: (k: number) => void;
  /** show the markers and slide the preview arrows in along their push directions */
  flyArrows: (ms?: number) => Promise<void>;
  /** Index of the last frame (29 for a 30-frame run; some cached runs have fewer). */
  last: () => number;
  play: (once?: boolean) => void;
  pause: () => void;
  /** Show / hide the push markers and preview arrows (hidden at load in present mode; panel(true) shows them). */
  marks: (on: boolean) => void;
  /** Show the undeformed sheet instead of the frames. */
  flat: (on: boolean) => void;
  panel: (on: boolean) => void;
  working: (on: boolean) => void;
  result: (on: boolean) => void;
  arrows: (a: { true?: boolean; pred?: boolean; rev?: boolean }) => void;
  scrollPanel: (selector: string) => void;
  camera: (stepS?: number | Partial<Record<CamStatus, number>>) => Promise<void>;
  lightbox: (stepKey: string | null) => void;
  orbit: (deg: number, ms?: number) => Promise<void>;
  reset: () => void;
  data: () => PresentData | null;
}

/** Keyboard fallback for a hand-driven recording. */
export function bindKeys(api: PresentApi): () => void {
  const onKey = (e: KeyboardEvent) => {
    if (e.target && (e.target as HTMLElement).tagName === 'INPUT') return;
    const d = api.data();
    switch (e.key) {
      case 'n': api.panel(true); break;
      case 'N': api.panel(false); break;
      case 'w': api.working(true); break;
      case 'r': api.result(true); api.play(true); break;
      case 'p': api.play(true); break;
      case 't': api.arrows({ true: true }); break;
      case 'y': api.arrows({ pred: true }); break;
      case 'u': api.arrows({ rev: true }); break;
      case 'U': api.arrows({ rev: false }); break;
      case 'm': api.scrollPanel('table.metrics'); break;
      case 'c': if (d?.camera) void api.camera(1.5); break;
      case 'o': void api.orbit(25, 2500); break;
      case '[': api.setFrame(Math.max(0, currentFrame - 1)); break;
      case ']': api.setFrame(Math.min(api.last(), currentFrame + 1)); break;
      case '0': api.reset(); break;
      default: return;
    }
    e.preventDefault();
  };
  window.addEventListener('keydown', onKey);
  return () => window.removeEventListener('keydown', onKey);
}

export let currentFrame = 0;
export function noteFrame(k: number): void { currentFrame = k; }
