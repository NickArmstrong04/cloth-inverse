/** Backend client + .npy float16 frame decoder. */

export interface ForceInput {
  u: number;
  v: number;
  direction: [number, number, number];
  magnitude: number;
  duration: number;
}

export interface ForceEx extends ForceInput {
  sigma_cells: number;
  radius_cells: number;
  impulse: number;
}

export interface Metric {
  pos_cm: number;
  dir_deg: number;
  impulse_rel: number;
}

export interface JobResult {
  frames: string;
  frame_shape: number[];
  true_forces: ForceEx[];
  predicted_forces: ForceEx[];
  reversal_forces: ForceEx[];
  metrics: { per_force: Metric[]; mean: Metric; reference: Metric };
  timings: { sim_s: number; infer_s: number };
  grid: number;
  cached: boolean;
}

export interface JobStatus {
  status: 'queued' | 'simulating' | 'camera' | 'predicting' | 'done' | 'failed';
  progress: number;
  result?: JobResult;
  error?: string;
}

/** Camera path (backend camera.py): photo -> SAM 3.1 -> VGGT -> decoded grid -> M21-c. */
export interface CameraPrediction {
  predicted_forces: ForceEx[];
  metrics: { per_force: Metric[]; mean: Metric };
}

export interface CameraResult {
  id?: string;
  stem: string;
  live: boolean;
  true_forces: ForceEx[];
  images: { photo: string; mask: string | null };
  mask_source: string;
  grids: { camera: string; true: string; shape: number[] };
  camera: CameraPrediction;
  mesh: CameraPrediction;
  coverage: { camera_sees: number | null; decoded: number | null; scale_fix: number | null; lift_err_visible_cm: number | null };
  reference: { camera: { pos_cm: number; dir_deg: number; pos_cm_median: number }; mesh: { pos_cm: number; dir_deg: number } };
  timings?: Record<string, number>;
  adjustments?: string[];
}

export interface GalleryIndex {
  n: number;
  entries: { id: string; stem: string; camera: Metric; mesh: Metric; coverage: CameraResult['coverage'] }[];
  summary: { camera: Metric & { pos_cm_median: number }; mesh: Metric & { pos_cm_median: number } };
  source: string;
}

export async function getGalleryIndex(): Promise<GalleryIndex> {
  const r = await fetch('gallery/index.json');
  if (!r.ok) throw new Error(`gallery index: ${r.status}`);
  return (await r.json()) as GalleryIndex;
}

export async function getGalleryEntry(id: string): Promise<CameraResult> {
  const r = await fetch(`gallery/${id}.json`);
  if (!r.ok) throw new Error(`gallery ${id}: ${r.status}`);
  return (await r.json()) as CameraResult;
}

export async function postCamera(forces: ForceInput[]): Promise<string> {
  const r = await fetch(`${API}/camera`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ forces }),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error((detail as { detail?: string }).detail ?? `POST /camera: ${r.status}`);
  }
  return ((await r.json()) as { job_id: string }).job_id;
}

export async function getCameraJob(jobId: string): Promise<{ status: JobStatus['status']; progress: number; result?: CameraResult; error?: string }> {
  const r = await fetch(`${API}/jobs/${jobId}`);
  if (!r.ok) throw new Error(`GET /jobs: ${r.status}`);
  return await r.json();
}

const API = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

export async function postSimulate(forces: ForceInput[]): Promise<string> {
  const r = await fetch(`${API}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ forces }),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error((detail as { detail?: string }).detail ?? `POST /simulate: ${r.status}`);
  }
  return ((await r.json()) as { job_id: string }).job_id;
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const r = await fetch(`${API}/jobs/${jobId}`);
  if (!r.ok) throw new Error(`GET /jobs: ${r.status}`);
  return (await r.json()) as JobStatus;
}

function f16bits(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

/** Parse np.save float16 bytes -> {data, shape}. */
export function parseNpyF16(buf: ArrayBuffer): { data: Float32Array; shape: number[] } {
  const dv = new DataView(buf);
  const major = dv.getUint8(6);
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const hoff = major === 1 ? 10 : 12;
  const header = String.fromCharCode(...new Uint8Array(buf, hoff, hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1] ?? '';
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? '';
  if (descr !== '<f2') throw new Error(`expected <f2 frames, got ${descr}`);
  const dims = shape.split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  const count = dims.reduce((a, b) => a * b, 1);
  const raw = new Uint16Array(buf, hoff + hlen, count);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = f16bits(raw[i]);
  return { data: out, shape: dims };
}

export function b64ToFrames(b64: string): { data: Float32Array; shape: number[] } {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return parseNpyF16(bytes.buffer);
}

/** az/el degrees -> unit 3D direction. */
export function dirFromAzEl(azDeg: number, elDeg: number): [number, number, number] {
  const az = (azDeg * Math.PI) / 180;
  const el = (elDeg * Math.PI) / 180;
  return [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)];
}
