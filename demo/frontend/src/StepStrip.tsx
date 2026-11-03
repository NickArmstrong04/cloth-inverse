import { useEffect, useRef, useState } from 'react';
import type { CamStatus, CameraJob, JobResult } from './api';

/** The camera path, one tile per step, in the order the backend reports them (camera.py). */
export const STEPS = [
  { key: 'simulating', title: 'Re-simulate on paper', model: 'Newton / Warp VBD',
    blurb: 'Your two pushes on a paper sheet that creases and settles to rest: the physics the camera model was trained on. Pushes outside its range are clamped.' },
  { key: 'render', title: 'Two photos, 20° apart', model: 'renderer · coded print',
    blurb: 'The sheet wears a printed code so a camera can tell which point of the sheet each pixel shows.' },
  { key: 'sam', title: 'Find the sheet', model: 'SAM 3.1 · Meta Model API',
    blurb: 'Hosted SAM 3.1 segments the sheet in each photo from a text prompt. Only pixels inside the mask go further.' },
  { key: 'vggt', title: 'Depth from the pair', model: 'VGGT · Meta open weights',
    blurb: 'VGGT lifts the photo pair to depth; the known 20° camera motion sets the metric scale.' },
  { key: 'decode', title: 'Read the print', model: 'coded-texture decoder',
    blurb: 'Each printed cell names its place on the sheet, giving a 3D point for every decoded vertex.' },
  { key: 'grid', title: 'Lift to a grid', model: 'known-size check · harmonic fill',
    blurb: 'Decoded points become a 256×256 grid; the sheet\'s known 50 cm size corrects the scale and the hidden side is filled in smoothly.' },
  { key: 'predict', title: 'Guess the pushes', model: 'M21-c · CNN trained on camera grids',
    blurb: 'The inverse model reads the camera-built grid and predicts both pushes: where, which way, how hard.' },
  { key: 'narrate', title: 'Verdict', model: 'Muse Spark · Meta Model API',
    blurb: 'Muse Spark writes the plain-language verdict from the numbers alone; every number it uses is checked against them.' },
] as const;

export type StepKey = (typeof STEPS)[number]['key'];

const ORDER: CamStatus[] = ['queued', 'simulating', 'render', 'sam', 'vggt', 'decode', 'grid', 'predict', 'narrate', 'done'];

export interface CamState {
  id: string;
  job: CameraJob | null;
  startedAt: number;
  error?: string;
}

function img(b64: string | null | undefined): string | undefined {
  return b64 ? `data:image/jpeg;base64,${b64}` : undefined;
}

function s(x: number | null | undefined, d = 1): string {
  return x == null ? '…' : x.toFixed(d);
}

/** 64x64 visibility mask (uint8, base64) as a small image: seen = light, filled in = dark. */
function VisThumb({ vis, onClick }: { vis: string; onClick: () => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const bytes = Uint8Array.from(atob(vis), (ch) => ch.charCodeAt(0));
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const im = ctx.createImageData(64, 64);
    for (let k = 0; k < 64 * 64; k++) {
      const v = bytes[k] ? 235 : 70;
      const j = 63 - Math.floor(k / 64); // v up
      const o = (j * 64 + (k % 64)) * 4;
      im.data[o] = v; im.data[o + 1] = v; im.data[o + 2] = bytes[k] ? 215 : 90; im.data[o + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
  }, [vis]);
  return <canvas ref={ref} width={64} height={64} className="thumb" onClick={onClick} title="Seen by the camera (light) vs filled in (dark)" />;
}

export function StepStrip({ cam, mesh }: { cam: CamState | null; mesh: JobResult | null }) {
  const [now, setNow] = useState(Date.now());
  const [big, setBig] = useState<{ src: string; caption: string } | null>(null);
  const [stepStart, setStepStart] = useState<{ key: string; at: number } | null>(null);
  const status = cam?.job?.status ?? (cam ? 'queued' : null);
  const stages = cam?.job?.stages ?? {};
  const result = cam?.job?.result ?? null;
  const running = status != null && status !== 'done' && status !== 'failed';

  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, [running]);

  useEffect(() => {
    if (status && (!stepStart || stepStart.key !== status)) setStepStart({ key: status, at: Date.now() });
  }, [status]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { setBig(null); }, [cam?.id]);

  if (!cam) return null;

  const rank = (k: CamStatus) => ORDER.indexOf(k);
  const state = (key: StepKey): 'pending' | 'running' | 'done' | 'failed' => {
    if (key in stages) return 'done';
    if (status === 'failed') return rank(key) <= rank(stepStart?.key as CamStatus) ? 'failed' : 'pending';
    if (status === key) return 'running';
    return 'pending';
  };
  const elapsed = stepStart ? ((now - stepStart.at) / 1000).toFixed(0) : '0';

  const open = (src: string | undefined, caption: string) => src && setBig({ src, caption });

  const body = (key: StepKey) => {
    const st = state(key);
    if (st === 'running') return <div className="tile-wait">working… {elapsed} s</div>;
    if (st === 'failed') return <div className="tile-fail">{cam.error ?? cam.job?.error ?? 'failed'}</div>;
    if (st === 'pending') return <div className="tile-wait">waiting</div>;
    switch (key) {
      case 'simulating': {
        const a = stages.simulating!;
        return (
          <>
            <div className="tile-text">{a.adjustments.length === 0 ? 'Your pushes used as given.' : a.adjustments.join(' ')}</div>
            <div className="tile-cap">{s(a.elapsed_s, 0)} s</div>
          </>
        );
      }
      case 'render': {
        const a = stages.render!;
        return (
          <>
            <img className="thumb" src={img(a.v0)} alt="photo, view 1" onClick={() => open(img(a.v0), 'Photo 1 of the stereo pair (coded print)')} />
            <div className="tile-cap">view 1 of 2 · {s(a.elapsed_s, 0)} s</div>
          </>
        );
      }
      case 'sam': {
        const a = stages.sam!;
        return (
          <>
            <img className="thumb" src={img(a.overlay_v0)} alt="SAM 3.1 mask" onClick={() => open(img(a.overlay_v0), `SAM 3.1 mask, prompt "${a.prompt ?? 'fabric'}" (view 1)`)} />
            {a.fallback
              ? <div className="tile-cap warn">renderer mask: SAM 3.1 unavailable</div>
              : <div className="tile-cap">prompt "{a.prompt}" · {a.k} found · {s(a.latency_s)} s</div>}
          </>
        );
      }
      case 'vggt': {
        const a = stages.vggt!;
        return (
          <>
            <img className="thumb" src={img(a.depth_v0)} alt="VGGT depth" onClick={() => open(img(a.depth_v0), 'VGGT depth of view 1 (yellow near, blue far)')} />
            <div className="tile-cap">{a.depth_range_cm ? `${s(a.depth_range_cm[0], 0)}–${s(a.depth_range_cm[1], 0)} cm` : 'metric depth'} · scale ×{s(a.scale_geo, 2)} · {s(a.elapsed_s, 0)} s</div>
          </>
        );
      }
      case 'decode': {
        const a = stages.decode!;
        return (
          <>
            <img className="thumb" src={img(a.lattice_v0)} alt="decoded print" onClick={() => open(img(a.lattice_v0), 'Decoded vertices of the print, coloured by their place on the sheet')} />
            <div className="tile-cap">{a.cells_found} of {a.cells_total} cells read · {s(a.elapsed_s, 0)} s</div>
          </>
        );
      }
      case 'grid': {
        const a = stages.grid!;
        return (
          <>
            <VisThumb vis={a.vis} onClick={() => {}} />
            <div className="tile-cap">{(100 * a.visible_frac).toFixed(0)}% seen, rest filled{a.scale_fix ? ` · scale fix ×${s(a.scale_fix, 2)}` : ''}</div>
          </>
        );
      }
      case 'predict': {
        if (!result) return <div className="tile-wait">…</div>;
        const c = result.camera.metrics.mean, m = result.mesh.metrics.mean, r = result.reference;
        return (
          <>
            <div className="tile-text">
              <b>from the photos:</b> {s(c.pos_cm)} cm / {s(c.dir_deg, 0)}° off<br />
              <small>typical {s(r.camera.pos_cm)} cm / {s(r.camera.dir_deg, 0)}°</small><br />
              <b>from the exact sheet:</b> {s(m.pos_cm)} cm / {s(m.dir_deg, 0)}°
              {mesh && <><br /><small>mesh path above: {s(mesh.metrics.mean.pos_cm)} cm / {s(mesh.metrics.mean.dir_deg, 0)}°</small></>}
            </div>
          </>
        );
      }
      case 'narrate': {
        const n = stages.narrate!;
        return n.ok && n.text
          ? <div className="tile-text">{n.text}<div className="tile-cap">{n.model ?? 'Muse Spark'} · {s(n.latency_s ?? n.elapsed_s, 0)} s</div></div>
          : <div className="tile-text">{result
              ? `The camera guess is ${s(result.camera.metrics.mean.pos_cm)} cm and ${s(result.camera.metrics.mean.dir_deg, 0)}° off; the same model on the exact sheet is ${s(result.mesh.metrics.mean.pos_cm)} cm and ${s(result.mesh.metrics.mean.dir_deg, 0)}° off.`
              : '…'}<div className="tile-cap warn">{n.error?.includes('META_API_KEY') ? 'Muse Spark not configured' : 'Muse Spark verdict unavailable'}</div></div>;
      }
    }
  };

  return (
    <div className="under-hood">
      <div className="uh-head">
        <b>Under the hood: the same pushes seen through a camera</b>
        <span className="notice">
          {status === 'failed' ? `stopped: ${cam.error ?? cam.job?.error ?? ''}`
            : status === 'done' ? `done in ${s(result?.timings?.total_s, 0)} s`
              : `running (${elapsed} s on this step) · the GPU container wakes on demand`}
        </span>
      </div>
      <div className="uh-tiles">
        {STEPS.map((st, i) => (
          <div key={st.key} className={`tile ${st.key} ${state(st.key)}`} title={st.blurb}>
            <div className="tile-n">{i + 1}</div>
            <div className="tile-title">{st.title}</div>
            <div className="tile-model">{st.model}</div>
            {body(st.key)}
          </div>
        ))}
      </div>
      {big && (
        <div className="lightbox" onClick={() => setBig(null)}>
          <img src={big.src} alt={big.caption} />
          <div className="lightbox-cap">{big.caption} · click to close</div>
        </div>
      )}
    </div>
  );
}
