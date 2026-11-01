import { useCallback, useEffect, useRef, useState } from 'react';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { ACCENTS, REVERSAL, SceneView, type PlacedForce, type Stage } from './Scene';
import { b64ToFrames, dirFromAzEl, getJob, postSimulate, type JobResult } from './api';

interface ForceForm extends PlacedForce {
  az: number;
  el: number;
}

const DEFAULTS: ForceForm[] = [
  { u: 0.3, v: 0.6, az: -68, el: 57, direction: [0, 0, 1], magnitude: 0.8, duration: 0.5 },
  { u: 0.7, v: 0.4, az: 90, el: -72, direction: [0, 0, 1], magnitude: 0.5, duration: 0.4 },
];

function withDir(f: ForceForm): ForceForm {
  return { ...f, direction: dirFromAzEl(f.az, f.el) };
}

function sep(a: ForceForm, b: ForceForm): number {
  return Math.hypot(a.u - b.u, a.v - b.v);
}

const COLOUR = ['red', 'blue'];
// Job polling interval. 3 s (was 500 ms) to keep requests to the backend
// low; the backend scales to zero when idle, so a job takes ~30-120 s anyway.
const POLL_MS = 3000;

function hint(stage: Stage, active: number, placed: boolean[], locked: boolean[]): string {
  const k = active + 1;
  if (stage === 'running') {
    return 'Simulating the real physics, then running the inverse model. The GPU backend wakes on demand, so the first push after a pause can take a minute or two. Camera stays free.';
  }
  if (locked[active]) {
    const other = active === 0 ? 1 : 0;
    return locked[other]
      ? 'Both positions are locked — sheet clicks do nothing. Untick a lock to move a push.'
      : `Push ${k} is locked — sheet clicks move push ${other + 1} instead.`;
  }
  if (stage === 'done') {
    return `Solid arrows: your pushes. Faded: predicted. Green: reversal. Push ${k} selected — click the sheet to move it (clears this result).`;
  }
  if (!placed[active]) {
    return `Push ${k} selected — click the sheet to place it (${COLOUR[active]} marker). Drag to orbit, wheel to zoom, right-drag to pan.`;
  }
  if (!placed[0] || !placed[1]) {
    return `Push ${k} selected — click the sheet to move it. Select the other card to place push ${2 - active}.`;
  }
  return `Push ${k} selected — click the sheet to move it, or set its direction, magnitude and duration, then press Push.`;
}

export default function App() {
  const [forms, setForms] = useState<ForceForm[]>(DEFAULTS.map(withDir));
  const [stage, setStage] = useState<Stage>('place1');
  const [active, setActive] = useState(0);
  // True once the user has picked a card themselves (click, slider, focus):
  // sheet clicks then keep moving that push instead of advancing to the other.
  const [pinned, setPinned] = useState(false);
  const [placed, setPlaced] = useState([false, false]);
  // Locked pushes ignore sheet clicks; a click then falls through to the
  // other push if that one is unlocked.
  const [locked, setLocked] = useState([false, false]);
  const allPlaced = placed[0] && placed[1];
  const [frames, setFrames] = useState<Float32Array | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(true);
  const [job, setJob] = useState<{ id: string; progress: number; error?: string } | null>(null);
  const [result, setResult] = useState<JobResult | null>(null);
  const [showTrue, setShowTrue] = useState(true);
  const [showPred, setShowPred] = useState(true);
  const [showRev, setShowRev] = useState(true);
  const [dark, setDark] = useState(true);
  const [cachedNote, setCachedNote] = useState(false);
  const controlsRef = useRef<OrbitControlsImpl>(null!);

  useEffect(() => {
    document.documentElement.dataset.theme = dark ? '' : 'light';
  }, [dark]);

  // Sheet clicks always move the active (selected) push, except mid-job.
  // While the user is editing a card (pinned), clicks keep moving that push;
  // only the untouched initial flow advances to the other push after a
  // placement. Moving a patch after a result clears the stale result;
  // orbit-drag never clicks.
  const onPlace = useCallback(
    (u: number, v: number) => {
      if (stage === 'running') return;
      const target = !locked[active] ? active : !locked[active === 0 ? 1 : 0] ? (active === 0 ? 1 : 0) : -1;
      if (target < 0) return;
      const nplaced = [placed[0] || target === 0, placed[1] || target === 1];
      setForms((fs) => fs.map((f, k) => (k === target ? { ...f, u, v } : f)));
      setPlaced(nplaced);
      if (nplaced[0] && nplaced[1]) {
        if (stage === 'done') {
          setResult(null);
          setFrames(null);
          setFrame(0);
          setPlaying(false);
        }
        setStage('ready');
      } else if (pinned || locked[active]) {
        setStage(target === 0 ? 'place1' : 'place2');
      } else {
        const other = target === 0 ? 1 : 0;
        setActive(other);
        setStage(other === 0 ? 'place1' : 'place2');
      }
    },
    [active, stage, placed, pinned, locked],
  );

  // Any interaction with a card (click, slider drag, keyboard focus) selects
  // that push for sheet clicks.
  const select = (k: number) => {
    setActive(k);
    setPinned(true);
    if (stage === 'place1' || stage === 'place2') {
      setStage(k === 0 ? 'place1' : 'place2');
    }
  };

  const push = useCallback(async () => {
    try {
      const id = await postSimulate(
        forms.map((f) => ({ u: f.u, v: f.v, direction: f.direction, magnitude: f.magnitude, duration: f.duration })),
      );
      setJob({ id, progress: 0 });
      setStage('running');
      setResult(null);
    } catch (e) {
      setJob({ id: '', progress: 0, error: String(e) });
    }
  }, [forms]);

  useEffect(() => {
    if (!job || job.error || stage !== 'running') return;
    const t = setInterval(async () => {
      try {
        const s = await getJob(job.id);
        if (s.status === 'done' && s.result) {
          clearInterval(t);
          const { data } = b64ToFrames(s.result.frames);
          setFrames(data);
          setFrame(29);
          setResult(s.result);
          setCachedNote(!!s.result.cached);
          setStage('done');
          setPlaying(false);
          setJob(null);
        } else if (s.status === 'failed') {
          clearInterval(t);
          setJob({ id: job.id, progress: 1, error: s.error ?? 'job failed' });
          setStage('ready');
        } else {
          setJob({ id: job.id, progress: s.progress });
        }
      } catch (e) {
        clearInterval(t);
        setJob({ id: job.id, progress: 1, error: String(e) });
        setStage('ready');
      }
    }, POLL_MS);
    return () => clearInterval(t);
  }, [job, stage]);

  useEffect(() => {
    if (!playing || !frames) return;
    const t = setInterval(() => {
      setFrame((f) => (f >= 29 ? (loop ? 0 : 29) : f + 1));
    }, 100);
    return () => clearInterval(t);
  }, [playing, frames, loop]);

  useEffect(() => {
    if (frame >= 29 && !loop) setPlaying(false);
  }, [frame, loop]);

  const tooClose = sep(forms[0], forms[1]) < 0.2;
  const reset = () => {
    setResult(null);
    setFrames(null);
    setFrame(0);
    setPlaying(false);
    setPlaced([false, false]);
    setActive(0);
    setPinned(false);
    setLocked([false, false]);
    setStage('place1');
    setJob(null);
  };

  const update = (k: number, patch: Partial<ForceForm>) => {
    select(k);
    setForms((fs) => fs.map((f, i) => (i === k ? withDir({ ...f, ...patch }) : f)));
  };

  return (
    <div className="app">
      <div className="stage-wrap">
        <SceneView
          frames={frames}
          frame={frame}
          stage={stage}
          placed={placed}
          allForms={forms}
          result={result ? { predicted: result.predicted_forces, reversal: result.reversal_forces, showTrue, showPred, showRev } : null}
          onPlace={onPlace}
          controlsRef={controlsRef}
          dark={dark}
        />
        <div className="stage-hint">{hint(stage, active, placed, locked)}</div>
      </div>
      <div className="panel">
        <div className="topbar">
          <h2>Cloth inverse demo</h2>
          <div className="spacer" />
          <button className="ghost" onClick={() => setDark((d) => !d)}>{dark ? 'Light' : 'Dark'}</button>
          <button className="ghost" onClick={() => controlsRef.current?.reset()}>Reset view</button>
        </div>

        {forms.map((f, k) => (
          <div className={`force-card c${k + 1}${k === active ? ' active' : ''}`} key={k}
            onPointerDownCapture={() => select(k)} onFocusCapture={() => select(k)}
            title={k === active ? 'Selected — sheet clicks move this push' : 'Click or edit to select this push'}>
            <h3>Push {k + 1}{k === active ? ' (selected)' : ''} — {locked[k] ? 'position locked' : `click sheet to ${placed[k] ? 'move' : 'place'}`}</h3>
            <div className="row"><label>position</label><output>({f.u.toFixed(2)}, {f.v.toFixed(2)})</output>
              <label title="Locked: sheet clicks never move this push">
                <input type="checkbox" checked={locked[k]}
                  onChange={(e) => setLocked((l) => l.map((x, i) => (i === k ? e.target.checked : x)))} /> lock
              </label>
            </div>
            <Slider label="azimuth" min={-180} max={180} step={1} value={f.az} unit="°"
              onChange={(v) => update(k, { az: v })} />
            <Slider label="elevation" min={-90} max={90} step={1} value={f.el} unit="°"
              onChange={(v) => update(k, { el: v })} />
            <div className="row"><label>direction</label>
              <output>[{f.direction.map((x) => x.toFixed(2)).join(', ')}]</output></div>
            <Slider label="magnitude" min={0.15} max={1.25} step={0.01} value={f.magnitude} unit=" N"
              onChange={(v) => update(k, { magnitude: v })} />
            <Slider label="duration" min={0.3} max={0.7} step={0.01} value={f.duration} unit=" s"
              onChange={(v) => update(k, { duration: v })} />
          </div>
        ))}
        {tooClose && <div className="warn">Patches are closer than 0.2 in (u, v) — move one.</div>}

        <button onClick={push} disabled={stage !== 'ready' || tooClose || !allPlaced}>
          {stage === 'running' ? 'Working…' : 'Push'}
        </button>
        {job?.error && <div className="err">{job.error}</div>}
        {(stage === 'running' || stage === 'done') && (
          <button className="ghost" onClick={reset}>Start over</button>
        )}

        {frames && (
          <div>
            <h3>Playback</h3>
            <div className="row">
              <button className="ghost" onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</button>
              <input type="range" min={0} max={29} step={1} value={frame}
                onChange={(e) => setFrame(Number(e.target.value))} style={{ flex: 1 }} />
              <output>{frame}/29</output>
            </div>
            <div className="row">
              <label><input type="checkbox" checked={loop} onChange={(e) => setLoop(e.target.checked)} /> loop</label>
              <label><input type="checkbox" checked={showTrue} onChange={(e) => setShowTrue(e.target.checked)} /> true</label>
              <label><input type="checkbox" checked={showPred} onChange={(e) => setShowPred(e.target.checked)} /> predicted</label>
              <label><input type="checkbox" checked={showRev} onChange={(e) => setShowRev(e.target.checked)} /> reversal</label>
            </div>
          </div>
        )}

        {result && (
          <div>
            <h3>Model vs your pushes</h3>
            {cachedNote && <div className="notice">Served from cache (GPU busy) — nearest precomputed result.</div>}
            <div className="legend">
              <span><span className="swatch" style={{ background: ACCENTS[0] }} />true 1</span>
              <span><span className="swatch" style={{ background: ACCENTS[1] }} />true 2</span>
              <span><span className="swatch" style={{ background: REVERSAL }} />reversal</span>
            </div>
            <p className="notice">How far off is the model's guess of each push? Lower is better.</p>
            <table className="metrics">
              <thead><tr>
                <th />
                <th title="Distance on the sheet between where you pushed and where the model thinks you pushed, in centimetres (sheet is 50 cm wide).">
                  location off<br /><small>cm</small></th>
                <th title="Angle between your push direction and the predicted direction, in degrees (0 = same direction, 90 = perpendicular).">
                  angle off<br /><small>degrees</small></th>
                <th title="How much the predicted push strength (force x duration) differs from yours, as a percentage of yours.">
                  strength off<br /><small>% of yours</small></th>
              </tr></thead>
              <tbody>
                {result.metrics.per_force.map((m, k) => (
                  <tr key={k} title="Each predicted push is matched to the closest true push (the assignment in metrics.py).">
                    <td>push {k + 1}</td><td>{m.pos_cm.toFixed(1)}</td><td>{m.dir_deg.toFixed(0)}</td><td>{(100 * m.impulse_rel).toFixed(0)}%</td>
                  </tr>
                ))}
                <tr title="Average of the two pushes.">
                  <td>average</td><td>{result.metrics.mean.pos_cm.toFixed(1)}</td>
                  <td>{result.metrics.mean.dir_deg.toFixed(0)}</td><td>{(100 * result.metrics.mean.impulse_rel).toFixed(0)}%</td>
                </tr>
                <tr title="What the model (M16-a) scores on average over its held-out test set. Your numbers near these are normal.">
                  <td>typical (test set)</td><td>{result.metrics.reference.pos_cm.toFixed(1)}</td>
                  <td>{result.metrics.reference.dir_deg.toFixed(0)}</td><td>{(100 * result.metrics.reference.impulse_rel).toFixed(0)}%</td>
                </tr>
              </tbody>
            </table>
            <div className="notice">
              sim {result.timings.sim_s.toFixed(1)} s · inference {result.timings.infer_s.toFixed(2)} s · grid {result.grid}
            </div>
          </div>
        )}

        <details className="how">
          <summary>How it works</summary>
          <p>1. The backend simulates your two pushes on a 0.5 m sheet (Newton/VBD, zero gravity).</p>
          <p>2. It observes the full deformed mesh: positions and velocities.</p>
          <p>3. The inverse model (M16-a: CNN, hybrid strain/curvature input) recovers both pushes.</p>
          <p>4. Predicted pushes appear faded next to yours.</p>
          <p>5. Reversal: the predicted pushes negated — what would undo the deformation.</p>
          <p>Who benefits: robots handling garments can infer the grasp that crumpled them.
            Sheet-goods lines can plan pushes or undo unwanted deformations.
            The method recovers unseen loads from observed shape alone.</p>
        </details>

        <details className="how">
          <summary>Camera path (under construction)</summary>
          <p>What a camera would see: synthetic photo of a simulated sheet,
            the renderer's oracle mask, and the SAM 3 mask (prompt "fabric",
            IoU 0.98). This is the perception input being built — the model
            above runs on the simulator mesh, not on this image.</p>
          <img src="camera-tile.png" alt="Synthetic photo, oracle mask, SAM 3 mask" style={{ width: '100%' }} />
        </details>

        <footer className="stack">
          PyTorch · Newton/Warp VBD sim · M16-a cnn_reg+hybrid (1.0 cm / 4.3° / 0.156) · React + three.js
        </footer>
      </div>
    </div>
  );
}

function Slider({ label, min, max, step, value, unit, onChange }: {
  label: string; min: number; max: number; step: number; value: number; unit: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="row">
      <label>{label}</label>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} />
      <output>{value.toFixed(step < 1 ? 2 : 0)}{unit}</output>
    </div>
  );
}
