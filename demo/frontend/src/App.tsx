import { useCallback, useEffect, useRef, useState } from 'react';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { ACCENTS, REVERSAL, SceneView, type PlacedForce, type Stage } from './Scene';
import { b64ToFrames, dirFromAzEl, getCameraJob, getJob, getModels, postCamera, postSimulate, type JobResult, type ModelInfo } from './api';
import { StepStrip, type CamState } from './StepStrip';
import * as THREE from 'three';
import { PRESENT, PRESENT_FLAT, bindKeys, loadPresent, noteFrame, replayCamera, type PresentApi, type PresentData } from './present';

interface ForceForm extends PlacedForce {
  az: number;
  el: number;
}

/** Inverse of dirFromAzEl: a unit direction -> azimuth / elevation in degrees. */
function azElFromDir(d: [number, number, number]): { az: number; el: number } {
  const az = (Math.atan2(d[1], d[0]) * 180) / Math.PI;
  const el = (Math.asin(Math.max(-1, Math.min(1, d[2]))) * 180) / Math.PI;
  return { az: Math.round(az), el: Math.round(el) };
}

// The model menu (mirrors the backend's MODELS; /models adds availability). Default M16-a.
// No multi-guess refinement models (M22) by decision 2026-11-03.
const MODEL_MENU: ModelInfo[] = [
  { name: 'M16a', label: 'Positions + final velocities', physics: 'cloth', available: true,
    blurb: 'M16-a: the full mesh with its velocities at the instant the pushes end (cloth, zero gravity; ~20 s per push, plus 1-2 min when the server is cold).',
    reference: { pos_cm: 1.0, dir_deg: 4.3, impulse_rel: 0.156 } },
  { name: 'M20a3', label: 'Positions only', physics: 'paper', available: false,
    blurb: 'M20-a3: a creased paper sheet settled to rest, exact geometry, no velocities (~2 min per push, plus 1-2 min when the server is cold).',
    reference: { pos_cm: 2.8, dir_deg: 6.6, impulse_rel: null } },
  { name: 'M21c', label: 'Pseudo camera', physics: 'paper', available: false,
    blurb: 'M21-c: same paper setting, trained on grids rebuilt from photos, the camera path\'s model (~2 min per push, plus 1-2 min when the server is cold).',
    reference: { pos_cm: 3.4, dir_deg: 7.3, impulse_rel: null } },
];

const DEFAULTS: ForceForm[] = [
  // Default sides and directions swapped 2026-11-03: push 1 now pushes down (el -72) from
  // 0.7/0.4, push 2 up (el 57) from 0.3/0.6; magnitudes and durations stay with their push.
  { u: 0.7, v: 0.4, az: 90, el: -72, direction: [0, 0, 1], magnitude: 0.8, duration: 0.5 },
  { u: 0.3, v: 0.6, az: -68, el: 57, direction: [0, 0, 1], magnitude: 0.5, duration: 0.4 },
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
    return 'Solid arrows: your pushes. Faded: predicted. Green: reversal. The pushes and the sheet are locked with this result — press "Run another simulation" to reset the sheet.';
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
  // Frames per result: 30 normally, fewer in some cached runs (never index past the last one).
  const nFrames = frames ? Math.max(1, Math.round(frames.length / (64 * 64 * 3))) : 30;
  const last = nFrames - 1;
  const [frame, setFrame] = useState(0);
  const [flyIn, setFlyIn] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(true);
  const [job, setJob] = useState<{ id: string; progress: number; error?: string; startedAt?: number } | null>(null);
  const [now, setNow] = useState(Date.now());
  const [result, setResult] = useState<JobResult | null>(null);
  // The camera path (step strip under the canvas): started when the mesh result lands.
  const [cam, setCam] = useState<CamState | null>(null);
  const [model, setModel] = useState('M16a');
  const [menu, setMenu] = useState<ModelInfo[]>(MODEL_MENU);
  const [showTrue, setShowTrue] = useState(true);

  useEffect(() => {
    if (PRESENT) return;
    getModels().then((m) => setMenu(m.models)).catch(() => {});
  }, []);
  // Presentation mode (video): static data, panel/strip hidden until cued (present.ts).
  const [panelIn, setPanelIn] = useState(!PRESENT);
  const presentData = useRef<PresentData | null>(null);
  const [cursor, setCursor] = useState<{ x: number; y: number; down: boolean }>({ x: -100, y: -100, down: false });
  const [cursorOn, setCursorOn] = useState(false); // presentation mode: the fake cursor only when the capture asks for it
  const [marksOff, setMarksOff] = useState(PRESENT);
  const [flat, setFlat] = useState(PRESENT_FLAT);
  const [showPred, setShowPred] = useState(true);
  const [showRev, setShowRev] = useState(true);
  const [dark, setDark] = useState(true);
  const [cachedNote, setCachedNote] = useState(false);
  // True while the shown result is the precomputed run from the demo video (public/present/*),
  // loaded by the "View the demo video's run" button with no backend call (2026-11-03).
  const [demoRun, setDemoRun] = useState(false);
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
      if (stage === 'running' || stage === 'done') return;  // locked with a result (2026-11-03)
      const target = !locked[active] ? active : !locked[active === 0 ? 1 : 0] ? (active === 0 ? 1 : 0) : -1;
      if (target < 0) return;
      const nplaced = [placed[0] || target === 0, placed[1] || target === 1];
      setForms((fs) => fs.map((f, k) => (k === target ? { ...f, u, v } : f)));
      setPlaced(nplaced);
      if (nplaced[0] && nplaced[1]) {
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
    if (PRESENT) {
      setJob({ id: 'present', progress: 0.3 });
      setStage('running');
      setResult(null);
      setCam(null);
      return;
    }
    try {
      const id = await postSimulate(
        forms.map((f) => ({ u: f.u, v: f.v, direction: f.direction, magnitude: f.magnitude, duration: f.duration })),
        model,
      );
      setJob({ id, progress: 0, startedAt: Date.now() });
      setStage('running');
      setResult(null);
      setCam(null);
    } catch (e) {
      setJob({ id: '', progress: 0, error: String(e) });
    }
  }, [forms, model]);

  // Same pushes through the camera path (paper re-sim -> photos -> SAM 3.1 -> VGGT -> M21-c),
  // shown step by step under the canvas. On demand only, after a mesh result.
  const runCamera = useCallback(() => {
    if (!result || cam) return;
    const fs = result.true_forces.map((f) => ({ u: f.u, v: f.v, direction: f.direction, magnitude: f.magnitude, duration: f.duration }));
    setCam({ id: '', job: null, startedAt: Date.now() });
    postCamera(fs)
      .then((cid) => setCam({ id: cid, job: null, startedAt: Date.now() }))
      .catch((e) => setCam({ id: '', job: { status: 'failed', progress: 0 }, startedAt: Date.now(), error: String(e) }));
  }, [result, cam]);

  // The precomputed run from the demo video (cache entry 10, the same files presentation
  // mode replays): shown at once, pushes set and locked, no backend call. Judges skip the wait.
  // With another model selected, the same pushes through that model (entry_<model>.json,
  // 2026-11-03); without such a file the M16-a run is shown and the menu follows it.
  const showDemoRun = useCallback(async () => {
    try {
      const d = presentData.current ?? (await loadPresent());
      presentData.current = d;
      const entry = (model !== 'M16a' && d.entries[model]) || d.entry;
      const { data } = b64ToFrames(entry.frames);
      setForms(entry.true_forces.map((t) => {
        const dir = t.direction as [number, number, number];
        return withDir({ u: t.u, v: t.v, magnitude: t.magnitude, duration: t.duration, direction: dir, ...azElFromDir(dir) });
      }));
      setPlaced([true, true]);
      setPinned(true);
      setModel(entry.model ?? 'M16a');
      setFrames(data);
      setFrame(0);
      setResult(entry);
      setCachedNote(false);
      setDemoRun(true);
      setJob(null);
      setCam(null);
      setStage('done');
      setLoop(false);
      setPlaying(true);
    } catch (e) {
      setJob({ id: '', progress: 0, error: `demo run: ${e}` });
    }
  }, [model]);

  // For the demo run the camera path is the recorded job, replayed tile by tile (~1 s each).
  const replayDemoCamera = useCallback(() => {
    const d = presentData.current;
    if (!d?.camera || cam) return;
    void replayCamera(d.camera.job, 1.0, setCam);
  }, [cam]);

  useEffect(() => {
    if (stage !== 'running') return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [stage]);

  useEffect(() => {
    if (PRESENT || !job || job.error || stage !== 'running') return;
    const t = setInterval(async () => {
      try {
        const s = await getJob(job.id);
        if (s.status === 'done' && s.result) {
          clearInterval(t);
          const { data } = b64ToFrames(s.result.frames);
          setFrames(data);
          // Play the pushes from the start (was: sit on the end state), 2026-11-03.
          setFrame(0);
          setResult(s.result);
          setCachedNote(!!s.result.cached);
          setDemoRun(false);
          setStage('done');
          setLoop(false);  // play once through (2026-11-03)
          setPlaying(true);
          setJob(null);
          // Paper models clamp the pushes to their training range: the cards and the true
          // arrows then show what was actually simulated (result.adjustments says what changed).
          if (s.result.adjustments?.length && s.result.true_forces.length === 2) {
            setForms(s.result.true_forces.map((t) => {
              const d = t.direction as [number, number, number];
              return withDir({ u: t.u, v: t.v, magnitude: t.magnitude, duration: t.duration, direction: d, ...azElFromDir(d) });
            }));
          }
          // The camera path no longer starts by itself (2026-11-03): see runCamera / the button.
        } else if (s.status === 'failed') {
          clearInterval(t);
          setJob({ id: job.id, progress: 1, error: s.error ?? 'job failed' });
          setStage('ready');
        } else {
          setJob({ id: job.id, progress: s.progress, startedAt: job.startedAt });
        }
      } catch (e) {
        clearInterval(t);
        setJob({ id: job.id, progress: 1, error: String(e) });
        setStage('ready');
      }
    }, POLL_MS);
    return () => clearInterval(t);
  }, [job, stage, forms]);

  // Camera-path job: poll until done/failed; the strip re-renders as stages arrive.
  useEffect(() => {
    if (PRESENT || !cam || !cam.id) return;
    const st = cam.job?.status;
    if (st === 'done' || st === 'failed') return;
    const id = cam.id;
    const t = setInterval(async () => {
      try {
        const j = await getCameraJob(id);
        setCam((c) => (c && c.id === id ? { ...c, job: j } : c));
        if (j.status === 'done' || j.status === 'failed') clearInterval(t);
      } catch (e) {
        clearInterval(t);
        setCam((c) => (c && c.id === id ? { ...c, job: { status: 'failed', progress: 1 }, error: String(e) } : c));
      }
    }, 2000);
    return () => clearInterval(t);
  }, [cam?.id, cam?.job?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!playing || !frames) return;
    const t = setInterval(() => {
      setFrame((f) => (f >= last ? (loop ? 0 : last) : f + 1));
    }, 100);
    return () => clearInterval(t);
  }, [playing, frames, loop, last]);

  useEffect(() => {
    if (frame >= last && !loop) setPlaying(false);
  }, [frame, loop, last]);

  // Presentation mode: load the static entry, show it on its last frame with the pushes placed
  // (stage 'ready' so Push is live), and expose window.__present for the capture script.
  useEffect(() => {
    if (!PRESENT) return;
    document.documentElement.dataset.theme = '';
    loadPresent().then((d) => {
      presentData.current = d;
      const { data } = b64ToFrames(d.entry.frames);
      const n = Math.max(1, Math.round(data.length / (64 * 64 * 3)));
      setFrames(data);
      setFrame(PRESENT_FLAT ? 0 : n - 1);
      noteFrame(PRESENT_FLAT ? 0 : n - 1);
      setForms(d.entry.true_forces.map((t) => {
        const dir = t.direction as [number, number, number];
        return withDir({ u: t.u, v: t.v, magnitude: t.magnitude, duration: t.duration, direction: dir, ...azElFromDir(dir) });
      }));
      setPlaced([true, true]);
      setPinned(true);
      setStage('ready');
      setShowTrue(false); setShowPred(false); setShowRev(false);
      setLoop(false);
    }).catch((e) => setJob({ id: '', progress: 0, error: `present data: ${e}` }));
    const onMove = (e: MouseEvent) => setCursor((c) => ({ ...c, x: e.clientX, y: e.clientY }));
    const onDown = () => setCursor((c) => ({ ...c, down: true }));
    const onUp = () => setCursor((c) => ({ ...c, down: false }));
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mousedown', onDown);
    window.addEventListener('mouseup', onUp);
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mousedown', onDown); window.removeEventListener('mouseup', onUp); };
  }, []);

  useEffect(() => {
    if (!PRESENT) return;
    const api: PresentApi = {
      setFrame: (k) => { const kk = Math.min(last, Math.max(0, k)); setPlaying(false); setFrame(kk); noteFrame(kk); },
      last: () => last,
      play: (once = true) => { setLoop(!once); setFrame(0); noteFrame(0); setPlaying(true); },
      pause: () => setPlaying(false),
      marks: (on) => setMarksOff(!on),
      flat: (on) => setFlat(on),
      panel: (on) => setPanelIn(on),
      cursor: (on) => setCursorOn(on),
      flyArrows: (ms = 900) => new Promise((resolve) => {
        setMarksOff(false);
        setFlyIn(0);
        const t0 = performance.now();
        const tick = () => {
          const u = Math.min(1, (performance.now() - t0) / ms);
          setFlyIn(1 - Math.pow(1 - u, 3)); // ease out
          if (u < 1) requestAnimationFrame(tick); else resolve();
        };
        requestAnimationFrame(tick);
      }),
      working: (on) => { if (on) { setJob({ id: 'present', progress: 0.3 }); setStage('running'); setResult(null); setCam(null); } else { setJob(null); setStage('ready'); } },
      result: (on) => {
        const d = presentData.current;
        if (on && d) { setResult(d.entry); setCachedNote(false); setStage('done'); setJob(null); setShowTrue(true); setShowPred(true); setShowRev(false); }
        else { setResult(null); setStage('ready'); }
      },
      arrows: (a) => { if (a.true != null) setShowTrue(a.true); if (a.pred != null) setShowPred(a.pred); if (a.rev != null) setShowRev(a.rev); },
      scrollPanel: (selector) => document.querySelector(selector)?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
      camera: (stepS = 1.5) => {
        const d = presentData.current;
        if (!d?.camera) return Promise.resolve();
        return replayCamera(d.camera.job, stepS, setCam);
      },
      lightbox: (stepKey) => {
        if (stepKey == null) (document.querySelector('.lightbox') as HTMLElement | null)?.click();
        else (document.querySelector(`.tile.${stepKey} .thumb`) as HTMLElement | null)?.click();
      },
      orbit: (deg, ms = 2000) => new Promise((resolve) => {
        const c = controlsRef.current;
        if (!c) { resolve(); return; }
        const cam = c.object;
        const axis = new THREE.Vector3(0, 0, 1);
        const total = (deg * Math.PI) / 180;
        const t0 = performance.now();
        let done = 0;
        const tick = () => {
          const u = Math.min(1, (performance.now() - t0) / ms);
          const target = total * (1 - Math.cos(Math.PI * u)) / 2; // ease in-out
          cam.position.applyAxisAngle(axis, target - done);
          done = target;
          c.update();
          if (u < 1) requestAnimationFrame(tick); else resolve();
        };
        requestAnimationFrame(tick);
      }),
      reset: () => { setResult(null); setCam(null); setJob(null); setStage('ready'); setFrame(last); noteFrame(last); setPlaying(false); setShowTrue(false); setShowPred(false); setShowRev(false); setPanelIn(false); },
      data: () => presentData.current,
    };
    (window as unknown as { __present: PresentApi }).__present = api;
    return bindKeys(api);
  });

  const tooClose = sep(forms[0], forms[1]) < 0.2;
  // After Push the pushes and the sheet are locked with the result (2026-11-03): editing a
  // card or clicking the sheet used to clear the result by accident. "Run another simulation"
  // resets the sheet and unlocks the cards, keeping the pushes as they are.
  const frozen = stage === 'running' || stage === 'done';
  const another = () => {
    setResult(null);
    setFrames(null);
    setFrame(0);
    setPlaying(false);
    setStage('ready');
    setJob(null);
    setCam(null);
    setDemoRun(false);
  };
  const reset = () => {
    setDemoRun(false);
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
    setCam(null);
  };

  const update = (k: number, patch: Partial<ForceForm>) => {
    if (frozen) return;
    select(k);
    setForms((fs) => fs.map((f, i) => (i === k ? withDir({ ...f, ...patch }) : f)));
  };

  return (
    <div className={`app${PRESENT ? ' present' : ''}${panelIn ? '' : ' panel-out'}`}>
      <div className="stage-wrap">
        <div className="scene">
          <SceneView
            frames={flat ? null : frames}
            frame={frame}
            stage={stage}
            placed={marksOff ? [false, false] : placed}
            allForms={forms}
            result={result ? { predicted: result.predicted_forces, reversal: result.reversal_forces, showTrue, showPred, showRev } : null}
            onPlace={onPlace}
            controlsRef={controlsRef}
            dark={dark}
            flyIn={flyIn}
            previewAsTrue={PRESENT}
          />
          {!PRESENT && (
            <div className="stage-hint">
              {hint(stage, active, placed, locked)}
            </div>
          )}
        </div>
        <StepStrip cam={cam} mesh={result} />
      </div>
      <div className="panel">
        <div className="topbar">
          <h2>Uncrumple</h2>
          <div className="spacer" />
          <button className="ghost" onClick={() => setDark((d) => !d)}>{dark ? 'Light' : 'Dark'}</button>
          <button className="ghost" onClick={() => controlsRef.current?.reset()}>Reset view</button>
        </div>

        {forms.map((f, k) => (
          <div className={`force-card c${k + 1}${k === active ? ' active' : ''}`} key={k}
            onPointerDownCapture={() => select(k)} onFocusCapture={() => select(k)}
            title={k === active ? 'Selected — sheet clicks move this push' : 'Click or edit to select this push'}>
            <h3>Push {k + 1}{k === active ? ' (selected)' : ''} — {frozen ? 'locked with the result' : locked[k] ? 'position locked' : `click sheet to ${placed[k] ? 'move' : 'place'}`}</h3>
            <div className="row"><label>position</label><output>({f.u.toFixed(2)}, {f.v.toFixed(2)})</output>
              <label title="Locked: sheet clicks never move this push">
                <input type="checkbox" checked={locked[k]} disabled={frozen}
                  onChange={(e) => setLocked((l) => l.map((x, i) => (i === k ? e.target.checked : x)))} /> lock
              </label>
            </div>
            <Slider disabled={frozen} label="azimuth" min={-180} max={180} step={1} value={f.az} unit="°"
              onChange={(v) => update(k, { az: v })} />
            <Slider disabled={frozen} label="elevation" min={-90} max={90} step={1} value={f.el} unit="°"
              onChange={(v) => update(k, { el: v })} />
            <div className="row"><label>direction</label>
              <output>[{f.direction.map((x) => x.toFixed(2)).join(', ')}]</output></div>
            <Slider disabled={frozen} label="magnitude" min={0.15} max={1.25} step={0.01} value={f.magnitude} unit=" N"
              onChange={(v) => update(k, { magnitude: v })} />
            <Slider disabled={frozen} label="duration" min={0.3} max={0.7} step={0.01} value={f.duration} unit=" s"
              onChange={(v) => update(k, { duration: v })} />
          </div>
        ))}
        {tooClose && <div className="warn">Patches are closer than 0.2 in (u, v) — move one.</div>}

        <div className="row" title={menu.find((m) => m.name === model)?.blurb}>
          <label htmlFor="model">model</label>
          <select id="model" value={model} disabled={frozen}
            onChange={(e) => setModel(e.target.value)}>
            {menu.map((m) => (
              <option key={m.name} value={m.name} disabled={!m.available}>
                {m.label}{m.available ? '' : ' (checkpoint not deployed)'}
              </option>
            ))}
          </select>
        </div>
        <div className="notice">{menu.find((m) => m.name === model)?.blurb}</div>

        <button onClick={push} disabled={stage !== 'ready' || tooClose || !allPlaced}>
          {stage === 'running' ? 'Working…' : 'Push'}
        </button>
        {stage !== 'running' && stage !== 'done' && !PRESENT && (
          <button className="highlight" onClick={showDemoRun}>View the demo video's run (precomputed, no wait)</button>
        )}
        {stage === 'running' && job?.startedAt && !job.error && (
          <div className="notice">
            {Math.max(0, Math.round((now - job.startedAt) / 1000))} s elapsed · typically ~{MODEL_MENU.find((m) => m.name === model)?.physics === 'paper' ? '2 min' : '20 s'} per push,
            1-2 min more if the server is starting up.
          </div>
        )}
        {job?.error && <div className="err">{job.error}</div>}
        {stage === 'done' && (
          <button onClick={another}>Run another simulation</button>
        )}
        {(stage === 'running' || stage === 'done') && (
          <button className="ghost" onClick={reset}>Start over</button>
        )}
        {stage === 'done' && result && !cam && !PRESENT && !demoRun && (
          <button className="ghost" onClick={runCamera}>Run the camera path on these pushes (~3-5 min)</button>
        )}
        {stage === 'done' && result && !cam && !PRESENT && demoRun && presentData.current?.camera && (
          <button className="ghost" onClick={replayDemoCamera}>Show the camera path for this run (recorded, ~10 s)</button>
        )}

        {frames && (
          <div>
            <h3>Playback</h3>
            <div className="row">
              <button className="ghost" onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</button>
              <input type="range" min={0} max={last} step={1} value={frame}
                onChange={(e) => setFrame(Number(e.target.value))} style={{ flex: 1 }} />
              <output>{frame}/{last}</output>
            </div>
            <div className="row toggles">
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
            {demoRun && (result.model ?? 'M16a') === 'M16a' && <div className="notice">The precomputed run from the demo video: pushes, simulation and prediction as recorded, no server call.</div>}
            {demoRun && result.model && result.model !== 'M16a' && <div className="notice">The demo video's pushes through {menu.find((m) => m.name === result.model)?.label ?? result.model}, precomputed on the live backend{result.source?.recorded ? ` (${result.source.recorded})` : ''}: no server call.</div>}
            {cachedNote && <div className="err">Not a live prediction: the simulation timed out or the GPU was busy, so this is the nearest
              precomputed result re-scored against your pushes. Press Start over and Push again once the server is warm.</div>}
            {result.model && result.model !== 'M16a' && (
              <div className="notice">{menu.find((m) => m.name === result.model)?.label ?? result.model} on the paper physics
                {result.adjustments?.length ? `. Pushes clamped to its training range: ${result.adjustments.join('; ')}.` : '.'}</div>
            )}
            <div className="legend">
              <span><span className="swatch" style={{ background: ACCENTS[0] }} />true 1</span>
              <span><span className="swatch" style={{ background: ACCENTS[1] }} />true 2</span>
              <span><span className="swatch" style={{ background: REVERSAL }} />reversal</span>
              <span><span className="swatch faded" style={{ background: ACCENTS[0] }} /><span className="swatch faded" style={{ background: ACCENTS[1] }} />predicted (faded)</span>
            </div>
            <p className="notice">Solid arrows are the pushes you set; the faded arrows next to them are the model's guess of each push from the end state alone.</p>
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
                <tr title="What this model scores on average over its held-out test set. Your numbers near these are normal.">
                  <td>typical ({menu.find((m) => m.name === result.model)?.label ?? result.model ?? 'M16-a'}, test set)</td><td>{result.metrics.reference.pos_cm.toFixed(1)}</td>
                  <td>{result.metrics.reference.dir_deg.toFixed(0)}</td>
                  <td>{result.metrics.reference.impulse_rel == null ? '–' : `${(100 * result.metrics.reference.impulse_rel).toFixed(0)}%`}</td>
                </tr>
                {cam?.job?.result && (
                  <tr title="The camera path under the canvas: the same pushes re-simulated on paper, photographed, and read back with SAM 3.1, VGGT and the print decoder before M21-c guesses. Typical: 7.4 cm / 17°.">
                    <td>from the camera</td><td>{cam.job.result.camera.metrics.mean.pos_cm.toFixed(1)}</td>
                    <td>{cam.job.result.camera.metrics.mean.dir_deg.toFixed(0)}</td>
                    <td>{(100 * cam.job.result.camera.metrics.mean.impulse_rel).toFixed(0)}%</td>
                  </tr>
                )}
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
          <p>6. Under the canvas, the same pushes go through a camera: paper sheet, two photos, SAM 3.1 finds the sheet,
            VGGT gives depth, the printed code gives correspondences, M21-c guesses from that grid, Muse Spark writes the verdict.</p>
          <p>Who benefits: robots handling garments can infer the grasp that crumpled them.
            Sheet-goods lines can plan pushes or undo unwanted deformations.
            The method recovers unseen loads from observed shape alone.</p>
        </details>

        <footer className="stack">
          PyTorch · Newton/Warp VBD sim · M16-a cnn_reg+hybrid (1.0 cm / 4.3° / 0.156) · camera path: SAM 3.1 + VGGT + M21-c + Muse Spark · React + three.js
        </footer>
      </div>
      {PRESENT && (
        <div className={`fake-cursor${cursor.down ? ' down' : ''}`} style={{ left: cursor.x, top: cursor.y, display: cursorOn ? undefined : 'none' }}>
          <svg width="22" height="30" viewBox="0 0 22 30"><path d="M1 1 L1 23 L6.5 18 L10 27 L14 25.5 L10.5 17 L18 17 Z" fill="#fff" stroke="#000" strokeWidth="1.4" strokeLinejoin="round" /></svg>
        </div>
      )}
    </div>
  );
}

function Slider({ label, min, max, step, value, unit, onChange, disabled }: {
  label: string; min: number; max: number; step: number; value: number; unit: string;
  onChange: (v: number) => void; disabled?: boolean;
}) {
  return (
    <div className="row">
      <label>{label}</label>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))} />
      <output>{value.toFixed(step < 1 ? 2 : 0)}{unit}</output>
    </div>
  );
}
