import { useEffect, useState } from 'react';
import { ACCENTS, MESH_PRED } from './Scene';
import {
  getCameraJob, getGalleryEntry, getGalleryIndex, postCamera,
  type CameraResult, type ForceInput, type GalleryIndex, type Metric,
} from './api';

const POLL_MS = 3000;

const STAGE_TEXT: Record<string, string> = {
  queued: 'Waiting for the GPU…',
  simulating: 'Simulating your pushes on paper until it comes to rest (about 2 minutes)…',
  camera: 'Photographing the sheet, SAM 3.1 mask, VGGT depth, decoding the print…',
  predicting: 'Running the inverse model…',
};

function pct(x: number | null | undefined): string {
  return x == null ? '–' : `${Math.round(100 * x)}%`;
}

function Row({ label, m, title }: { label: string; m: Metric; title: string }) {
  return (
    <tr title={title}>
      <td>{label}</td><td>{m.pos_cm.toFixed(1)}</td><td>{m.dir_deg.toFixed(0)}</td>
      <td>{(100 * m.impulse_rel).toFixed(0)}%</td>
    </tr>
  );
}

/** Camera mode: precomputed gallery (instant) plus a live run of the user's pushes. */
export function CameraPanel({
  entry,
  onEntry,
  view,
  setView,
  showMesh,
  setShowMesh,
  forms,
}: {
  entry: CameraResult | null;
  onEntry: (e: CameraResult) => void;
  view: 'camera' | 'true';
  setView: (v: 'camera' | 'true') => void;
  showMesh: boolean;
  setShowMesh: (b: boolean) => void;
  forms: ForceInput[];
}) {
  const [index, setIndex] = useState<GalleryIndex | null>(null);
  const [pos, setPos] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [live, setLive] = useState<{ id: string; status: string; t0: number } | null>(null);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    getGalleryIndex().then(setIndex).catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (!index || live) return;
    const id = index.entries[pos]?.id;
    if (!id) return;
    getGalleryEntry(id).then(onEntry).catch((e) => setErr(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, pos]);

  useEffect(() => {
    if (!live) return;
    const t = setInterval(async () => {
      setNow(Date.now());
      try {
        const s = await getCameraJob(live.id);
        if (s.status === 'done' && s.result) {
          clearInterval(t);
          onEntry(s.result);
          setLive(null);
        } else if (s.status === 'failed') {
          clearInterval(t);
          setErr(`${s.error ?? 'live run failed'} — the gallery above is unaffected.`);
          setLive(null);
        } else {
          setLive((l) => (l ? { ...l, status: s.status } : l));
        }
      } catch (e) {
        clearInterval(t);
        setErr(String(e));
        setLive(null);
      }
    }, POLL_MS);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live?.id]);

  const runLive = async () => {
    setErr(null);
    try {
      const id = await postCamera(forms);
      setLive({ id, status: 'queued', t0: Date.now() });
    } catch (e) {
      setErr(String(e));
    }
  };

  const n = index?.n ?? 0;
  const go = (d: number) => n && setPos((p) => (p + d + n) % n);

  return (
    <div className="camera">
      <p className="notice">
        Here the model never sees the simulator's mesh. It gets what a camera would: a stereo photo pair of
        printed paper. SAM 3.1 (Meta Model API) finds the sheet, VGGT (Meta, open weights) measures depth,
        the printed code says which point of the sheet each pixel is, and M21-c recovers the two pushes.
      </p>

      <div className="row">
        <button className="ghost" onClick={() => go(-1)} disabled={!n || !!live}>◀</button>
        <output style={{ flex: 1, textAlign: 'center' }}>
          {entry?.live ? 'your live run' : n ? `test sheet ${pos + 1} of ${n}` : 'loading gallery…'}
        </output>
        <button className="ghost" onClick={() => go(1)} disabled={!n || !!live}>▶</button>
      </div>

      {entry && (
        <>
          <div className="tiles">
            <figure>
              <img src={`data:image/jpeg;base64,${entry.images.photo}`} alt="Synthetic photo of the printed sheet" />
              <figcaption>photo (left of the stereo pair)</figcaption>
            </figure>
            {entry.images.mask && (
              <figure>
                <img src={`data:image/jpeg;base64,${entry.images.mask}`} alt="Segmentation mask over the photo" />
                <figcaption>{entry.mask_source}</figcaption>
              </figure>
            )}
          </div>
          <div className="row">
            <label>3D view</label>
            <label><input type="radio" checked={view === 'camera'} onChange={() => setView('camera')} /> camera's sheet</label>
            <label><input type="radio" checked={view === 'true'} onChange={() => setView('true')} /> true sheet</label>
          </div>
          <div className="row">
            <label><input type="checkbox" checked={showMesh} onChange={(e) => setShowMesh(e.target.checked)} /> show perfect-information guess</label>
          </div>
          <div className="legend">
            <span><span className="swatch" style={{ background: ACCENTS[0] }} />true / camera guess 1</span>
            <span><span className="swatch" style={{ background: ACCENTS[1] }} />true / camera guess 2</span>
            <span><span className="swatch" style={{ background: MESH_PRED }} />from the true mesh</span>
          </div>
          <table className="metrics">
            <thead><tr>
              <th />
              <th title="Distance on the 50 cm sheet between the true and the predicted push, in centimetres.">location off<br /><small>cm</small></th>
              <th title="Angle between the true and the predicted push direction.">angle off<br /><small>degrees</small></th>
              <th title="Predicted push strength (force x duration) vs the true one.">strength off<br /><small>% of true</small></th>
            </tr></thead>
            <tbody>
              <Row label="from the camera" m={entry.camera.metrics.mean}
                title="M21-c on the grid rebuilt from the photos (SAM 3.1 + VGGT + print decode)." />
              <Row label="from the true mesh" m={entry.mesh.metrics.mean}
                title="The same model given the simulator's exact sheet: the perfect-information limit." />
              <tr title="M21-c over 175 camera views of 100 held-out sheets (hosted SAM 3.1, RESULTS.md M21).">
                <td>typical camera</td><td>{entry.reference.camera.pos_cm.toFixed(1)}</td>
                <td>{entry.reference.camera.dir_deg.toFixed(0)}</td><td>–</td>
              </tr>
            </tbody>
          </table>
          <div className="notice">
            The camera sees {pct(entry.coverage.camera_sees)} of the sheet; the print was decoded on {pct(entry.coverage.decoded)}.
            The hidden side is filled in smoothly, which is most of the gap to the true mesh.
            {entry.coverage.scale_fix && entry.coverage.scale_fix !== 1 ? ` Depth scale corrected ×${entry.coverage.scale_fix.toFixed(2)} from the sheet's known size.` : ''}
          </div>
          {entry.adjustments && entry.adjustments.length > 0 && (
            <div className="notice">Adjusted to the paper training range: {entry.adjustments.join('; ')}.</div>
          )}
          {entry.timings && (
            <div className="notice">
              {Object.entries(entry.timings).map(([k, v]) => `${k.replace(/_s$/, '')} ${v.toFixed(0)} s`).join(' · ')}
            </div>
          )}
        </>
      )}

      <button onClick={runLive} disabled={!!live}>
        {live ? 'Running live…' : 'Run the camera path live on my two pushes'}
      </button>
      {live && (
        <div className="notice">
          {STAGE_TEXT[live.status] ?? live.status} {Math.round((now - live.t0) / 1000)} s
          (about 3 minutes in all; longer if the GPU was asleep).
        </div>
      )}
      {!live && (
        <p className="notice">
          Uses the positions, magnitudes and durations of your two pushes (Mesh tab). Paper creases under opposing
          pushes, so each push is turned to within 30° of the other and tilted at most 35°, as in training.
        </p>
      )}
      {err && <div className="err">{err}</div>}
      {index && (
        <p className="notice">
          Gallery: the first {index.n} held-out sheets in name order (no picking). Mean over them: camera{' '}
          {index.summary.camera.pos_cm.toFixed(1)} cm / {index.summary.camera.dir_deg.toFixed(0)}°, true mesh{' '}
          {index.summary.mesh.pos_cm.toFixed(1)} cm / {index.summary.mesh.dir_deg.toFixed(0)}°.
        </p>
      )}
    </div>
  );
}
