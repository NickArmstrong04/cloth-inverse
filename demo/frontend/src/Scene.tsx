import { useEffect, useMemo, useRef, type RefObject } from 'react';
import { Canvas, type ThreeEvent } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import * as THREE from 'three';
import type { ForceEx } from './api';

export const GRID = 64;
export const SHEET = 0.5;
export const ACCENTS = ['#ff5a5a', '#3aa0ff'];
export const REVERSAL = '#37d67a';

export interface PlacedForce {
  u: number;
  v: number;
  direction: [number, number, number];
  magnitude: number;
  duration: number;
}

export type Stage = 'place1' | 'place2' | 'ready' | 'running' | 'done';

function gridPos(frames: Float32Array | null, frame: number, i: number, j: number, out: THREE.Vector3): void {
  if (!frames) {
    out.set(((i / (GRID - 1)) - 0.5) * SHEET, ((j / (GRID - 1)) - 0.5) * SHEET, 0);
    return;
  }
  const o = (frame * GRID * GRID + j * GRID + i) * 3;
  out.set(frames[o], frames[o + 1], frames[o + 2]);
}

function SheetMesh({
  frames,
  frame,
  stage,
  onPlace,
}: {
  frames: Float32Array | null;
  frame: number;
  stage: Stage;
  onPlace: (u: number, v: number) => void;
}) {
  const mesh = useRef<THREE.Mesh>(null);
  const picking = stage !== 'running';

  const { positions, colors, uvs, index } = useMemo(() => {
    const n = GRID * GRID;
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const uvs = new Float32Array(n * 2);
    for (let j = 0; j < GRID; j++) {
      for (let i = 0; i < GRID; i++) {
        const k = j * GRID + i;
        uvs[k * 2] = i / (GRID - 1); // material u, valid from either face
        uvs[k * 2 + 1] = j / (GRID - 1); // material v
      }
    }
    const index: number[] = [];
    for (let j = 0; j < GRID - 1; j++) {
      for (let i = 0; i < GRID - 1; i++) {
        const a = j * GRID + i;
        const b = a + 1;
        const c = a + GRID;
        const d = c + 1;
        index.push(a, c, b, b, c, d);
      }
    }
    return { positions, colors, uvs, index };
  }, []);

  useEffect(() => {
    const m = mesh.current;
    if (!m) return;
    const g = m.geometry;
    const v = new THREE.Vector3();
    let zmin = Infinity;
    let zmax = -Infinity;
    const zs = new Float32Array(GRID * GRID);
    for (let j = 0; j < GRID; j++) {
      for (let i = 0; i < GRID; i++) {
        gridPos(frames, frame, i, j, v);
        const k = j * GRID + i;
        positions[k * 3] = v.x;
        positions[k * 3 + 1] = v.y;
        positions[k * 3 + 2] = v.z;
        zs[k] = v.z;
        if (v.z < zmin) zmin = v.z;
        if (v.z > zmax) zmax = v.z;
      }
    }
    const span = Math.max(zmax - zmin, 1e-6);
    const cA = new THREE.Color('#2f6fed');
    const cB = new THREE.Color('#e8e4da');
    const cC = new THREE.Color('#e2483d');
    const c = new THREE.Color();
    for (let k = 0; k < GRID * GRID; k++) {
      const t = (zs[k] - zmin) / span;
      if (t < 0.5) c.lerpColors(cA, cB, t * 2);
      else c.lerpColors(cB, cC, (t - 0.5) * 2);
      colors[k * 3] = c.r;
      colors[k * 3 + 1] = c.g;
      colors[k * 3 + 2] = c.b;
    }
    g.attributes.position.needsUpdate = true;
    g.attributes.color.needsUpdate = true;
    g.computeVertexNormals();
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1);
  }, [frames, frame, positions, colors]);

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    if (!picking) return;
    e.stopPropagation();
    if (!e.uv) return;
    onPlace(THREE.MathUtils.clamp(e.uv.x, 0.02, 0.98), THREE.MathUtils.clamp(e.uv.y, 0.02, 0.98));
  };

  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    g.setIndex(index);
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1);
    return g;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => () => geom.dispose(), [geom]);

  return (
    <mesh ref={mesh} geometry={geom} onClick={handleClick} frustumCulled={false}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} roughness={0.85} metalness={0} />
    </mesh>
  );
}

function Marker({ at, color }: { at: THREE.Vector3; color: string }) {
  return (
    <mesh position={at}>
      <sphereGeometry args={[0.008, 16, 16]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

function ForceArrow({
  at,
  dir,
  impulse,
  color,
  opacity,
  incoming = false,
}: {
  at: THREE.Vector3;
  dir: [number, number, number];
  impulse: number;
  color: string;
  opacity: number;
  /** Head on the anchor, tail back along -dir: a push arriving at the dot. */
  incoming?: boolean;
}) {
  const d = useMemo(() => new THREE.Vector3(...dir).normalize(), [dir]);
  const len = Math.max(0.03, 0.34 * impulse);
  const quat = useMemo(() => new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d), [d]);
  const pos = useMemo(
    () => at.clone().addScaledVector(d, incoming ? -(len + 0.009) : 0.004),
    [at, d, incoming, len],
  );
  const shaftLen = len * 0.72;
  const headLen = len * 0.28;
  return (
    <group position={pos} quaternion={quat}>
      <mesh position={[0, shaftLen / 2, 0]}>
        <cylinderGeometry args={[0.0035, 0.0035, shaftLen, 10]} />
        <meshBasicMaterial color={color} transparent={opacity < 1} opacity={opacity} />
      </mesh>
      <mesh position={[0, shaftLen + headLen / 2, 0]}>
        <coneGeometry args={[0.009, headLen, 14]} />
        <meshBasicMaterial color={color} transparent={opacity < 1} opacity={opacity} />
      </mesh>
    </group>
  );
}

function anchorOf(frames: Float32Array | null, frame: number, u: number, v: number): THREE.Vector3 {
  const i = Math.min(GRID - 1, Math.max(0, Math.round(u * (GRID - 1))));
  const j = Math.min(GRID - 1, Math.max(0, Math.round(v * (GRID - 1))));
  const out = new THREE.Vector3();
  gridPos(frames, frame, i, j, out);
  return out;
}

export function SceneView({
  frames,
  frame,
  stage,
  placed,
  allForms,
  result,
  onPlace,
  controlsRef,
  dark,
}: {
  frames: Float32Array | null;
  frame: number;
  stage: Stage;
  placed: boolean[];
  allForms: PlacedForce[];
  result: { predicted: ForceEx[]; reversal: ForceEx[]; showTrue: boolean; showPred: boolean; showRev: boolean } | null;
  onPlace: (u: number, v: number) => void;
  controlsRef: RefObject<OrbitControlsImpl>;
  dark: boolean;
}) {
  const last = frames ? 29 : 0;
  const markers = useMemo(
    () => allForms.map((f, k) => (placed[k] ? { at: anchorOf(frames, 0, f.u, f.v), color: ACCENTS[k % 2] } : null)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [allForms, placed, frames === null],
  );
  // Before a result: one arrow per placed push arriving at its marker, length
  // proportional to the impulse (magnitude x duration), so the chosen angle
  // and strength are visible while the sliders move.
  const previewArrows = useMemo(() => {
    if (result) return [];
    return allForms.map((f, k) =>
      placed[k]
        ? { at: anchorOf(null, 0, f.u, f.v), dir: f.direction, impulse: f.magnitude * f.duration, color: ACCENTS[k % 2], opacity: 0.95 }
        : null,
    );
  }, [result, allForms, placed]);
  const arrows = useMemo(() => {
    if (!result) return [];
    const items: { at: THREE.Vector3; dir: [number, number, number]; impulse: number; color: string; opacity: number }[] = [];
    result.predicted.forEach((f, k) => {
      if (!result.showPred) return;
      items.push({ at: anchorOf(frames, last, f.u, f.v), dir: f.direction, impulse: f.impulse, color: ACCENTS[k % 2], opacity: 0.45 });
    });
    result.reversal.forEach((f) => {
      if (!result.showRev) return;
      items.push({ at: anchorOf(frames, last, f.u, f.v), dir: f.direction, impulse: f.impulse, color: REVERSAL, opacity: 0.9 });
    });
    return items;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result, frames === null]);

  const trueArrows = useMemo(() => {
    if (!result || !result.showTrue) return [];
    return allForms.map((f, k) => ({
      at: anchorOf(frames, last, f.u, f.v),
      dir: f.direction,
      impulse: f.magnitude * f.duration,
      color: ACCENTS[k % 2],
      opacity: 1,
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result, allForms, frames === null]);

  return (
    <Canvas camera={{ position: [0.58, -0.72, 0.7], fov: 42 }} style={{ background: dark ? '#101418' : '#eef1f5' }}>
      <ambientLight intensity={0.9} />
      <directionalLight position={[0.6, -0.4, 0.9]} intensity={1.6} />
      <directionalLight position={[-0.5, 0.5, -0.6]} intensity={0.35} />
      <SheetMesh frames={frames} frame={frame} stage={stage} onPlace={onPlace} />
      {markers.map((m, k) => m && <Marker key={k} at={m.at} color={m.color} />)}
      {previewArrows.map((a, k) => a && <ForceArrow key={`v${k}`} {...a} incoming />)}
      {trueArrows.map((a, k) => (
        <ForceArrow key={`t${k}`} {...a} />
      ))}
      {arrows.map((a, k) => (
        <ForceArrow key={`p${k}`} {...a} />
      ))}
      <OrbitControls ref={controlsRef} makeDefault enablePan enableDamping />
    </Canvas>
  );
}
