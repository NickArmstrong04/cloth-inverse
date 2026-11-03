# Demo frontend

React 18 + Vite + TypeScript + three.js (fiber/drei). Single page: free-camera
3D sheet, force cards, Push → 30-frame playback, true/predicted/reversal
arrows, metrics with M16-a references, and the "under the hood" step strip (photo -> SAM 3.1 ->
VGGT -> M21-c -> Muse Spark) fed by POST /camera. Live site: https://cloth-inverse.vercel.app

## Run against the local backend

```bash
cd demo/frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev   # -> http://localhost:5173/
```

(`VITE_API_URL` defaults to `http://localhost:8000` when unset.)

## Production build / Vercel

```bash
npm run build   # tsc + vite build -> dist/
```

Deploy: `vercel` from `demo/frontend/`, or import the repo in the
Vercel dashboard with root directory `demo/frontend`. Set the
`VITE_API_URL` environment variable to the Modal https URL that `modal deploy`
prints (set on the Vercel project), then redeploy. `vercel.json` holds the SPA fallback.
