"""Modal deployment for the demo backend. Modal-only code lives here;
``app.py`` runs unchanged under plain ``uvicorn`` (and later Baseten).

Layout on the container mirrors the repo (``/app/demo/backend`` +
``/app/cloth_data_generation/...``) so ``app.py``'s repo-relative imports
work untouched. The checkpoint arrives via a Volume, not the image.

One-time setup (Nick, from WSL):
    modal setup                                            # first time only
    modal volume create cloth-demo-checkpoints
    modal volume put cloth-demo-checkpoints \
        demo/backend/checkpoints/M16a/model.pt M16a/model.pt
    modal volume put cloth-demo-checkpoints \
        demo/backend/checkpoints/M16a/metrics.json M16a/metrics.json

Camera path (POST /camera, camera.py), one-time additions:
    modal volume put cloth-demo-checkpoints \
        demo/backend/checkpoints/M21c/model.pt M21c/model.pt
    modal secret create meta-api META_API_KEY=<key>          # hosted SAM 3.1
VGGT-1B (~5 GB) downloads to the volume (HF_HOME=/vol/hf) on the first camera job.

Deploy:
    cd <repo> && modal deploy demo/backend/modal_app.py     # -> https URL
"""

from __future__ import annotations

import os

import modal

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

image = (
    modal.Image.debian_slim(python_version="3.12")  # matches the verified WSL env (numpy 2.5.x needs >=3.12)
    .apt_install("git")
    .pip_install(
        "fastapi==0.162.0",
        "uvicorn==0.54.0",
        "numpy==2.5.4",
        "h5py==3.16.0",
        "torch==2.11.0",
        "newton==1.7.0",
        "warp-lang==1.18.0",
        "warp-nn==0.4.1",
        # camera path: versions of the Bouchet sam3env the chain was measured with (RESULTS.md M21)
        "torchvision==0.26.0",
        "scipy==1.18.1",
        "pillow",
        "psutil==7.2.2",
        "einops==0.8.2",
        "safetensors==0.8.0",
        "huggingface_hub==2.4.0",
    )
    # vggt pins numpy<2 in its pyproject, but the chain was measured with numpy 2.5.4
    # (sam3env has vggt next to numpy 2.5.4); its real deps are pinned above.
    .pip_install("vggt @ git+https://github.com/facebookresearch/vggt@a288dd0",
                 extra_options="--no-deps")
    .add_local_dir(os.path.join(_REPO, "demo", "backend"), "/app/demo/backend")
    .add_local_dir(os.path.join(_REPO, "cloth_data_generation", "transfer", "cloth_sim"),
                   "/app/cloth_data_generation/transfer/cloth_sim")
    .add_local_file(os.path.join(_REPO, "cloth_data_generation", "transfer",
                                 "generate_sample.py"),
                    "/app/cloth_data_generation/transfer/generate_sample.py")
    .add_local_file(os.path.join(_REPO, "cloth_data_generation", "transfer",
                                 "generate_batch.py"),
                    "/app/cloth_data_generation/transfer/generate_batch.py")
    .add_local_dir(os.path.join(_REPO, "cloth_data_generation", "inverse"),
                   "/app/cloth_data_generation/inverse",
                   ignore=["runs", "__pycache__"])
)

app = modal.App("cloth-demo")
vol = modal.Volume.from_name("cloth-demo-checkpoints", create_if_missing=True)


# Scale to zero when idle (no warm container): the first request after idle
# pays the cold start (container + Warp + checkpoint, ~1-2 min). Credits over
# latency for now.
@app.function(image=image, gpu="A10G", min_containers=0, scaledown_window=120,
              timeout=600, volumes={"/vol": vol},
              secrets=[modal.Secret.from_name("meta-api")])
@modal.asgi_app()
def web():
    import sys  # noqa: E402
    sys.path.insert(0, "/app/demo/backend")
    os.environ.setdefault("CKPT_DIR", "/vol/M16a")
    os.environ.setdefault("CAMERA_CKPT", "/vol/M21c")
    os.environ.setdefault("HF_HOME", "/vol/hf")
    os.environ.setdefault(
        "CORS_ORIGINS",
        "http://localhost:5173,https://cloth-inverse.vercel.app,"
        "https://cloth-inverse-js3blwak2-nick-armstrong-s-projects.vercel.app")
    from app import create_app  # noqa: E402
    return create_app()
