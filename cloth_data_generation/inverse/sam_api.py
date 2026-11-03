"""Hosted SAM 3.1 (Meta Model API) client and mask decoder, pure Python + numpy.

The API (`POST https://api.meta.ai/v1/responses`, model `sam-3.1`) answers with a special-token
lane, one record per instance:

  <0f>0<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H|><|mask;x=0;y=0;data=<mh>,<mw>,<enc>payload|>,1<|box..

The mask is an mh x mw raster at the box's top-left (mh = y2 - y1 + 1, inclusive box). We request
`mask_encoding: one_bit` (`!` payloads): an adaptive binary range coder over a 12-pixel causal
context, packed in a base85 variant. This is a port of `decodeRaster` / `unpack` from the
npm package @meta-sam/parser 0.1.0 (dist/mask-codec.js, SAM License); the `~` lossless
encoding is not ported. The PyPI `meta-sam-parser` 0.0.1 is an empty placeholder.

Key from the environment: META_API_KEY (never written to disk by this module).

  python sam_api.py --png view.png --prompt fabric          # prints instances, writes nothing
  python sam_api.py --photos ../../cloth_results/photos/rest20_coded   # masks for the photo chain
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import urllib.request

import numpy as np

URL = "https://api.meta.ai/v1/responses"

_EXCLUDED = set('"\\,;<>|')
_ALPHABET = "".join(chr(c) for c in range(0x21, 0x7F) if chr(c) not in _EXCLUDED)
_VALUE = {ch: i for i, ch in enumerate(_ALPHABET[:85])}
_M32 = 0xFFFFFFFF
_TOP, _BOTTOM = 1 << 24, 1 << 16


def _unpack(payload: str) -> bytes:
    """Base85 variant: 5-digit length prefix, 4 bytes per 5 digits, short tail."""
    digit = _VALUE.__getitem__
    length = 0
    for ch in payload[:5]:
        length = length * 85 + digit(ch)
    rem = length % 4
    expected = 5 + (length // 4) * 5 + (rem + 1 if rem else 0)
    if len(payload) != expected:
        raise ValueError(f"mask payload length {len(payload)} != {expected} from its prefix")
    out = bytearray()
    src = 5
    for _ in range(length // 4):
        v = 0
        for ch in payload[src:src + 5]:
            v = v * 85 + digit(ch)
        out += v.to_bytes(4, "big")
        src += 5
    if rem:
        v = 0
        for i in range(5):
            v = v * 85 + (digit(payload[src + i]) if i < rem + 1 else 84)
        out += v.to_bytes(4, "big")[:rem]
    return bytes(out)


class _RangeDecoder:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 4
        self.low, self.range, self.scaled = 0, _M32, 0
        self.code = int.from_bytes(data[:4], "big")

    def frequency(self, total: int) -> int:
        self.scaled = self.range // total
        if self.scaled == 0:
            raise ValueError("mask payload has invalid coding state")
        v = ((self.code - self.low) & _M32) // self.scaled
        return total - 1 if v >= total else v

    def decode(self, cum: int, freq: int) -> None:
        self.low = (self.low + self.scaled * cum) & _M32
        self.range = self.scaled * freq
        while True:
            if (self.low ^ ((self.low + self.range) & _M32)) < _TOP:
                pass
            elif self.range < _BOTTOM:
                self.range = (-self.low) & (_BOTTOM - 1)
            else:
                break
            self.code = ((self.code << 8) | self.data[self.pos]) & _M32
            self.pos += 1
            self.low = (self.low << 8) & _M32
            self.range = (self.range << 8) & _M32


def decode_one_bit(payload: str, width: int, height: int) -> np.ndarray:
    """`!`-payload -> (height, width) uint8 0/1 raster (row-major, as the npm decoder)."""
    if not payload.startswith("!"):
        raise ValueError("one_bit payloads start with '!' (lossless '~' is not ported)")
    dec = _RangeDecoder(_unpack(payload[1:]))
    counts = [1] * (2 * 4096)
    pw = width + 4                       # 2 columns of padding each side, 2 rows on top
    p = bytearray(pw * (height + 2))
    out = np.zeros(height * width, np.uint8)
    for y in range(height):
        row = (y + 2) * pw + 2
        for x in range(width):
            q = row + x
            a, b = q - pw, q - 2 * pw
            ctx = (p[q - 1] | p[q - 2] << 1 | p[a - 2] << 2 | p[a - 1] << 3 | p[a] << 4
                   | p[a + 1] << 5 | p[a + 2] << 6 | p[b - 2] << 7 | p[b - 1] << 8 | p[b] << 9
                   | p[b + 1] << 10 | p[b + 2] << 11)
            o = ctx << 1
            zero, one = counts[o], counts[o + 1]
            bit = 0 if dec.frequency(zero + one) < zero else 1
            if bit:
                dec.decode(zero, one)
            else:
                dec.decode(0, zero)
            p[q] = bit
            out[y * width + x] = bit
            counts[o + bit] += 14
            if counts[o] + counts[o + 1] >= 4096:
                counts[o] = counts[o] >> 1 or 1
                counts[o + 1] = counts[o + 1] >> 1 or 1
    return out.reshape(height, width)


_RECORD = re.compile(r"(\d+)<\|box;([^|]*)\|><\|mask;([^|]*)\|>")


def parse_output(text: str) -> list[dict]:
    """Special-token lane of one image -> [{id, box (x1,y1,x2,y2 inclusive), mask (H,W) bool}]."""
    insts = []
    for oid, box, mask in _RECORD.findall(text):
        bf = dict(kv.split("=", 1) for kv in box.split(";"))
        mf = dict(kv.split("=", 1) for kv in mask.split(";"))
        x1, y1, x2, y2, W, H = (int(bf[k]) for k in ("x1", "y1", "x2", "y2", "w", "h"))
        mh, mw, payload = mf["data"].split(",", 2)
        crop = decode_one_bit(payload, int(mw), int(mh))
        full = np.zeros((H, W), bool)
        full[y1:y1 + int(mh), x1:x1 + int(mw)] = crop.astype(bool)
        insts.append({"id": int(oid), "box": (x1, y1, x2, y2), "mask": full,
                      "conf": float(bf["c"]) if "c" in bf else None})
    return insts


def segment(png: str, prompt: str, key: str | None = None, timeout: float = 60) -> tuple[list[dict], float]:
    """One hosted SAM 3.1 call on a local PNG/JPEG. Returns (instances, wall seconds)."""
    key = key or os.environ["META_API_KEY"]
    mime = "image/png" if png.lower().endswith(".png") else "image/jpeg"
    with open(png, "rb") as f:
        url = f"data:{mime};base64," + base64.b64encode(f.read()).decode()
    body = {"model": "sam-3.1", "stream": False, "metadata": {"mask_encoding": "one_bit"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": url}]}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    dt = time.time() - t0
    text = "".join(c.get("text", "") for o in resp.get("output", []) for c in o.get("content", []))
    return parse_output(text), dt


FALLBACK = ["fabric", "a piece of cloth", "cloth"]


def write_masks(photos: str, prompts: list[str]) -> dict:
    """segment.py-compatible masks for a photo.py set: <photos>/sam3/<stem>__fabric.png (0/255, the
    largest instance; the file name keeps 'fabric' because lift_stereo / decode_print /
    grid_from_views read that name). Prompts are tried in order until one finds something."""
    from PIL import Image
    import glob as _glob
    out = os.path.join(photos, "sam3")
    os.makedirs(out, exist_ok=True)
    lp = os.path.join(out, "sam31_api_log.json")
    log = json.load(open(lp)) if os.path.exists(lp) else {}
    for png in sorted(_glob.glob(os.path.join(photos, "*_v[0-9]*.png"))):
        stem = os.path.basename(png)[:-4]
        if not re.fullmatch(r".*_v\d+", stem):
            continue
        if os.path.exists(os.path.join(out, f"{stem}__fabric.png")):
            continue  # resume: mask written by an earlier run
        for p in prompts:
            for attempt in range(4):  # the API occasionally stalls past the read timeout
                try:
                    insts, dt = segment(png, p)
                    break
                except (TimeoutError, OSError) as e:
                    print(f"{stem} {p}: {type(e).__name__}, retry {attempt + 1}", flush=True)
                    time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(f"{stem}: SAM 3.1 API failed 4 times")
            if insts:
                break
        log[stem] = {"prompt": p, "k": len(insts), "s": round(dt, 2)}
        if insts:
            m = max(insts, key=lambda i: int(i["mask"].sum()))["mask"]
            Image.fromarray(m.astype(np.uint8) * 255).save(os.path.join(out, f"{stem}__fabric.png"))
        print(stem, log[stem], flush=True)
        with open(lp, "w") as f:
            json.dump(log, f, indent=1)
    with open(os.path.join(out, "sam31_api_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png")
    ap.add_argument("--photos", help="photo.py set: write sam3/<stem>__fabric.png for every view")
    ap.add_argument("--prompt", default="fabric")
    a = ap.parse_args()
    if a.photos:
        write_masks(a.photos, [a.prompt] + [p for p in FALLBACK if p != a.prompt])
        return
    insts, dt = segment(a.png, a.prompt)
    print(f"{len(insts)} instance(s) in {dt:.2f} s")
    for i in insts:
        print(f"  id {i['id']} box {i['box']} mask px {int(i['mask'].sum())}")


if __name__ == "__main__":
    main()
