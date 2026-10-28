"""Save one late frame of the force GIF as PNG for inspection."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # run from transfer/: python tools/<name>.py
from PIL import Image

im = Image.open(
    "/home/narmstrong/cloth/sample_seed0_forces.gif")
print("frames:", im.n_frames)
im.seek(im.n_frames - 1)
im.save("/home/narmstrong/cloth/force_last_frame.png")
print("saved last frame")
