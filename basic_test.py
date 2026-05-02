"""Standalone smoke test for EmbedSeg.

Loads ``BranchedERFNet`` the same way ``server.py`` does and runs a forward
pass + clustering on a small synthetic input. Does NOT spin up the IPC
server -- the goal here is to verify the dev shell + model assembly without
the network plumbing.

Run from the repo root:
    nix develop --impure --command python basic_test.py
"""

import sys

# server.py reads sys.argv[1] at import time; inject a placeholder so
# importing it from this file doesn't crash.
if len(sys.argv) < 2:
    sys.argv.append("ipc:///tmp/embedseg_basic_test.ipc")

import numpy  # noqa: E402

from server import setup  # noqa: E402


def main() -> None:
    processor, info = setup()
    print(f"setup: {info}")
    assert "cuda" in info["device"] or "GPU" in info["device"], (
        f"Not on GPU! info['device']={info['device']!r}"
    )

    # Image segmentation model: 5-D NCZYX. H/W must be divisible by tile size.
    numpy.random.seed(0)
    data = numpy.random.random_sample((1, 1, 1, 256, 256)).astype(numpy.float32)
    out = processor(data)

    arr = out.cpu().numpy() if hasattr(out, "cpu") else out
    print(f"process: {type(arr)} {arr.shape} dtype={arr.dtype} max={arr.max()}")


if __name__ == "__main__":
    main()
