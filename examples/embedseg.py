# /usr/bin/env python
"""
This example uses a server within the environment defined on `https://github.com/afermg/EmbedSeg.git`.

Run `nix run github:afermg/EmbedSeg/nahual-wrap -- ipc:///tmp/embedseg.ipc` from any directory,
or `nix develop --impure --command bash -c "python server.py ipc:///tmp/embedseg.ipc"` from
the root of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("embedseg", signature=("dict", "numpy"))
address = "ipc:///tmp/embedseg.ipc"

# %% Load model server-side
parameters = {
    # Optional overrides; defaults give a working 2-D BranchedERFNet
    # at random init (or pass `weights` to load a checkpoint).
    # "weights": "/path/to/checkpoint.pth",
    # "n_y": 256,
    # "n_x": 256,
    # "n_sigma": 2,
    # "input_channels": 1,
    # "device": 0,
}
response = setup(parameters, address=address)
print(response)
# Example: {'device': 'cuda:0', 'n_y': 256, 'n_x': 256, 'n_sigma': 2,
#           'input_channels': 1, 'num_classes': [4, 1],
#           'expected_tile_size': 8, 'weights': None}

# %% Define custom data
# 5-D NCZYX. H/W must be divisible by `expected_tile_size` (8).
tile_size = 256
numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, 1, 1, tile_size, tile_size)).astype(
    numpy.float32
)
result = process(data, address=address)
print(f"Shape: {result.shape}, Max: {result.max()}")
# Expected: Shape: (1, 256, 256), Max: 0  (random weights -> empty mask)
