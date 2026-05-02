"""Nahual server for EmbedSeg.

EmbedSeg is a PyTorch-based instance segmentation model that learns
spatial embeddings + per-pixel sigma + seediness scores per pixel and then
clusters those embeddings into instances. This wrap loads the standard 2-D
``BranchedERFNet`` backbone and (optionally) loads a checkpoint from disk.

If ``weights`` is ``None`` the model is left at random init -- the wrap will
still answer requests and return an instance map (typically all zeros for
random weights), which is sufficient for smoke / wiring tests.

Run with:
    nix run --impure . -- ipc:///tmp/embedseg.ipc
or:
    python server.py ipc:///tmp/embedseg.ipc
"""

import sys
from functools import partial
from typing import Callable

import numpy
import pynng
import torch
import trio
from nahual.preprocess import pad_channel_dim, validate_input_shape
from nahual.server import responder

# server.py captures argv[1] at import time; basic_test.py injects a
# placeholder before importing this module.
address = sys.argv[1]


def setup(
    weights: str | None = None,
    n_y: int = 256,
    n_x: int = 256,
    n_sigma: int = 2,
    input_channels: int = 1,
    num_classes: tuple[int, int] | list[int] = (4, 1),
    pixel_y: float = 1.0,
    pixel_x: float = 1.0,
    seed_thresh: float = 0.9,
    fg_thresh: float = 0.5,
    min_mask_sum: int = 0,
    min_unclustered_sum: int = 0,
    min_object_size: int = 36,
    device: int | None = None,
    expected_tile_size: int = 8,
) -> tuple[Callable, dict]:
    """Load a 2-D ``BranchedERFNet`` and the matching ``Cluster`` postprocess.

    Parameters
    ----------
    weights : str | None
        Path to an EmbedSeg checkpoint ``.pth`` file. If ``None`` the network
        keeps its random initialisation (smoke-test friendly).
    n_y, n_x : int
        Grid size used by ``Cluster``. Inputs may be smaller; the cluster
        only indexes the top-left ``H x W`` corner of the grid.
    n_sigma : int
        Number of sigma channels (2 for 2-D, matches upstream defaults).
    input_channels : int
        Channels expected by the encoder (1 = grayscale).
    num_classes : list[int]
        Branched-ERFNet decoder widths. Defaults to ``[4, 1]`` (2-D variant).
    pixel_y, pixel_x : float
        Physical pixel spacings used by the spatial-embedding clustering.
    seed_thresh, fg_thresh : float
        Standard EmbedSeg clustering thresholds.
    min_mask_sum, min_unclustered_sum, min_object_size : int
        Clustering pruning knobs.
    device : int | None
        CUDA device index. ``None`` -> ``cuda:0`` if available, else CPU.
    expected_tile_size : int
        Required divisor for input H/W (EmbedSeg/ERFNet downsamples by 8).
    """
    if device is None:
        device = 0
    if torch.cuda.is_available():
        torch_device = torch.device(int(device))
    else:
        torch_device = torch.device("cpu")

    from EmbedSeg.models.BranchedERFNet import BranchedERFNet
    from EmbedSeg.utils.utils import Cluster

    model = BranchedERFNet(
        num_classes=list(num_classes), input_channels=input_channels
    )
    model.init_output(n_sigma=n_sigma)
    model = model.to(torch_device).eval()

    if weights is not None:
        state = torch.load(weights, map_location=torch_device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)

    cluster = Cluster(
        grid_y=n_y,
        grid_x=n_x,
        pixel_y=pixel_y,
        pixel_x=pixel_x,
        device=torch_device,
        one_hot=False,
    )

    info = {
        "device": str(torch_device),
        "n_y": n_y,
        "n_x": n_x,
        "n_sigma": n_sigma,
        "input_channels": input_channels,
        "num_classes": list(num_classes),
        "expected_tile_size": expected_tile_size,
        "weights": weights,
    }

    processor = partial(
        process,
        model=model,
        cluster=cluster,
        device=torch_device,
        n_sigma=n_sigma,
        seed_thresh=seed_thresh,
        fg_thresh=fg_thresh,
        min_mask_sum=min_mask_sum,
        min_unclustered_sum=min_unclustered_sum,
        min_object_size=min_object_size,
        expected_channels=input_channels,
        expected_tile_size=expected_tile_size,
    )
    return processor, info


def process(
    pixels: numpy.ndarray,
    model,
    cluster,
    device: torch.device,
    n_sigma: int,
    seed_thresh: float,
    fg_thresh: float,
    min_mask_sum: int,
    min_unclustered_sum: int,
    min_object_size: int,
    expected_channels: int,
    expected_tile_size: int,
) -> numpy.ndarray:
    """Run EmbedSeg on a 5-D NCZYX numpy array and return per-tile labels.

    Output shape: ``(N, H, W)`` with int32 instance ids (0 = background).
    """
    if pixels.ndim != 5:
        raise ValueError(f"Expected NCZYX (5D) array, got shape {pixels.shape}")
    _, _, _, *input_yx = pixels.shape
    validate_input_shape(input_yx, expected_tile_size)

    # Drops Z (assumes single Z slice) and pads channels up to expected_channels.
    pixels = pad_channel_dim(pixels, expected_channels)
    torch_tensor = torch.from_numpy(pixels.copy()).float().to(device)

    with torch.no_grad():
        output = model(torch_tensor)

    # output: (N, 2 + n_sigma + 1, H, W). Cluster operates per-sample.
    labels = numpy.zeros(
        (output.shape[0], output.shape[2], output.shape[3]), dtype=numpy.int32
    )
    for i in range(output.shape[0]):
        instance_map = cluster.cluster(
            output[i],
            n_sigma=n_sigma,
            seed_thresh=seed_thresh,
            fg_thresh=fg_thresh,
            min_mask_sum=min_mask_sum,
            min_unclustered_sum=min_unclustered_sum,
            min_object_size=min_object_size,
        )
        labels[i] = instance_map.cpu().numpy().astype(numpy.int32)
    return labels


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"EmbedSeg server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(responder, setup=setup), sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
