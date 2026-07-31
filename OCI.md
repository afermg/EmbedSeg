# EmbedSeg Nahual OCI image

Build the reproducible archive and load it into Podman or Docker:

```console
nix build .#oci-image
podman load < result                         # or: docker load < result
```

The image is tagged `nahual/embedseg:local` and listens on TCP port 5555.

```console
podman run --rm --device nvidia.com/gpu=all -p 5555:5555 \
  nahual/embedseg:local
```

For Docker, replace the CDI device option with `--gpus all`. CPU operation is
supported. With Nahual and NumPy installed on the host, run:

```console
NAHUAL_DEVICE=cpu python oci/smoke_test.py
```

No pretrained checkpoint is bundled or published by this repository, so the
smoke test validates the complete network and clustering path with deterministic
random initialization. A trusted checkpoint can be mounted read-only and
selected with the setup request's `weights` parameter.
