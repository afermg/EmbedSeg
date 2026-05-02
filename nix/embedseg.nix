{
  lib,
  buildPythonPackage,
  hatchling,
  hatch-vcs,
  # Runtime deps from pyproject.toml. We omit `jupyter` (huge transitive
  # closure) -- not used at inference time -- and `pycocotools` which is
  # only used for COCO eval.
  matplotlib,
  numpy,
  scipy,
  tifffile,
  numba,
  tqdm,
  pandas,
  seaborn,
  scikit-image,
  colorspacious,
  # Imported by the codebase but not declared:
  torch,
  torchvision,
}:
buildPythonPackage {
  pname = "embedseg";
  version = "0.2.5";

  src = ./..; # use --impure when building.

  pyproject = true;

  # Upstream pyproject.toml does `sources = ["EmbedSeg"]` which strips the
  # `EmbedSeg/` prefix from the wheel layout, so subpackages land at the
  # top of site-packages instead of under `EmbedSeg/`. Drop that override
  # so the package imports as `EmbedSeg.<sub>`.
  postPatch = ''
    sed -i '/^sources = \["EmbedSeg"\]/d' pyproject.toml
  '';

  build-system = [
    hatchling
    hatch-vcs
  ];

  dependencies = [
    matplotlib
    numpy
    scipy
    tifffile
    numba
    tqdm
    pandas
    seaborn
    scikit-image
    colorspacious
    torch
    torchvision
  ];

  # hatch-vcs needs a git tag; we don't have one in the source tree the way
  # this is sourced. Pre-set the version env var.
  env = {
    SETUPTOOLS_SCM_PRETEND_VERSION = "0.2.5";
    HATCH_BUILD_HOOKS_ENABLE = "false";
  };

  pythonImportsCheck = [
    "EmbedSeg"
    "EmbedSeg.models.BranchedERFNet"
    "EmbedSeg.utils.utils"
  ];

  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  meta = {
    description = "EmbedSeg: instance segmentation via spatial embeddings.";
    homepage = "https://github.com/juglab/EmbedSeg";
    license = lib.licenses.bsd3;
  };
}
