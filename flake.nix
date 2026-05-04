{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in
      with pkgs;
      rec {
        apps.default =
          let
            python_with_pkgs = python3.withPackages (pp: [
              (inputs.nahual-flake.packages.${system}.nahual)
              packages.embedseg
              pp.torch
              pp.torchvision
              pp.numpy
              pp.scipy
              pp.scikit-image
              pp.tifffile
              pp.numba
              pp.tqdm
              pp.matplotlib
            ]);
            runServer = pkgs.writeScriptBin "runserver.sh" ''
              #!${pkgs.bash}/bin/bash
              # PYTHONSAFEPATH=1 (Python 3.11+) keeps Python from prepending
              # the script's directory to sys.path so the in-tree `EmbedSeg/`
              # source tree never shadows the nix-built package.
              export PYTHONSAFEPATH=1
              ${python_with_pkgs}/bin/python ${self}/server.py ''${@:-"ipc:///tmp/embedseg.ipc"}
            '';
          in
          {
            type = "app";
            program = "${runServer}/bin/runserver.sh";
          };

        packages = {
          embedseg = pkgs.python3.pkgs.callPackage ./nix/embedseg.nix { };
        };

        devShells = {
          default =
            let
              python_with_pkgs = python3.withPackages (pp: [
                (inputs.nahual-flake.packages.${system}.nahual)
                packages.embedseg
                pp.torch
                pp.torchvision
                pp.numpy
                pp.scipy
                pp.scikit-image
                pp.tifffile
                pp.numba
                pp.tqdm
                pp.matplotlib
                pp.pandas
                pp.seaborn
                pp.pyyaml
              ]);
            in
            mkShell {
              packages = [
                python_with_pkgs
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
              ];
              shellHook = ''
                # PYTHONSAFEPATH=1 (Python 3.11+) keeps Python from prepending
                # the script's directory to sys.path so `python basic_test.py`
                # never picks up the in-tree `EmbedSeg/` source tree instead
                # of the nix-built package.
                export PYTHONSAFEPATH=1
                export PYTHONDONTWRITEBYTECODE=1
              '';
            };
        };
      }
    );
}
