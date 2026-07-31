{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
    nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        embedseg = pkgs.python3.pkgs.callPackage ./nix/embedseg.nix {};
        python_with_pkgs = pkgs.python3.withPackages (pp: [
          inputs.nahual-flake.packages.${system}.nahual
          embedseg
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
        runServer = pkgs.writeScriptBin "nahual-embedseg" ''
          #!${pkgs.bash}/bin/bash
          export PYTHONSAFEPATH=1
          exec ${python_with_pkgs}/bin/python ${self}/server.py \
            "''${1:-tcp://0.0.0.0:5555}"
        '';
        embedsegApp = {
          type = "app";
          program = "${runServer}/bin/nahual-embedseg";
        };
      in
        with pkgs; rec {
          packages =
            {inherit embedseg;}
            // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              oci-image = import ./nix/oci-image.nix {
                inherit pkgs;
                name = "embedseg";
                title = "Nahual EmbedSeg";
                description = "EmbedSeg instance segmentation served through Nahual";
                source = "https://github.com/afermg/EmbedSeg";
                revision = self.rev or self.dirtyRev or "unknown";
                server = runServer;
                entrypoint = embedsegApp.program;
              };
            };
          inherit python_with_pkgs;
          scripts.runServer = runServer;
          apps = rec {
            embedseg = embedsegApp;
            default = embedseg;
          };
          devShells.default = mkShell {
            packages = [
              python_with_pkgs
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              python3Packages.pandas
              python3Packages.seaborn
              python3Packages.pyyaml
            ];
            shellHook = ''
              export PYTHONSAFEPATH=1
              export PYTHONDONTWRITEBYTECODE=1
            '';
          };
        }
    );
}
