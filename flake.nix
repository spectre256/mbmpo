{
  description = "Dev shell for CSSE490 projects";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      formatter.${system} = pkgs.nixpkgs-fmt;

      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          (pkgs.python3.withPackages (py-pkgs: with py-pkgs; [
            torch
            torchvision
            gymnasium
            mujoco
            imageio
            numpy
            pandas
            matplotlib
            seaborn
            ipython
            pygame
            jupytext
            jupyter
            tqdm
          ]))
          # pandoc
          # texliveFull
        ];
      };
    };
}
