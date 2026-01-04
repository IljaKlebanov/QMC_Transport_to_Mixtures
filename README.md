This repository contains the code used to produce the numerical results in the paper

Transporting higher-order quadrature rules – Quasi-Monte Carlo points and sparse grids for mixture distributions
by Tim Sullivan and Ilja Klebanov.

The paper is available at:

https://link.springer.com/article/10.1007/s11222-025-10764-x

https://arxiv.org/abs/2308.10081

Notes on the implementation

The code is organized so that you can either run the full set of experiments yourself or use the cached results provided in the .mat files.

Files whose names start with MAIN_ reproduce the numerical results reported in the paper; the remaining files contain helper functions.

For the sparse grid experiments, you will need a sparse grids MATLAB toolbox (for example, by adding it to the MATLAB path):
https://sites.google.com/view/sparse-grids-kit

If you encounter any errors or inconsistencies, I would be very grateful if you could let me know.
