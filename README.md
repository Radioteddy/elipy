# Elipy

Post-processing tool for ABINIT >= 9.8.1 aiming for calculation of energy-resolved Eliashberg function

## Description

Since version 9.8.1 ABINIT provides easy-to-use methods within EPH package for the calculation of electron-phonon matrix elements on rectangular k- and q-point grids (optdriver 7, eph_task 11). The present package is a tool for calculation of electron-energy-resolved Eliashberg function on arbitrary electron and phonon grids. 

$$
\alpha^2 F(\epsilon,\epsilon',\omega)=\frac{1}{N_\mathbf{k}N_\mathbf{q}}\sum_{\mathbf{k}\mathbf{q}mn\nu}\left|g_{mn\nu}(\mathbf{k},\mathbf{q})\right|^2\delta(\epsilon_{\mathbf{k}n}-\epsilon)\delta(\epsilon_{\mathbf{k+q}m}-\epsilon')\delta(\omega_{\mathbf{q}\nu}-\omega)
$$

The above-written definition does not include electron density of states at Fermi level. One needs to divide $\alpha^2 F$ over $N_F$ to get physically meaningful results. 

The current implemetation supports Gaussian representation of delta-functions. Project uses mpi4py for many-core parallelization and Numba for acceleration of procedures dealing with iteration over numpy arrays.

Elipy is insipired by ElectronPhononCoupling and Abipy projects, and uses some of their utility functions. Apologies for code copy-pasting, it allows to keep dependency list as short as possible.  

## Getting Started

### Dependencies

* Numpy
* Scipy
* netCDF4
* Numba
* mpi4py

The actual versions of required packages are stored in pyproject.toml file.

### Installation

Installation based on [Poetry](https://python-poetry.org) package and dependency manager.
```
git clone https://github.com/Radioteddy/elipy.git
cd elipy
poetry install
```

### Executing program

* Use mpirun, mpiexec, srun,... for program execution
```
mpiexec -n X python filename.py > log 2> err
```

## Authors

[Fedor Akhmetov](https://github.com/Radioteddy)

## Version History

* 0.1.5
    * Working Release
* 0.1.0
    * Initial Release

## Acknowledgments
* [ElectronPhononCoupling](https://github.com/GkAntonius/ElectronPhononCoupling)
* [abipy](https://github.com/abinit/abipy)
