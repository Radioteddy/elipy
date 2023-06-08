The minimal example of Eliahsberg function calculation in Al.

`indata/` contains all necessary input files: squared absolute values of matrix elements |g|^2 in full BZ for both k- and q-point grids `indata/GSTORE_small.nc`; electron eigenvalues in full k-BZ `indata/EIG_small.nc`; phonon eigenvalues in full q-BZ `indata/anaddb_PHFRQ_small`.

From v0.1.7, all eigenvalues are restored from `GSTORE.nc` file, files with eigenvalues are no more needed (still can be used for consistency).