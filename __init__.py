"""
elipy: a post-processing tool for ABINIT EPH module

Provides `Eliashberg` class object contaning methods for calculation of energy-dependent Eliashberg function a2F from e-ph matrix elements |g_kq|^2

Usage: 
1. Create file with input parameters `yourfile.py`, e.g.
-----------------------------------------------------------
from elipy import Eliashberg

gkq_input = 'indata\in_GSTORE.nc'
eeig_input = 'indata\in_EIG.nc'
pheig_input = 'indata\in_PHFRQ'

a2f_class = Eliashberg(gkq_input, eeig_input, pheig_input)
a2f_class.compute_a2f()
-----------------------------------------------------------
2. Run the caculation in the following way: `mpirun -n 4 python yourfile.py > log`
3. At the end of calculation you will get netCDF4 file `eliashberg_eew.nc' with a2F(e,e',w)
"""

from .core.eliashberg import *
from .core import kpt_utils as kpt_utils
from .core import grid as grid
from .core import constants as constants
from .core import a2f_calculator as a2f_calculator