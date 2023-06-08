from elipy import Elisahberg

# only first file used from now (v.0.1.7)
test_files = [
    'indata/GSTORE_small.nc',
    'indata/EIG_small.nc',
    'indata/anaddb_PHFRQ_small',
]

# initialize Elaishberg class containing methods for a2f calculation
# Setup input files, parameters of energy grid (now default values are used)
# if we do not set up parameters for e', e values will be used by default
# parameters of phonon grid (phonon window will be taken from phonon eigenvalues analysis)

test = Elisahberg(test_files[0], ewindow=[-0.4, 0.4], esmear=0.01, epoints=500, eunits='Ha',  
                  phsmear=2., phpoints=400, phunits='meV') #
# call a2f calculator
test.compute_a2f()