from scipy.constants import physical_constants as pc

Ha_eV = pc['hartree-electron volt relationship'][0]
eV_Ha = pc['electron volt-hartree relationship'][0]
THz_Ha = pc['hertz-hartree relationship'][0]*1e9
inv_cm_Ha = pc['inverse meter-hartree relationship'][0]*1e2

unit_conversion = {
    'Ha': 1e0,
    'eV': eV_Ha,
    'Thz': THz_Ha,
    'cm-1': inv_cm_Ha,
    'meV': eV_Ha*1e-3,
}

# egird defaults in Ha units
default_emin = -0.4
default_emax = 0.4
default_esmear = 0.01 # Ha
default_epoints = 50

# phgrid defaults in meV
# default min and max are taken from eigenfrequencies file
default_phsmear = 2e-5 # ~0.544 meV
default_phpoints = 100 # higher resolution over phonon frequencies

# small addition to highest phonon eigenvalue for phonon grid
ph_delta = eV_Ha * 1e-2 # 10 meV
