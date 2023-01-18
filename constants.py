from scipy.constants import physical_constants as pc

eV_Ha = pc['electron volt-hartree relationship'][0]
THz_Ha = pc['hertz-hartree relationship'][0]*1e9
inv_cm_Ha = pc['inverse meter-hartree relationship'][0]*1e2

unit_conversion = {
    'eV': eV_Ha,
    'Thz': THz_Ha,
    'cm-1': inv_cm_Ha,
    'meV': eV_Ha*1e-3,
}

# small addition to highest phonon eigenvalue for phonon grid
ph_delta = 10 * eV_Ha # 10 meV

# egird defaults in Ha units
default_emin = -0.4
default_emax = 0.4
default_esmear = 0.01
default_epoints = 50

# phgrid defaults in meV
# default min and max are taken from eigenfrequencies file
default_phsmear = 0.1
default_phpoints = 100 # higher resolution over phonon frequencies
