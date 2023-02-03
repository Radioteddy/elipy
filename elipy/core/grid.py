from numpy import linspace
from .constants import *

class Grid:
    """ provides linear grid for electron and phonon energies
    """    
    def __init__(self, g_min: float, g_max: float, smear: float,
                 npoints: int, unit: str = 'Ha'):
        """__init__ parameters of grid

        Parameters
        ----------
        g_min : float
            lowest grid point
        g_max : float
            largest grid point
        smear : float
            parameter for gaussian smearing
        npoints : int
            number of grid points
        unit : str, optional
           energy units, by default 'Ha'
        """
        self.unit = unit
        self.npoints = npoints
        if self.unit == 'Ha':
            self.g_min = g_min
            self.g_max = g_max
            self.smear = smear
        elif self.unit in unit_conversion.keys():       
            self.g_min = g_min * unit_conversion[self.unit]
            self.g_max = g_max * unit_conversion[self.unit]
            self.smear = smear * unit_conversion[self.unit]
        else:
            raise ValueError('unit is not allowed')
        self.grid = self.make_grid()
                
    def make_grid(self):
        return linspace(self.g_min, self.g_max, self.npoints)