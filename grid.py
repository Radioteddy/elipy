from numpy import linspace
from .constants import *

class Grid:
    def __init__(self, g_min: float, g_max: float, smear: float,
                 npoints: int, unit: str = 'Ha'):
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