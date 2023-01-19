# These functions are taken from abipy package (almost) without any changes
# https://abinit.github.io/abipy/_modules/abipy/core/kpoints.html
# the reason to take them instead of import is keeping minimalistic requirements
import numpy as np
from numba import njit

@njit()
def is_integer(x, atol=1e-8):
    """
    True if all x is integer within the absolute tolerance atol.
    """
    int_x = np.around(x)
    return np.allclose(int_x, x, atol=atol)

@njit()
def issamek(k1, k2, atol=1e-8):
    """
    True if k1 and k2 are equal modulo a lattice vector.
    """
    k1 = np.asarray(k1)
    k2 = np.asarray(k2)
    #if k1.shape != k2.shape:

    return is_integer(k1 - k2, atol=atol)

@njit()
def get_kq2k(kpts: np.ndarray, kqpt: np.ndarray, atol_kdiff=1e-8) -> tuple([int, float]):
    """
    Find idx{k+q} --> idx{k}, g0

    Args:
        kpts: array of k-points in fractional coordinates
        kqpt: (k+q)-point in fractional coordinates
        atol_kdiff: Tolerance used to compare k-points.

    """
    for ikq, kpt in enumerate(kpts):
        if issamek(kqpt, kpt, atol=atol_kdiff):
            g0 = np.rint(kqpt - kpt)
            break
    return (ikq, g0)