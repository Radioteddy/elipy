# These functions are taken from abipy package (almost) without any changes
# https://abinit.github.io/abipy/_modules/abipy/core/kpoints.html
# the reason to take them instead of import is keeping minimalistic requirements
# and accelerate up to 2-3 orders of magnitude thanks to Numba jit-compilation
import numpy as np
from numba import njit

@njit()
def is_integer(x, atol=1e-8):
    """
    True if all x is integer within the absolute tolerance atol.
    """
    int_x = np.empty_like(x)
    np.round(x, 0, int_x)
    return np.all(np.abs(int_x-x)<=(1e-8+1e-5*np.abs(x)))

@njit()
def issamek(k1, k2, atol=1e-8):
    """
    True if k1 and k2 are equal modulo a lattice vector.
    """
    k1 = np.asarray(k1)
    k2 = np.asarray(k2)
    return is_integer(k1 - k2, atol=atol)

@njit()
def get_kq2k(kpts: np.ndarray, kpts_sub: np.ndarray, qpt: np.ndarray, atol_kdiff=1e-8) -> tuple([int, float]):
    """
    Find idx{k+q} --> idx{k}, g0

    Args:
        kpts: array of k-points in full BZ
        kpts: array of subset k-points for which we want to find mapping
        qpt: q-point
        atol_kdiff: Tolerance used to compare k-points.

    """
    k2kqg = dict()
    if np.all(np.abs(qpt) <= 1e-6):
        g0 = np.zeros(3)
        for ik, _ in enumerate(kpts_sub):
            k2kqg[ik] = (ik, g0)
    else:
        for ik, kpoint in enumerate(kpts_sub):
            kqpt = kpoint + qpt
            for ikq, kpt in enumerate(kpts):
                if issamek(kqpt, kpt, atol=atol_kdiff):
                    g0 = np.rint(kqpt - kpt)
                    k2kqg[ik] = (ikq, g0)
                    break
    return k2kqg