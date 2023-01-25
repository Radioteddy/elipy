# These functions are taken from abipy package (almost) without any changes
# https://abinit.github.io/abipy/_modules/abipy/core/kpoints.html
# the reason to take them instead of import is keeping minimalistic requirements
# and accelerate up to 2-3 orders of magnitude thanks to Numba jit-compilation
import numpy as np
from numba import njit

@njit()
def is_integer(x: np.ndarray, atol: float=1e-8) -> bool:
    """is_integer check if vector has integer coordinates within tolerance atol

    Parameters
    ----------
    x : np.ndarray(3,)
        input vector
    atol : float, optional
        tolerance, by default 1e-8

    Returns
    -------
    bool
        True if vector has integer coordinates, False otherwise
    """
    int_x = np.empty_like(x)
    np.round(x, 0, int_x)
    return np.all(np.abs(int_x-x)<=(atol+1e-5*np.abs(x)))

@njit()
def issamek(k1: np.ndarray, k2: np.ndarray, atol: float=1e-8) -> bool:
    """issamek checks if two k-points are equal modulo a lattice vector.

    Parameters
    ----------
    k1 : np.ndarray(3,)
        first k-point
    k2 : np.ndarray(3,)
        second k-point
    atol : float, optional
        tolerance, by default 1e-8

    Returns
    -------
    bool
        True if two k-points are the same
    """
    k1 = np.asarray(k1)
    k2 = np.asarray(k2)
    return is_integer(k1 - k2, atol=atol)

@njit()
def get_kq2k(kpts: np.ndarray, kpts_sub: np.ndarray, qpt: np.ndarray, atol_kdiff=1e-8) -> dict:
    """get_kq2k Find idx{k+q} --> idx{k} for k-point subset

    Parameters
    ----------
    kpts : np.ndarray(nk,3)
        array of k-points in the whole BZ
    kpts_sub : np.ndarray(nk_sub,3)
        array of k-point subset
    qpt : np.ndarray(3,)
        q-point for which we map BZ
    atol : float, optional
        tolerance, by default 1e-8
        
    Returns
    -------
    {int: (int, np.ndarray(3,))}
        dictionary which keys are k-point ind, values[0] is k+q ind, values[1] is G vector
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