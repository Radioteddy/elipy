import numpy as np
from numba import njit
from numba.typed import List

from .grid import Grid
from .kpt_utils import *
# from .mpi import master

@njit()
def gaussian(x: float, x0: float, sigma: float) -> np.float_:
    """gaussian Gaussian weight for given energy point x and eigenvalue x0

    Parameters
    ----------
    x : float
    x0 : float
    sigma : float

    Returns
    -------
    np.float_
    """    
    denom = 1/(2*sigma**2)
    pref = np.sqrt(denom/np.pi)
    return np.exp(-denom * (x-x0)**2) * pref

# @njit(parallel=True)
@njit()
def get_eew_term(eps_k: np.float_, eps_kq: np.float_, w_q: np.float_, 
                   sigma_eps: np.float_, sigma_eps_pr: np.float_, sigma_w: np.float_, 
                    eps_arr: np.ndarray, eps_pr_arr: np.ndarray, w_arr: np.ndarray,
                    Neps: int, Neps_pr: int, Nw: int, a2f_eew: np.ndarray) -> np.ndarray:
    """get_eew_term calculate gaussian weights for all eigenvalues

    Parameters
    ----------
    eps_k : np.float_
        electron eigenvalue at k-point
    eps_kq : np.float_
        electron eigenvalue at k+q-point
    w_q : np.float_
        phonon eigenvalue at q-point
    sigma_eps : np.float_
        electron smearing for e
    sigma_eps_pr : np.float_
        electron smearing for e'
    sigma_w : np.float_
        phonon smearing
    eps_arr : np.ndarray(Neps)
        energy grid e
    eps_pr_arr : np.ndarray(Neps_pr)
        energy grid e'
    w_arr : np.ndarray(Nw)
        frequency grid w
    Neps : int
        number of points in e grid
    Neps_pr : int
        number of points in e' grid
    Nw : int
        number of points in w grid
    a2f_eew : np.ndarray(Neps, Neps_pr, Nw)
        array for gaussian weights, being rewritten every time

    Returns
    -------
    np.ndarray
        gaussian weights
    """
    # we don't require Nw, may change later
    for i in range(Neps):
        for j in range(Neps_pr):
            a2f_eew[i,j,:] = ( gaussian(w_arr[:], w_q, sigma_w) 
                            * gaussian(eps_pr_arr[j], eps_kq, sigma_eps_pr)
                            * gaussian(eps_arr[i], eps_k, sigma_eps) )
    return a2f_eew

@njit()
def calculate_chunk(gkq_chunk: np.ndarray, eps_eig: np.ndarray, ph_eig: np.ndarray, 
                    nonzero_dims: tuple, ikqpts: list,
                    egrid: np.ndarray, e1grid: np.ndarray, phgrid: np.ndarray,
                    esmear: np.float_, e1smear: np.float_, phsmear: np.float_,
                    epoints: int, e1points: int, phpoints: int) -> np.ndarray:
    """calculate_chunk calculate a2F values for given |g|^2 chunk

    Parameters
    ----------
    gkq_chunk : np.ndarray
        chunk of |g|^2 values
    eps_eig : np.ndarray
        electron eigenvalues
    ph_eig : np.ndarray
        phonon eigenvalues
    nonzero_dims : tuple
        indicies where |g|^2 != 0
    ikqpts : list
        maps of k->k+q points for every q
    egrid : np.ndarray
        energy grid e
    e1grid : np.ndarray
        energy grid e'
    phgrid : np.ndarray
        frequency grid w
    esmear : np.float_
        electron smearing for e
    e1smear : np.float_
        electron smearing for e'
    phsmear : np.float_
        phonon smearing
    epoints : int
        number of points in e grid
    e1points : int
        number of points in e' grid
    phpoints : int
        number of points in w grid
    Returns
    -------
    np.ndarray
        a2F values for |g|^2 chunk
    """
    # countainers for a2f values
    a2f_vals = np.zeros((epoints, e1points, phpoints))
    a2f_temp = np.empty((epoints, e1points, phpoints))    
    nk, nq, nnu, nb, nbp = nonzero_dims
    for i in range(len(nk)):
        ik, iq, inu, ib, ibp = nk[i], nq[i], nnu[i], nb[i], nbp[i]
        ikq = ikqpts[iq][ik][0]
        w_q = ph_eig[iq, inu]
        eps_k = eps_eig[ik,ib]
        eps_kq = eps_eig[ikq,ibp]
        g_kq = gkq_chunk[ik,iq,inu,ib,ibp]
        get_eew_term(eps_k, eps_kq, w_q,
                esmear, e1smear, phsmear,
                egrid, e1grid, phgrid,
                epoints, e1points, phpoints,
                a2f_temp)
        a2f_vals += a2f_temp * g_kq
    return a2f_vals

    
def get_a2f_chunk(gkq_chunk: np.ndarray, kpts: np.ndarray, kpts_chunk: np.ndarray, qpts: np.ndarray,
                  eps_eig: np.ndarray, ph_eig: np.ndarray, 
                  egrid: Grid, e1grid: Grid, phgrid: Grid) -> np.ndarray:
    """get_a2f_chunk wrapper for calculate_chunk function

    Parameters
    ----------
    gkq_chunk : np.ndarray
        chunk of |g|^2 values
    kpts : np.ndarray
        k-points in full BZ
    kpts_chunk : np.ndarray
        chunk of k-points
    qpts : np.ndarray
        q-points in full BZ
    eps_eig : np.ndarray
        electron eigenvalues
    ph_eig : np.ndarray
        phonon eigenvalues
    egrid : Grid
        electron grid e
    e1grid : Grid
        electron grid e'
    phgrid : Grid
        phonon grid w

    Returns
    -------
    np.ndarray
        a2F values for |g|^2 chunk
    """
    # mapping of BZ: for every q get k+q -> k 
    ikqpts = List()
    [ikqpts.append(get_kq2k(kpts, kpts_chunk, qpt)) for qpt in qpts]
    # take only nonzero values of gkq_chunk
    where_nonzero = np.nonzero(gkq_chunk)
    a2f_vals = calculate_chunk(gkq_chunk, eps_eig, ph_eig,
                               where_nonzero, ikqpts, 
                               egrid.grid, e1grid.grid, phgrid.grid,
                               egrid.smear, e1grid.smear, phgrid.smear,
                               egrid.npoints, e1grid.npoints, phgrid.npoints)
    return a2f_vals
