import numpy as np
from numba import njit #, prange
from .grid import Grid
from .kpt_utils import *
from .mpi import master

@njit()
def gaussian(x: float, x0: float, sigma: float) -> np.float_:
    denom = 1/(2*sigma**2)
    pref = np.sqrt(denom/np.pi)
    return np.exp(-denom * (x-x0)**2) * pref

# @njit(parallel=True)
@njit()
def get_eew_term(eps_k: np.float_, eps_kq: np.float_, w_q: np.float_, 
                   sigma_eps: np.float_, sigma_eps_pr: np.float_, sigma_w: np.float_, 
                    eps_arr: np.ndarray, eps_pr_arr: np.ndarray, w_arr: np.ndarray,
                    Neps: int, Neps_pr: int, Nw: int, a2f_eew: np.ndarray) -> np.ndarray:
    # a2f_eew = np.empty((ne, ne_pr, nw)) 
    # we don't need to create a2f array every time - just pass once created as in and out
    # we don't require Nw, may change later
    for i in range(Neps):
        for j in range(Neps_pr):
            a2f_eew[i,j,:] = ( gaussian(w_arr[:], w_q, sigma_w) 
                            * gaussian(eps_pr_arr[j], eps_kq, sigma_eps_pr)
                            * gaussian(eps_arr[i], eps_k, sigma_eps) )
    return a2f_eew

@njit()
def get_a2f_chunk(gkq_chunk: np.ndarray, kpts: np.ndarray, qpts: np.ndarray,
                  eps_eig: np.ndarray, ph_eig: np.ndarray, 
                  epsilon: Grid, epsilon_pr: Grid, omega: Grid) -> np.ndarray:
    # get energy grids and smearing
    e_range = epsilon.grid
    epr_range = epsilon_pr.grid
    w_range = omega.grid
    a2f_vals = np.zeros((epsilon.npoints, epsilon_pr.npoints, omega.npoints))
    a2f_temp = np.empty((epsilon.npoints, epsilon_pr.npoints, omega.npoints))
    Nq, Nk, Nnu, Nb, Nbpr = gkq_chunk.shape[:-1] # dimensions of summation
    for iq in range(Nq):
        print(f'qpt {iq+1}/{Nq}: {qpts[iq][0]:.8e} {qpts[iq][1]:.8e} {qpts[iq][2]:.8e}')
        ikq, _ = get_kq2k(kpts, qpts[iq])
        for ik in range(Nk):
            for inu in range(Nnu):
                w_q = ph_eig[iq, inu]
                for ib in range(Nb):
                    eps_k = eps_eig[ik,ib]
                    for ibp in range(Nbpr):
                        eps_kq = eps_eig[ikq,ibp]
                        g_kq = gkq_chunk[ik,iq,inu,ib,ibp]
                        get_eew_term(eps_k,eps_kq,w_q,
                                        epsilon.smear,epsilon_pr.smear,omega.smear,
                                        e_range, epr_range, w_range,
                                        epsilon.npoints, epsilon_pr.npoints, omega.npoints,
                                        a2f_temp)
                        a2f_vals += a2f_temp * g_kq
    return a2f_vals
