import numpy as np
from numba import njit #, prange
from collections import OrderedDict
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
def calculate_chunk(gkq_chunk: np.ndarray, eps_eig: np.ndarray, ph_eig: np.ndarray, 
                    nonzero_dims: tuple, ikqpts: list,
                    egrid: np.ndarray, e1grid: np.ndarray, phgrid: np.ndarray,
                    esmear: np.float_, e1smear: np.float_, phsmear: np.float_,
                    epoints: int, e1points: int, phpoints: int) -> np.ndarray:
    # countainers for a2f values
    a2f_vals = np.zeros((egrid.npoints, e1grid.npoints, phgrid.npoints))
    a2f_temp = np.empty((egrid.npoints, e1grid.npoints, phgrid.npoints))    
    nk, nq, nnu, nb, nbp = nonzero_dims
    for i in range(len(nk)):
        ik, iq, inu, ib, ibp = nk[i], nq[i], nnu[i], nb[i], nbp[i]
        ikq = ikqpts[ik][0]
        w_q = ph_eig[iq, inu]
        eps_k = eps_eig[ik,ib]
        eps_kq = eps_eig[ikq,ibp]
        g_kq = gkq_chunk[ik,iq,inu,ib,ibp]
        get_eew_term(eps_k,eps_kq,w_q,
                esmear, e1smear, phsmear,
                egrid, e1grid, phgrid,
                epoints, e1points, phpoints,
                a2f_temp)
        a2f_vals += a2f_temp * g_kq
    return a2f_vals

    
def get_a2f_chunk(gkq_chunk: np.ndarray, kpts: np.ndarray, kpts_chunk: np.ndarray, qpts: np.ndarray,
                  eps_eig: np.ndarray, ph_eig: np.ndarray, 
                  egrid: Grid, e1grid: Grid, phgrid: Grid) -> np.ndarray:
    # mapping of BZ: for every q get k+q -> k 
    ikqpts = [get_kq2k(kpts, kpts_chunk, qpt) for qpt in qpts]
    # take only nonzero values of gkq_chunk
    where_nonzero = np.nonzero(gkq_chunk)
    a2f_vals = calculate_chunk(gkq_chunk, eps_eig, ph_eig,
                               where_nonzero, ikqpts, 
                               egrid.grid, e1grid.grid, phgrid.grid,
                               egrid.smear, e1grid.smear, phgrid.smear,
                               egrid.npoints, e1grid.npoints, phgrid.npoints)
    return a2f_vals

    # Nk, Nq, Nnu, Nb, Nbpr = gkq_chunk.shape # dimensions of summation
    # a2f_vals = np.zeros((egrid.npoints, e1grid.npoints, phgrid.npoints))
    # a2f_temp = np.empty((egrid.npoints, e1grid.npoints, phgrid.npoints))

    # nk, nq, nnu, nb, nbp = where_nonzero
    # for i in range(len(nk)):
    #     ik, iq, inu, ib, ibp = nk[i], nq[i], nnu[i], nb[i], nbp[i]
    #     ikq = ikqpts[ik][0]
    #     w_q = ph_eig[iq, inu]
    #     eps_k = eps_eig[ik,ib]
    #     eps_kq = eps_eig[ikq,ibp]
    #     g_kq = gkq_chunk[ik,iq,inu,ib,ibp]
    #     get_eew_term(eps_k,eps_kq,w_q,
    #             egrid.smear, e1grid.smear, phgrid.smear,
    #             egrid.grid, e1grid.grid, phgrid.grid,
    #             egrid.npoints, e1grid.npoints, phgrid.npoints,
    #             a2f_temp)
    #     a2f_vals += a2f_temp * g_kq



    # for iq in range(Nq):
    #     if master:
    #         print(f'qpt {iq}/{Nq}: {qpts[iq][0]:.8e} {qpts[iq][1]:.8e} {qpts[iq][2]:.8e}')
    #     ikq_g0 = get_kq2k(kpts, qpts[iq])
    #     if ikq_g0:
    #         ikq, _ = ikq_g0
    #     else:
    #         continue
    #     for ik in range(Nk):
    #         for inu in range(Nnu):
    #             w_q = ph_eig[iq, inu]
    #             for ib in range(Nb):
    #                 eps_k = eps_eig[ik,ib]
    #                 for ibp in range(Nbpr):
    #                     eps_kq = eps_eig[ikq,ibp]
    #                     g_kq = gkq_chunk[ik,iq,inu,ib,ibp]
    #                     get_eew_term(eps_k,eps_kq,w_q,
    #                                     e_smear, epr_smear, w_smear,
    #                                     e_range, epr_range, w_range,
    #                                     e_points, epr_points, w_points,
    #                                     a2f_temp)
    #                     a2f_vals += a2f_temp * g_kq
