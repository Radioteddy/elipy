from pathlib import Path
from time import time

import numpy as np

from .files_handling import *
from .constants import *
from .messages import *
from .grid import Grid
from .a2f_calculator import get_a2f_chunk

from .mpi import MPI, comm, size, rank, master_only, mpi_watch, master

class Elisahberg:
    def __init__(self, gkq_file: str, eig_file: str, pheig_file: str,
                 ewindow: list = [default_emin, default_emax], esmear: float = default_esmear, 
                 epoints: int = default_epoints, eunits: str = 'Ha',
                 e1window: list = None, e1smear: float = None,
                 e1points: int = None, e1units: str = None,
                 phwindow: list = None, phsmear: float = default_phsmear,
                 phpoints: int = default_phpoints, phunits: str = 'Ha',
                 out_file: str or Path = Path.cwd()/'eliashberg_eew.nc') -> None:
        self.gkq_file = gkq_file
        self.eig_file = eig_file
        self.pheig_file = pheig_file
        self.egrid = Grid(*ewindow, esmear, epoints, eunits)
        # use the same parameters for e and e1 by default
        e1window = ewindow if not e1window else e1window
        e1smear = esmear if not e1smear else e1smear
        e1points = epoints if not e1points else e1points
        e1units = eunits if not e1units else e1units
        self.e1grid = Grid(*e1window, e1smear, e1points, e1units)
        # phonon grid is set according to phonon eigenvalues by default. 
        # If phonon window is chosen, use these values
        if phwindow:
            self.phgrid = Grid(*phwindow, phsmear, phpoints, phunits)
        else:
            self.phgrid = None # will be initialized later
        self.__phsmear = phsmear
        self.__phpoints = phpoints
        self.__phunits = phunits
        self.out_file = out_file
    
    @master_only        
    def read_data(self):
        self.e_eigvals, e_kpts = get_electron_eigenvalues_kpoints(self.eig_file)
        self.ph_eigvals = get_phonon_eigenvalues(self.pheig_file)
        gkq_vals, self.g_kpts, self.g_qpts, self.efermi = get_gkq_values(self.gkq_file)
                # check data consistency
        assert np.array_equal(e_kpts, self.g_kpts), "GS and EPH k-point grids are inconsistent"
        assert self.e_eigvals.shape[1] == gkq_vals.shape[3], 'numbers of bands in GS and EPH are inconsistent'
        # TODO: 1) now code checks only sizes of q-point grid for phonons and eph but not grids themself
        assert len(self.ph_eigvals) == len(self.g_qpts), "PHONON and EPH q-grids are inconsistent"
        # if everything is OK, make electron eigenvalues Fermi energy centered
        self.e_eigvals = self.e_eigvals - self.efermi
        # g_kq has shape `[nq,nk,...]`, we swap 0th and 1st dimensions for convenience
        self.gkq_vals = np.swapaxes(gkq_vals, 0, 1)
        # and set up variables for all dimensions

    @mpi_watch
    def broadcast_dims(self):
        # Broadcast all dimensions from master to other cpu-s
        
        if master:
            dim_arr = np.array(self.gkq_vals.shape[:-1], dtype=int)
        else:
            dim_arr = np.empty(4, dtype=int)
        comm.Bcast([dim_arr, MPI.INT])
        self.nkpt, self.nqpt, self.nbranch, self.nband = dim_arr
    
    @mpi_watch
    def broadcast_kqpoints(self):
        # Broadcast k- and q-points arrays
        
        if not master:
            self.g_kpts = np.empty((self.nkpt, 3), dtype=np.float64)
            self.g_qpts = np.empty((self.nqpt, 3), dtype=np.float64)
        comm.Bcast([self.g_kpts, MPI.DOUBLE])
        comm.Bcast([self.g_qpts, MPI.DOUBLE])
                    
    @mpi_watch
    def broadcast_eigvals(self):
        # Broadcast electron and phonon eigenvalues
        
        if not master:
            self.e_eigvals = np.empty((self.nkpt, self.nband), dtype=np.float64)
            self.ph_eigvals = np.empty((self.nqpt, self.nbranch), dtype=np.float64)
        comm.Bcast([self.e_eigvals, MPI.DOUBLE])
        comm.Bcast([self.ph_eigvals, MPI.DOUBLE])
        # if phonon grid is not set, do it now    
        phmin = np.amin(self.ph_eigvals)
        phmin = 0 if phmin >= 0 else phmin - ph_delta
        if not self.phgrid:           
            self.phgrid = Grid(phmin, np.amax(self.ph_eigvals) + ph_delta,
                               self.__phsmear*unit_conversion[self.__phunits], self.__phpoints, 'Ha')


    # for some reason, scattering of n-dimensional array doesn't work
    # @mpi_watch
    # def scatter_gkq_vals(self):
    #     # scatter gkq values to cpu-s over nkpt dimension
    #     # TODO: think about scattering over nqpt as well
    #     if not master:
    #         self.gkq_vals = None
    #     # find dimension of chunks per cpu and number dimension displacemen
    #     ave, res = divmod(self.nkpt, size)
    #     counts = np.array([ave + 1 if p < res else ave for p in range(size)], dtype=int)
    #     # self because will call it later
    #     self.__displ = np.array([sum(counts[:p]) for p in range(size)], dtype=int)
    #     # allocate space for chunk
    #     self.gkq_chunk = np.empty((counts[rank], self.nqpt, self.nbranch, self.nband, self.nband))
    #     rest_dims = self.nqpt*self.nbranch*self.nband*self.nband
    #     comm.Scatterv([self.gkq_vals, counts*rest_dims, self.__displ*rest_dims, MPI.DOUBLE], self.gkq_chunk, root=0)

    # here array is flatten to 1d and back to nd, but it may brake everything at some point
    @mpi_watch
    def scatter_gkq_vals(self):
        # scatter gkq values to cpu-s over nkpt dimension
        
        # TODO: think about scattering over nqpt as well
        if master: 
            gkq_flatten = self.gkq_vals.flatten()
        else:
            gkq_flatten = None
        # find dimension of chunks per cpu and number dimension displacemen
        ave, res = divmod(self.nkpt, size)
        counts = np.array([ave + 1 if p < res else ave for p in range(size)], dtype=int)
        # self because will call it later
        self.__displ = np.array([sum(counts[:p]) for p in range(size)], dtype=int)
        rest_dims = self.nqpt*self.nbranch*self.nband*self.nband
        # allocate space for chunk
        self.gkq_chunk = np.empty(counts[rank]*rest_dims)
        comm.Scatterv([gkq_flatten, counts*rest_dims, self.__displ*rest_dims, MPI.DOUBLE], self.gkq_chunk, root=0)    
        self.gkq_chunk = self.gkq_chunk.reshape((counts[rank], self.nqpt, self.nbranch, self.nband, self.nband))
        
    @mpi_watch
    def sum_chunks(self):
        # allocate array summing chunks from all cpu-s
        self.a2f_vals = np.empty((self.egrid.npoints,
                                self.e1grid.npoints,
                                self.phgrid.npoints)) 
        # get the proper subset of kpoints
        if rank != size-1:
            kpts_chunk = self.g_kpts[self.__displ[rank]:self.__displ[rank+1],:]
        else:
            kpts_chunk = self.g_kpts[self.__displ[rank]:,:]

        # calculate a2f values for given chunck        
        a2f_chunk = get_a2f_chunk(self.gkq_chunk, self.g_kpts, kpts_chunk, self.g_qpts, 
                                  self.e_eigvals, self.ph_eigvals,
                                  self.egrid, self.e1grid, self.phgrid)
        comm.Reduce([a2f_chunk, MPI.DOUBLE], [self.a2f_vals, MPI.DOUBLE], op=MPI.SUM, root=0)       
            
    @master_only
    def write_netcdf(self):
        # multiply Eliashberg function to (almost) correct prefactor
        self.a2f_vals = self.a2f_vals * (2/self.nkpt/self.nqpt) # division to g(Ef) is up to user!
        # save computed Eliashberg function to netCDF file
        store_a2f_values(self.out_file, self.efermi, 
                         self.egrid, self.e1grid, self.phgrid, self.a2f_vals)
        
    def compute_a2f(self):
        # gather all previous methods together into united workflow
        # and add status messages to log file
        if master:
        # we don't need to log everything 
            start = time()           
        print_header()

        self.read_data()
        print_read_status(self.eig_file, self.pheig_file, self.gkq_file)
        
        self.broadcast_dims()
        self.broadcast_kqpoints()
        self.broadcast_eigvals()
        self.scatter_gkq_vals()
        print_variables(self.egrid, self.e1grid, self.phgrid)
        
        # print_computation()
        self.sum_chunks()
        
        self.write_netcdf()
        print_save_status(self.out_file)
        
        if master:
            elapsed = time() - start
            print_complete(elapsed)