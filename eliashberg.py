import numpy as np
from netCDF4 import Dataset

from files_handling import *
from constants import *
from grid import Grid
from a2f_calculator import get_a2f_chunk

from mpi import MPI, comm, size, rank, master_only, mpi_watch, master

class Elisahberg:
    def __init__(self, gkq_file: str, eig_file: str, pheig_file: str,
                 egrid: dict = {'g_min': default_emin, 'g_max': default_emax, 
                                'smear': default_esmear, 'npoints': default_epoints,
                                'units': 'Ha'},
                 e1grid: dict = None,
                 phgrid: dict = None,
                 out_file: str = 'eliashberg_eew.nc') -> None:
        self.gkq_file = gkq_file
        self.eig_file = eig_file
        self.pheig_file = pheig_file
        self.egrid = Grid(**egrid)
        if e1grid:
            self.e1grid = Grid(**e1grid)
        else:
            self.e1grid = Grid(**egrid) # same grids for e and e'
        if phgrid:
            self.phgrid = Grid(**phgrid)
        else:
            self.phgrid = None # will be initialized later
        self.out_file = out_file
    
    @master_only        
    def read_data(self):
        self.e_eigvals, e_kpts = get_electron_eigenvalues_kpoints(self.eig_file)
        self.ph_eigvals = get_phonon_eigenvalues(self.pheig_file)
        gkq_vals, self.g_kpts, self.g_qpts = get_gkq_values(self.gkq_file)
                # check data consistency
        assert np.array_equal(e_kpts, self.g_kpts), "GS and EPH k-point grids are inconsistent"
        assert self.e_eigvals.shape[1] == gkq_vals.shape[3], 'numbers of bands in GS and EPH are inconsistent'
        # TODO: 1) now code checks only sizes of q-point grid for phonons and eph but not grids themself
        assert len(self.ph_eigvals) == len(self.g_qpts), "PHONON and EPH q-grids are inconsistent"
        # if everything is OK, move further
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
        self.nkpt, self.nqpt, self.nbranch, self.nband = dim_arr[0], dim_arr[1], dim_arr[2], dim_arr[3] 
    
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
        if not self.phgrid:
            self.phgrid = Grid(
                0,
                np.amax(self.ph_eigvals) + ph_delta,
                default_phsmear, 
                default_phpoints,
                unit='meV'
            )
            
    @mpi_watch
    def scatter_gkq_vals(self):
        # scatter gkq values to cpu-s over nkpt dimension
        # TODO: think about scattering over nqpt as well
        if not master:
            self.gkq_vals = None
        # find dimension of chunks per cpu and number dimension displacemen
        ave, res = divmod(self.nkpt, size)
        counts = np.array([ave + 1 if p < res else ave for p in range(size)], dtype=int)
        # self because will call it later
        self.__displ = np.array([sum(counts[:p]) for p in range(size)], dtype=int)
        # allocate space for chunk
        self.gkq_chunk = np.empty((counts[rank], self.nqpt, self.nbranch, self.nband, self.nband))
        rest_dims = self.nqpt*self.nbranch*self.nband*self.nband
        comm.Scatterv([self.gkq_vals, counts*rest_dims, self.__displ*rest_dims, MPI.DOUBLE], self.gkq_chunk, root=0)

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
        a2f_chunk = get_a2f_chunk(self.gkq_chunk, kpts_chunk, self.g_qpts, 
                                  self.e_eigvals, self.ph_eigvals,
                                  self.egrid, self.e1grid, self.phgrid)
        comm.Reduce([a2f_chunk, MPI.DOUBLE], [self.a2f_vals, MPI.DOUBLE], op=MPI.SUM, root=0)       
    
    @master_only
    def write_netcdf(self):
        # multiply Eliashberg function to (almost) correct prefactor
        self.a2f_vals = self.a2f_vals * (2/self.nkpt/self.nqpt) # division to g(Ef) is up to user!
        # save computed Eliashberg function to netCDF file
        ncout = Dataset(self.out_file,'w') 
        ncout.createDimension('number_of_epoints', self.egrid.npoints)
        ncout.createDimension('number_of_e1points',self.e1grid.npoints)
        ncout.createDimension('number_of_frequencies', self.phgrid.npoints)
        evar = ncout.createVariable('energy','float64',('number_of_epoints'))
        evar[:] = self.egrid.grid
        e1var = ncout.createVariable('energy_pr','float64',('number_of_e1points'))
        e1var[:] = self.e1grid.grid
        phvar = ncout.createVariable('frequency','float64',('number_of_frequencies'))
        phvar[:] = self.phgrid.grid
        a2fvar = ncout.createVariable('a2f','float64',('number_of_epoints','number_of_e1points','number_of_frequencies'))
        a2fvar.setncattr('units','mm')
        a2fvar[:] = self.a2f_vals
        ncout.close()      
        
    def compute_a2f(self):
        # gather all previous methods together into united workflow
        self.read_data()
        self.broadcast_dims()
        self.broadcast_kqpoints()
        self.broadcast_eigvals()
        self.scatter_gkq_vals()
        self.sum_chunks()
        self.write_netcdf()
