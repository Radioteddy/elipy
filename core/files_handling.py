from pathlib import Path

import numpy as np
import netCDF4 as nc

from .grid import Grid

def check_in_path(path: str or Path, file_type: str) -> Path:
    # pathlib.Path representation of input file
    if isinstance(path, Path):
        pass
    elif isinstance(path, str):
        path = Path(path)
    else:
        raise TypeError('file_path should be str or pathlib.Path object')    
    # check if file really exists and make it absolute
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Path to {file_type} file '{str(path)}' does not exist")
    return path
    
def check_out_path(path: str or Path) -> Path:
    # pathlib.Path representation of output file
    if isinstance(path, Path):
        pass
    elif isinstance(path, str):
        path = Path(path)
    else:
        raise TypeError('file_path should be str or pathlib.Path object')
    # check if directory to output file exists, and create it if necessary    
    if not path.parents[0].exists():
        path.parents[0].mkdir(parents=True, exist_ok=True)
    return path
                
def get_electron_eigenvalues_kpoints(eigenval_path: str or Path) -> tuple([np.ndarray, np.ndarray]):
    eigenval_path = check_in_path(eigenval_path, 'Electron eigenvalues')
    if 'EIG' in eigenval_path.name:
        file=nc.Dataset(eigenval_path, 'r')
        eignvalues = np.ma.getdata(file.variables['Eigenvalues'][:])[0,:,:]
        kpoints = np.ma.getdata(file.variables['Kptns'][:])
        file.close()
    elif ('GSTORE' in eigenval_path.name) or ('A2F' in eigenval_path.name):
        file=nc.Dataset(eigenval_path, 'r')
        # currently for spin-independent eigenvalues only
        eignvalues = np.ma.getdata(file.variables['eigenvalues'][:])[0,:,:]
        kpoints = np.ma.getdata(file.variables['reduced_coordinates_of_kpoints'][:])
        file.close()
    else:
        raise ValueError('incorrect netCDF file!')
    return eignvalues, kpoints

def get_phonon_eigenvalues(eigenval_path: str or Path) -> tuple([np.ndarray, np.ndarray]):
    eigenval_path = check_in_path(eigenval_path, 'Phonon eigenvalues')
    if 'PHFRQ' in eigenval_path.name:
        eignvalues = np.loadtxt(eigenval_path, usecols=(1,2,3))
    else:
        raise ValueError('incorrect phonon modes file!')
    return eignvalues

def get_gkq_values(gkq_path: str) -> tuple([np.ndarray, np.ndarray, np.ndarray]):
    gkq_path = check_in_path(gkq_path, 'EPH matrix elements')
    if 'GSTORE' in gkq_path.name:
        file = nc.Dataset(gkq_path, 'r')
        if len(file.groups) == 1:
            gkq = file.groups['gqk_spin1'].variables['gvals']
            gkq_vals = np.ma.getdata(gkq[:])
            # we don't need last dimension since we deal with |g|^2
            gkq_vals = gkq_vals.reshape(gkq_vals.shape[:-1]) 
            kpoints = np.ma.getdata(file.variables['gstore_kbz'][:])
            qpoints = np.ma.getdata(file.variables['gstore_qbz'][:])
            fermi_energy = np.float64(np.ma.getdata(file.variables['fermi_energy'][:]))
            assert (gkq_vals.shape[0], gkq_vals.shape[1]) == (qpoints.shape[0], kpoints.shape[0]), "only k(bz) q(bz) gkq file is currently supported"
        else:
            raise ValueError('Spin-dependent gkq is not supported yet')
        file.close()
    else:
        raise ValueError('incorrect netCDF file!')
    return gkq_vals, kpoints, qpoints, fermi_energy

def store_a2f_values(out_path: str or Path, egrid: Grid, e1grid: Grid, phgrid: Grid, a2f_vals: np.ndarray) -> None:
    out_path = check_out_path(out_path)
    ncout = nc.Dataset(out_path,'w') 
    ncout.createDimension('number_of_epoints', egrid.npoints)
    ncout.createDimension('number_of_e1points',e1grid.npoints)
    ncout.createDimension('number_of_frequencies', phgrid.npoints)
    evar = ncout.createVariable('energy','float64',('number_of_epoints'))
    evar[:] = egrid.grid
    e1var = ncout.createVariable('energy_pr','float64',('number_of_e1points'))
    e1var[:] = e1grid.grid
    phvar = ncout.createVariable('frequency','float64',('number_of_frequencies'))
    phvar[:] = phgrid.grid
    a2fvar = ncout.createVariable('a2f','float64',('number_of_epoints','number_of_e1points','number_of_frequencies'))
    a2fvar.setncattr('units','mm')
    a2fvar[:] = a2f_vals
    ncout.close()      
