from pathlib import Path

import numpy as np
import netCDF4 as nc

from .grid import Grid
from .kpt_utils import get_eigvals_bz

def check_in_path(path: str|Path, file_type: str = None) -> Path:
    """check_in_path converts input file path to pathlib.Path if necessary and makes it absolute

    Parameters
    ----------
    path : str | Path
        file path
    file_type : str
        which file

    Returns
    -------
    Path
        absolute path to file
    """    
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
    
def check_out_path(path: str|Path) -> Path:
    """check_out_path converts output file path to pathlib.Path if necessary and makes it absolute

    Parameters
    ----------
    path : str | Path
        file path

    Returns
    -------
    Path
        absolute path to file
    """
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
                
def get_electron_eigenvalues_kpoints(eigenval_path: str|Path) -> tuple([np.ndarray, np.ndarray]):
    """get_electron_eigenvalues_kpoints reads netCDF4 file with electron eigenvalues

    Parameters
    ----------
    eigenval_path : str | Path
        path to eigenvalues file

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        arrays of electron eigenvalues and k-points
    """
    eigenval_path = check_in_path(eigenval_path, 'Electron eigenvalues')
    if 'EIG' in eigenval_path.name:
        file=nc.Dataset(eigenval_path, 'r')
        eigenvalues = np.ma.getdata(file.variables['Eigenvalues'][:])[0,:,:]
        kpoints = np.ma.getdata(file.variables['Kptns'][:])
        file.close()
    elif ('GSTORE' in eigenval_path.name) or ('A2F' in eigenval_path.name):
        file=nc.Dataset(eigenval_path, 'r')
        # currently for spin-independent eigenvalues only
        eigenvalues = np.ma.getdata(file.variables['eigenvalues'][:])[0,:,:]
        kpoints = np.ma.getdata(file.variables['reduced_coordinates_of_kpoints'][:])
        file.close()
    else:
        raise ValueError('incorrect netCDF file!')
    return eigenvalues, kpoints

def get_phonon_eigenvalues(eigenval_path: str|Path) -> np.ndarray:
    """get_phonon_eigenvalues reads acsii file with phonon eigenvalues

    Parameters
    ----------
    eigenval_path : str | Path
        path to eigenvalues file

    Returns
    -------
    numpy.ndarray
        array of phonon eigenvalues
    """    
    eigenval_path = check_in_path(eigenval_path, 'Phonon eigenvalues')
    if 'PHFRQ' in eigenval_path.name:
        eigenvalues = np.loadtxt(eigenval_path, usecols=(1,2,3))
    else:
        raise ValueError('incorrect phonon modes file!')
    return eigenvalues

def get_gkq_values(gkq_path: str) -> dict:
    """get_gkq_values reads netCDF4 file with eph matrix elements

    Parameters
    ----------
    gkq_path : str
        path to |g|^2 file 

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.float_)
        arrays with |g|^2 values, k- and q-points, Fermi energy
    """    
    gkq_path = check_in_path(gkq_path, 'EPH matrix elements')
    read_data = {}
    if 'GSTORE' in gkq_path.name:
        file = nc.Dataset(gkq_path, 'r')
        # read electron eigenenergies in IBZ and restore values for full BZ
        k_ibz2bz = np.ma.getdata(file.variables['gstore_kbz2ibz'][:])
        e_eigs_ibz = np.ma.getdata(file.variables['eigenvalues'][:])[0,:,:]
        e_eigs_bz = get_eigvals_bz(e_eigs_ibz, k_ibz2bz)
        read_data['e_eigs'] = e_eigs_bz
        # read phonon frequencies in IBZ and restore values for full BZ
        q_ibz2bz = np.ma.getdata(file.variables['gstore_qbz2ibz'][:])
        ph_eigs_ibz = np.ma.getdata(file.variables['phfreqs_ibz'][:])
        ph_eigs_bz = get_eigvals_bz(ph_eigs_ibz, q_ibz2bz)
        read_data['ph_eigs'] = ph_eigs_bz
        if len(file.groups) == 1:
            gkq = file.groups['gqk_spin1'].variables['gvals']
            gkq_vals = np.ma.getdata(gkq[:])
            # check if we got g instead of |g|^2 and calculate |g|^2
            if gkq_vals.shape[-1] == 2:
                gkq_vals_temp = np.empty(gkq_vals.shape[:-1])
                gkq_vals_temp = gkq_vals[:,:,:,:,:,0]**2 + gkq_vals[:,:,:,:,:,1]**2
                gkq_vals = gkq_vals_temp
            else:
                # since we deal with |g|^2 we don't need last dimension
                gkq_vals = gkq_vals.reshape(gkq_vals.shape[:-1]) 
            kpoints = np.ma.getdata(file.variables['gstore_kbz'][:])
            qpoints = np.ma.getdata(file.variables['gstore_qbz'][:])
            fermi_energy = np.float64(np.ma.getdata(file.variables['fermi_energy'][:]))
            assert (gkq_vals.shape[0], gkq_vals.shape[1]) == (qpoints.shape[0], kpoints.shape[0]), "only k(bz) q(bz) gkq file is currently supported"
            read_data['kpts'] = kpoints
            read_data['qpts'] = qpoints
            read_data['gkq'] = gkq_vals
            read_data['efermi'] = fermi_energy
        else:
            raise ValueError('Spin-dependent gkq is not supported yet')
        file.close()
    else:
        raise ValueError('incorrect netCDF file!')
    return read_data

def store_a2f_values(out_path: str|Path, efermi: np.float_,
                     egrid: Grid, e1grid: Grid, phgrid: Grid, a2f_vals: np.ndarray) -> None:
    """store_a2f_values saves output data no netCDF4 file

    Parameters
    ----------
    out_path : str | Path
        path to output file
    efermi : np.float_
        Fermi energy of a system
    egrid : Grid
        electron energy grid e
    e1grid : Grid
        electron energy grid e'
    phgrid : Grid
        phonon frequency grid w
    a2f_vals : np.ndarray
        a2F(e,e',w) values
    """
    out_path = check_out_path(out_path)
    ncout = nc.Dataset(out_path,'w')
    efvar = ncout.createVariable('Fermi_energy', 'float64') 
    efvar[0] = efermi
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
