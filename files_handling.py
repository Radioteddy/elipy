import numpy as np
import netCDF4 as nc

def get_electron_eigenvalues_kpoints(eigenval_path: str) -> tuple([np.ndarray, np.ndarray]):
    if 'EIG' in eigenval_path:
        file=nc.Dataset(eigenval_path, 'r')
        eignvalues = np.ma.getdata(file.variables['Eigenvalues'][:])
        kpoints = np.ma.getdata(file.variables['Kptns'][:])
        file.close()
    elif ('GSTORE' in eigenval_path) or ('A2F' in eigenval_path):
        file=nc.Dataset(eigenval_path, 'r')
        # currently for spin-independent eigenvalues only
        eignvalues = np.ma.getdata(file.variables['eigenvalues'][:])[0,:,:]
        kpoints = np.ma.getdata(file.variables['reduced_coordinates_of_kpoints'][:])
        file.close()
    else:
        raise ValueError('incorrect netCDF file!')
    return eignvalues, kpoints

def get_phonon_eigenvalues(eigenval_path: str) -> tuple([np.ndarray, np.ndarray]):
    if 'PHFRQ' in eigenval_path:
        eignvalues = np.loadtxt(eigenval_path, usecols=(1,2,3))
    else:
        raise ValueError('incorrect phonon modes file!')
    return eignvalues

def get_gkq_values(gkq_path: str) -> tuple([np.ndarray, np.ndarray, np.ndarray]):
    if 'GSTORE' in gkq_path:
        file = nc.Dataset(gkq_path, 'r')
        if len(file.groups) == 1:
            gkq = file.groups['gqk_spin1'].variables['gvals']
            gkq_vals = np.ma.getdata(gkq[:])
            # we don't need last dimension since we deal with |g|^2
            gkq_vals = gkq_vals.reshape(gkq_vals.shape[:-1]) 
            kpoints = np.ma.getdata(file.variables['gstore_kbz'][:])
            qpoints = np.ma.getdata(file.variables['gstore_qbz'][:])
            assert (gkq_vals.shape[0], gkq_vals.shape[1]) == (qpoints.shape[0], kpoints.shape[0]), "only k(bz) q(bz) gkq file is currently supported"
        else:
            raise ValueError('Spin-dependent gkq is not supported yet')
        file.close()
    else:
        raise ValueError('incorrect netCDF file!')
    return gkq_vals, kpoints, qpoints