from datetime import datetime, timedelta
# import importlib.metadata
from .mpi import master_only

@master_only
def print_header():
    # __version__ = importlib.metadata.version("elipy")
    __version__ = "0.1.0"
    header_message = f"""
elipy v{__version__} -- post-processing tool for ABINIT EPH package
Started at: {datetime.now()}
    """
    print(header_message)

@master_only
def print_read_status(e_file, w_file, g_file):
    read_status_message = f"""
Electron energy values: {str(e_file)}
Phonon frequency values: {str(w_file)}
Electron-phonon matrix elements: {str(g_file)}
    """
    print(read_status_message)
   
@master_only   
def print_variables(egrid, e1grid, phgrid):
    variables_message = f"""
-------------------------------------------------------------------------------------
                    Variables that govern the present computation
-------------------------------------------------------------------------------------
e_grid:
    e_window    {egrid.g_emin}  {egrid.g_emax} {egrid.unit}
    e_smearing  {egrid.smear} {egrid.unit}
    e_npoints    {egrid.npoints}
e1_grid:
    e1_window    {e1grid.g_emin}  {e1grid.g_emax} {e1grid.unit}
    e1_smearing  {e1grid.smear} {e1grid.unit}
    e1_npoints    {e1grid.npoints}
ph_grid:
    ph_window    {phgrid.g_emin}  {phgrid.g_emax} {phgrid.unit}
    ph_smearing  {phgrid.smear} {phgrid.unit}
    ph_npoints    {phgrid.npoints}
    """
    print(variables_message)

@master_only
def print_computation():
    computation_message = """
-------------------------------------------------------------------------------------
                                Computation progress
-------------------------------------------------------------------------------------
    """
    print(computation_message)

@master_only
def print_save_status(out_file):
    save_status_message = f"""
Writing energy-resolved Eliashberg function values to netcdf file: {str(out_file)}
    """
    print(save_status_message)
    
@master_only    
def print_complete(elapsed):
    complete_message = f"""
Calculation completed.
Calculation time is: {str(timedelta(seconds=elapsed))}
    """
    print(complete_message)