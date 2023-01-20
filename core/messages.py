from datetime import datetime, timedelta
# import importlib.metadata
from pathlib import Path
from .grid import Grid
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
def print_read_status(e_file: Path, w_file: Path, g_file: Path) -> None:
    read_status_message = f"""
Electron energy values: {str(e_file)}
Phonon frequency values: {str(w_file)}
Electron-phonon matrix elements: {str(g_file)}
    """
    print(read_status_message)
   
@master_only   
def print_variables(egrid: Grid, e1grid: Grid, phgrid: Grid) -> None:
    variables_message = f"""
-------------------------------------------------------------------------------------
                    Variables that govern the present computation
-------------------------------------------------------------------------------------

all values in atomic units

e_grid:
    e_window    {egrid.g_min}  {egrid.g_max}
    e_smearing  {egrid.smear} 
    e_npoints    {egrid.npoints}
e1_grid:
    e1_window    {e1grid.g_min}  {e1grid.g_max}
    e1_smearing  {e1grid.smear} 
    e1_npoints    {e1grid.npoints}
ph_grid:
    ph_window    {phgrid.g_min}  {phgrid.g_max}
    ph_smearing  {phgrid.smear} 
    ph_npoints    {phgrid.npoints}
    """
    print(variables_message)

@master_only
def print_computation() -> None:
    computation_message = """
-------------------------------------------------------------------------------------
                                Computation progress
-------------------------------------------------------------------------------------
    """
    print(computation_message)

@master_only
def print_save_status(out_file: Path) -> None:
    save_status_message = f"""
Writing energy-resolved Eliashberg function values to netcdf file: {str(out_file)}
    """
    print(save_status_message)
    
@master_only    
def print_complete(elapsed: float) -> None:
    complete_message = f"""
Calculation completed.
Calculation time is: {str(timedelta(seconds=elapsed))}
    """
    print(complete_message)