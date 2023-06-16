from datetime import datetime, timedelta
import importlib.metadata
from pathlib import Path
from .grid import Grid
from .mpi import master_only, size

@master_only
def print_header():
    __version__ = importlib.metadata.version("elipy")
    # __version__ = "0.1.8"
    header_message = f"""
elipy v{__version__} -- post-processing tool for ABINIT EPH package
Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """
    print(header_message, flush=True)

@master_only
def print_mpi_info(num_kpoints: int) -> None:
    mpi_message = f"""
Number of cpu-s used is: {size}
K-point parallelization: avg. {num_kpoints} k-points per cpu
    """
    print(mpi_message, flush=True)

@master_only
def print_read_status(g_file: Path) -> None:
    read_status_message = f"""
Electron-phonon matrix elements, electron and phonon eigenvalues:\n{str(g_file)}
    """
    print(read_status_message, flush=True)
   
@master_only   
def print_variables(egrid: Grid, phgrid: Grid, e1grid: Grid = None) -> None:
    if e1grid:
        variables_message = f"""
    -------------------------------------------------------------------------------------
                        Variables that govern the present computation
    -------------------------------------------------------------------------------------

    all values in atomic units

    e_grid:
        e_window    {egrid.g_min:.3e}  {egrid.g_max:.3e}
        e_smearing  {egrid.smear:.3e} 
        e_npoints    {egrid.npoints}
    e1_grid:
        e1_window    {e1grid.g_min:.3e}  {e1grid.g_max:.3e}
        e1_smearing  {e1grid.smear:.3e} 
        e1_npoints    {e1grid.npoints}
    ph_grid:
        ph_window    {phgrid.g_min:.3e}  {phgrid.g_max:.3e}
        ph_smearing  {phgrid.smear:.3e} 
        ph_npoints    {phgrid.npoints}
    -------------------------------------------------------------------------------------

    -------------------------------------------------------------------------------------
        """
    else:
        variables_message = f"""
    -------------------------------------------------------------------------------------
                        Variables that govern the present computation
    -------------------------------------------------------------------------------------

    all values in atomic units

    e_grid:
        e_window    {egrid.g_min:.3e}  {egrid.g_max:.3e}
        e_smearing  {egrid.smear:.3e} 
        e_npoints    {egrid.npoints}
    ph_grid:
        ph_window    {phgrid.g_min:.3e}  {phgrid.g_max:.3e}
        ph_smearing  {phgrid.smear:.3e} 
        ph_npoints    {phgrid.npoints}
    -------------------------------------------------------------------------------------

    -------------------------------------------------------------------------------------
        """
        
    print(variables_message, flush=True)

@master_only
def print_computation() -> None:
    computation_message = """
-------------------------------------------------------------------------------------
                                Computation progress
-------------------------------------------------------------------------------------
    """
    print(computation_message, flush=True)

@master_only
def print_save_status(out_file: Path) -> None:
    save_status_message = f"""
Writing energy-resolved Eliashberg function values to netcdf file: {str(out_file)}
    """
    print(save_status_message, flush=True)
    
@master_only    
def print_complete(elapsed: float) -> None:
    complete_message = f"""
Calculation completed.
Calculation time is: {str(timedelta(seconds=elapsed))}
    """
    print(complete_message, flush=True)