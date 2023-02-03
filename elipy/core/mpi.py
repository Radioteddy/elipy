"""This file is taken from https://github.com/GkAntonius/ElectronPhononCoupling/blob/master/ElectronPhononCoupling"""

"""MPI distribution of work load"""
import sys, traceback
from contextlib import contextmanager
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

master = bool(rank == 0)

@contextmanager
def mpi_abort_if_exception():
    """mpi_abort_if_exception terminates all mpi process if an exception is raised.
    """
    try:
        yield
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        MPI.COMM_WORLD.Abort(1)

def mpi_watch(f):
    """mpi_watch is decorator. Terminates all mpi process if an exception is raised.

    Parameters
    ----------
    f : function

    Returns
    -------
    function
    """
    def g(*args, **kwargs):
        with mpi_abort_if_exception():
            return f(*args, **kwargs)
    return g

def master_only(f):
    """master_only is ecorator. Lets a function be executed only by master.

    Parameters
    ----------
    f : function

    Returns
    -------
    function
    """
    def g(*args, **kwargs):
        if master:
            with mpi_abort_if_exception():
                return f(*args, **kwargs)
        return
    return g