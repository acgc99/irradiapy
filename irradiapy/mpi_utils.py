"""Utilities for MPI-related functionalities."""

# pylint: disable=no-name-in-module, broad-except

import sys
import traceback
from functools import wraps
from pathlib import Path

from mpi4py import MPI


def broadcast_variables(root: int, comm: MPI.Comm, *variables) -> list:
    """Broadcasts variables.

    Parameters
    ----------
    root : int
        The rank of the process that will broadcast the variables.
    comm : MPI.Comm
        The MPI communicator.
    variables : tuple
        The variables to broadcast.

    Returns
    -------
    list
        The broadcasted variables.
    """
    return [comm.bcast(var, root=root) for var in variables]


def rm_file(path_file: Path, comm: MPI.Comm) -> None:
    """Remove a file."""
    rank = comm.Get_rank()
    if rank == 0:
        path_file.unlink(missing_ok=True)
    comm.Barrier()


def cp_file(original: Path, target: Path, comm: MPI.Comm) -> None:
    """Copy a file from `original` to `target`, overwriting `target` if it exists."""
    rank = comm.Get_rank()
    if rank == 0:
        print(f"\n\n\nCopying {original} to {target}")
        target.write_bytes(original.read_bytes())
    comm.Barrier()


def mv_file(original: Path, target: Path, comm: MPI.Comm) -> None:
    """Move a file from `original` to `target`, overwriting `target` if it exists."""
    rank = comm.Get_rank()
    if rank == 0:
        target.write_bytes(original.read_bytes())
        original.unlink()
    comm.Barrier()


def ap_rm_file(original: Path, target: Path, comm: MPI.Comm) -> None:
    """Append content from `original` to `target` and delete `original`."""
    rank = comm.Get_rank()
    if rank == 0:
        original_content = original.read_bytes()
        with target.open("ab") as f:
            f.write(original_content)
        original.unlink()
    comm.Barrier()


def mpi_safe_method(method):
    """Decorator that wraps an MPI-using method so any exception prints a
    traceback with the current rank and then calls MPI.Abort.

    The method should be a member of a class that has `comm` and `rank` attributes.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            tb = traceback.format_exc()
            sys.stderr.write(f"Rank {self.rank} raised an exception:\n{tb}\n")
            sys.stderr.flush()
            self.comm.Abort(1)

    return wrapper


class MPIExceptionHandlerMixin:
    """Provides a common MPI exception handler method when `mpi_safe_method` cannot be used.

    This is useful for `__init__` and `__post_init__` methods where decorators cannot be applied
    because `self.comm` and `self.rank` are not yet defined before the function is called. You can
    use this method after the `comm` and `rank` attributes are initialized in a try/except block.
    """

    def _handle_exception(self) -> None:
        tb = traceback.format_exc()
        sys.stderr.write(f"[Rank {self.rank}] Exception:\n{tb}\n")
        sys.stderr.flush()
        self.comm.Abort(1)


class MPITagAllocator:
    """A class to allocate unique tags for processes."""

    _next_tag = 0

    @classmethod
    def get_tag(cls):
        """Get a unique tag for the current process.

        Warning
        -------
        This method should not be called in a statement that is only executed by one rank,
        such as in `if rank == 0:`. It is designed to be called by all ranks
        to ensure that all ranks receive the same tag.
        """
        tag = cls._next_tag
        cls._next_tag += 1
        return tag
