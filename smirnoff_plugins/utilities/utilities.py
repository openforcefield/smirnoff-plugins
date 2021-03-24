import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Optional


@contextmanager
def temporary_cd(directory_path: Optional[str] = None):
    """Temporarily move the current working directory to the path
    specified. If no path is given, a temporary directory will be
    created, moved into, and then destroyed when the context manager
    is closed.

    Parameters
    ----------
    directory_path:
        The optional path to change to. If none is specified a random temporary
        directory will be changed to.
    """

    if directory_path is not None and len(directory_path) == 0:
        yield
        return

    old_directory = os.getcwd()

    try:

        if directory_path is None:

            with TemporaryDirectory() as new_directory:
                os.chdir(new_directory)
                yield

        else:

            os.chdir(directory_path)
            yield

    finally:
        os.chdir(old_directory)
