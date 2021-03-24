import os

from smirnoff_plugins.utilities import temporary_cd


def compare_paths(path_1: str, path_2: str) -> bool:
    """Checks whether two paths are the same.

    Parameters
    ----------
    path_1
        The first path.
    path_2
        The second path.

    Returns
    -------
        True if the paths are equivalent.
    """
    return os.path.normpath(path_1) == os.path.normpath(path_2)


def test_temporary_cd():
    """Tests that temporary cd works as expected"""

    original_directory = os.getcwd()

    # Move to the parent directory
    with temporary_cd(os.pardir):

        current_directory = os.getcwd()
        expected_directory = os.path.abspath(
            os.path.join(original_directory, os.pardir)
        )

        assert compare_paths(current_directory, expected_directory)

    assert compare_paths(os.getcwd(), original_directory)

    # Move to a temporary directory
    with temporary_cd():
        assert not compare_paths(os.getcwd(), original_directory)

    assert compare_paths(os.getcwd(), original_directory)

    # Move to the same directory
    with temporary_cd(""):
        assert compare_paths(os.getcwd(), original_directory)

    assert compare_paths(os.getcwd(), original_directory)

    with temporary_cd(os.curdir):
        assert compare_paths(os.getcwd(), original_directory)

    assert compare_paths(os.getcwd(), original_directory)
