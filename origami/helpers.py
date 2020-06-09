""" Helper functions and types for internal use only """

from typing import Union
from os import PathLike
from pathlib import Path

class PathTypeError(ValueError):
    """ A generic exception raised when the provided value
        for a path isn't in a type that can be converted
        into a Path. """

class FileTypeError(FileNotFoundError):
    """ Exception raised when the provided path doesn't
        lead to the expected file type.
        E.g. the function expected a directory but got a
        regular file. """


PathType = Union[PathLike, str]

def convert_path(path: PathType) -> Path:
    """ Tries to convert a PathType into
        a plausible path (if necessary) """
    if isinstance(path, (PathLike, str)):
        return Path(path)
    raise PathTypeError("Path should be either be a string or "\
                        "implement the PathLike interface !")
