""" Serialization relation stuff """
from __future__ import annotations
from abc import ABC
from typing import final, Any
from pathlib import Path

import os
import joblib # type: ignore

from .helpers import FileTypeError, PathType, convert_path

class Serializable(ABC):
    """ Interface to render a class serializable easily. """

    @final
    def dump(self, filepath: PathType, *args: Any, **kwargs: Any) -> None:
        """ Dumps the model and its cache into the designated file.

        :param filepath: Path to the file where to save the model.
         Should be in a writable, existing directory.
         With the default implementation provided, the compression method
         corresponding to one of the supported filename extensions
         (‘.z’, ‘.gz’, ‘.bz2’, ‘.xz’ or ‘.lzma’) will be used automatically

         see `joblib.dump <https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html#joblib.dump>`_.
        """
        filepath = convert_path(filepath)
        if not filepath.parent.is_dir():
            raise FileTypeError(f"{filepath.parent} is not in a valid directory.")
        if not os.access(str(filepath.parent.resolve().absolute()), os.W_OK):
            raise FileTypeError(f"{filepath.parent} is not writable.")
        self._dump(filepath, *args, **kwargs)

    @classmethod
    @final
    def load(cls, filepath: PathType, *args: Any, **kwargs: Any) -> Serializable:
        """ Loads back an instance of this class and its cache
            from the designated file.

        :param filepath: Path to the file where the model was saved.
         Should be readable.
        """
        filepath = convert_path(filepath)
        if not filepath.is_file():
            raise FileTypeError(f"{filepath} is not in a valid file.")
        if not os.access(str(filepath.resolve().absolute()), os.R_OK):
            raise FileTypeError(f"{filepath} is not readable.")
        obj = cls._load(filepath, *args, **kwargs)
        assert isinstance(obj, cls)
        return obj

    def _dump(self, filepath: Path, *args: Any, **kwargs: Any) -> None:
        """ Implementation of how to dump the instance to a file. """
        joblib.dump(value=self, filename=filepath)

    @classmethod
    def _load(cls, filepath: Path, *args: Any, **kwargs: Any) -> Serializable:
        """ Implementation of how to load the instance from a file. """
        return joblib.load(filename=filepath)
