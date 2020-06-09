""" DataModel related stuff """
from __future__ import annotations
from typing import TypeVar, Generic
from abc import ABC, abstractmethod
from pathlib import Path

from .Serializable import Serializable

class IntegrityError(ValueError):
    """ Raised by a data model when provided with
        data that fail the integrity test.
        Use this as a base class for more specific
        integrity error ! """

# DataType typevar
DT = TypeVar('DT')

class DataModel(Serializable, Generic[DT], ABC):
    """ An interface to check data integrity automatically """

    def __init__(self, *args, **kwargs):
        """ Default DataModel init just adds the kwargs as new attributes
            and checks for integrity. """
        self.__dict__.update(kwargs)

        if not self.check_integrity():
            raise IntegrityError("Data failed integrity check when "\
                                 f"initializing the {self.__class__.__name__} object.")

    @abstractmethod
    def check_integrity(self) -> bool:
        """ Checks the data integrity """
        raise NotImplementedError()

    @abstractmethod
    def _dump(self, filepath: Path, *args, **kwargs) -> None:
        """ Implementation of how to dump the data to a file. """
        raise NotImplementedError()

    @abstractmethod
    @classmethod
    def _load(cls, filepath: Path, *args, **kwargs) -> DataModel:
        """ Implementation of how to load the data from a file. """
        raise NotImplementedError()
