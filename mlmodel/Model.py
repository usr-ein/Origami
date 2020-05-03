from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Tuple, final, Any
from pathlib import Path
import joblib
from joblib import Memory
import numpy as np


class Model(ABC):
    """ A base class for any predictive model """
    trained: bool = False
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    _cache_memory: Memory

    def __init__(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], *args, **kwargs):
        """
        Initialises the model object

        :param input_shape: constraint on the last dimensions of input data (for both training and predicting)
        :param output_shape: constraint on the last dimensions of the output data (for predicting only)

        The input shape should match:

        >>> model = Model(input_shape=(10,3), output_shape=(42,))
        >>> model.train(np.random.random((300, 10, 4)))
            AssertionError: Wrong shape
        >>> model.train(np.random.random((300, 10, 3)))
        >>> model.predict(np.random.random((123, 10, 4)))
            AssertionError: Wrong shape

        And it's garanteed that the output shape will match too:

        >>> pred = model.predict(np.random.random((123, 10, 3)))
        >>> assert pred.shape[-1] == 42
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._cache_memory = Memory(location=None, verbose=False)

    @final
    def train(self, data: np.ndarray, *args: Any, **kwargs: Any) -> None:
        """
        Trains the model with the provided data and arguments.

        :param data: Data with which to train, should have the same last dimensions
         as the `input_shape`
        """
        assert isinstance(data, np.ndarray)
        # Forces the input to fit in the given shape
        assert data.shape[-len(self.input_shape):] == self.input_shape, "Wrong shape"
        self._train(data, *args, **kwargs)
        self.trained = True
        # Note: Cache isn't shared between "freshly created" instances and "loaded from file" instances !
        # However, cache IS shared between all instances loaded from files
        self._cache_memory = Memory(location=f"./model_cache", verbose=False)

    @final
    def predict(self, data: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """ Predict data from the model,

        :param data: Data from which to predict, should have the same last dimensions
         as the `input_shape`
        :return: Predicted data, will have the same last dimensions
         as the `output_shape`
        """
        assert isinstance(data, np.ndarray) and self.trained
        # Forces the input to fit in the given shape
        assert data.shape[-len(self.input_shape):] == self.input_shape, "Wrong shape"
        # Tries to use self._predict's cached results first
        predict_cached = self._cache_memory.cache(self._predict)
        output = predict_cached(data, *args, **kwargs)
        # Forces the output to fit in the given shape
        assert output.shape[-len(self.output_shape):] == self.output_shape, "Wrong shape"
        return output

    @final
    def dump(self, filepath: Path, *args: Any, **kwargs: Any) -> None:
        """ Dumps the model and its cache into the designated file.

        :param filepath: Path to the file where to save the model.
         Should be in a writable, existing directory.
         With the default implementation provided, the compression method
         corresponding to one of the supported filename extensions
         (‘.z’, ‘.gz’, ‘.bz2’, ‘.xz’ or ‘.lzma’) will be used automatically

         see `joblib.dump <https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html#joblib.dump>`_.
        """
        assert filepath.parent.is_dir() and os.access(str(filepath.parent.resolve().absolute()), os.W_OK)
        self._dump(filepath, *args, **kwargs)

    @classmethod
    @final
    def load(cls, filepath: Path, *args: Any, **kwargs: Any) -> Model:
        assert filepath.is_file()
        obj = cls._load(filepath, *args, **kwargs)
        assert isinstance(obj, cls)
        return obj

    def _dump(self, filepath: Path, *args: Any, **kwargs: Any) -> None:
        joblib.dump(value=self, filename=filepath)

    @classmethod
    def _load(cls, filepath: Path, *args: Any, **kwargs: Any) -> Model:
        return joblib.load(filename=filepath)

    def clear_cache(self):
        """ Clears the model's prediction cache """
        self._cache_memory.clear(warn=False)

    @abstractmethod
    def _train(self, data: np.ndarray, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _predict(self, data: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError()
