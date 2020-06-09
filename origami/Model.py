""" Model class related stuff """
from __future__ import annotations

from abc import ABC, abstractmethod
import warnings
import shutil
from typing import final, Any
from joblib import Memory # type: ignore

from .Serializable import Serializable
from .DataModel import DataModel, IntegrityError

class ModelNotTrained(UserWarning):
    """ Raised when trying to use an untrained model to predict """

class Model(Serializable, ABC):
    """ A base class for any predictive model """
    trained: bool = False
    _cache_memory: Memory

    def __init__(self, *args, **kwargs):
        """ Initialises the model object """
        self._cache_memory = Memory(location=None, verbose=False)

    @final
    def train(self, data: DataModel, /, *args: Any, **kwargs: Any) -> None:
        """
        Trains the model with the provided data and arguments.

        :param data: Data with which to train
        """
        if not isinstance(data, DataModel):
            raise TypeError("Argument data should be of type DataModel,"\
                           f"found {data.__class__.__name__}")
        if not data.check_integrity():
            raise IntegrityError("Data failed integrity check before training")
        self._train(data, *args, **kwargs)
        self.trained = True
        # Note: Cache isn't shared between "freshly created" instances and "loaded from file" instances !
        # However, cache IS shared between all instances loaded from files
        self._cache_memory = Memory(location=f"./model_cache", verbose=False)

    @final
    def predict(self, data: DataModel, /, *args: Any, check_output=True, **kwargs: Any) -> DataModel:
        """ Predict data from the model,

        :param data: Data from which to predict
        :param check_output: Whether to check the prediction's output integrity.
         May be frustrating when debugging, you may want to turn this off then.
        :return: Predicted data
        """
        if not self.trained:
            raise ModelNotTrained("Trying to predict using an untrained model !"\
                                  "Results may not be satisfying.")
        if not isinstance(data, DataModel):
            raise TypeError("Argument data should be of type DataModel,"\
                           f"found {data.__class__.__name__}")
        if not data.check_integrity():
            raise IntegrityError("Data failed integrity check before predicting")

        # Tries to use self._predict's cached results first
        predict_cached = self._cache_memory.cache(self._predict)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = predict_cached(data, *args, **kwargs)
        if check_output and not output.check_integrity():
            raise IntegrityError("Prediction failed integrity check")
        return output

    def clear_cache(self, *, soft=False):
        """ Clears the model's prediction cache """
        self._cache_memory.clear(warn=False)
        if not soft:
            shutil.rmtree(self._cache_memory.location, ignore_errors=True)

    @abstractmethod
    def _train(self, data: DataModel, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _predict(self, data: DataModel, *args: Any, **kwargs: Any) -> DataModel:
        raise NotImplementedError()
