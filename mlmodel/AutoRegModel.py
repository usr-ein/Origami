from .Model import Model
from typing import Union, Any, Dict
import numpy as np
import pandas as pd
from datetime import timedelta
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults


# noinspection PyMethodOverriding
class AutoRegModel(Model):
    """ Predicts times series based on correlation with time delayed versions """
    max_lag: int
    _model: VARResults

    def _train(self, data: np.ndarray, max_lag: int = 300, *args: Any, **kwargs: Any) -> None:
        data_fd = np.diff(data, axis=0)
        assert data_fd.shape[0] >= max_lag
        model = VAR(endog=data_fd)
        self.max_lag = max_lag
        self._model = model.fit(maxlags=max_lag, trend="n")

    def _predict(self, data: np.ndarray, steps: int = 200, *args: Any, **kwargs: Any) -> np.ndarray:
        data_fd = np.diff(data, axis=0)
        assert data_fd.shape[0] >= self.max_lag
        assert steps >= 0
        preds_fd = self._model.forecast(data_fd, steps=steps)
        preds = np.cumsum(preds_fd*1.005, axis=0) + data_fd[-1]
        preds = np.rint(preds)
        return preds

    def predict_duration(self, df: pd.DataFrame, duration: timedelta, return_dataframe: bool = True) \
            -> Union[pd.DataFrame, np.ndarray]:
        """
        Predicts values for a given duration in the future from a datetime-indexed DataFrame

        :param df: a datetime-indexed DataFrame
        :param duration: a Python's datetime.timedelta object specifying the duration to predict in the future.
         Might be rounded up to the match the median time between rows in the DataFrame.
        :param return_dataframe: Whether to return a pandas DataFrame or the plain NumPy data.
         Default: True
        :return: The predicted samples in the future

        Usage:

        >>> from matplotlib import pyplot as plt
        >>> from datetime import timedelta
        >>> df = pd.read_csv("data.csv")
        >>> model = AutoRegModel(
        ...     input_shape=(len(df.columns),),
        ...     output_shape=(len(df.columns),)
        ... )
        >>> model.train(df[:1000].values, max_lag=300)
        >>> duration = timedelta(hours=2)
        >>> df_preds = model.predict_duration(
        ...     df[:1000],
        ...     duration,
        ...     return_dataframe=True
        ... )
        >>> plt.plot(df[:1000], label="Past", c='orange')
        >>> plt.plot(df_preds, label="Future", c='green')
        >>> plt.show()

        """
        assert isinstance(df.index, pd.DatetimeIndex), "DataFrame's index should be a datetime !"
        data = df.values
        median_delta = np.median(np.diff(df.index.to_pydatetime()))
        steps = int(np.ceil(duration/median_delta)) + 1
        preds = self.predict(data, steps)
        if return_dataframe:
            indices = np.cumsum([median_delta] * steps) + df.index[-1]
            return pd.DataFrame(data=preds, index=indices, columns=df.columns)
        else:
            return preds

