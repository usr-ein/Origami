import unittest

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import timedelta
from time import time
from copy import deepcopy

from mlmodel import AutoRegModel


class TestAutoRegModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        df = pd.read_csv("tests/data_tests.csv")
        df['intervalTime(ms)'] = pd.to_datetime(df['intervalTime(ms)'], unit='ms')
        df.set_index(df['intervalTime(ms)'], inplace=True)
        interestings_vals = [
            'cfn-AutoScalingGroupFO_threads-Value',
            'cfn-DevOpsEC2Instance_cpuaverage-Value',
            'cfn-DMZEC2Instance_cpuaverage-Value',
            'cfn-MOEC2Instance_threads-Value',
            'rds_databaseconnections-Value',
            'cfn-MOEC2Instance_cpuaverage-Value',
            'cfn-DevOpsEC2Instance_threads-Value',
        ]
        df = df[interestings_vals]
        cls.df = df
        cls.original_model = AutoRegModel((len(df.columns),), (len(df.columns),))

    def setUp(self) -> None:
        self.model = deepcopy(self.original_model)

    def tearDown(self) -> None:
        self.model.clear_cache()

    def test_train(self):
        self.assertRaises(AssertionError, self.model.train, self.df.values[:200], max_lag=300)
        self.assertRaises(AssertionError, self.model.train, self.df, max_lag=300)
        self.assertRaises(AssertionError, self.model.train, self.df[self.df.columns[:-1]].values, max_lag=300)

        self.model.train(self.df[:1000].values, max_lag=300)

    def test_predict(self):
        self.model.train(self.df[:1000].values, max_lag=300)

        self.assertRaises(AssertionError, self.model.predict, self.df.values[:200], steps=5)
        self.assertRaises(AssertionError, self.model.predict, self.df, steps=5)
        self.assertRaises(AssertionError, self.model.predict, self.df[self.df.columns[:-1]].values, steps=5)
        self.assertRaises(AssertionError, self.model.predict, self.df.values[:301], steps=-1)
        self.model.predict(self.df.values[:301], steps=0)
        self.model.predict(self.df.values[:301], steps=30)

    def test_predict_duration(self):
        self.model.train(self.df[:1000].values, max_lag=300)

        duration = timedelta(hours=2)
        inputs = self.df[:1000]
        df_res = self.model.predict_duration(inputs, duration=duration, return_dataframe=True)
        np_res = self.model.predict_duration(inputs, duration=duration, return_dataframe=False)
        self.assertIsInstance(df_res, pd.DataFrame)
        self.assertIsInstance(np_res, np.ndarray)
        self.assertEqual(len(df_res), np_res.shape[0])

        assert all(idx > inputs.index[-1] for idx in df_res.index)

        assert (df_res.index[-1] - df_res.index[0]).to_pytimedelta() >= duration

    def test_prediction_caching(self):
        self.model.train(self.df[:1000].values, max_lag=300)

        t1 = time()
        p1 = self.model.predict(self.df.values[:301], steps=30)
        dt_1 = time() - t1

        t2 = time()
        p2 = self.model.predict(self.df.values[:301], steps=30)
        dt_2 = time() - t2

        t3 = time()
        p3 = self.model.predict(self.df.values[:301], steps=30)
        dt_3 = time() - t3

        t4 = time()
        self.model.predict(self.df.values[:301], steps=70)
        dt_4 = time() - t4

        p5 = self.model.predict(self.df.values[100:500], steps=30)

        self.assertLess(dt_2, dt_1)
        self.assertAlmostEqual(dt_2, dt_3, delta=dt_1/10)
        self.assertGreater(dt_4, dt_2)
        self.assertGreater(dt_4, dt_1)

        assert all([np.equal(p, p1).all() for p in [p2, p3]])
        assert not any([np.equal(p, p5).all() for p in [p1, p2, p3]])

    def test_dump_load(self):
        self.model.train(self.df[:1000].values, max_lag=300)

        def chrono_model(model):
            """ Times the model's prediction """
            t = time()
            pred = model.predict(self.df.values[100:500], steps=30)
            dt = time() - t
            return pred, dt

        def reload_model(self):
            """ Dumps, del and loads the model from a file """
            fp = Path(f"test_model{int(time())}.joblib.gz")
            self.model.dump(fp)
            del self.model
            self.model = AutoRegModel.load(fp)
            os.remove(fp)

        pred_1, dt_1 = chrono_model(self.model)
        pred_2, dt_2 = chrono_model(self.model)

        reload_model(self)

        pred_3, dt_3 = chrono_model(self.model)
        pred_4, dt_4 = chrono_model(self.model)

        reload_model(self)

        pred_5, dt_5 = chrono_model(self.model)

        # Cache before and after dumping/reloading isn't shared
        # however, once the model has been dumped/reloaded once, it is shared
        # I suspect that's because the dumped/reloaded versions are mapped to disk
        # while the freshly instantiated one isn't.

        self.assertGreater(dt_1/2, dt_2)
        self.assertAlmostEqual(dt_1, dt_3, delta=dt_2)
        self.assertAlmostEqual(dt_2, dt_4, delta=dt_2/10)
        self.assertAlmostEqual(dt_4, dt_5, delta=dt_2/10)

        self.assertTrue(np.equal(pred_1, pred_3).all())


if __name__ == '__main__':
    unittest.main()
