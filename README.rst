Origami
========================

A sturdy interface for predictive ML models

---------------

The goal of this module is to provide a standard way to interchange
data with ML inputs, facilitate the production cycle and
generally avoid common pitfalls.

Here are a few of these:

- Prediction's input data shape doesn't match what the model has been train on
- Output data shape from prediction doesn't match what the caller expects
- Calling the prediction method multiple times with the same input which could've
  been cached
- Hard to dump/load the model effectively due to the variety of ML models
- Hard to swap between multiple models that need to fulfill the same task anyway
- Overwhelming number of model parameters for training and prediction, need to read
  the whole doc just to know which one are not interesting.
- Enforcing typing isn't easy
