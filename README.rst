=======
Origami
=======

A sturdy interface for predictive ML models

Goals
-----

The goal of this module is to provide a standard way to interchange
data with ML inputs, facilitate the production cycle and
generally avoid common pitfalls.

Here are a few of these:

- Output data shape from prediction doesn't match what the caller expects
- Calling the prediction method multiple times with the same input which could've
  been cached
- Hard to dump/load the model effectively due to the variety of ML models
- Hard to swap between multiple models that need to fulfill the same task anyway
- Overwhelming number of model parameters for training and prediction, need to read
  the whole doc just to know which one are not interesting.
- Enforcing typing isn't easy


Data handling
-------------

In data science, data formats are the bread and butter of every project.
They are usually disregarded by junior data scientist as a loss of time, or
at least as something that should *just get the job done*.
However, as the name imply, 99.9% of datascience is just about coercing the
problem into the right data format. Once it's right, everything else is almost
automatic.

Of course, you could have funky models, GANs and whatnots but in the end, any
good ol' model could've cut it and you realise that most of the time you spent
was on designing a good data format.

Having that sudden realisation made me start designing a wrapper around Numpy's 
``ndarray`` specially designed for ML, but five days into this project, as I 
was researching Numpy's structured dtypes, I stumbled upon links to some very
well implemented higher level interfaces for ``ndarray``.

As it turns out, this kind of challenge has been relevant in another branch of
science for `at least two decade now <https://cdf.gsfc.nasa.gov/html/FAQ.html>`_ : climatology.
From their need, other science people developped tools like NetCDF, and eventually
`xarray <https://xarray.pydata.org/en/stable/why-xarray.html>`_ which, when discovering it,
I just dumped all my sources into a "Google First" folder.

The point of this is, CS is all about building on the shoulder of giants, and I happend to
stumbled upon one, so this project won't deal with anything of lower level than xarray's
objects.

The fight for good data format in ML isn't over yet though, and emphasizing as much models as
data format is a task any good ML library should strive for, even though it's not as sexy as 
faster, better, stronger models.
