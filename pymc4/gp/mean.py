"""
Mean functions for PyMC4's Gaussian Process Module.

"""

import tensorflow as tf
from .util import _default_dtype

__all__ = ["Zero", "Constant"]


class Mean:
    r"""Base Class for all the mean functions in GP."""

    def __init__(self, feature_ndims=1, dtype=_default_dtype()):
        self.feature_ndims = feature_ndims
        self._dtype = dtype

    def __call__(self, X):
        raise NotImplementedError("Your mean function should override this method")

    def __add__(self, mean2):
        return MeanAdd(self, mean2)

    def __mul__(self, mean2):
        return MeanProd(self, mean2)

    @property
    def dtype(self):
        return self._dtype


class MeanAdd(Mean):
    r"""Addition of two or more mean functions

    Parameters
    ----------
    mean1 : callable, pm.Mean
        First mean function
    mean2 : callable, pm.Mean
        Second mean function
    """

    def __init__(self, mean1, mean2):
        self.mean1 = mean1
        self.mean2 = mean2

    def __call__(self, X):
        return self.mean1(X) + self.mean2(X)


class MeanProd(Mean):
    r"""Product of two or more mean functions

    Parameters
    ----------
    mean1 : callable, pm.Mean
        First mean function
    mean2 : callable, pm.Mean
        Second mean function
    """

    def __init__(self, mean1, mean2):
        self.mean1 = mean1
        self.mean2 = mean2

    def __call__(self, X):
        return self.mean1(X) * self.mean2(X)


class Zero(Mean):
    r"""Zero mean

    Parameters
    ----------
    feature_ndims : int, optional
        number of rightmost dims to include in mean computation
    """

    def __call__(self, X):
        X = tf.convert_to_tensor(X)
        return tf.zeros(X.shape[: -self.feature_ndims], dtype=self._dtype)


class Constant(Mean):
    r"""Constant mean

    Parameters
    ----------
    coef : tensor, array-like, optional
        co-efficient to scale the mean
    feature_ndims : int, optional
        number of rightmost dims to include in mean computation
    """

    def __init__(self, coef=1.0, feature_ndims=1, dtype=_default_dtype()):
        self.coef = coef
        super().__init__(feature_ndims=feature_ndims, dtype=dtype)

    def __call__(self, X):
        X = tf.convert_to_tensor(X)
        return tf.ones(X.shape[: -self.feature_ndims], dtype=self._dtype) * self.coef
