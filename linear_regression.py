
from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            # We have seen in lecture 2 in this case: xi = (1, x1, . . . , xd)
            ones_to_concatenate = np.ones(X.shape[0])
            concatenated_X = np.r_['-1,2,0', ones_to_concatenate, X]
            # Compute the pseudo-inverse of concatenated_X:
            pseudo_inverse_x = np.linalg.pinv(concatenated_X)
        else:
            # Compute the pseudo-inverse of concatenated_X:
            pseudo_inverse_x = np.linalg.pinv(X)
        # We have seen in lecture 2 :  wˆ = X†y
        self.coefs_ = np.matmul(pseudo_inverse_x, y)
        return

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # We have seen in lecture 2 that: Xw = y
        if self.include_intercept_:
            # We have seen in lecture 2 in this case: xi = (1, x1, . . . , xd)
            ones_to_concatenate = np.ones(X.shape[0])
            concatenated_X = np.r_['-1,2,0', ones_to_concatenate, X]
            return  np.matmul(concatenated_X, self.coefs_)
        else:
            return np.matmul(X, self.coefs_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        y_predicted =self._predict(X)
        diff = y - y_predicted
        MSE = np.mean(diff ** 2)
        return MSE
