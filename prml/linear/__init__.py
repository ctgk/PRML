"""Linear machine learning models."""

from prml.linear._bayesian_logistic_regression import (
    BayesianLogisticRegression,
)
from prml.linear._bayesian_regression import BayesianRegression
from prml.linear._empirical_bayes_regression import EmpiricalBayesRegression
from prml.linear._fishers_linear_discriminant import FishersLinearDiscriminant
from prml.linear._least_squares_classifier import LeastSquaresClassifier
from prml.linear._linear_regression import LinearRegression
from prml.linear._logistic_regression import LogisticRegression
from prml.linear._perceptron import Perceptron
from prml.linear._ridge_regression import RidgeRegression
from prml.linear._softmax_regression import SoftmaxRegression
from prml.linear._variational_linear_regression import (
    VariationalLinearRegression,
)
from prml.linear._variational_logistic_regression import (
    VariationalLogisticRegression,
)


__all__ = [
    "BayesianLogisticRegression",
    "BayesianRegression",
    "EmpiricalBayesRegression",
    "LeastSquaresClassifier",
    "LinearRegression",
    "FishersLinearDiscriminant",
    "LogisticRegression",
    "Perceptron",
    "RidgeRegression",
    "SoftmaxRegression",
    "VariationalLinearRegression",
    "VariationalLogisticRegression",
]
