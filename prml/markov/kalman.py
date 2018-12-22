import numpy as np
from prml.rv.multivariate_gaussian import MultivariateGaussian as Gaussian
from prml.markov.state_space_model import StateSpaceModel


class Kalman(StateSpaceModel):
    """
    A class to perform kalman filtering or smoothing
    z : internal state
    x : observation

    z_1 ~ N(z_1|mu_0, P_0)\n
    z_n ~ N(z_n|A z_n-1, P)\n
    x_n ~ N(x_n|C z_n, S)

    Parameters
    ----------
    system : (Dz, Dz) np.ndarray
        system matrix aka transition matrix (A)
    cov_system : (Dz, Dz) np.ndarray
        covariance matrix of process noise
    measure : (Dx, Dz) np.ndarray
        measurement matrix aka observation matrix (C)
    cov_measure : (Dx, Dx) np.ndarray
        covariance matrix of measurement noise
    mu0 : (Dz,) np.ndarray
        mean parameter of initial hidden variable
    P0 : (Dz, Dz) np.ndarray
        covariance parameter of initial hidden variable

    Attributes
    ----------
    Dz : int
        dimensionality of hidden variable
    Dx : int
        dimensionality of observed variable
    """


    def __init__(self, system, cov_system, measure, cov_measure, mu0, P0):
        """
        construct Kalman model

        z_1 ~ N(z_1|mu_0, P_0)\n
        z_n ~ N(z_n|A z_n-1, P)\n
        x_n ~ N(x_n|C z_n, S)

        Parameters
        ----------
        system : (Dz, Dz) np.ndarray
            system matrix aka transition matrix (A)
        cov_system : (Dz, Dz) np.ndarray
            covariance matrix of process noise
        measure : (Dx, Dz) np.ndarray
            measurement matrix aka observation matrix (C)
        cov_measure : (Dx, Dx) np.ndarray
            covariance matrix of measurement noise
        mu0 : (Dz,) np.ndarray
            mean parameter of initial hidden variable
        P0 : (Dz, Dz) np.ndarray
            covariance parameter of initial hidden variable

        Attributes
        ----------
        hidden_state : list
            list of hidden states starting from the given hidden state
            [[mean, covariance], ..., [mean, covariance]]
        Dz : int
            dimensionality of hidden variable
        Dx : int
            dimensionality of observed variable
        """
        self.Dz = np.size(system, 0)
        self.Dx = np.size(measure, 0)

        self.system = system
        self.cov_system = cov_system
        self.measure = measure
        self.cov_measure = cov_measure

        self.hidden_state = [[mu0, P0]]

    def predict(self):
        """
        predict hidden state at current step given estimate at previous step

        Returns
        -------
        tuple ((Dz,) np.ndarray, (Dz, Dz) np.ndarray)
            tuple of mean and covariance of the estimate at current step
        """
        mu_prev, cov_prev = self.hidden_state[-1]
        mu = self.system @ mu_prev
        cov = self.system @ cov_prev @ self.system.T + self.cov_system
        self.hidden_state.append([mu, cov])
        return mu, cov

    def filter(self, observed):
        """
        bayesian update of current estimate given current observation

        Parameters
        ----------
        observed : (Dx,) np.ndarray
            current observation

        Returns
        -------
        tuple ((Dz,) np.ndarray, (Dz, Dz) np.ndarray)
            tuple of mean and covariance of the updated estimate
        """
        mu, cov = self.hidden_state[-1]
        innovation = observed - self.measure @ mu
        cov_innovation = self.cov_measure + self.measure @ cov @ self.measure.T
        kalman_gain = np.linalg.solve(cov_innovation, self.measure @ cov).T
        mu += kalman_gain @ innovation
        cov -= kalman_gain @ self.measure @ cov
        return mu, cov


    def smoothing(self):
        raise NotImplementedError


def kalman_filter(kalman:Kalman, observed_sequence:np.ndarray)->tuple:
    """
    perform kalman filtering given Kalman model and observed sequence

    Parameters
    ----------
    kalman : Kalman
        Kalman model
    observed_sequence : (T, Dx) np.ndarray
        sequence of observations

    Returns
    -------
    tuple ((T, Dz) np.ndarray, (T, Dz, Dz) np.ndarray)
        seuquence of mean and covariance at each time step
    """
    mean_sequence = []
    cov_sequence = []
    for obs in observed_sequence:
        kalman.predict()
        mean, cov = kalman.filter(obs)
        mean_sequence.append(mean)
        cov_sequence.append(cov)
    mean_sequence = np.asarray(mean_sequence)
    cov_sequence = np.asarray(cov_sequence)
    return mean_sequence, cov_sequence
