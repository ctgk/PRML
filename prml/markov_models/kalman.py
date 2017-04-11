import numpy as np
from .state_space_model import StateSpaceModel


class Kalman(StateSpaceModel):
    """
    A class to perform kalman filtering or smoothing
    """

    def __init__(
        self,
        transition,
        observation,
        process_noise,
        measurement_noise,
        init_state_mean,
        init_state_cov
    ):
        """
        construct state space model to perform kalman filtering or smoothing
        z_n ~ N(z_n|Az_n-1,Gamma)
        x_n ~ N(x_n|Cz_n,Sigma)
        z_1 ~ N(z_1|mu_0, P_0)

        Parameters
        ----------
        transition : (ndim_hidden, ndim_hidden) np.ndarray
            transition matrix of hidden variable (A)
        observation : (ndim_observe, ndim_hidden) np.ndarray
            observation matrix (C)
        process_noise : (ndim_hidden, ndim_hidden) np.ndarray
            covariance matrix of process noise (Gamma)
        measurement_noise : (ndim_observe, ndim_observe) np.ndarray
            covariance matrix of measurement noise (Sigma)
        init_state_mean : (ndim_hidden,) np.ndarray
            mean parameter of initial hidden variable (mu_0)
        init_state_cov : (ndim_hidden, ndim_hidden) np.ndarray
            covariance parameter of initial hidden variable (P_0)

        Attributes
        ----------
        ndim_hidden : int
            dimensionality of hidden variable
        ndim_observe : int
            dimensionality of observed variable
        """

        assert init_state_mean.ndim == 1
        assert (
            transition.ndim == observation.ndim == process_noise.ndim
            == measurement_noise.ndim == init_state_cov.ndim == 2
        )
        assert (
            transition.shape[0] == transition.shape[1]
            == process_noise.shape[0] == process_noise.shape[1]
            == observation.shape[1] == init_state_mean.size
            == init_state_cov.shape[0] == init_state_cov.shape[1]
        )
        assert (
            observation.shape[0] == measurement_noise.shape[0]
            == measurement_noise.shape[1]
        )

        self.ndim_hidden = init_state_mean.size
        self.ndim_observe = observation.shape[0]

        self.transition = transition
        self.process_noise = process_noise
        self.observation = observation
        self.measurement_noise = measurement_noise
        self.init_state_mean = init_state_mean
        self.init_state_cov = init_state_cov

    def filtering(self, seq):
        """
        kalman filter
        1. prediction
            p(z_n+1|x_1:n) = \int p(z_n+1|z_n)p(z_n|x_1:n)dz_n
        2. filtering
            p(z_n+1|x_1:n+1) \propto p(x_n+1|z_n+1)p(z_n+1|x_1:n)

        Parameters
        ----------
        seq : (N, ndim_observe) np.ndarray
            observed sequence

        Returns
        -------
        mean : (N, ndim_hidden) np.ndarray
            mean parameter of posterior hidden distribution
        cov : (N, ndim_hidden, ndim_hidden) np.ndarray
            covariance of posterior hidden distribution
        """
        kalman_gain = self.init_state_cov @ self.observation.T @ np.linalg.inv(
            self.observation @ self.init_state_cov @ self.observation.T
            + self.measurement_noise)
        mean = [self.init_state_mean + kalman_gain @ (seq[0] - self.observation @ self.init_state_mean)]
        cov = [(np.eye(self.ndim_observe) - kalman_gain @ self.observation) @ self.init_state_cov]
        for s in seq[1:]:
            mean_predict = self.transition @ mean[-1]
            cov_predict = (
                self.transition @ cov[-1] @ self.transition.T
                + self.process_noise)
            if np.logical_and.reduce(np.isnan(s)):
                mean.append(mean_predict)
                cov.append(cov_predict)
            else:
                kalman_gain = cov_predict @ self.observation.T @ np.linalg.inv(
                    self.observation @ cov_predict @ self.observation.T
                    + self.measurement_noise)
                mean.append(mean_predict + kalman_gain @ (s - self.observation @ mean_predict))
                cov.append(
                    (np.eye(self.ndim_observe) - kalman_gain @ self.observation)
                    @ cov_predict)
        mean = np.asarray(mean)
        cov = np.asarray(cov)
        return mean, cov

    def smoothing(self):
        raise NotImplementedError
