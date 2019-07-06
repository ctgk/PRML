import numpy as np


class Regression(object):
    r"""
    Linear Regression model

    .. math::

        p(y|{\bf x}) = \mathcal{N}(y|{\bf w}^{\rm T}{\bf x}, \sigma^2)


    Attributes
    ----------
    w : np.ndarray (D,)
        coefficient :math:`{\bf w}`
    sigma : float
        standard deviation of error :math:`\sigma`
    """

    def fit(self, X: np.ndarray, t: np.ndarray):
        r"""
        perform maximum likelihood estimation given input and output pairs

        .. math::

            \max_{{\bf w}, \sigma} \sum_i \log p(t_i|{\bf x}_i)

            {\bf w} = ({\bf X}^{\rm T}{\bf X})^{-1}{\bf X}^{\rm T}{\bf t}

            \sigma = \sqrt{
                {1\over N}
                \sum_{i=1}^N\left({\bf w}^{\rm T}{\bf x}_i - {\bf t}\right)^2
            }

        Parameters
        ----------
        X : np.ndarray (N, D)
            design matrix whose row represents each input
            :math:`{\bf X}=({\bf x}_0, ..., {\bf x}_{N-1})^{\rm T}`
        t : np.ndarray (N)
            corresponding target of each input :math:`{\bf t}`
        """
        self.w = np.linalg.pinv(X) @ t
        self.sigma = np.sqrt(np.mean(np.square(X @ self.w - t)))

    def predict(self, X: np.ndarray, return_std: bool = False):
        r"""
        predict output of corresponding input

        .. math:: y = {\bf w}^{\rm T}{\bf x}

        Parameters
        ----------
        X : np.ndarray (N, D)
            input
        return_std : bool, optional
            flag to return uncertainty, by default False

        Returns
        -------
        np.ndarray
            output :math:`y`
        """
        y = X @ self.w
        if return_std:
            y_std = self.sigma + np.zeros_like(y)
            return y, y_std
        return y
