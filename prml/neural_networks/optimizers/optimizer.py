class Optimizer(object):
    """
    Optimizer to train neural network
    """

    def __init__(self, network, learning_rate):
        """
        construct optimizer

        Parameters
        ----------
        network : Network
            network to train
        learning_rate : float
            update rate of parameters to be estimated

        Attributes
        ----------
        n_iter : int
            number of iterations performed
        """
        self.network = network
        self.learning_rate = learning_rate
        self.n_iter = 0

    def set_decay(self, decay_rate, decay_step):
        """
        set exponential decay parameters

        Parameters
        ----------
        decay_rate : float
            dacay rate of the learning rate
        decay_step : int
            steps to decay the learning rate
        """
        self.decay_rate = decay_rate
        self.decay_step = decay_step

    def increment_iteration(self):
        self.n_iter += 1
        if hasattr(self, "decay_rate"):
            if self.n_iter % self.decay_step == 0:
                self.learning_rate *= self.decay_rate
