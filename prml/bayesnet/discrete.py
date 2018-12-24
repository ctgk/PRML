import numpy as np
from prml.bayesnet.probability_function import ProbabilityFunction
from prml.bayesnet.random_variable import RandomVariable


class DiscreteVariable(RandomVariable):
    """
    Discrete random variable
    """

    def __init__(self, n_class:int, parent=None):
        """
        intialize a discrete random variable

        parameters
        ----------
        n_class : int
            number of classes
        parent : DiscreteProbability, optional
            parent node this variable came out from

        Attributes
        ----------
        message_from : dict
            dictionary of message from neighbor node and itself
        child : list of DiscreteProbability
            probability function this variable is conditioning
        proba : np.ndarray
            current estimate
        """
        self.n_class = n_class
        self.parent = parent
        self.message_from = {self: np.ones(n_class)}
        if parent is not None:
            self.message_from[parent] = parent.marginalize()
        self.child = []
        self.summarize_message()

    def __repr__(self):
        string = f"DiscreteVariable(prior={self.prior}"
        if np.allclose(self.prior, self.posterior):
            string += ")"
        elif self.is_observed():
            string += f", observed={self.posterior})"
        else:
            string += f", posterior={self.posterior})"
        return string

    def add_parent(self, parent):
        if self.parent is not None:
            raise ValueError("This variable already has its parent node")
        self.parent = parent
        self.message_from[parent] = parent.marginalize()

    def add_child(self, child):
        self.child.append(child)
        self.message_from[child] = np.ones(self.n_class)

    def is_observed(self):
        return np.allclose(self.message_from[self].sum(), 1)

    @property
    def proba(self):
        return self.posterior

    def receive_message(self, message, giver, proprange):
        self.message_from[giver] = message
        self.summarize_message()
        self.send_message(proprange, exclude=giver)

    def summarize_message(self):
        self.prior = self.message_from[self.parent]

        self.likelihood = np.copy(self.message_from[self])
        for func in self.child:
            self.likelihood *= self.message_from[func]

        self.posterior = self.prior * self.likelihood
        self.posterior /= self.posterior.sum()

    def send_message(self, proprange=-1, exclude=None):
        if self.parent is not exclude:
            self.parent.receive_message(self.likelihood, self, proprange)
        for func in self.child:
            if func is not exclude:
                if self.is_observed():
                    func.receive_message(self.message_from[self], self, proprange)
                else:
                    func.receive_message(self.prior, self, proprange)

    def observe(self, data:int, proprange=-1):
        """
        set observed data of this variable

        Parameters
        ----------
        data : int
            observed data of this variable
            This must be smaller than n_class and must be non-negative
        propagate : int, optional
            Range to propagate the observation effect to the other random variable using belief propagation alg.
            If proprange=1, the effect only propagate to the neighboring random variables.
            Default is -1, which is infinite range.
        """
        assert(0 <= data < self.n_class)
        self.receive_message(np.eye(self.n_class)[data], self, proprange=proprange)


class DiscreteProbability(ProbabilityFunction):
    """
    Discrete probability function
    """

    def __init__(self, table, *condition, out=None, name=None):
        """
        initialize discrete probability function

        Parameters
        ----------
        table : (K, ...) np.ndarray or array-like
            probability table
            If a discrete variable A is conditioned with B and C,
            table[a,b,c] give probability of A=a when B=b and C=c.
            Thus, the sum along the first axis should equal to 1.
            If a table is 1 dimensional, the variable is not conditioned.
        condition : tuple of DiscreteVariable, optional
            parent node, discrete variable this function is conidtioned by
            len(condition) should equal to (table.ndim - 1)
            (Default is (), which means no condition)
        out : DiscreteVariable, optional
            output of this discrete probability function
            Default is None which construct a new output instance
        name : str
            name of this discrete probability function
        """
        self.table = np.asarray(table)
        self.condition = condition
        if condition:
            for var in condition:
                var.add_child(self)
        self.message_from = {var: var.prior for var in condition}

        self.out = DiscreteVariable(len(table), self) if out is None else out
        self.message_from[self.out] = np.ones(len(self.table))

        self.name = name

    def __repr__(self):
        if self.name is not None:
            return self.name
        else:
            return super().__repr__()

    def receive_message(self, message, giver, proprange):
        if message is not None:
            self.message_from[giver] = message
        if proprange:
            self.send_message(proprange, exclude=giver)

    def marginalize(self):
        proba = np.copy(self.table)
        for var in reversed(self.condition):
            proba = np.sum(proba * self.message_from[var], axis=-1)
        return proba

    def send_message(self, proprange, exclude=None):
        proprange = proprange - 1

        def expand_dims(x, ndim, axis):
            shape = [-1 if i == axis else 1 for i in range(ndim)]
            return x.reshape(*shape)

        if self.out is not exclude:
            proba = self.marginalize()
            self.out.receive_message(proba, self, proprange)

        if proprange == 0: return

        likelihood = (self.table.transpose() * self.message_from[self.out]).transpose().sum(axis=0)
        for i, var in enumerate(self.condition):
            if var is exclude:
                continue
            proba = np.copy(likelihood)
            for j, var_ in enumerate(self.condition):
                if var_ is var:
                    continue
                proba *= expand_dims(self.message_from[var_], proba.ndim, j)
            axis = list(range(proba.ndim))
            axis.remove(i)
            message = np.sum(proba, axis=tuple(axis))
            var.receive_message(message, self, proprange - 1)


def discrete(table, *condition, out=None, name=None):
    """
    discrete probability function

    Parameters
    ----------
    table : (K, ...) np.ndarray or array-like
        probability table
        If a discrete variable A is conditioned with B and C,
        table[a,b,c] give probability of A=a when B=b and C=c.
        Thus, the sum along the first axis should equal to 1.
        If a table is 1 dimensional, the variable is not conditioned.
    condition : tuple of DiscreteVariable, optional
        parent node, discrete variable this function is conidtioned by
        len(condition) should equal to (table.ndim - 1)
        (Default is (), which means no condition)
    out : DiscreteVariable, optional
        output of this discrete probability function
        Default is None which construct a new output instance
    name : str
        name of the discrete probability function

    Returns
    -------
    DiscreteVariable
        output discrete random variable of discrete probability function
    """
    function = DiscreteProbability(table, *condition, out=out, name=name)
    return function.out
