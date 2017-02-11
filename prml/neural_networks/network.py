class Network(object):
    """
    Neural Network
    """

    def __init__(self, layers=None):
        """
        construct neural network object

        Parameters
        ----------
        layers : tuple of Layer or None
            layers constructing network
        """
        if layers is None:
            self.layers = []
            self.trainables = []
        else:
            self.layers = layers
            self.trainables = []
            for layer in layers:
                if layer.istrainable:
                    self.trainables.append(layer)

    def add(self, layer):
        """
        add subsequent layer

        Parameters
        ----------
        layer : Layer or tuple of Layer
            function to add to this model
        """
        if isinstance(layer, tuple):
            self.layers.extend(layer)
            for l in layer:
                if l.istrainable:
                    self.trainables.append(l)
        else:
            self.layers.append(layer)
            if layer.istrainable:
                self.trainables.append(layer)

    def forward(self, x, training=False):
        """
        feed forward computation without output activation

        Parameters
        ----------
        x : ndarray
            input of this network

        Returns
        -------
        output : ndarray
            output of this network without activating
        """
        for layer in self.layers:
            x = layer.forward(x, training)
            assert x.dtype == "float32"

        return x

    def backward(self, delta):
        """
        backpropagate output error

        Parameters
        ----------
        delta : ndarray
            output error
        """
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            assert delta.dtype == "float32"
