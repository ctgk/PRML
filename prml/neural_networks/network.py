from collections import OrderedDict


class Network(object):
    """
    Neural Network
    """

    def __init__(self, layers=None):
        """
        construct neural network object

        Parameters
        ----------
        layers : OrderedDict of Layer or None
            layers constructing network
        """
        if layers is None:
            self.layers = OrderedDict()
            self.trainables = []
        else:
            self.layers = layers
            self.trainables = []
            for layer in layers.values():
                if layer.istrainable:
                    self.trainables.append(layer)

    def add(self, layer, name=None):
        """
        add subsequent layer

        Parameters
        ----------
        layer : Layer
            function to add to this model
        name : str
            name of the input layer
        """
        if name is None:
            name = str(len(self.layers))
        self.layers[name] = layer
        if layer.istrainable:
            self.trainables.append(layer)

    def forward(self, x, training=False, name=None):
        """
        feed forward computation without output activation

        Parameters
        ----------
        x : ndarray
            input of this network
        name : str
            feature to extract

        Returns
        -------
        output : ndarray
            output of the layers
        """
        for key, layer in self.layers.items():
            x = layer.forward(x, training)
            assert x.dtype == "float32"
            if key == name:
                return x

        return x

    def backward(self, delta):
        """
        backpropagate output error

        Parameters
        ----------
        delta : ndarray
            output error

        Returns
        -------
        delta : ndarray
            input error
        """
        for layer in reversed(self.layers.values()):
            delta = layer.backward(delta)
            assert delta.dtype == "float32"
        return delta
