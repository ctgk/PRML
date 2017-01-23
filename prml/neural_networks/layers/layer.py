class Layer(object):
    """Layer to construct neural network"""

    def __init__(self):
        """
        construct Layer object

        Attributes
        ----------
        istrainable : bool
            flag indicating whether this layer is trainable or not
        """
        self.istrainable = False
