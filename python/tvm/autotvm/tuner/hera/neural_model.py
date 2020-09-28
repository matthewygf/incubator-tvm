class NeuralNetwork():
    def __init__(self, framework) -> None:
        self.framework = framework

    def classify(self, xs):
        """
        Classify whether the configs should be needed.

        Parameters
        ----------
        xs: Array of int
            The indexes of configs from a larger set
        Returns
        -------
        a reduced set of configs
        """
        raise NotImplementedError()