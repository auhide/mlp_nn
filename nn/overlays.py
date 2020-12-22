from neural_network.nn import NeuralNetwork
from neural_network.optimizers import *
from neural_network.exceptions import OptimizerDoesNotExist


DEFAULT_ARCH = {
    0: 3,
    1: 3,
    3: 3
}


def neuralnet_with_optimizer(optimizer):

    class NeuralNet(optimizer, NeuralNetwork):
        pass

    return NeuralNet


class NeuralNetFactory:
    """
    A Factory class that returns a Neural Network with a certain optimizer.
    For that you have to call the `get_nn()` class method.
    """

    @classmethod
    def get_nn(cls, optimizer="sgd"):
        """Initiates a class with a certain `optimizer`

        Args:
            optimizer (str, optional): The optimization algorithm. Defaults to "sgd".

        Returns:
            type: Returns the NeuralNetwork class, inheriting a certain optimizer
        """
        
        if optimizer == "sgd":
            return neuralnet_with_optimizer(SGDOptimizer)
        
        elif optimizer == "gd":
            return neuralnet_with_optimizer(GDOptimizer)

        elif optimizer == "sgdm":
            return neuralnet_with_optimizer(SGDMOptimizer)

        elif optimizer == "adagrad":
            return neuralnet_with_optimizer(AdaGrad)

        else:
            raise OptimizerDoesNotExist(
                f"""Optimizer {optimizer} is not supported by `NeuralNetFactory`."""
            )

    @classmethod
    def define_nn(cls, optimizer="sgd", architecture_dict=DEFAULT_ARCH, **kwargs):
        """Defining the architecture and all hyperparameters of the NN

        Args:
            optimizer (str, optional): The optimization algorithm used in the NN. Defaults to "sgd".
            architecture_dict (dict, optional): Architecture of the NN. Defaults to DEFAULT_ARCH.
            **kwargs: X, y and all Hyperparameters

        Returns:
            NeuralNet: The defined Neural Network
        """
        NeuralNet = cls.get_nn(optimizer=optimizer)
        nn = NeuralNet(**kwargs)
        
        for layer_i, neurons in architecture_dict.items():
            nn.add_layer(neurons)

        return nn
