from nn.neural_network.nn import NeuralNetwork
from nn.neural_network.optimizers import *
from nn.neural_network.exceptions import (
    OptimizerDoesNotExist, 
    WrongNNArchitecture
)


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
            return neuralnet_with_optimizer(AdaGradOptimizer)

        elif optimizer == "adam":
            return neuralnet_with_optimizer(AdamOptimizer)

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
        if not cls._architecture_is_valid(architecture_dict):
            raise WrongNNArchitecture("All layers' neurons must be greater than 1")

        NeuralNet = cls.get_nn(optimizer=optimizer)
        nn = NeuralNet(**kwargs)
        
        for layer_i, neurons in architecture_dict.items():
            nn.add_layer(neurons)

        return nn

    @classmethod
    def _architecture_is_valid(cls, architecture):
        """Loops through the number of the neurons of each hidden layer.
        If the neurons of a layer are less than 2, the architecture is not valid. 

        Args:
            architecture (dict): NN Architecture

        Returns:
            bool: Whether the NN Architecture is valid or not
        """
        
        for layer_i, neurons in architecture.items():

            if int(layer_i) > 0 and neurons < 2:
                return False

        return True
