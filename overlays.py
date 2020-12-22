from neural_network.nn import NeuralNetwork
from neural_network.optimizers import *
from neural_network.exceptions import OptimizerDoesNotExist



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

