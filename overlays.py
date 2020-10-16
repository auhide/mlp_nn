from alt_network.nn import NeuralNetwork
from alt_network.optimizers import *



def neuralnet_with_optimizer(optimizer):

    class NeuralNet(NeuralNetwork, optimizer):
        pass

    return NeuralNet


class OptimizedNN:

    def get_nn(self, optimizer="sgd"):
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

