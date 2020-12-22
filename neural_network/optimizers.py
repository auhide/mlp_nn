import numpy as np


# TODO: Implement the Adam Optimizer

class GDOptimizer:

    def _backpropagation(self):
        self._forward(self.X)
        self._reformat_output(self.y)

        for _ in range(self.epochs):
            self._gradient_descent(self.X)

    def _gradient_descent(self, X):

        for i in reversed(range(len(self._layers))):
            
            if self._layers[i] == self._layers[-1]:
                self._layers[i].errors = self.expected_output - self.output
                self._layers[i].create_deltas()

            else:
                next_layer = self._layers[i+1]
                self._layers[i].errors = next_layer.deltas.dot(next_layer.weights.T)
                self._layers[i].create_deltas()

        self._update_weights(X)


class SGDOptimizer(GDOptimizer):

    def _backpropagation(self):

        for epoch in range(self.epochs):
            for X, y in self._batch_split():
                self._forward(X)
                self._reformat_output(y)

                self._gradient_descent(X)

    def _batch_split(self):
        
        for i in range(0, len(self.y), self.batch_size):
            yield self.X[i:i+self.batch_size], self.y[i:i+self.batch_size]


class SGDMOptimizer(SGDOptimizer):
    """
    The idea in a nutshell:
        (accumulator) = (old accumulator)*(momentum) + (gradient)
        (new weights) = (old weights) - (learning rate)*(accumulator)

    (old accumulator) is always the average of the previous gradients

    Using the momentum of the Gradient for faster convergence.
    """

    # SGD looks like:
    # new weights = old weights - learning rate * gradient
    # `gradient` is calculated in `_gradient_descent()`
    # `new weights` are calculated in `_update_weights()`

    # Therefore we'll need to override the `_update_weights()` method only
    def _update_weights(self, X):

        for i in range(len(self._layers)):
            curr_input = np.atleast_2d(
                X if i == 0 else self._layers[i-1].output
            )
            gradient = self._layers[i].deltas.T.dot(curr_input)
            
            # Calculating the average gradient
            self._accumulator = np.average(gradient)

            # Creating the new accumulator based on the previous gradients
            self._accumulator = self._accumulator * self.momentum + gradient

            # Changing the weights
            self._layers[i].weights += self._accumulator.T * self.l_rate


class AdaGrad(SGDOptimizer):
    """
    Stochastic optimization method that adapts the learning rate based on the
    steps (epochs) it's taking. - https://www.paperswithcode.com/method/adagrad
    
    Here:
        (learning rate) = (prev. learning rate) / sqrt(alpha + epsilon)
        Where:
            alpha = sum of squared weight gradients
            epsilon = a small positive number; used because in some cases alpha
            becomes extremely small
    """

    # TODO: Fix the weights updation (Warning: Overflow encountered in `exp()`)
    def _update_weights(self, X):

        for i in range(len(self._layers)):
            curr_input = np.atleast_2d(
                X if i == 0 else self._layers[i-1].output
            )
            gradient = self._layers[i].deltas.T.dot(curr_input)

            # Calculating alpha
            alpha = np.sum(gradient ** 2)

            # Changing the learning rate
            self.l_rate = self.l_rate / np.sqrt(alpha + self.epsilon)

            # Updating the weights
            self._layers[i].weights += gradient.T * self.l_rate
