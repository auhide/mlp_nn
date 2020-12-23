import numpy as np



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


class AdaGradOptimizer(SGDOptimizer):
    """
    Stochastic optimization method that adapts the learning rate based on the
    steps (epochs) it's taking. - https://d2l.ai/chapter_optimization/adagrad.html
    Paper: https://stanford.edu/~jduchi/projects/DuchiHaSi10_colt.pdf
    
    Here:
        (adapted learning rate) = (learning rate) / sqrt(alpha + epsilon)
        Where:
            alpha = sum of squared weight gradients
            epsilon = a small positive number; used because in some cases alpha
            becomes extremely small
    """

    def _update_weights(self, X):

        for i in range(len(self._layers)):
            curr_input = np.atleast_2d(
                X if i == 0 else self._layers[i-1].output
            )
            gradient = self._layers[i].deltas.T.dot(curr_input)

            # Calculating alpha
            alpha = np.sum(gradient ** 2)

            # Changing the learning rate
            adapted_l_rate = self.l_rate / np.sqrt(alpha + self.epsilon)

            # Updating the weights
            self._layers[i].weights += gradient.T * adapted_l_rate


class AdamOptimizer(SGDOptimizer):
    """
    Paper: https://arxiv.org/pdf/1412.6980.pdf
    Page 2 is where the algorithm steps are written.
    """
    
    # TODO: Make Adam Work
    def _update_weights(self, X):

        for i in range(len(self._layers)):
                curr_input = np.atleast_2d(
                    X if i == 0 else self._layers[i-1].output
                )
                gradient = self._layers[i].deltas.T.dot(curr_input)
                
                if i == 0:
                    m = np.zeros(gradient.shape)
                    v = np.zeros(gradient.shape)

                else:
                    m, v = self._add_or_remove_weights(m, v, gradient)

                # Biased first moment estimate
                m = self.beta_1 * m + (1 - self.beta_1) * gradient
                # Biased second raw moment estimate
                v = self.beta_2 * v + (1 - self.beta_2) * (gradient ** 2)

                # Bias-corrected
                m_hat = m / (1 - self.beta_1)
                v_hat = v / (1 - self.beta_2)

                weight_change = m_hat / (np.sqrt(v_hat) + self.epsilon)

                self._layers[i].weights += self.l_rate * weight_change.T

    def _add_or_remove_weights(self, m, v, gradient):
        shape_diff = gradient.shape[1] - m.shape[1]

        if shape_diff == 0:
            return m, v

        if shape_diff < 0:
            m = self._rem_col(m, shape_diff)
            v = self._rem_col(v, shape_diff)

            return m, v

        elif shape_diff > 0:
            m = self._add_col(m, shape_diff)
            v = self._add_col(v, shape_diff)

            return m, v

    def _rem_col(self, array, cols_to_remove):
        bool_remove_arr = [False for _ in range(array.shape[1])]
        bool_remove_arr[-1] = True

        # Removing the last column
        array = array.compress(np.logical_not(bool_remove_arr), axis=1)
        
        return array

    def _add_col(self, array, cols_to_add):
        new_cols = np.zeros((1, array.shape[0]))

        return np.hstack((array, np.atleast_2d(new_cols).T))
