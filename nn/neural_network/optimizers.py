import numpy as np



class GDOptimizer:

    def _backpropagation(self):
        self._forward(self.X)
        
        if self._layers[-1].n_neurons > 1:
            self._reformat_output(self.y)

        else:
            self.expected_output = self.y.reshape((-1, 1))

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
                
                if self._layers[-1].n_neurons > 1:
                    self._reformat_output(y)
                
                else:
                    self.expected_output = y.reshape((-1, 1))

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
                    m, v = self._change_shapes_of_biases(m, v, gradient)

                # Biased first moment estimate
                m = self.beta_1 * m + (1 - self.beta_1) * gradient
                # Biased second raw moment estimate
                v = self.beta_2 * v + (1 - self.beta_2) * (gradient ** 2)

                # Bias-corrected
                m_hat = m / (1 - self.beta_1)
                v_hat = v / (1 - self.beta_2)

                weight_change = m_hat / (np.sqrt(v_hat) + self.epsilon)

                self._layers[i].weights += self.l_rate * weight_change.T

    def _change_shapes_of_biases(self, m, v, gradient):
        shape_diff_cols = gradient.shape[1] - m.shape[1]
        shape_diff_rows = gradient.shape[0] - m.shape[0]

        if shape_diff_cols == 0:
            return m, v

        if shape_diff_cols < 0:
            m = self._rem_col_or_row(m, shape_diff_cols, 1)
            v = self._rem_col_or_row(v, shape_diff_cols, 1)

        elif shape_diff_cols > 0:
            m = self._add_cols(m, shape_diff_cols)
            v = self._add_cols(v, shape_diff_cols)

        if shape_diff_rows < 0:
            m = self._rem_col_or_row(m, shape_diff_cols, 0)
            v = self._rem_col_or_row(v, shape_diff_cols, 0)

        elif shape_diff_rows > 0:
            m = self._add_rows(m, shape_diff_rows)
            v = self._add_rows(v, shape_diff_rows)

        return m, v


    def _rem_col_or_row(self, array, number_to_remove, axis):
        """
        Removes row or a column from `array`.
        """
        number_to_remove = abs(number_to_remove)

        if axis == 0:
            for _ in range(number_to_remove):
                array = array[:-1, :]

        else:
            for _ in range(number_to_remove):
                array = array[:, :-1]

        return array

    def _add_cols(self, arr, cols_to_add):
        new_cols = np.zeros((1, arr.shape[1] + cols_to_add))

        for _ in range(cols_to_add):
            arr = np.concatenate((arr, new_cols.T), axis=1)

        return arr

    def _add_rows(self, arr, rows_to_add):
        new_rows = np.zeros((1, arr.shape[0]))

        for _ in range(rows_to_add):
            arr = np.concatenate((arr, new_rows))

        return arr