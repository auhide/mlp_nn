
# TODO: Add Momentum to the SGD and implement the Adam Optimizer

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
        # self._forward(X)


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