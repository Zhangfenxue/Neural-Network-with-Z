import DL.Function as fun

class Network:
    def __init__(self, layers, activation):
        self.layers = layers
        self.activation = activation
        self.weights = [np.random.randn(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        self.biases = [np.random.randn(layers[i + 1]) for i in range(len(layers) - 1)]

        def forward(self, x):
            a = x
            for w, b, activation in zip(self.weights, self.biases, self.activations):
                z = np.dot(a, w) + b
                a = activation(z)
            return a

        def train(self, X_train, Y_train, epochs, learning_rate=0.01):
            for epoch in range(epochs):
                # Forward pass
                a = X_train
                activations = [a]
                zs = []
                for w, b, activation in zip(self.weights, self.biases, self.activations):
                    z = np.dot(a, w) + b
                    zs.append(z)
                    a = activation(z)
                    activations.append(a)

                # Calculate loss
                loss = fun.CEE(Y_train, activations[-1])
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss}")

                # Backward pass
                delta = activations[-1] - Y_train
                for l in range(len(self.weights) - 1, -1, -1):
                    gradient_w = np.dot(activations[l].T, delta)
                    gradient_b = np.sum(delta, axis=0)
                    if l > 0:
                        delta = np.dot(delta, self.weights[l].T) * self.activations[l - 1](zs[l - 1], derivative=True)
                    self.weights[l] -= learning_rate * gradient_w
                    self.biases[l] -= learning_rate * gradient_b