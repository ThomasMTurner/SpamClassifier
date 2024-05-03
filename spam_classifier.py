import numpy as np
import os
import h5py

# Sigmoid activation function implementation - takes single scalar input.
def sigmoid(x, derivative=False):
    if not derivative:
        return 1 / (1 + np.exp(-x))
    else:
        return x * (1 - x)
    
def ReLU(x, derivative=False):
    if not derivative:
        return np.maximum(0, x)
    else:
        return np.where(x > 0, 1, 0)


# Simple implementation of Layer object - with forward pass defined as a method.
class Layer:
    def __init__(self, input_dim, output_dim,  activation_function, weight_initialisation=None, bias_initialisation=None, position=None):
        if weight_initialisation is None:
            # Random weight initialisation
            self.weights = np.random.randn(input_dim, output_dim)
        elif weight_initialisation == "xavier":
            # Normalised Xavier initialisation - if we choose a deeper architecture this will prevent vanishing gradients.
            self.weights = np.random.normal(0, np.sqrt(2.0 / (input_dim + output_dim)), (input_dim, output_dim)) 
        elif weight_initialisation == "normal":
            # Randomised normal distribution initialisation
            self.weights = np.random.normal(0, 1, (input_dim, output_dim))
        
        if bias_initialisation is None:
            # Random bias initialisation
            self.biases = np.random.randn(output_dim) 
        elif bias_initialisation == "zero":
            # Zero bias
            self.biases = np.zeros(output_dim)
    
        self.activation = activation_function # Setting activation function for the layer
        self.input_dim = input_dim # Saving input and output dimensions
        self.output_dim = output_dim
        self.position = position # Saving position in the network topology

    def forward(self, x):
        self.X = x
        self.Y = self.activation(np.dot(x, self.weights) + self.biases)
        return self.Y
    
    def backward(self, dL_dZ, learning_rate, momentum):
        # Compute gradients for weights, bias, loss & inputs.
        #dL_dZ = dL_dZ * self.activation(self.Y, derivative=True)
        dL_dZ = dL_dZ * self.Y * (1 - self.Y)
        dL_dB = np.sum(dL_dZ, axis=0)
        dL_dX = np.dot(dL_dZ, self.weights.T)

        if dL_dZ.shape == (1,):
            dL_dW = (self.X.T * dL_dZ).reshape(-1, 1)

        else: 
            dL_dW = np.dot(self.X.T, np.resize(dL_dZ, self.X.shape[0]))

        # Using stochastic gradient descent
        if momentum is not None:
            # Initialize velocity if not already initialized
            if not hasattr(self, 'velocity_w'):
                self.velocity_w = np.zeros_like(self.weights)
                self.velocity_b = np.zeros_like(self.biases)

            # Update velocity
            self.velocity_w = momentum * self.velocity_w + learning_rate * dL_dW
            self.velocity_b = momentum * self.velocity_b + learning_rate * dL_dB

            # Update weights and biases (parameters) using velocity
            self.weights -= self.velocity_w
            self.biases -= self.velocity_b

        
        # Using standard gradient descent
        else:
            # Update weights and biases (parameters)
            self.weights -= learning_rate * dL_dW
            self.biases -= learning_rate * dL_dB
        
        # Return new output gradient to backpropagate
        return dL_dX
    

    def __repr__(self):
        return f"Layer {self.position} contains {self.output_dim} neurons, {self.input_dim * self.output_dim} weights, with activation {self.activation.__name__}"


class ANN:
    def __init__(self, learning_rate, momentum):
        self.layers = []
        self.learning_rate = learning_rate
        self.momentum = momentum

    def backprop(self, feature, target):
        self.input = feature

        # Forward pass
        self.forward()
        output = self.input

        # Binary cross-entropy loss
        #loss = -(target * np.log(output) + (1 - target) * np.log(1 - output))

        # Binary cross-entropy loss derivative
        #output_gradient = -(target / output) + (1 - target) / (1 - output)

        # Mean squared loss
        loss = np.mean((target - output)**2)

        # Mean squared loss derivative
        output_gradient = -2 * (target - output) 

        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, self.learning_rate, self.momentum)

        return loss


    def forward(self):
        for layer in self.layers:
            output = layer.forward(self.input)
            self.input = output
    
    def train(self, x, y, num_epochs):
        for _ in range(num_epochs):
            # Use a subset of the data for training on each epoch (selected randomly)
            subset_size = int(x.shape[0] * 0.8)
            indices = np.random.choice(x.shape[0], subset_size, replace=False)
            x_temp = x[indices]
            y_temp = y[indices] 
            for j in range(subset_size):
                feature = x_temp[j]
                label = y_temp[j]
                loss = self.backprop(feature, label)
                print("Obtained loss: ", loss)  
       
        print(f"Training complete, final loss: {loss}")

        
    def predict(self, x):
        predictions = []
        for i in x:
            self.input = i
            self.forward()
            if self.input < 0 or self.input > 1:
                raise Exception("Output out of bounds, something wrong with the forward pass")
            if self.input > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return np.array(predictions)

        
    def build(self, *layers): 
        for i, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise Exception("All layers must be of type 'Layer'")
            else:
                layer.position = i + 1
                self.layers.append(layer)

    # Save model weights into given output path
    def save(self, path):
        if os.path.exists(path):
            os.remove(path)
        
        with h5py.File(path, 'w') as hf:
            hf.attrs[f"Number of layers"] = len(self.layers)
            for i, layer in enumerate(self.layers):
                hf.create_dataset(f"layer {i + 1} weights", data=layer.weights)
                hf.create_dataset(f"layer {i + 1} biases", data=layer.biases)
                hf.attrs.create(f"layer {i + 1} activation", data=layer.activation.__name__, dtype=h5py.string_dtype())
    
            

    # Load model weights from given input path
    def load(self, path):
        print("Loading model parameters")
        with h5py.File(path, 'r') as hf:
            for i in range(hf.attrs["Number of layers"]):
                weights = np.array(hf[f"layer {i + 1} weights"])
                biases = np.array(hf[f"layer {i + 1} biases"])
                activation = globals()[hf.attrs[f"layer {i + 1} activation"]]
                layer = Layer(weights.shape[0], weights.shape[1], activation)
                layer.weights = weights
                layer.biases = biases
                self.layers.append(layer)


    def __repr__(self):
        out = "ANN with Layers:\n"
        for i, layer in enumerate(self.layers):
            out += f"Layer {i + 1}: " + str(layer) + "\n"

        return out


# Simple extraction function to get label vector & feature vectors from training data set (path currently is "data/training_spam_copy.csv")
def extract(training_path):
    training_spam = np.loadtxt(open(training_path), delimiter=",").astype(int)
    labels = training_spam[:, 0]
    features = training_spam[:, 1:]
    return labels, features

def test_accuracy(predictions, labels, name):
    accuracy = np.count_nonzero(predictions == labels)/labels.shape[0]
    print(f"Accuracy on {name} data is: {accuracy}")


if __name__ == "__main__":
    # Extract features & their supervised labels.
    training_labels, training_features = extract("data/training_spam.csv")
    testing_labels, testing_features = extract("data/testing_spam.csv")
    
    # Set hyperparameters here. momentum=None defaults to standard gradient descent.
    a = ANN(learning_rate=0.08, momentum=None)
            
    # Define our hidden layers & output. Build the network with these layers. Initialisation=None defaults to random weights.
    hidden1 = Layer(input_dim=54,    output_dim=128,   activation_function=sigmoid, weight_initialisation=None, bias_initialisation=None)
    output  = Layer(input_dim=128,   output_dim=1,     activation_function=sigmoid, weight_initialisation=None, bias_initialisation=None)
    a.build(hidden1, output)

    # Training procedure - should be omitted from the final submission.
    a.train(training_features, training_labels, num_epochs=200)

    # Saving the trained model
    a.save("model_data.h5")
   
    # Testing that model has saved correctly
    classifier = ANN(learning_rate=0.1, momentum=None)
    classifier.load("model_data.h5")
    print(classifier)

    testing_accuracy = test_accuracy(a.predict(testing_features), testing_labels, "testing")
    training_accuracy = test_accuracy(a.predict(training_features), training_labels, "training")
    saved_accuracy = test_accuracy(classifier.predict(testing_features), testing_labels, "saved model testing")


