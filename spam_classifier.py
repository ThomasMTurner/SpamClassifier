import numpy as np

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
    def __init__(self, input_dim, output_dim,  activation_function, position=None):
        ## TO DO: observe different weight initialisation methods for the optimisation section.
        self.weights = np.random.randn(input_dim, output_dim) # Initialising weight matrix with random values 
        self.biases = np.random.randn(output_dim) # Initialising bias vector with random values.
        self.activation = activation_function # Setting activation function for the layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.position = position

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
        
        '''
        # Update weights and biases (parameters)
        self.weights -= learning_rate * dL_dW
        self.biases -= learning_rate * dL_dB
        '''

        # Return new output gradient to backpropagate
        return dL_dX
    

    def __repr__(self):
        return f"Layer {self.position} contains {self.output_dim} neurons, {self.input_dim * self.output_dim} weights, with activation {self.activation.__name__}"


class ANN:
    def __init__(self, optimiser, learning_rate, momentum):
        self.layers = []
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.momentum = momentum

    def backprop(self, feature, target):
        self.input = feature

        # Forward pass
        self.forward()
        output = self.input

        # Binary cross-entropy loss
        loss = -(target * np.log(output) + (1 - target) * np.log(1 - output))

        # Output layer gradient using binary cross-entropy loss derivative.
        output_gradient = -(target / output) + (1 - target) / (1 - output)

        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, self.learning_rate, self.momentum)

        return loss


    def forward(self):
        for layer in self.layers:
            output = layer.forward(self.input)
            self.input = output
    
    def train(self, x, y, num_epochs):
        if x.shape[1] != self.layers[0].input_dim:
            raise Exception("Incorrect input shape for given epoch number")
        
        initial_loss = None
        loss = None

        # Random shuffling of training set to improve generalisation.
        '''
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        '''
    
        for _ in range(num_epochs):
            for j in range(x.shape[0]):
                feature = x[j]
                label = y[j]
                if initial_loss is None:
                    initial_loss = self.backprop(feature, label)
                    print("Obtained loss: ", initial_loss)
                else:
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
        print("Made call to save")
        layers = {}
        for layer in self.layers:
            print("Saving layer: ", layer)
            layers[f"Layer {layer.position} {layer.activation.__name__}"] = layer.weights
       
        np.savez(path, **layers)
            

    # Load model weights from given input path
    def load(self, path):
        weights = np.load(path)
        layers = []
        for layer_id, weight_mat in weights.items():
            layers.append(Layer(input_dim=weight_mat.shape[0], 
                                output_dim=weight_mat.shape[1], 
                                activation_function=globals()[layer_id.split()[2]]))

        print("Loaded layers: ", layers)
        self.build(*layers)


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


if __name__ == "__main__":
    # Extract features & their supervised labels.
    training_labels, training_features = extract("data/training_spam.csv")
    testing_labels, testing_features = extract("data/testing_spam.csv")
    
    # Set hyperparameters here. Optimiser=None defaults to standard gradient descent.
    a = ANN(optimiser=None, learning_rate=0.15, momentum=0.9)
            
    # Define our hidden layers & output. Build the network with these layers.
    hidden1 = Layer(input_dim=54,   output_dim=128,   activation_function=sigmoid)
    hidden2 = Layer(input_dim=128,  output_dim=64,   activation_function=sigmoid)
    output  = Layer(input_dim=64,   output_dim=1,    activation_function=sigmoid)
    a.build(hidden1, hidden2, output)

    # Training procedure - should be omitted from the final submission.
    a.train(training_features, training_labels, num_epochs=100)

    # Saving the trained model
    # a.save("newest_model.npz")
   
    # Testing that model has saved correctly
    # classifier = ANN(optimiser=None, learning_rate=0.06)
    # classifier.load("newest_model.npz")
    # print(classifier)

    #saved_predictions = classifier.predict(testing_features)
    training_predictions = a.predict(training_features)
    accuracy = 0
    for i, prediction in enumerate(training_predictions):
        if prediction == training_labels[i]:
            accuracy += 1

    print("Obtained accuracy: ", (accuracy / len(training_features)) * 100, "%")

    
    '''
    trained_accuracy = 0

    trained_predictions = a.predict(testing_features)
    print(trained_predictions)
    
    for i, prediction in enumerate(trained_predictions):
        if prediction == testing_labels[i]:
            trained_accuracy += 1

    #trained_accuracy = np.count_nonzero(trained_predictions == testing_labels)/testing_labels.shape[0]
    
    
    #print(f"Test set accuracy for saved model: ", (saved_accuracy / len(testing_features)) * 100, "%")
    print(f"Test set accuracy for trained model: ", (trained_accuracy / len(testing_features)) * 100, "%")
    

    # Optimisations:
    # 1. Random shuffling of training set for each epoch.
    # 2. Hyperparameter tuning.
    # 3. SGD as opposed to standard - momentum, random noise, adaptive learning rate.
    # 4. Regularisation techniques.
    '''



