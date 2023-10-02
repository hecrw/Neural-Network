import numpy as np
import pandas as pd

data = pd.read_csv("digit-recognizer/train.csv")
data = np.array(data)
row, column = data.shape
np.random.shuffle(data)

num_test_samples = 1000

#in both testing and training data x typically represents the features (pixel values) and y is the label
data_test = data[:num_test_samples].T
x_test = data_test[1:] / 255
y_test = data_test[0]

data_train = data[num_test_samples:].T
x_train = data_train[1:] / 255
y_train = data_train[0]

#initializing random weights and biases
def initialize():
    # Dimensions for one hidden layer with 10 neurons
    input_size = 784
    hidden_size = 10
    output_size = 10 

    W1 = np.random.uniform(0, 1, size=(hidden_size, input_size)) - 0.5
    b1 = np.random.uniform(0, 1, size=(hidden_size, 1)) - 0.5
    W2 = np.random.uniform(0, 1, size=(output_size, hidden_size))
    b2 = np.random.uniform(0, 1, size=(output_size, 1))

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forProp(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    
    return Z1, A1, Z2, A2


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backProp(Z1, A1, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / row * dZ2.dot(A1.T)
    db2 = 1 / row * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * (Z1 > 0) 
    dW1 = 1 / row * dZ1.dot(X.T)
    db1 = 1 / row * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dw1, db1, dw2, db2, learning_rate):
    learning_rate = 0.1
    W1 = W1 - learning_rate * dw1
    W2 = W2 - learning_rate * dw2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

def gradient_descent(X, Y, learning_rate, Epoch):
    W1, b1, W2, b2 = initialize()
    for epoch in range(Epoch):
        Z1, A1, Z2, A2 = forProp(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backProp(Z1, A1, A2, W2, X, Y)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        # get accuracy 
        if epoch % 10 == 0:
            predicted = np.argmax(A2, axis=0)
            correct_predictions = np.sum(predicted == Y)
            accuracy = (correct_predictions / Y.size) * 100
            print(f"Epoch: {epoch}")
            print(f"Accuracy: {accuracy}%")
            
            
gradient_descent(x_train, y_train, 0.1, 1000)

            