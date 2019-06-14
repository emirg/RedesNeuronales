# Recurrent Neural Network from Scratch in Python 3

import copy
import numpy as np

# np.random.seed(0)

# Sigmoid Activation Function
# To be applied at Hidden Layers and Output Layer
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

# Derivative of Sigmoid Function
# Used in calculation of Back Propagation Loss
def sigmoidPrime(z):
    return z * (1-z)


# Generate Input Dataset
int_to_binary = {}
binary_dim = 8

# Calculate the largest value which can be attained
# 2^8 = 256
max_val = (2**binary_dim)

# Calculate Binary values for int from 0 to 256
binary_val = np.unpackbits(np.array([range(max_val)], dtype=np.uint8).T, axis=1)

# Function to map Integer values to Binary values
for i in range(max_val):
    int_to_binary[i] = binary_val[i]
    # print('\nInteger value: ',i)
    # print('binary value: ', binary_val[i])


# NN variables
# Learning rate controls how quickly or slowly a neural network model learns a problem.
#This means that a learning rate of 0.1, a traditionally common default value, 
# would mean that weights in the network are updated 0.1 * (estimated weight error)
#  or 10% of the estimated weight error each time the weights are updated.


learning_rate = 0.1

# Inputs: Values to be added bit by bit
inputLayerSize = 2

# Hidden Layer with 16 neurons
hiddenLayerSize = 16

# Output at one time step is 1 bit
outputLayerSize = 1

# Initialize Weights
# Weight of first Synapse (Synapse_0) from Input to Hidden Layer at Current Timestep
W1 = 2 * np.random.random((inputLayerSize, hiddenLayerSize)) - 1

# Weight of second Synapse (Synapse_1) from Hidden Layer to Output Layer
W2 = 2 * np.random.random((hiddenLayerSize, outputLayerSize)) - 1

# Weight of Synapse (Synapse_h) from Current Hidden Layer to Next Hidden Layer in Timestep
# carry in 
W_h = 2 * np.random.random((hiddenLayerSize, hiddenLayerSize)) - 1


# Initialize Updated Weights Values
W1_update = np.zeros_like(W1)
W2_update = np.zeros_like(W2)
W_h_update = np.zeros_like(W_h)


# Iterate over 10,000 samples for Training
for j in range(10000):
    # ----------------------------- Compute True Values for the Sum (a+b) [binary encoded] --------------------------
    # Generate a random sample value for 1st input
    a_int = np.random.randint(max_val/2)
    # Convert this Int value to Binary
    a = int_to_binary[a_int]

    # Generate a random sample value for 2nd input
    b_int = np.random.randint(max_val/2)
    # Map Int to Binary
    b = int_to_binary[b_int]

    # True Answer a + b = c
    c_int = a_int + b_int
    
    #if a_int>b_int:
    #    c_int = a_int
    #else:
    #    c_int=b_int
    c = int_to_binary[c_int]
    # Array to save predicted outputs (binary encoded)
    d = np.zeros_like(c)

    # Initialize overall error to "0"
    overallError = 0

    # Save the values of dJdW1 and dJdW2 computed at Output layer into a list
    output_layer_deltas = list()

    # Save the values obtained at Hidden Layer of current state in a list to keep track
    hidden_layer_values = list()

    # Initially, there is no previous hidden state. So append "0" for that
    hidden_layer_values.append(np.zeros(hiddenLayerSize))

    # ----------------------------- Compute the Values for (a+b) using RNN [Forward Propagation] ----------------------
    # position: location of the bit amongst 8 bits; starting point "0"; "0 - 7"
    for position in range(binary_dim):
        # Generate Input Data for RNN
        # Take the binary values of "a" and "b" generated for each iteration of "j"

        # With increasing value of position, the bit location of "a" and "b" decreases from "7 -> 0"
        # and each iteration computes the sum of corresponding bit of "a" and "b".
        # ex. for position = 0, X = [a[7],b[7]], 7th bit of a and b.
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])

        # Actual value for (a+b) = c, c is an array of 8 bits, so take transpose to compare bit by bit with X value.
        y = np.array([[c[binary_dim - position - 1]]]).T

        # Values computed at current hidden layer
        # [dot product of Input(X) and Weights(W1)] + [dot product of previous hidden layer values and Weights (W_h)]
        # W_h: weight from previous step hidden layer to current step hidden layer
        # W1: weights from current step input to current hidden layer
        hidden_layer_values_new = sigmoid(np.dot(X,W1) + np.dot(hidden_layer_values[-1],W_h))


        # The new output using new Hidden layer values
        output_calculated = sigmoid(np.dot(hidden_layer_values_new, W2))

        # Calculate the error (target - calculared)
        output_error = y - output_calculated

        # Save the error deltas at each step as it will be propagated back
        # S'()
        output_layer_deltas.append((output_error)*sigmoidPrime(output_calculated))

        # Save the sum of error at each binary position
        overallError += np.abs(output_error[0])

        # Round off the values to nearest "0" or "1" and save it to a list
        d[binary_dim - position - 1] = np.round(output_calculated[0][0])

        # Save the hidden layer to be used later
        hidden_layer_values.append(copy.deepcopy(hidden_layer_values_new))

    future_layer_1_delta = np.zeros(hiddenLayerSize)
    
# ----------------------------------- Back Propagating the Error Values to All Previous Time-steps ---------------------
    for position in range(binary_dim):
        # a[0], b[0] -> a[1]b[1] ....
        X = np.array([[a[position], b[position]]])
        # The last step Hidden Layer where we are currently a[0],b[0]
        layer_1 = hidden_layer_values[-position - 1]
        # The hidden layer before the current layer, a[1],b[1]
        prev_hidden_layer = hidden_layer_values[-position-2]
        # Errors at Output Layer, a[1],b[1]
        output_layer_delta = output_layer_deltas[-position-1]
        layer_1_delta = (future_layer_1_delta.dot(W_h.T) + output_layer_delta.dot(W2.T)) * sigmoidPrime(layer_1)

        # Update all the weights and try again
        W2 += np.atleast_2d(layer_1).T.dot(output_layer_delta) * learning_rate
        W_h+= np.atleast_2d(prev_hidden_layer).T.dot(layer_1_delta)* learning_rate
        W1 += X.T.dot(layer_1_delta) * learning_rate

        future_layer_1_delta = layer_1_delta

    # Print out the Progress of the RNN
    if (j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")