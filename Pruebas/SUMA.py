import numpy as np
import json
import copy

# Generate Input Dataset
int_to_binary = {}
binary_dim = 16

# Calculate the largest value which can be attained
# 2^8 = 256
#2^(16-1) por el signo 
max_val = (2**(binary_dim-1))

# Calculate Binary values for int from 0 to 256
#binary_val = np.unpackbits(np.array([range(max_val)], dtype=np.uint8).T, axis=1)

def unpackbits(x,num_bits):
    xshape = list(x.shape)
    #print(xshape)
    x = x.reshape(-1,1)
    #print(x)
    to_and_aux = 2**np.arange(num_bits)
    to_and=2**np.arange(num_bits)
    for i in range(len(to_and_aux)):
        to_and[i]=to_and_aux[num_bits-(i+1)]
    aux=(x & to_and).astype(bool).astype(int)
    return aux

#creo un arreglo 
binary_val = unpackbits(
    np.array([range(max_val)],dtype=np.uint16).T,binary_dim)

def complemento2(binario):
    for i in range(binary_dim):        
        if binario[i]==0:
            binario[i]=1
        else:
            binario[i]=0
    
    uno=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    suma= np.add(binario,uno)
    return suma

# NN variables
# Learning rate controls how quickly or slowly a neural network model learns a problem.
#This means that a learning rate of 0.1, a traditionally common default value, 
# would mean that weights in the network are updated 0.1 * (estimated weight error)
#  or 10% of the estimated weight error each time the weights are updated.
learning_rate = 0.1

# Inputs: Values to be added bit by bit
inputLayerSize = 2

# Hidden Layer with 16 neurons
hiddenLayerSize = 2*binary_dim

# Output at one time step is 1 bit
outputLayerSize = 1

# Function to map Integer values to Binary values
for i in range(max_val):
    int_to_binary[i] = binary_val[i]
    # print('\nInteger value: ',i)
    # print('binary value: ', binary_val[i])

class NeuralNetwork():

    # Constructor

    def __init__(self):
        # Initialize Weights
        # Weight of first Synapse (Synapse_0) from Input to Hidden Layer at Current Timestep
        self.W1 = 2 * np.random.random((inputLayerSize, hiddenLayerSize)) - 1
        # Weight of second Synapse (Synapse_1) from Hidden Layer to Output Layer
        self.W2 = 2 * np.random.random((hiddenLayerSize, outputLayerSize)) - 1
        # Weight of Synapse (Synapse_h) from Current Hidden Layer to Next Hidden Layer in Timestep
        self.W_h = 2 * np.random.random((hiddenLayerSize, hiddenLayerSize)) - 1
        # Save the values obtained at Hidden Layer of current state in a list to keep track
        self.hidden_layer_values = list()



    # Funcion de activacion
  
    # Sigmoid Activation Function
    # To be applied at Hidden Layers and Output Layer

    def sigmoid(self,z):
        return (1 / (1 + np.exp(-z)))

    # Derivative of Sigmoid Function
    # Used in calculation of Back Propagation Loss
    def sigmoidPrime(self,z):
        return z * (1-z)

    def think(self,input_1, input_2,):
        # Convert this Int value to Binary
        if input_1<0:
            input_1=abs(input_1)
            a=int_to_binary[input_1]
            a=complemento2(a)
        else:
            a = int_to_binary[input_1]
        #print(a)

        # Map Int to Binary
        if input_2<0:
            input_2=abs(input_2)
            b = int_to_binary[input_2]
            b=complemento2(b)
        else:
            b = int_to_binary[input_2]
        
        #print(b)

        
        c_int = input_1 + input_2 
        c = int_to_binary[c_int]
        
        # Array to save predicted outputs (binary encoded)
        d = np.zeros(binary_dim)

        # Initially, there is no previous hidden state. So append "0" for that
        self.hidden_layer_values.append(np.zeros(hiddenLayerSize))

        # ---[Forward Propagation] ----------------------

        for position in range(binary_dim):

            X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])

            hidden_layer_values_new = self.sigmoid(np.dot(X,self.W1) + np.dot(self.hidden_layer_values[-1],self.W_h))

            output_calculated = self.sigmoid(np.dot(hidden_layer_values_new, self.W2))

            d[binary_dim - position - 1] = np.round(output_calculated[0][0])

            self.hidden_layer_values.append(copy.deepcopy(hidden_layer_values_new))
        
        return d


    # Funcion para entrenar
    def entrenar(self,training_iterations):
        # Iterate for Training
        for j in range(training_iterations):
            # ----------------------------- Compute True Values for the Sum (a+b) [binary encoded] --------------------------
            # Generate a random sample value for 1st input
            a_int = np.random.randint(max_val/4)
            # Convert this Int value to Binary
            #a = int_to_binary[a_int]
            #print(a)

            # Generate a random sample value for 2nd input
            b_int = np.random.randint(max_val/4)
            # Map Int to Binary
            #b = int_to_binary[b_int]
            #print(b)
            signo= np.random.randint(6)
            #print(signo)
            #if signo<3:
            a_int=a_int*-1
                
            if a_int<0:
                a_int=abs(a_int)
                a=int_to_binary[a_int]
                a=complemento2(a)
                a_int=a_int*-1
            else:
                a = int_to_binary[a_int]
            #print(a)

            # Map Int to Binary
            if b_int<0:
                b_int=abs(b_int)
                b = int_to_binary[b_int]
                b=complemento2(b)
                b=-1*b_int
            else:
                b = int_to_binary[b_int]

            # True Answer a + b = c
            
            
            #if a_int>b_int:
            #    c_int = a_int
            #else:
            #    c_int=b_int

            c_int = a_int + b_int

            if c_int<0:
                c_int=abs(c_int)
                c = int_to_binary[c_int]
                c=complemento2(c)
                c_int=c_int*-1
            else:
                c = int_to_binary[c_int]

            
            # Array to save predicted outputs (binary encoded)
            d = np.zeros_like(c)

            # Initialize overall error to "0"
            overallError = 0

            # Save the values of dJdW1 and dJdW2 computed at Output layer into a list
            output_layer_deltas = list()

            # Initially, there is no previous hidden state. So append "0" for that
            self.hidden_layer_values.append(np.zeros(hiddenLayerSize))

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
                hidden_layer_values_new = self.sigmoid(np.dot(X,self.W1) + np.dot(self.hidden_layer_values[-1],self.W_h))


                # The new output using new Hidden layer values
                output_calculated = self.sigmoid(np.dot(hidden_layer_values_new, self.W2))

                # Calculate the error (target - calculared)
                output_error = y - output_calculated

                # Save the error deltas at each step as it will be propagated back
                output_layer_deltas.append((output_error)*self.sigmoidPrime(output_calculated))

                # Save the sum of error at each binary position
                overallError += np.abs(output_error[0])

                # Round off the values to nearest "0" or "1" and save it to a list
                d[binary_dim - position - 1] = np.round(output_calculated[0][0])

                # Save the hidden layer to be used later
                self.hidden_layer_values.append(copy.deepcopy(hidden_layer_values_new))

            future_layer_1_delta = np.zeros(hiddenLayerSize)
            
        # ----------------------------------- Back Propagating the Error Values to All Previous Time-steps ---------------------
            for position in range(binary_dim):
                # a[0], b[0] -> a[1]b[1] ....
                X = np.array([[a[position], b[position]]])
                # The last step Hidden Layer where we are currently a[0],b[0]
                layer_1 = self.hidden_layer_values[-position - 1]
                # The hidden layer before the current layer, a[1],b[1]
                prev_hidden_layer = self.hidden_layer_values[-position-2]
                # Errors at Output Layer, a[1],b[1]
                output_layer_delta = output_layer_deltas[-position-1]
                layer_1_delta = (future_layer_1_delta.dot(self.W_h.T) + output_layer_delta.dot(self.W2.T)) * self.sigmoidPrime(layer_1)

                # Update all the weights and try again
                self.W2 += np.atleast_2d(layer_1).T.dot(output_layer_delta) * learning_rate
                self.W_h+= np.atleast_2d(prev_hidden_layer).T.dot(layer_1_delta)* learning_rate
                self.W1 += X.T.dot(layer_1_delta) * learning_rate

                future_layer_1_delta = layer_1_delta

            # Print out the Progress of the RNN 
            
            if (j % 1000 == 0):
                print("Error:" + str(overallError))
                print("Pred:" + str(d))
                print("True:" + str(c))
                out = 0
                verdad=0
                signo=1
                if c_int<0 :
                    signo=-1
                    d=complemento2(d)
                else:
                    if a_int<0 | b_int:
                        c=complemento2(c)
                        
                    
                
                for index, x in enumerate(reversed(d)):
                    #print(index)
                    out += x * pow(2, index)

                for index, x in enumerate(reversed(c)):
                    #print(index)
                    verdad += x * pow(2, index)

                out=signo*out
                verdad=verdad*signo
                print("verdad "+str(verdad))
                print(str(a_int) + " + " + str(b_int) + " = " + str(out))
                print("------------")

# Main 
if __name__ == "__main__":
    neural_network = NeuralNetwork()
    neural_network.entrenar(100000)
    while(True):
        user_input_one = str(input("Entrada uno (p1): "))
        user_input_two = str(input("Entrada dos (p2): "))
        user_input_one=int(user_input_one)
        user_input_two=int(user_input_two)
        correct_int=user_input_one+user_input_two

        d=neural_network.think(user_input_one, user_input_two)
    
        print("Pred:" + str(d))
        out = 0
        signo=1
        if correct_int<0:
            d=complemento2(d)
            signo=-1
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        out=signo*out
        print(str(user_input_one) + " + " + str(user_input_two) + " = " + str(out))
        print("------------")
        print("True:" + str(correct_int))