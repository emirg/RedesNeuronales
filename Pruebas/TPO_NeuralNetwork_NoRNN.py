# Recurrent Neural Network from Scratch in Python 3
import copy
import numpy as np
import csv
#import pandas as pd

class NeuralNetwork():

    def __init__(self):
        self.sum = False  # sum = True / max = False
        self.int_to_binary = {}
        self.binary_dim = 8
        self.max_val = (2 ** self.binary_dim)
        self.binary_val = np.unpackbits(np.array([range(self.max_val)], dtype=np.uint8).T, axis=1)
        for i in range(self.max_val):
            self.int_to_binary[i] = self.binary_val[i]

        self.epochs = 300

        self.training_size = 100000

        self.dataSetInputs = list()
        self.dataSetOutputs = list()

        # NN variables
        self.learning_rate = 0.1

        # Inputs: Values to be added bit by bit
        self.inputLayerSize = 2

        # Hidden Layer with 64 neurons
        self.hiddenLayerSize = 64

        # Output at one time step is 1 bit
        self.outputLayerSize = 1

        # Initialize Weights
        # Weight of first Synapse (Synapse_0) from Input to Hidden Layer at Current Timestep
        self.W1 = 2 * np.random.random((self.inputLayerSize, self.hiddenLayerSize)) - 1

        # Weight of second Synapse (Synapse_1) from Hidden Layer to Output Layer
        self.W2 = 2 * np.random.random((self.hiddenLayerSize, self.hiddenLayerSize)) - 1

        # Weight of second Synapse (Synapse_1) from Hidden Layer to Output Layer
        self.W3 = 2 * np.random.random((self.hiddenLayerSize, self.outputLayerSize)) - 1


        # Initialize Updated Weights Values
        self.W1_update = np.zeros_like(self.W1)
        self.W2_update = np.zeros_like(self.W2)
        self.W3_update = np.zeros_like(self.W3)

        self.hidden_layer_1_values = list()
        self.hidden_layer_2_values = list()

        self.minError = 999

    # Sigmoid Activation Function
    # To be applied at Hidden Layers and Output Layer
    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    # Derivative of Sigmoid Function
    # Used in calculation of Back Propagation Loss
    def sigmoidPrime(self, z):
        return z * (1 - z)




    def train(self):
        for j in range(self.training_size):  # 100000 anduvo bastante bien para la suma
            # ----------------------------- Compute True Values for the Sum (a+b) [binary encoded] --------------------------
            # Generate a random sample value for 1st input
            if self.sum:
                a_int = np.random.randint(self.max_val / 2)
                b_int = np.random.randint(self.max_val / 2)
            else:
                a_int = np.random.randint(self.max_val)
                b_int = np.random.randint(self.max_val)

            # Convert this Int value to Binary
            a = self.int_to_binary[a_int]

            # Map Int to Binary
            b = self.int_to_binary[b_int]

            # True Answer a + b = c
            if self.sum:
                c_int = a_int + b_int
            else:
                c_int = max(a_int, b_int)

            c = self.int_to_binary[c_int]

            # Array to save predicted outputs (binary encoded)
            d = np.zeros_like(c)

        for j in range(self.epochs):

            # Initialize overall error to "0"
            overallError = 0

            # Save the values of dJdW1 and dJdW2 computed at Output layer into a list
            output_layer_deltas = list()

            # Initially, there is no previous hidden state. So append "0" for that
            self.hidden_layer_1_values.append(np.zeros(self.hiddenLayerSize))
            self.hidden_layer_2_values.append(np.zeros(self.hiddenLayerSize))

            # ----------------------------- Compute the Values for (a+b) using RNN [Forward Propagation] ----------------------
            # position: location of the bit amongst 8 bits; starting point "0"; "0 - 7"
            for position in range(self.binary_dim):
                # Generate Input Data for RNN
                # Take the binary values of "a" and "b" generated for each iteration of "j"

                # With increasing value of position, the bit location of "a" and "b" decreases from "7 -> 0"
                # and each iteration computes the sum of corresponding bit of "a" and "b".
                # ex. for position = 0, X = [a[7],b[7]], 7th bit of a and b.
                X = np.array([[a[self.binary_dim - position - 1], b[self.binary_dim - position - 1]]])

                # Actual value for (a+b) = c, c is an array of 8 bits, so take transpose to compare bit by bit with X value.
                y = np.array([[c[self.binary_dim - position - 1]]]).T

                # Values computed at current hidden layer
                # [dot product of Input(X) and Weights(W1)] + [dot product of previous hidden layer values and Weights (W_h)]
                # W_h: weight from previous step hidden layer to current step hidden layer
                # W1: weights from current step input to current hidden layer
                layer_1 = self.sigmoid(np.dot(X, self.W1))

                # The new output using new Hidden layer values
                layer_2 = self.sigmoid(np.dot(layer_1, self.W2))

                layer_3 = self.sigmoid(np.dot(layer_2, self.W3))

                # Calculate the error
                output_error = y - layer_3 # y_j - a_j

                # Save the error deltas at each step as it will be propagated back
                output_layer_deltas.append((output_error) * self.sigmoidPrime(layer_3)) # delta_j

                # Save the sum of error at each binary position
                overallError += np.abs(output_error[0])


                # Round off the values to nearest "0" or "1" and save it to a list
                d[self.binary_dim - position - 1] = np.round(layer_3[0][0])

                # Save the hidden layer to be used later
                self.hidden_layer_1_values.append(copy.deepcopy(layer_1))
                self.hidden_layer_2_values.append(copy.deepcopy(layer_2))



            # ----------------------------------- Back Propagating the Error Values to All Previous Time-steps ---------------------
            for position in range(self.binary_dim):
                # a[0], b[0] -> a[1]b[1] ....
                X = np.array([[a[position], b[position]]])  # 1x2

                # The last step Hidden Layer where we are currently a[0],b[0]
                layer_1 = self.hidden_layer_1_values[-position - 1] # Values from Forward Propagation
                layer_2 = self.hidden_layer_2_values[-position - 1] # Values from Forward Propagation

                # Errors at Output Layer, a[1],b[1]
                output_layer_delta = output_layer_deltas[-position - 1] # Error calculation for output layer

                layer_2_delta = output_layer_delta.dot(self.W3.T) * self.sigmoidPrime(layer_2) #

                layer_1_delta = layer_2_delta.dot(self.W2.T) * self.sigmoidPrime(layer_1)


                # Update all the weights and try again
                self.W1_update += X.T.dot(layer_1_delta)
                self.W2_update += layer_1.T.dot(layer_2_delta)
                self.W3_update += layer_2.T.dot(output_layer_delta)



            # Update the weights with the values
            self.W1 += self.W1_update * self.learning_rate
            self.W2 += self.W2_update * self.learning_rate
            self.W3 += self.W3_update * self.learning_rate

            # Clear the updated weights values
            self.W1_update *= 0
            self.W2_update *= 0
            self.W3_update *= 0


            #if overallError < self.minError:
            #    self.minError = overallError
            #    print(self.minError)
            #    self.saveModel()

            # Print out the Progress of the RNN
            if (j % 10000 == 0):
                print("Iteracion: " + str(j))
                print("Error:" + str(overallError))
                print("Pred:" + str(d))
                print("True:" + str(c))
                out = 0
                for index, x in enumerate(reversed(d)):
                    out += x * pow(2, index)
                if self.sum:
                    print(str(a_int) + " + " + str(b_int) + " = " + str(out))
                else:
                    print("max(" + str(a_int) + " , " + str(b_int) + ") = " + str(out))
                print("------------")

    def predict(self):
        # Solicito al usuario dos numeros para comprobar las predicciones de la red
        user_input_one = int(input("Entrada uno (p1): "))
        user_input_two = int(input("Entrada dos (p2): "))

        a = self.int_to_binary[user_input_one]
        b = self.int_to_binary[user_input_two]

        c_int = user_input_one + user_input_two
        # c_int = max(user_input_one, user_input_two)
        c = self.int_to_binary[c_int]
        d = np.zeros_like(c)


        for position in range(self.binary_dim):
            X = np.array([[a[self.binary_dim - position - 1], b[self.binary_dim - position - 1]]])

            layer_1 = self.sigmoid(np.dot(X, self.W1) )

            # The new output using new Hidden layer values
            layer_2 = self.sigmoid(np.dot(layer_1, self.W2))

            layer_3 = self.sigmoid(np.dot(layer_2, self.W3))

            # Round off the values to nearest "0" or "1" and save it to a list
            d[self.binary_dim - position - 1] = np.round(layer_3[0][0])

        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)

        print("Pred:" + str(d))
        print("True:" + str(c))
        if self.sum:
            print(str(user_input_one) + " + " + str(user_input_two) + " = " + str(out))
        else:
            print("max(" + str(user_input_one) + " , " + str(user_input_two) + ") = " + str(out))
        print("------------")

    def saveModel(self):
        np.savetxt("models/modelo_max_v1/W1.csv", self.W1, delimiter=",")
        np.savetxt("models/modelo_max_v1/W2.csv", self.W2, delimiter=",")
        np.savetxt("models/modelo_max_v1/W3.csv", self.W3, delimiter=",")
        np.savetxt("models/modelo_max_v1/W_h_layer_1.csv", self.W_h_layer_1, delimiter=",")
        np.savetxt("models/modelo_max_v1/W_h_layer_2.csv", self.W_h_layer_2, delimiter=",")

    def loadModel(self):
        with open("models/modelo_max_v1/W1.csv", 'r') as f:
            W1 = list(csv.reader(f, delimiter=","))
        self.W1 = np.array(W1, dtype=np.float)
        f.close()

        with open("models/modelo_max_v1/W2.csv", 'r') as f:
            W2 = list(csv.reader(f, delimiter=","))
        self.W2 = np.array(W2, dtype=np.float)
        f.close()

        with open("models/modelo_max_v1/W3.csv", 'r') as f:
            W3 = list(csv.reader(f, delimiter=","))
        self.W3 = np.array(W3, dtype=np.float)
        f.close()

        with open("models/modelo_max_v1/W_h_layer_1.csv", 'r') as f:
            W_h_layer_1 = list(csv.reader(f, delimiter=","))
        self.W_h_layer_1 = np.array(W_h_layer_1, dtype=np.float)
        f.close()

        with open("models/modelo_max_v1/W_h_layer_2.csv", 'r') as f:
            W_h_layer_2 = list(csv.reader(f, delimiter=","))
        self.W_h_layer_2 = np.array(W_h_layer_2, dtype=np.float)
        f.close()


if __name__ == "__main__":
    print("Inicializando Neural Network...")
    neural_network = NeuralNetwork()

    print("Entrenando...")
    neural_network.train()

    #print("Cargando modelo...")
    #neural_network.loadModel() # Carga el max

    #print("Min error:" + str(neural_network.minError))

    #print("Guardando modelo...")
    #neural_network.saveModel()

    print("Listo! Pregunte!")
    for i in range(10):
        neural_network.predict()
