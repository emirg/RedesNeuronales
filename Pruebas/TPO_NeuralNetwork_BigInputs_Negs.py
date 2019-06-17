# Recurrent Neural Network from Scratch in Python 3
import copy
import numpy as np
import csv
import os
from bitarray import bitarray

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

def to_bin(val, bits):
    # val es un int positivo o negativo
    # bits es la cantidad de bits que utiliza el numero
    # retorna un 1D array
    val_bin =  np.binary_repr(val, bits) # Int -> Str
    val_bin = np.array(list(val_bin),dtype=int) # Str -> 1D Array
    return val_bin

#def to_int(bin):
    # bin es un 1-D array de 1/0's
    #bin_inv = np.invert(bin.astype(bool)).astype(np.uint8) # Invierto
    #num = ''
    #for x, val in enumerate(reversed(bin_inv)):
        #num = num + str(val)
    #print()



class NeuralNetwork():

    def __init__(self):
        self.sum = False  # sum = True / max = False
        self.int_to_binary = {}
        self.binary_dim = 16
        self.max_val = (2 ** self.binary_dim)
        # self.max_val = (2**(self.binary_dim - 1)) - 1
        # self.min_val = -(2**(self.binary_dim - 1))

        #self.binary_val = np.unpackbits(np.array([range(self.max_val)], dtype=np.uint8).T, axis=1) # Original de numpy
        self.binary_val = unpackbits(np.array([range(self.max_val)],dtype=np.uint16).T,self.binary_dim)
        for i in range(self.max_val):
            self.int_to_binary[i] = self.binary_val[i]

        #self.epochs = 300

        self.training_size = 500000

        #self.dataSetInputs = list()
        #self.dataSetOutputs = list()

        # NN variables
        self.learning_rate = 0.1

        # Inputs: Values to be added bit by bit
        self.inputLayerSize = self.binary_dim * 2

        # Hidden Layer with 64 neurons
        self.hiddenLayerSize = 256

        # Output at one time step is 1 bit
        self.outputLayerSize = self.binary_dim

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

        #self.minError = 999

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
                a_int = np.random.randint(-(self.max_val / 4),(self.max_val / 4) - 1) # Esto es temporal, despues capaz nos conviene tener un self.max_val
                b_int = np.random.randint(-(self.max_val / 4),(self.max_val / 4) - 1) # y un self.min_val para que estas partes queden mas prolijas
            else:
                a_int = np.random.randint(-(self.max_val/2),(self.max_val/2) - 1)
                b_int = np.random.randint(-(self.max_val/2),(self.max_val/2) - 1)

            a = to_bin(a_int,self.binary_dim)
            b = to_bin(b_int, self.binary_dim)

            # True Answer
            if self.sum:
                c_int = a_int + b_int
            else:
                c_int = max(a_int, b_int)

            c = to_bin(c_int, self.binary_dim)

            # Array to save predicted outputs (binary encoded)
            d = np.zeros_like(c)

            # Initialize overall error to "0"
            overallError = 0

            # Save the values of dJdW1 and dJdW2 computed at Output layer into a list
            output_layer_deltas = list()

            # Initially, there is no previous hidden state. So append "0" for that
            #self.hidden_layer_1_values.append(np.zeros(self.hiddenLayerSize))
            #self.hidden_layer_2_values.append(np.zeros(self.hiddenLayerSize))

            # ----------------------------- Compute the Values for (a+b) using RNN [Forward Propagation] ----------------------

            X = np.array([a, b]).reshape(1, self.inputLayerSize)  # (16,) o (1,16)?

            # Actual value for (a+b) = c, c is an array of 8 bits, so take transpose to compare bit by bit with X value.
            y = np.array(c).T

            layer_1 = self.sigmoid(np.dot(X, self.W1))

            # The new output using new Hidden layer values
            layer_2 = self.sigmoid(np.dot(layer_1, self.W2))

            layer_3 = self.sigmoid(np.dot(layer_2, self.W3))

            # Calculate the error
            output_error = y - layer_3  # y_j - a_j

            # Save the sum of error at each binary position
            overallError += np.abs(output_error)

            # Round off the values to nearest "0" or "1" and save it to a list
            d = np.round(layer_3).reshape(self.binary_dim).astype('int16')

            # ----------------------------------- Back Propagating the Error Values to All Previous Time-steps ---------------------

            # a[0], b[0] -> a[1]b[1] ....
            # X = np.array([a, b])  # 1x2

            # Errors at Output Layer, a[1],b[1]
            output_layer_delta = (output_error) * self.sigmoidPrime(layer_3)

            layer_2_delta = output_layer_delta.dot(self.W3.T) * self.sigmoidPrime(layer_2)  #

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

            # if overallError < self.minError:
            #    self.minError = overallError
            #    print(self.minError)
            #    self.saveModel()

            # Print out the Progress of the RNN
            if (j % 10000 == 0):
                print("Iteracion: " + str(j))
                #print("Error:" + str(overallError))
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
        print("NO ESTA IMPLEMENTADO EL PASAJE DE COMPLEMENTO A DOS -> ENTERO")
        print("IGNORAR LOS ENTEROS Y COMPARAR CON PRED Y TRUE")
        # Solicito al usuario dos numeros para comprobar las predicciones de la red
        user_input_one = int(input("Entrada uno (p1): "))
        user_input_two = int(input("Entrada dos (p2): "))

        a = to_bin(user_input_one, self.binary_dim)
        b = to_bin(user_input_two, self.binary_dim)

        # True Answer
        if self.sum:
            c_int = user_input_one + user_input_two
        else:
            c_int = max(user_input_one, user_input_two)

        c = to_bin(c_int, self.binary_dim)
        d = np.zeros_like(c)

        X = np.array([a, b]).reshape(1, self.inputLayerSize)

        layer_1 = self.sigmoid(np.dot(X, self.W1))

        # The new output using new Hidden layer values
        layer_2 = self.sigmoid(np.dot(layer_1, self.W2))

        layer_3 = self.sigmoid(np.dot(layer_2, self.W3))

        # Round off the values to nearest "0" or "1" and save it to a list
        d = np.round(layer_3).reshape(self.binary_dim).astype('int16')


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


    def saveModel(self,modelname):
        dir = "models/" + modelname
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.savetxt(dir + "/W1.csv", self.W1, delimiter=",")
        np.savetxt(dir + "/W2.csv", self.W2, delimiter=",")
        np.savetxt(dir + "/W3.csv", self.W3, delimiter=",")


    def loadModel(self,modelname):
        with open("models/" + modelname + "/W1.csv", 'r') as f:
            W1 = list(csv.reader(f, delimiter=","))
        self.W1 = np.array(W1, dtype=np.float)
        f.close()

        with open("models/" + modelname + "/W2.csv", 'r') as f:
            W2 = list(csv.reader(f, delimiter=","))
        self.W2 = np.array(W2, dtype=np.float)
        f.close()

        with open("models/" + modelname + "/W3.csv", 'r') as f:
            W3 = list(csv.reader(f, delimiter=","))
        self.W3 = np.array(W3, dtype=np.float)
        f.close()

def menu(neural_network):
    print("---Neural Network---")
    print("1) Elegir problema")
    print("2) Entrenar")
    print("3) Predecir")
    print("4) Guardar modelo")
    print("5) Cargar modelo")
    print("0) Salir")
    user_input = int(input("Opcion: "))
    if user_input == 1:
        print("1) Suma")
        print("2) Max")
        problema = int(input("Opcion: "))
        if problema == 1:
            neural_network.sum = True
        else:
            neural_network.sum = False
    if user_input == 2:
        neural_network.train()
    if user_input == 3:
        neural_network.predict()
    if user_input == 4:
        modelo = str(input("Nombre del modelo: "))
        neural_network.saveModel(modelo)
    if user_input == 5:
        modelo = str(input("Nombre del modelo: "))
        neural_network.loadModel(modelo)

    return user_input

if __name__ == "__main__":
    print("Inicializando Neural Network...")
    neural_network = NeuralNetwork()

    opcion = 1
    while opcion > 0:
        opcion = menu(neural_network)





