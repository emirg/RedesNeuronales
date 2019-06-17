import pandas as pd
import numpy as np
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Implementacion de Redes Neuronales desde 0


class NeuralNetwork():

    # Constructor
    # Inicializo los pesos con valores aleatorios
    # Inicializo el bias con un valor aleatorio
    def __init__(self):
        # np.random.seed(1) # Util mas que nada para testing/debugging
        self.synaptic_weights_1 = 2 * np.random.random((2, 1)) - 1
        self.bias_1 = 2 * np.random.random((1,)) - 1

        self.synaptic_weights_2 = 2 * np.random.random((2, 1)) - 1
        self.bias_2 = 2 * np.random.random((1,)) - 1

    # Funcion de activacion -> hardlim === Escalon
    def hardlim(self, x):
        if x >= 0:
            return 1
        else:
            return 0

    def sigmoid(self,x):
        output = 1 / (1 + np.exp(-x))
        return output

    # convert output of sigmoid function to its derivative
    def sigmoid_output_to_derivative(self,output):
        return output * (1 - output)

    # Funcion para predecir
    # Recibe los inputs (en forma de vector/matriz) y hace el proceso de una neurona:
    #   - Producto punto entre pesos y entradas = (n = p1 * w1 + p2 * w2 + ... + pq * wq)
    #   - Aplica funcion de activacion al producto punto obtenido (n)
    def think(self, inputs):
        # inputs = inputs.astype(float) # Convierte a float
        output = self.hardlim(np.dot(inputs, self.synaptic_weights_1) + self.bias_1)  # Producto punto y Funcion de activacion
        return output

    # Funcion para entrenar
    def entrenar(self, training_inputs, training_outputs, training_iterations):

        # Una iteracion = Una epoca
        for iteration in range(training_iterations):  # Para cada epoca
            for q in range(len(training_inputs)):  # Para cada dato de entrenamiento

                # Obtengo lo que la red cree que es la respuesta
                output = self.think(training_inputs[q])  # Esto devuelve 1 o 0

                # Calculo el error entre la salida de la red y la salida real
                error = training_outputs[q] - output
                # print("error: ",error)

                # Ajusto los pesos y bias
                adjustments = (error * self.sigmoid_output_to_derivative(output))* training_inputs[q]
                self.synaptic_weights_1 += np.array([adjustments]).T
                self.bias_1 += error


def cargarDatos():
    #read data file
    df = pd.read_csv('data/SolucionSuma.csv', header=None)
    df.rename(columns={0: 'idSolucion', 1: 'idProblema', 2: 'parametrosEntrada', 3: 'salida'}, inplace=True)
    #df.to_csv('data/SolucionSuma.csv', index=False) # save to new csv file

    #check data has been read in properly
    df.head()

    #create a dataframe with all training data except the target column
    train_X = df.drop(columns=['idSolucion', 'idProblema', 'salida'])


    # new data frame with split value columns
    new = train_X["parametrosEntrada"].str.split(",", n=1, expand=True)

    # making separate first name column from new data frame
    train_X["entradaUno"] = new[0].astype(int)

    # making separate last name column from new data frame
    train_X["entradaDos"] = new[1].astype(int)

    # Dropping old Name columns
    train_X.drop(columns=["parametrosEntrada"], inplace=True)


    test_X = train_X.iloc[ 0:5 ,]
    #print(test_X.head())

    train_X = train_X.get_values()
    test_X = test_X.get_values()

    #check that the target variable has been removed
    #print(train_X.dtypes)

    #create a dataframe with only the target column
    train_Y = df[['salida']].astype(int)
    train_Y = train_Y.get_values()
    #print(train_y)
    return train_X,test_X,train_Y


# Main
if __name__ == "__main__":
    train_X, test_X, train_Y = cargarDatos()  # Cargo el CSV a una estructura en python


    # Inicializo red neuronal
    neural_network = NeuralNetwork()
    print("Pesos y bias inicial: ")
    print(" W:", neural_network.synaptic_weights_1)
    print(" b:", neural_network.bias_1)

    # Entro la red con los datos del JSON
    neural_network.entrenar(train_X, train_Y, 1000)

    print("Pesos y bias luego del entrenamiento: ")
    print(" W:", neural_network.synaptic_weights)
    print(" b:", neural_network.bias)

    # Solicito al usuario dos numeros para comprobar las predicciones de la red
    user_input_one = str(input("Entrada uno (p1): "))
    user_input_two = str(input("Entrada dos (p2): "))

    print("Respuesta: ")
    respuesta = neural_network.think(np.array([int(user_input_one, 10), int(user_input_two, 10)]))
    # Simplemente para ver el numero maximo y no un 0/1
    if respuesta == 0:
        print(user_input_one)
    else:
        print(user_input_two)