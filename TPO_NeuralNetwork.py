import numpy as np
import csv
import os

def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val

def to_bin(val, bits):
    # val es un int positivo o negativo
    # bits es la cantidad de bits que utiliza el numero
    # retorna un 1D array
    val_bin =  np.binary_repr(val, bits) # Int -> Str
    val_bin = np.array(list(val_bin),dtype=int) # Str -> 1D Array
    return val_bin


class NeuralNetwork():

    def __init__(self):
        self.sum = False  # sum = True / max = False
        self.int_to_binary = {}
        self.binary_dim = 16
        self.max_val = (2 ** self.binary_dim)
        # self.max_val = (2**(self.binary_dim - 1)) - 1
        # self.min_val = -(2**(self.binary_dim - 1))

        self.training_size = 500000

        #self.dataSetInputs = list()
        #self.dataSetOutputs = list()

        # NN variables
        self.learning_rate = 0.1

        # Tamaño de las inputs
        self.inputLayerSize = self.binary_dim * 2

        # Tamaño de las Hidden Layers
        self.hiddenLayerSize = 256

        # Tamaño de las outputs
        self.outputLayerSize = self.binary_dim

        # Pesos
        # Pesos entre la Entrada y la Primer Capa: (32x256)
        self.W1 = 2 * np.random.random((self.inputLayerSize, self.hiddenLayerSize)) - 1

        # Pesos entre la Primer Capa y la Segunda Capa: (256x256)
        self.W2 = 2 * np.random.random((self.hiddenLayerSize, self.hiddenLayerSize)) - 1

        # Pesos entre la Segunda Capa y la Salida: (256x16)
        self.W3 = 2 * np.random.random((self.hiddenLayerSize, self.outputLayerSize)) - 1

        # Initialize Updated Weights Values
        self.W1_update = np.zeros_like(self.W1)
        self.W2_update = np.zeros_like(self.W2)
        self.W3_update = np.zeros_like(self.W3)

        self.hidden_layer_1_values = list()
        self.hidden_layer_2_values = list()

        self.error=0

    # Sigmoid
    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    # Derivada de Sigmoid
    def sigmoidPrime(self, z):
        return z * (1 - z)

    def train(self):
        for j in range(self.training_size):
            # Genera dos numeros random
            if self.sum:
                a_int = np.random.randint(-(self.max_val / 4),(self.max_val / 4) - 1) # Esto es temporal, despues capaz nos conviene tener un self.max_val
                b_int = np.random.randint(-(self.max_val / 4),(self.max_val / 4) - 1) # y un self.min_val para que estas partes queden mas prolijas
            else:
                a_int = np.random.randint(-(self.max_val/2),(self.max_val/2) - 1)
                b_int = np.random.randint(-(self.max_val/2),(self.max_val/2) - 1)

            # Pasaje de entero a binario
            a = to_bin(a_int,self.binary_dim)
            b = to_bin(b_int, self.binary_dim)

            signo=1

            # Valor verdadero de la consulta
            if self.sum:
                c_int = a_int + b_int
            else:
                c_int = max(a_int, b_int)
            if(c_int<0):
                signo=-1
            c = to_bin(c_int, self.binary_dim)

            # Variable donde se almacenara el output
            d = np.zeros_like(c)

            # Error inicial
            overallError = 0

            # ----------------------------- Forward Propagation -----------------------------

            # Creamos el vector de inputs con las entradas correspondientes
            X = np.array([a, b]).reshape(1, self.inputLayerSize)

            # Obtenemos el valor real para comparar
            y = np.array(c).T

            # Calculamos la predicción pasando las entradas a traves de la red
            layer_1 = self.sigmoid(np.dot(X, self.W1))

            layer_2 = self.sigmoid(np.dot(layer_1, self.W2))

            layer_3 = self.sigmoid(np.dot(layer_2, self.W3))

            # Cálculo del error
            output_error = y - layer_3

            # Actualizamos el error
            overallError += np.abs(output_error[0][0])

            # Redondeamos los valores a 0 o 1, y ajustamos la forma del vector
            d = np.round(layer_3).reshape(self.binary_dim).astype('int16')

            # ----------------------------- BackPropagating -----------------------------

            # Obtenemos los deltas para cada capa con sus respectivas entradas
            output_layer_delta = (output_error) * self.sigmoidPrime(layer_3)

            layer_2_delta = output_layer_delta.dot(self.W3.T) * self.sigmoidPrime(layer_2)  #

            layer_1_delta = layer_2_delta.dot(self.W2.T) * self.sigmoidPrime(layer_1)

            # Actualizamos las matrices utilizadas en la actualización de pesos
            self.W1_update += X.T.dot(layer_1_delta)
            self.W2_update += layer_1.T.dot(layer_2_delta)
            self.W3_update += layer_2.T.dot(output_layer_delta)

            # Actualizamos los pesos
            self.W1 += self.W1_update * self.learning_rate
            self.W2 += self.W2_update * self.learning_rate
            self.W3 += self.W3_update * self.learning_rate

            #
            self.W1_update *= 0
            self.W2_update *= 0
            self.W3_update *= 0

            # Muestra el progreso
            if (j % 10000 == 0):
                print("Iteracion: " + str(j))
                self.error=0
                for k in range(self.binary_dim):
                    #print(overallError[0][k])
                    self.error+=overallError[0][k]
                print("Error:" + str(self.error))
                print("Pred:" + str(d))
                print("True:" + str(c))
                out = 0
                for index, x in enumerate(reversed(d)):
                    out += x * pow(2, index)

                if(signo==-1):
                    out=twos_comp(out,self.binary_dim)
                if self.sum:
                    print(str(a_int) + " + " + str(b_int) + " = " + str(out))
                else:
                    print("max(" + str(a_int) + " , " + str(b_int) + ") = " + str(out))
                print("------------")
            if j==self.training_size:
                self.error=0
                for k in range(self.binary_dim):
                    #print(overallError[0][k])
                    self.error+=overallError[0][k]

    def predict(self):
        # Solicito al usuario dos numeros para comprobar las predicciones de la red
        print("ERROR : "+str(self.error))
        user_input_one = int(input("Entrada uno (p1): "))
        user_input_two = int(input("Entrada dos (p2): "))

        # Pasaje de enteros a binario
        a = to_bin(user_input_one, self.binary_dim)
        b = to_bin(user_input_two, self.binary_dim)

        signo=1

        # Valor verdadero de la consulta
        if self.sum:
            c_int = user_input_one + user_input_two
        else:
            c_int = max(user_input_one, user_input_two)
        if(c_int<0):
            signo=-1
        c = to_bin(c_int, self.binary_dim)
        d = np.zeros_like(c)

        X = np.array([a, b]).reshape(1, self.inputLayerSize)

        # Calculo de la prediccion
        layer_1 = self.sigmoid(np.dot(X, self.W1))

        layer_2 = self.sigmoid(np.dot(layer_1, self.W2))

        layer_3 = self.sigmoid(np.dot(layer_2, self.W3))

        # Redondeo a 0 o 1
        d = np.round(layer_3).reshape(self.binary_dim).astype('int16')

        # Pasaje de binario a entero
        out = 0
        if (signo == -1):
            out = twos_comp(out, self.binary_dim)
        else:
            for index, x in enumerate(reversed(d)):
                out += x * pow(2, index)

        print("Pred:" + str(d))
        print("True:" + str(c))
        if self.sum:
            print(str(user_input_one) + " + " + str(user_input_two) + " = " + str(out))
        else:
            print("max(" + str(user_input_one) + " , " + str(user_input_two) + ") = " + str(out))
        print("------------")


    # Metodo utilizado para guardar los pesos de una red
    def saveModel(self,modelname):
        dir = "models/" + modelname
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.savetxt(dir + "/W1.csv", self.W1, delimiter=",")
        np.savetxt(dir + "/W2.csv", self.W2, delimiter=",")
        np.savetxt(dir + "/W3.csv", self.W3, delimiter=",")

    # Metodo utilizado para cargar los pesos a una red
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

# Interfaz
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
    while opcion != 0:
        opcion = menu(neural_network)