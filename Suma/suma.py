import numpy as np

class NeuralNetwork():

    # Constructor
    # Inicializo los pesos en 1 -> Los pesos estan en forma de vector/matriz -> [1,1]
    # Inicializo el bias en 0
    def __init__(self):
        self.synaptic_weights = np.array([1,1])
        self.bias = 0

    # Funcion de activacion -> Lineal
    def linear(self, x):
        return x
        # Nota: Las redes neuronales con funcion lineal no pueden ser entrenadas

    # Funcion para predecir
    # Recibe los inputs (en forma de vector/matriz) y hace el proceso de una neurona:
    #   - Producto punto entre pesos y entradas = (p1 * w1 + p2 * w2)
    #   - Aplica funcion de activacion al producto punto obtenido (n)
    def think(self, inputs):
        inputs = inputs.astype(float) # Convierte a float 
        output = self.linear(np.dot(inputs, self.synaptic_weights)) # Producto punto y Funcion de activacion
        return output

    # Funcion para entrenar
    # En este caso no es necesario porque usando la funcion lineal el modelo no se puede entrenar
    # def entrenar(self, inputs, outputs)

if __name__ == "__main__":

    # initializing the neuron class
    neural_network = NeuralNetwork()
    user_input_one = str(input("Entrada uno (p1): "))
    user_input_two = str(input("Entrada dos (p2): "))
    # user_input_three = str(input("User Input Three: "))
    print("Respuesta: ")
    print(neural_network.think(
        np.array([user_input_one, user_input_two])))
    
