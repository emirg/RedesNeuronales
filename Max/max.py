import numpy as np
import json

entrada = [] # Entradas de entrenamiento
salida = [] # Salidas de entrenamiento

class NeuralNetwork():

    # Constructor
    # Inicializo los pesos con valores aleatorios
    # Inicializo el bias con un valor aleatorio
    def __init__(self):
        #np.random.seed(1) # Util mas que nada para testing/debugging
        self.synaptic_weights =  2 * np.random.random((2, 1)) - 1
        #self.bias = np.random.random()
        self.bias = 2 * np.random.random((1, )) - 1

    # Funcion de activacion -> hardlim === Escalon 
    def hardlim(self, x):
        if x >= 0:
            return 1
        else:
            return 0

    # Funcion para predecir
    # Recibe los inputs (en forma de vector/matriz) y hace el proceso de una neurona:
    #   - Producto punto entre pesos y entradas = (n = p1 * w1 + p2 * w2 + ... + pq * wq) 
    #   - Aplica funcion de activacion al producto punto obtenido (n)
    def think(self, inputs):
        # inputs = inputs.astype(float) # Convierte a float 
        output = self.hardlim(np.dot(inputs, self.synaptic_weights) + self.bias) # Producto punto y Funcion de activacion
        return output

    # Funcion para entrenar
    def entrenar(self, training_inputs, training_outputs, training_iterations):

        # Una iteracion = Una epoca
        for iteration in range(training_iterations): # Para cada epoca
            for q in range(len(training_inputs)): # Para cada dato de entrenamiento

                # Obtengo lo que la red cree que es la respuesta 
                output = self.think(training_inputs[q]) # Esto devuelve 1 o 0

                # Calculo el error entre la salida de la red y la salida real
                error = training_outputs[q] - output
                #print("error: ",error)
                
                # Ajusto los pesos y bias
                adjustments = np.dot(error,np.array(training_inputs[q]).T)
                self.synaptic_weights += np.array([adjustments]).T
                self.bias += error

# Metodo para cargar el JSON a python
def loadData():
    with open('ProblemaMax.json') as json_file:  
        data = json.load(json_file)
        for p in data:
            entradas = p["ParametrosEntrada"].split(",") # Divido los parametros
            entradaUno = int(entradas[0],10) # Los convierto a int en base 10
            entradaDos = int(entradas[1],10)
            entrada.append([entradaUno,entradaDos]) # Los guardo como un arreglo
            salida.append(p["Salida"]) # Guardo la salida
            # print("Dato:" , entrada[0]," --> ", p["Salida"])
        
# Metodo encargado de mapear los valores de salida (int) a 0/1        
def dataParsing():
    for x in range(len(entrada)):
        if salida[x] == entrada[x][0]:
            salida[x] = 0
        else:
            salida[x] = 1
    
    
# Main 
if __name__ == "__main__":
    loadData() # Cargo el JSON a una estructura en python
    dataParsing() # Mapeo las salidas a un numero 0 o 1

    # Convierto las entradas a un vector    
    entrada = np.array(entrada)
    
    # Inicializo red neuronal
    neural_network = NeuralNetwork()
    print("Pesos y bias inicial: ")
    print(" W:", neural_network.synaptic_weights)
    print(" b:", neural_network.bias)

    # Entro la red con los datos del JSON 
    neural_network.entrenar(entrada, salida, 1000)

    print("Pesos y bias luego del entrenamiento: ")
    print(" W:", neural_network.synaptic_weights)
    print(" b:", neural_network.bias)

    # Solicito al usuario dos numeros para comprobar las predicciones de la red
    user_input_one = str(input("Entrada uno (p1): "))
    user_input_two = str(input("Entrada dos (p2): "))

    print("Respuesta: ")
    respuesta = neural_network.think(np.array([int(user_input_one,10), int(user_input_two,10)]))
    # Simplemente para ver el numero maximo y no un 0/1
    if respuesta == 0:
        print(user_input_one)
    else:
        print(user_input_two)
    
