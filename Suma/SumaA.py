import numpy as np
import json
import copy

entrada = [] # Entradas de entrenamiento
salida = [] # Salidas de entrenamiento

class NeuralNetwork():

    # Constructor
    # Inicializo los pesos con valores aleatorios
    # Inicializo el bias con un valor aleatorio
    def __init__(self):

        # input variables
        self.alpha = 0.1
        self.input_dim = 2
        self.hidden_dim = 64
        self.output_dim = 1

       # initialize neural network weights
        self.synaptic_weights_0 = 2*np.random.random((self.input_dim,self.hidden_dim)) - 1
        self.synaptic_weights_1 = 2*np.random.random((self.hidden_dim,self.output_dim)) - 1
        self.synaptic_weights_bias = 2*np.random.random((self.hidden_dim,self.hidden_dim)) - 1

        self.synaptic_weights_0_update = np.zeros_like(self.synaptic_weights_0)
        self.synaptic_weights_1_update = np.zeros_like(self.synaptic_weights_1)
        self.synaptic_weights_bias_update = np.zeros_like(self.synaptic_weights_bias)

        self.layer_2_deltas = list()
        self.layer_1_values = list()
        

    # Funcion de activacion -> hardlim === Escalon 
    # compute sigmoid nonlinearity
    def sigmoid(self,x):
        output = 1/(1+np.exp(-x))
        return output

    # convert output of sigmoid function to its derivative
    def sigmoid_output_to_derivative(self,output):
        return output*(1-output)

    # Funcion para predecir
    # Recibe los inputs (en forma de vector/matriz) y hace el proceso de una neurona:
    #   - Producto punto entre pesos y entradas = (n = p1 * w1 + p2 * w2 + ... + pq * wq) 
    #   - Aplica funcion de activacion al producto punto obtenido (n)
    def think(self, inputs):
        # hidden layer (input ~+ prev_hidden)
        #print((inputs[0])[0])
        #print((inputs[1])[15])
        #print(self.synaptic_weights_0)
        for position in range(binary_dim):
            X = np.array([[((inputs[0])[0])[binary_dim - position - 1],((inputs[1])[0])[binary_dim - position - 1]]])

        output= self.sigmoid(np.dot(X,self.synaptic_weights_0) + np.dot(self.layer_1_values[-1],self.synaptic_weights_bias))
        # output layer (new binary representation)
        output = np.round(self.sigmoid(np.dot(output,self.synaptic_weights_1)))
        d = np.zeros_like((inputs[0]))
        for position in range(binary_dim):
            (d[0])[binary_dim - position - 1] = np.round(output[0][0])
        return d
    
    # Funcion para entrenar
    def entrenar(self, training_iterations):
        for j in range(training_iterations):
            # generate a simple addition problem (a + b = c)
            a_int = np.random.randint(256/2) # int version
            a = int2binary[a_int] # binary encoding

            b_int = np.random.randint(256/2) # int version
            b = int2binary[b_int] # binary encoding

            # true answer
            c_int = a_int + b_int
            c = int2binary[c_int]

            # where we'll store our best guess (binary encoded)
            d = np.zeros_like(c)

            overallError = 0
            
            self.layer_1_values.append(np.zeros(self.hidden_dim))
            
            # moving along the positions in the binary encoding
            for position in range(binary_dim):
                
                # generate input and output
                X = np.array([[(a[0])[binary_dim - position - 1],(b[0])[binary_dim - position - 1]]])
                #print(X)
                y = np.array([[(c[0])[binary_dim - position - 1]]]).T

                # hidden layer (input ~+ prev_hidden)
                # print(X)
                layer_1 = self.sigmoid(np.dot(X,self.synaptic_weights_0) + np.dot(self.layer_1_values[-1],self.synaptic_weights_bias))

                # output layer (new binary representation)
                layer_2 = self.sigmoid(np.dot(layer_1,self.synaptic_weights_1))

                # did we miss?... if so, by how much?
                layer_2_error = y - layer_2
                self.layer_2_deltas.append((layer_2_error)*self.sigmoid_output_to_derivative(layer_2))
                overallError += np.abs(layer_2_error[0])
            
                # decode estimate so we can print it out
                (d[0])[binary_dim - position - 1] = np.round(layer_2[0][0])
                
                # store hidden layer so we can use it in the next timestep
                self.layer_1_values.append(copy.deepcopy(layer_1))
            
            #inicializo en cero
            future_layer_1_delta = np.zeros(self.hidden_dim)
            
            for position in range(binary_dim):
                
                X = np.array([[(a[0])[position],(b[0])[position]]])
                layer_1 = self.layer_1_values[-position-1]
                prev_layer_1 = self.layer_1_values[-position-2]
                
                # error at output layer
                layer_2_delta = self.layer_2_deltas[-position-1]
                # error at hidden layer
                layer_1_delta = (future_layer_1_delta.dot(self.synaptic_weights_bias.T) + layer_2_delta.dot(self.synaptic_weights_1.T)) * self.sigmoid_output_to_derivative(layer_1)

                # let's update all our weights so we can try again
                self.synaptic_weights_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
                self.synaptic_weights_bias_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
                self.synaptic_weights_0 += X.T.dot(layer_1_delta)
                
                future_layer_1_delta = layer_1_delta
            
            self.synaptic_weights_0 += self.synaptic_weights_0_update * self.alpha
            self.synaptic_weights_1 += self.synaptic_weights_1_update * self.alpha
            self.synaptic_weights_bias += self.synaptic_weights_bias_update * self.alpha    

            self.synaptic_weights_0_update *= 0
            self.synaptic_weights_1_update *= 0
            self.synaptic_weights_bias_update *= 0

             # print out progress
            if(j % 1000 == 0):
                print ("Error:" + str(overallError))
                print ("Pred:" + str(d))
                print ("True:" + str(c))
                out = 0
                for index,x in enumerate(reversed(d[0])):
                    out += x*pow(2,index)
                print (str(a_int) + " + " + str(b_int) + " = " + str(out))
                print ("------------")

# training dataset generation
int2binary = {}
binary_dim = 16

def unpackbits(x,num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1,1])
    to_and_aux = 2**np.arange(num_bits).reshape([1,num_bits])
    to_and=2**np.arange(num_bits).reshape([1,num_bits])
    for i in range(len(to_and_aux[0])):
        to_and[0][i]=to_and_aux[0][num_bits-(i+1)]
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

largest_number = pow(2,binary_dim)
#creo un arreglo 
binary = unpackbits(
    np.array([range(largest_number)],dtype=np.uint16).T,binary_dim)


#print(largest_number)
for i in range(largest_number):
    int2binary[i] = binary[i]

    

# Metodo para cargar el JSON a python
def loadData():
    with open('ProblemaSuma.json') as json_file:  
        data = json.load(json_file)
        for p in data:
            entradas = p["ParametrosEntrada"].split(",") # Divido los parametros
            entradaUno = int(entradas[0],10) # Los convierto a int en base 10
            entradaUno =int2binary[entradaUno]
            entradaUno= unpackbits(entradaUno,binary_dim)
            entradaDos = int(entradas[1],10)
            entradaDos =int2binary[entradaDos]
            entradaDos = unpackbits(entradaDos,binary_dim)
            entrada.append([entradaUno,entradaDos]) # Los guardo como un arreglo
            salida.append(p["Salida"]) # Guardo la salida
            # print("Dato:" , entrada[0]," --> ", p["Salida"])
        
    
# Main 
if __name__ == "__main__":
    #loadData() # Cargo el JSON a una estructura en python

    # Convierto las entradas a un vector    
    #entrada = np.array(entrada)
    
    # Inicializo red neuronal
    neural_network = NeuralNetwork()
    #print("Pesos y bias inicial: ")
    #print(" W_0:", neural_network.synaptic_weights_0)
    #print(" W_1:", neural_network.synaptic_weights_1)
    #print(" b:", neural_network.synaptic_weights_bias)

    # Entro la red con los datos del JSON 
    neural_network.entrenar(10000)

    #rint("Pesos y bias luego del entrenamiento: ")
    #print(" W:", neural_network.synaptic_weights)
    #print(" b:", neural_network.bias)

    # Solicito al usuario dos numeros para comprobar las predicciones de la red
    user_input_one = str(input("Entrada uno (p1): "))
    user_input_two = str(input("Entrada dos (p2): "))
    user_input_one=int(user_input_one)
    user_input_two=int(user_input_two)
    user_input_one=int2binary[user_input_one]
    print(user_input_one)
    user_input_two=int2binary[user_input_two]
    print(user_input_two)

    print("Respuesta: ")
    d=neural_network.think(np.array([user_input_one, user_input_two]))
    out = 0
    for index,x in enumerate(reversed(d[0])):
        out += x*pow(2,index)
    print(str(out))