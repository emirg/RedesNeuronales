import copy, numpy as np
#np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

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
binary = unpackbits(
    np.array([range(largest_number)],dtype=np.uint16).T,16)
for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 32
output_dim = 1


# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(10000):
    
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(256/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(256/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    #if a_int>b_int:
    #    c_int=a_int
    #else:
    #    c_int=b_int
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        
        # generate input and output
        X = np.array([[(a[0])[binary_dim - position - 1],(b[0])[binary_dim - position - 1]]])
        #print(X)
        y = np.array([[(c[0])[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        # decode estimate so we can print it out
        (d[0])[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    
    future_layer_1_delta = np.zeros(hidden_dim)
    
    for position in range(binary_dim):
        
        X = np.array([[(a[0])[position],(b[0])[position]]])
        #print(X)
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        #print(X)
        #print(layer_1_delta)
        #print(synapse_0_update)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print ("Error:" + str(overallError))
        print ("Pred:" + str(d))
        print ("True:" + str(c))
        #print("weights_0 "+  str(synapse_0))
        #print("weights_1 "+  str(synapse_1))
        #print("weights_bias "+  str(synapse_h))
        out = 0
        #print((d[0])[0])
        for index,x in enumerate(reversed(d[0])):
            #print(x)
            out += x*pow(2,index)
        print (str(a_int) + " + " + str(b_int) + " = " + str(out))        
        #print ("max ( "+str(a_int) + "  " + str(b_int) + ") = " + str(out))
        print ("------------")

def think(a_int,b_int):
    a = int2binary[a_int] # binary encoding
    b = int2binary[b_int] # binary encoding
    d = np.zeros_like(a) # result

    for position in range(binary_dim):
        # generate input and output
        X = np.array([[(a[0])[binary_dim - position - 1],(b[0])[binary_dim - position - 1]]])

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        (d[0])[binary_dim - position - 1] = np.round(layer_2[0][0])

    return d

# Main 
if __name__ == "__main__":
 
    user_input_one = str(input("Entrada uno (p1): "))
    user_input_two = str(input("Entrada dos (p2): "))
    user_input_one=int(user_input_one)
    user_input_two=int(user_input_two)
    #user_input_one=int2binary[user_input_one]
    #user_input_two=int2binary[user_input_two]
    d=think(user_input_one,user_input_two)
    c_int = user_input_one + user_input_two
    c = int2binary[c_int]
    print(c)
    
    print(d)
    print("Respuesta: ")
    out = 0
    for index,x in enumerate(reversed(d[0])):
        out += x*pow(2,index)

    print (str(user_input_one) + " + " + str(user_input_two) + " = " + str(out))        
    #print ("max ( "+str(a_int) + "  " + str(b_int) + ") = " + str(out))
    print ("------------")
