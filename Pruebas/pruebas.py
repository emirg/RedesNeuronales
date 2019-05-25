import copy, numpy as np

int2binary={}
#print(int2binary)
binary_dim=8

largest_number= pow(2,binary_dim)


#convierte los int en binarios de 16 bits

def unpackbits(x,num_bits):
  xshape = list(x.shape)
  x = x.reshape([-1,1])
  to_and_aux = 2**np.arange(num_bits).reshape([1,num_bits])
  to_and=2**np.arange(num_bits).reshape([1,num_bits])
  for i in range(len(to_and_aux[0])):
      to_and[0][i]=to_and_aux[0][num_bits-(i+1)]
  return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])


A  = (np.array([range(largest_number)],dtype=np.uint16)).T
binary =unpackbits(A, binary_dim)
print('hola')
print(((binary[0])[0])[0])
#print(A)




for i in range(largest_number):
    int2binary[i] = binary[i]

a = int2binary[15]
print('imprimo a')
print(a)
print(range(len(a[0])))
aux=[0,0,0,0,0,0,0,0]
print(aux)
for i in range(len(a[0])):
    print(a[0][8-(i+1)])
    print(i+1)
    aux[i]=a[0][8-(i+1)]

print('imprimo aux')
reversed(a)
print(aux)

b=int2binary[1]
#print(b)
for position in range(binary_dim):
    #print(binary_dim - position - 1)
    X = np.array([[(a[0])[binary_dim - position - 1],(b[0])[binary_dim - position - 1]]])

print('X')
print(X)

# Main 
if __name__ == "__main__":
    a_int = 256 # int version
    #print(int2binary[1])
