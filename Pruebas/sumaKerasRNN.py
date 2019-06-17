import pandas as pd
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping

# Implementacion de Red Neuronal (Secuencial) con Keras


chars = '0123456789-, ' # Caracteres que se pueden encontrar en un input
TRAINING_SIZE = 50000 # Tamaño del entrenamiento
REVERSE = False # Si True entonces cuando genera una cadena tambien genera la inversa
DIGITS = 5 # Cantidad de digitos que pueden tener los numeros en el input
# Las inputs son de la forma "numero,numero" y pueden tener simbolos de negativo delante
# MAXLEN establece la longitud maxima que puede tener una input
MAXLEN = 1 + DIGITS + 1 + 1 + DIGITS # Ej: -1234,-5678

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


# Clase para el one hot encoding
class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars)) #0123456789+-,
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One-hot encode given string C.

        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.

        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)



ctable = CharacterTable(chars)

#read data file
df = pd.read_csv('data/SolucionSuma.csv', header=None)
#df.rename(columns={0: 'idSolucion', 1: 'idProblema', 2: 'parametrosEntrada', 3: 'salida'}, inplace=True)
#df.to_csv('data/SolucionSuma.csv', index=False) # save to new csv file

#check data has been read in properly
df.head()

#create a dataframe with all training data except the target column
inputs = df.drop(columns=[0,1,3]).get_values().astype(str)
outputs = df[[3]].get_values().astype(str)

inputs = inputs.reshape(1,102).tolist()[0] # Perdon por esta monstruosidad
outputs = outputs.reshape(1,102).tolist()[0] # Perdon otra vez

seen = set()
print('Generating more data...')
while len(inputs) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                    for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    signo_1,signo_2 = '',''
    if np.random.random() > 0.5:
        signo_1 = '-'
    if np.random.random() > 0.5:
        signo_2 = '-'

    a = signo_1 + str(a)
    b = signo_2 + str(b)
    # Skip any addition questions we've already seen
    # Also skip any such that x+Y == Y+x (hence the sorting).
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    # Pad the data with spaces such that it is always MAXLEN.
    q = '{},{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(int(a) + int(b))
    # Answers can be of maximum size DIGITS + 1.
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
        # space used for padding.)
        query = query[::-1]
    inputs.append(query)
    outputs.append(ans)
print('Total addition questions:', len(inputs))
print('Total addition answers:', len(outputs))

print('Vectorization...')
x = np.zeros((len(inputs), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(outputs), DIGITS + 2, len(chars)), dtype=np.bool)
for i, sentence in enumerate(inputs):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(outputs):
    y[i] = ctable.encode(sentence, DIGITS + 2)

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]



RNN = layers.LSTM
LAYERS = 2 # Cantidad de layers internas
HIDDEN_SIZE = 128 # Todavia no se que es esto
BATCH_SIZE = 64 # The batch size is a number of samples processed before the model is updated.




print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
# As the decoder RNN's input, repeatedly provide with the last output of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(layers.RepeatVector(DIGITS + 2))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy']
model.summary()



# Entrenamiento
for iteration in range(1, 350):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)

def predecir():
    # Solicito al usuario dos numeros para comprobar las predicciones de la red
    user_input_one = str(input("Entrada uno (p1): "))
    user_input_two = str(input("Entrada dos (p2): "))
    query = '{},{}'.format(user_input_one,user_input_two)
    query = ctable.encode(query, MAXLEN)
    query = np.expand_dims(query, axis=0)
    respuesta = model.predict(query)
    respuesta = ctable.decode(respuesta[0])
    return respuesta

model.save('models/sumaKerasRNN_1.h5')
model.save_weights('models/pesos_sumaKerasRNN_1.h5')
respuesta = predecir()
print(respuesta)
