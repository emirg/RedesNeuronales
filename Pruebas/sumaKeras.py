import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Implementacion de Red Neuronal (Secuencial) con Keras

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
print(test_X.head())

#check that the target variable has been removed
#print(train_X.dtypes)

#create a dataframe with only the target column
train_y = df[['salida']].astype(int)

#view dataframe
#print(train_y.dtypes)

#create model
model = Sequential()

#get number of columns in training data
n_cols = train_X.shape[1]
print(n_cols)
#add model layers
model.add(Dense(200, activation='relu', input_shape=(n_cols,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1))

#compile model using mse as a measure of model performance
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=10)
#train model
model.fit(train_X, train_y, validation_split=0.2, epochs=4000,callbacks=[early_stopping_monitor])

#example on how to use our newly trained model on how to make predictions on unseen data (we will pretend our new data is saved in a dataframe called 'test_X').
test_y_predictions = model.predict(test_X)
print(test_y_predictions.astype(int))