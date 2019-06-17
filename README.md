# TPO Redes Neuronales Hornero
En este trabajo se implementa una **Red Neuronal Artificial** con el objetivo que la misma sea capaz de resolver distintos problemas planteados en la plataforma *Hornero*, la cual es una aplicación web desarrollada por estudiantes y docentes de la Facultad de Informática de la Universidad Nacional del Comahue para gestionar torneos de programación. En cada torneo se selecciona un conjunto de ejercicios y la aplicación permite a los jugadores resolver los problemas con cualquier lenguaje de programación. 

### Caracteristicas de la RNA
* Dos entradas numéricas:
    Se considero que las entradas son dos números enteros en un rango de -32768 a 32767 los cuales se representaron en binario (16 bits complemento a dos)
* Dos capas ocultas de 256 neuronas cada una 
* Matrices de Pesos:
    * Pesos entre la Entrada y la Primer Capa: (32x256)
    * Pesos entre la Primer Capa y la Segunda Capa: (256x256)
    * Pesos entre la Segunda Capa y la Salida: (256x16)

* Una salida de 16 bits 
* Función de Activación: Sigmoid
* Tasa de aprendizaje: 10\%

## Uso

### Prerequisitos
El programa solo hace uso de los paquetes *numpy*, *csv* y *os*.
### Ejecución
`$ python TPO_NeuralNetwork.py`
### Modelos
Se proveen dos modelos ya entrenados para resolver los problemas de Suma y Max. Para utilizarlos, basta con elegir la opción `Cargar Modelo` y escribir el nombre de la carpeta que contiene los archivos necesarios.

Tambien es posible entrenar la red con otro dataset y guardar los pesos mediante la opción `Guardar Modelo`
