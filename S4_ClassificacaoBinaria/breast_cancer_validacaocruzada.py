# Importações
import pandas as pd
import numpy as np
import math
import seaborn as sea
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

#Importando o scikit_learn a partir do keras
#Esse wrapper acessa o scikit_learn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#  Importação da base de dados
previsores = pd.read_csv('entradas.csv')
classe = pd.read_csv('saidas.csv')

# Criação de uma função da rede neural:
def criarRede():
    classificador = Sequential()
    
    #1ª Camada 
    classificador.add(Dense(units = round((previsores.shape[1]+2)/2),
                        activation='relu', 
                        kernel_initializer='random_uniform',
                        input_dim=previsores.shape[1]))
    #Camada de dropout
    classificador.add(Dropout(0.2))
    #2ª camada
    classificador.add(Dense(units = round((previsores.shape[1]+2)/2),
                        activation='relu', 
                        kernel_initializer='random_uniform'))
    #Camada de dropout
    classificador.add(Dropout(0.2))
    #Camada de saída
    classificador.add(Dense(units=1, 
                  activation='sigmoid'))
    
    #Especificar mais detalhadamente o otimizador
    #Parece que antes havia até mesmo uma taxa de aprendizado que pode variar
    otimizador = keras.optimizers.Adam(learning_rate=0.0001,
                                   decay = 0.001,
                                   clipvalue=0.5)
    
    #Compilção da rede neural:
    classificador.compile(optimizer=otimizador,
                      loss = 'binary_crossentropy',
                      metrics= ['binary_accuracy'])
    return classificador

# Classificador do Keras
classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 100,
                                batch_size = 10)

#resultados 
resultados = cross_val_score(estimator = classificador, X = previsores,y= classe, cv=10,scoring = 'accuracy')

media = resultados.mean()

#Quanto maior este valor, maior o overfitting
desviopadrao = resultados.std()
                             
                             