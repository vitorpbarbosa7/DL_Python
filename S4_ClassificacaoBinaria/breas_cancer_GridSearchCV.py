# Importações
import pandas as pd
import numpy as np
import math
import seaborn as sea
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

#Importação do tuning, (GridSearchCV)
from sklearn.model_selection import GridSearchCV

#  Importação da base de dados
previsores = pd.read_csv('https://raw.githubusercontent.com/vitorpbarbosa7/Udemy-Redes-Neurais/master/entradas.csv')
classe = pd.read_csv('https://raw.githubusercontent.com/vitorpbarbosa7/Udemy-Redes-Neurais/master/saidas.csv')

# Criação de uma função da rede neural:
def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    
    #1ª Camada 
    classificador.add(Dense(units = neurons,
                        activation=activation, 
                        kernel_initializer=kernel_initializer,
                        input_dim=30))
    #Camada de dropout
    classificador.add(Dropout(0.2))
    #2ª camada
    classificador.add(Dense(units = neurons,
                        activation=activation, 
                        kernel_initializer=kernel_initializer))
    #Camada de dropout
    classificador.add(Dropout(0.2))
    #Camada de saída
    classificador.add(Dense(units=1, 
                  activation='sigmoid'))
    
    #Compilção da rede neural:
    classificador.compile(optimizer=optimizer,
                      loss = loss,
                      metrics= ['binary_accuracy'])
    return classificador

# Classificador do Keras
classificador = KerasClassifier(build_fn = criarRede)  

#Parametros para a rede:
parametros = {'batch_size': [10,30],
              'epochs': [50, 100],
              'optimizer': ['adam','sgd'],
              'loss': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu','tanh'],
              'neurons': [16, 8]}

#GridSearchCV
grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)

#Fit com o GridSearch
grid_search = grid_search.fit(previsores, classe)