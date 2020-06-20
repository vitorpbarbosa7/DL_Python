#Base de dados de câncer de mama
import pandas as pd
import numpy as np
import math

#Carregar os dados
df_entradas = pd.read_csv('entradas.csv')
df_entradas.info()

df_saidas = pd.read_csv('saidas.csv')

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(df_entradas, 
                                                                                              df_saidas,
                                                                                              test_size = 0.25)

import keras

#Arquitetura da rede neural
from keras.models import Sequential

#Rede neural fully connected
from keras.layers import Dense

#Criação da rede:
round(previsores_treinamento.shape[1] + 2)

classificador = Sequential()
classificador.add(Dense(units = round(previsores_treinamento.shape[1]+2),
                        activation='relu', 
                        kernel_initializer='random_uniform',
                        input_dim=previsores_treinamento.shape[1]))

#Camada de saída
classificador.add(Dense(units=1, 
                  activation='sigmoid'))

#Compilção da rede neural:
classificador.compile(optimizer='adam',
                      loss = 'binary_crossentropy',
                      metrics= ['binary_accuracy'])

#Fit a rede
classificador.fit(previsores_treinamento, 
                  classe_treinamento,
                  batch_size = 10,
                  epochs=100)

#Previsões:
round(previsoes)

previsoes = np.around(classificador.predict(previsores_teste))

resultado = classificador.evaluate(previsores_teste, classe_teste)

from sklearn.metrics import accuracy_score, confusion_matrix

acuracia = accuracy_score(classe_teste, previsoes)
matriz_confusao = confusion_matrix(classe_teste,previsoes)
