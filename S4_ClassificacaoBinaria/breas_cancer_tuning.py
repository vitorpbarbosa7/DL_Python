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

classificador = Sequential()

#1ª Camada 
classificador.add(Dense(units = 8,
                    activation='relu', 
                    kernel_initializer='normal',
                    input_dim=30))
#Camada de dropout
classificador.add(Dropout(0.2))
#2ª camada
classificador.add(Dense(units = 8,
                    activation='relu', 
                    kernel_initializer='normal'))
#Camada de dropout
classificador.add(Dropout(0.2))
#Camada de saída
classificador.add(Dense(units=1, 
              activation='sigmoid'))

#Compilção da rede neural:
classificador.compile(optimizer='adam',
                  loss = 'binary_crossentropy',
                  metrics= ['binary_accuracy'])


#Classificador
classificador.fit(previsores, 
                  classe, 
                  batch_size = 10, 
                  epochs = 100)

# Classificação de novos dados:
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 45000, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 20.00, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

previsaao = classificador.predict(novo)
