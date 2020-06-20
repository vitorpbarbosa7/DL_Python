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

#Gravar a rede neural: (estrutura da rede)
classificador_json = classificador.to_json()

with open('classificador_breast_cancer.json','w') as json_file:
    json_file.write(classificador_json)
    
#Gravar os pesos:

    