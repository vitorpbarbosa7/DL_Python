#Base de dados de câncer de mama
import pandas as pd
import numpy as np
import math

#Carregar os dados
df_entradas = pd.read_csv('entradas.csv')
df_entradas.info()

df_saidas = pd.read_csv('saidas.csv')


# %% Divis�o entre base de dados de treinamento e de teste 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_entradas, 
                                                    df_saidas,
                                                    test_size = 0.25)
# %% Rede neural
from tensorflow import keras

#Arquitetura da rede neural
from keras.models import Sequential

#Rede neural fully connected
from keras.layers import Dense
#%% Toda arquitetura e hiperpar�metros da rede
#Network creation
classificador = Sequential()

#Adicao da primeira camada oculta do tipo Dense, ou seja, fully connected
classificador.add(Dense(units = round(X_train.shape[1]/2 + 2),
                        activation='relu', #Geralmente bom desempenho para Deep Learning 
                        kernel_initializer='random_uniform', #Inicilializacao dos pesos
                        input_dim=X_train.shape[1]))

#Output layer
# Para classifica��o bin�ria, utilizar sigmoid � adequado porque retorna uma 
#probabilidade 
classificador.add(Dense(units=1, 
                  activation='sigmoid'))    

#Compilacao da rede neural:
#Defini��o do otimizador separadamente:
opt = keras.optimizers.Adam(learning_rate = 0.001)
#Classifica��o bin�ria posso utilizar crossnetropy
classificador.compile(optimizer=opt,
                      loss = 'binary_crossentropy',
                      metrics= ['binary_accuracy']) #Sempre no formato de lista ou dicion�rio

#Visualizar a arquitetura total da rede mais seus hiperpar�metros definidos para o treinamento
classificador.summary()

# %% Importante coment�rio de Batch_size :
# The strategy should be first to maximize batch_size as large as the memory permits, especially 
# when you are using GPU (4~11GB). Normally batch_size=32 or 64 should be fine, but in some cases,
#  you'd have to reduce to 8, 4, or even 1. The training code will throw exceptions if there is not
#  enough memory to allocate, so you know when to stop increasing the batch_size.

# https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network#

#Treinamento da rede
classificador.fit(X_train, 
                  y_train,
                  batch_size = 8, #Utilizar sempre um bin�rio
                  epochs=100,
                  verbose = 2, #Mostrar tudo para cada linha de treinamento na �poca
                  validation_split = 1) #Dividir entre valida��o e treinamento durante o treinamento

#%% Previs�es da rede
#Previsões:
previsoes = classificador.predict(X_test)

# Se definirmos o threshold em 0.1:    
previsoes_bit = np.array(list(map(lambda x: 0 if x<0.1 else 1, previsoes)))

#Divertido para aprender a utilizar map h� uma maneira mais f�cil:
previsoes_bit = (previsoes > 0.1)

# count number of Trues (Contar quantos s�o as previs�es iguais)
list(previsoes_bit == np.ndarray.ravel(y_test.values)).count(True)

# %% M�tricas comuns, acur�cia e matriz de confus�o
from sklearn.metrics import accuracy_score, confusion_matrix

acuracia = accuracy_score(y_test, previsoes_bit)
matriz_confusao = confusion_matrix(y_test,previsoes_bit)

# %% Relat�rio de m�tricas do sklearn
from sklearn.metrics import classification_report

report_metrics = classification_report(y_test, previsoes_bit)


# %%Avalia��o do modelo com evaluate: (Fun��o evaluate do pr�prio keras)
#Retorna o valor da fun��o de perda e o valor da acur�cia
resultado  = classificador.evaluate(X_test, y_test)
