#Base de dados de cÃ¢ncer de mama
import pandas as pd
import numpy as np
import math

#Carregar os dados
df_entradas = pd.read_csv('entradas.csv')
df_entradas.info()

df_saidas = pd.read_csv('saidas.csv')


# %% Divisão entre base de dados de treinamento e de teste 
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
#%% Toda arquitetura e hiperparâmetros da rede
#Network creation
classificador = Sequential()

#Adicao da primeira camada oculta do tipo Dense, ou seja, fully connected
classificador.add(Dense(units = round(X_train.shape[1]/2 + 2),
                        activation='relu', #Geralmente bom desempenho para Deep Learning 
                        kernel_initializer='random_uniform', #Inicilializacao dos pesos
                        input_dim=X_train.shape[1],
                        use_bias = True))

#Adição de mais uma camada oculta:
classificador.add(Dense(units = round(X_train.shape[1]/2 + 2),
                        activation='relu', #Geralmente bom desempenho para Deep Learning 
                        kernel_initializer='random_uniform',
                        use_bias = True)), #Inicilializacao dos pesos

#Output layer
# Para classificação binária, utilizar sigmoid é adequado porque retorna uma 
#probabilidade 
classificador.add(Dense(units=1, 
                  activation='sigmoid',
                  use_bias = True))    

#Compilacao da rede neural:
#Definição do otimizador separadamente:
opt = keras.optimizers.Adam(learning_rate = 0.001, 
                            decay = 0.001, # Taxa de decaimento da taxa de aprendizado
                            clipvalue = 2) # Truncar os pesos entre 0.5 e -0.5, no intuito de evitar a explosão destes valores 
#Classificação binária posso utilizar crossnetropy
classificador.compile(optimizer=opt,
                      loss = 'binary_crossentropy',
                      metrics= ['binary_accuracy']) #Sempre no formato de lista ou dicionário

#Visualizar a arquitetura total da rede mais seus hiperparâmetros definidos para o treinamento
classificador.summary()

# %% Importante comentário de Batch_size :
# The strategy should be first to maximize batch_size as large as the memory permits, especially 
# when you are using GPU (4~11GB). Normally batch_size=32 or 64 should be fine, but in some cases,
#  you'd have to reduce to 8, 4, or even 1. The training code will throw exceptions if there is not
#  enough memory to allocate, so you know when to stop increasing the batch_size.

# https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network#

#Treinamento da rede
classificador.fit(X_train, 
                  y_train,
                  batch_size = 8, #Utilizar sempre uma base de 2
                  epochs=100,
                  verbose = 2, #Mostrar tudo para cada linha de treinamento na época
                  validation_split = 1) #Dividir entre validação e treinamento durante o treinamento

# %%Retornar os pesos da rede treinada
#Primeira camada
layer_1_weights = classificador.layers[0].get_weights()
layer_1_weights_entrada = classificador.layers[0].get_weights()[0]
layer_1_bias = classificador.layers[0].get_weights()[1]

#Segunda camada
layer_2_weights = classificador.layers[1].get_weights()
layer_2_weights_layer_1 = classificador.layers[1].get_weights()[0]
layer_2_bias = classificador.layers[1].get_weights()[1]

#Terceira camada
layer_3_weights = classificador.layers[2].get_weights()
layer_3_weights_layer_2 = classificador.layers[2].get_weights()[0]
layer_3_bias = classificador.layers[2].get_weights()[1]
#%% Previsões da rede
#PrevisÃµes:
previsoes = classificador.predict(X_test)

# Se definirmos o threshold em 0.1:    
previsoes_bit = np.array(list(map(lambda x: 0 if x<0.1 else 1, previsoes)))

# count number of Trues (Contar quantos são as previsões iguais)
list(previsoes_bit == np.ndarray.ravel(y_test.values)).count(True)

#Divertido para aprender a utilizar map, mas há uma maneira mais fácil:
previsoes_bit = (previsoes > 0.1)
# %% Métricas comuns, acurácia e matriz de confusão
from sklearn.metrics import accuracy_score, confusion_matrix

acuracia = accuracy_score(y_test, previsoes_bit)
matriz_confusao = confusion_matrix(y_test,previsoes_bit)

# %% Relatório de métricas do sklearn
from sklearn.metrics import classification_report

report_metrics = classification_report(y_test, previsoes_bit)
print(report_metrics)


# %%Avaliação do modelo com evaluate: (Função evaluate do próprio keras)
#Retorna o valor da função de perda e o valor da acurácia
resultado  = classificador.evaluate(X_test, y_test)
