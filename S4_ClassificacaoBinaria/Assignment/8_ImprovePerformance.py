#Base de dados de câncer de mama
import pandas as pd
import numpy as np
import math

# Rede neural
from tensorflow import keras
#Arquitetura da rede neural
from keras.models import Sequential
#Rede neural fully connected
from keras.layers import Dense
#Dropout:
from keras.layers import Dropout

#Carregar os dados
df_entradas = pd.read_csv('entradas.csv')
df_saidas = pd.read_csv('saidas.csv')

X = df_entradas.values
y = df_saidas.values

# %%Valida��o cruzada
#O wrapper do keras � uma fun��o da rotina keras respons�vel por chamar uma segunda subrotina, ou seja, o scikit learn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#%% Cria��o da rede dentro de uma fun��o (a fun��o retornar� o classificador)
def criar_rede():

    #Network creation
    classificador = Sequential()
    
    #Adicao da primeira camada oculta do tipo Dense, ou seja, fully connected
    classificador.add(Dense(units = 16,
                        activation='relu', #Geralmente bom desempenho para Deep Learning 
                        kernel_initializer='random_uniform', #Inicilializacao dos pesos
                        input_dim=30,
                        use_bias = True))
    
    #Camada de dropout introduzida com o prop�sito de reduzir o overfitting
    #Possui portanto efeito de regulariza��o, assim como ocorre com Ridge e Lasso Regression
    #Introdu��o de um bias para reduzir a variancia 
    classificador.add(Dropout(rate = 0.2)) #20 % dos neur�nios ir�o zerar
    
    #Adi��o de mais uma camada oculta:
    classificador.add(Dense(units = 8,
                        activation='relu', #Geralmente bom desempenho para Deep Learning 
                        kernel_initializer='random_uniform',
                        use_bias = True)), #Inicilializacao dos pesos
    
    #Adi��o da camada de dropout para zerar neur�nios da camada anteriormente definida? 
    #Geralmente eh importante se preocupar em reduzir o overfitting de camadas maiores, 
    #Como essas duas camadas bem grandes
    classificador.add(Dropout(rate = 0.2))
    
    #Output layer
    # Para classifica��o bin�ria, utilizar sigmoid � adequado porque retorna uma 
    #probabilidade 
    classificador.add(Dense(units=1, 
                  activation='sigmoid',
                  use_bias = True))    
    
    #Compilacao da rede neural:
    #Defini��o do otimizador separadamente:
    opt = keras.optimizers.Adam(learning_rate = 0.001, 
                            decay = 0.001, # Taxa de decaimento da taxa de aprendizado
                            clipvalue = 1) # Truncar os pesos entre 0.5 e -0.5, no intuito de evitar a explos�o destes valores 
    #Classifica��o bin�ria posso utilizar crossnetropy
    classificador.compile(optimizer=opt,
                      loss = 'binary_crossentropy',
                      metrics= ['binary_accuracy']) #Sempre no formato de lista ou dicion�rio
    
    #Visualizar a arquitetura total da rede mais seus hiperpar�metros definidos para o treinamento
    classificador.summary()
    
    return classificador


# %% Local para executar a rede
# Importante coment�rio de Batch_size :
# The strategy should be first to maximize batch_size as large as the memory permits, especially 
# when you are using GPU (4~11GB). Normally batch_size=32 or 64 should be fine, but in some cases,
#  you'd have to reduce to 8, 4, or even 1. The training code will throw exceptions if there is not
#  enough memory to allocate, so you     know when to stop increasing the batch_size.

# https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network#

#Chamar a rede neural criada atrav�s da fun��o criar_rede() dentro do KerasClassifier
classificador = KerasClassifier(build_fn = criar_rede,
                                epochs = 100,
                                batch_size = 8)

# %%Padronizacao dos dados
from sklearn.preprocessing import StandardScaler

scale_x = StandardScaler()

X = scale_x.fit_transform(X)

# %%Fit do modelo
resultados = cross_val_score(estimator = classificador,
                             X = X,
                             y = y, 
                             cv=5,
                             scoring = 'accuracy')

resultado = resultados.mean()
std = resultados.std()

