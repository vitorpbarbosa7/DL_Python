#Base de dados de câncer de mama
import pandas as pd
import numpy as np
import math

#Keras from tensorflow
import keras 
# Rede neural
from tensorflow import keras
#Arquitetura da rede neural
from keras.models import Sequential
#Rede neural fully connected
from keras.layers import Dense
#Import dropout layer from Keras
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

# %%GridSearch para tuning dos par�metros
# Pesquisa em grade para encontrar os melhores par�metros
from sklearn.model_selection import GridSearchCV 

#%% Cria��o da rede dentro de uma fun��o (a fun��o retornar� o classificador)
# Anteriormente ao GridSearchCV, havi�mos criado a rede neural de forma est�tica. 
# Agora vamos passar como hyperpar�metros:
#optimizer, 
def criar_rede(optimizer, loss, kernel_initializer, activation, neurons):

    #Network creation
    classificador = Sequential()
    
    #Adicao da primeira camada oculta do tipo Dense, ou seja, fully connected
    classificador.add(Dense(units = neurons,
                        activation=activation, #Geralmente bom desempenho para Deep Learning 
                        kernel_initializer=kernel_initializer, #Inicilializacao dos pesos
                        input_dim=30,
                        use_bias = True))
    
    #Camada de dropout introduzida com o prop�sito de reduzir o overfitting
    #Possui portanto efeito de regulariza��o, assim como ocorre com Ridge e Lasso Regression
    #Introdu��o de um bias para reduzir a variancia 
    classificador.add(Dropout(rate = 0.2)) #20 % dos neur�nios ir�o zerar
    
    #Adi��o de mais uma camada oculta:
    classificador.add(Dense(units = neurons,
                        activation=activation, #Geralmente bom desempenho para Deep Learning 
                        kernel_initializer=kernel_initializer,
                        use_bias = True)), #Inicilializacao dos pesos
    
    #Adi��o da camada de dropout para zerar neur�nios da camada anteriormente definida? 
    #Geralmente eh importante se preocupar em reduzir o overfitting de camadas maiores, 
    #Como essas duas camadas bem grandes
    classificador.add(Dropout(rate = 0.2))
    
    #Output layer
    # Para classifica��o bin�ria, utilizar sigmoid � adequado porque retorna uma probabilidade 
    classificador.add(Dense(units=1, 
                  activation='sigmoid',
                  use_bias = True))    
    
    #Compilacao da rede neural:
    #Classifica��o bin�ria posso utilizar crossnetropy
    #Antes est�vamos utilizando otimizador fixo, e agora ser� otimizador do GridSearchCV
    classificador.compile(optimizer=optimizer,
                      loss = loss,
                      metrics= ['binary_accuracy']) #Sempre no formato de lista ou dicion�rio
    
    #Visualizar a arquitetura total da rede mais seus hiperpar�metros definidos para o treinamento
    classificador.summary()
    
    #O importante � que esta funcao retorne o classificador
    return classificador


#%% Vamos rodar um GridSearch pequeno aqui para teste, mas a m�quina n�o guenta tudo isso n�o
    
#Na utiliza��o de GridSearchCV, ao instanciar o objeto classificador a partir da classe KerasClassifier, s� preciso inserir o build_fn para criar a fun��o 
# e n�o as �pocas e batch size
classificador = KerasClassifier(build_fn=criar_rede)

# %%Hyperparameters
#Hyperpar�metros s�o inseridos no formato de dicion�rio. 
#Chaves com o nome do hyperpar�metro e valor na forma de lista

#Observa-se que eu selecionei os piores parametros s� para testar mesmo
parameters = {'batch_size': [16,32], 
              'epochs': [30],
              'optimizer': ['SGD'],
              'loss': ['poisson'],
              'kernel_initializer': ['random_normal'],
              'activation': ['tanh'],
              'neurons': [16,20]} 

#Parametros mais decentes:
parameters_2 = {'batch_size': [16,32], 
              'epochs': [50],
              'optimizer': ['Adam'],
              'loss': ['binary_crossentropy'],
              'kernel_initializer': ['random_uniform'],
              'activation': ['relu'],
              'neurons': [16,20]} 



# %% # Cria��o do GridSearchCV
# - No pr�prio GridSearchCV est� implementado a Valida��o Cruzada 

gridsearch = GridSearchCV(estimator = classificador, 
                          param_grid = parameters, 
                          scoring = 'accuracy',
                          cv = 5)

gridsearch_2 = GridSearchCV(estimator = classificador, 
                          param_grid = parameters_2, 
                          scoring = 'accuracy',
                          cv = 5)

# %% O treinamento do modelo � realizado a partir do objeto gridsearch instanciado a partir da classe GridSearchCV, logo a esta classe possui a a��o, o m�todo, a fun��o fit()
gridsearch = gridsearch.fit(X = X, 
                            y = y)

# %%Par�metros decentes:
gridsearch_2 = gridsearch_2.fit(X = X, 
                                y = y)

# %%Leitura dos melhores par�metros
best_params = gridsearch.best_params_
best_accuracy = gridsearch.best_score_  

# %%
best_params_2 = gridsearch_2.best_params_
best_accuracy_2 = gridsearch_2.best_score_