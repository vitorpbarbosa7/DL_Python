#Base de dados de cÃ¢ncer de mama
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

# %%Validação cruzada
#O wrapper do keras é uma função da rotina keras responsável por chamar uma segunda subrotina, ou seja, o scikit learn
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# %%GridSearch para tuning dos parâmetros
# Pesquisa em grade para encontrar os melhores parâmetros
from sklearn.model_selection import GridSearchCV 

#%% Criação da rede dentro de uma função (a função retornará o classificador)
# Anteriormente ao GridSearchCV, haviámos criado a rede neural de forma estática. 
# Agora vamos passar como hyperparâmetros:
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
    
    #Camada de dropout introduzida com o propósito de reduzir o overfitting
    #Possui portanto efeito de regularização, assim como ocorre com Ridge e Lasso Regression
    #Introdução de um bias para reduzir a variancia 
    classificador.add(Dropout(rate = 0.2)) #20 % dos neurônios irão zerar
    
    #Adição de mais uma camada oculta:
    classificador.add(Dense(units = neurons,
                        activation=activation, #Geralmente bom desempenho para Deep Learning 
                        kernel_initializer=kernel_initializer,
                        use_bias = True)), #Inicilializacao dos pesos
    
    #Adição da camada de dropout para zerar neurônios da camada anteriormente definida? 
    #Geralmente eh importante se preocupar em reduzir o overfitting de camadas maiores, 
    #Como essas duas camadas bem grandes
    classificador.add(Dropout(rate = 0.2))
    
    #Output layer
    # Para classificação binária, utilizar sigmoid é adequado porque retorna uma probabilidade 
    classificador.add(Dense(units=1, 
                  activation='sigmoid',
                  use_bias = True))    
    
    #Compilacao da rede neural:
    #Classificação binária posso utilizar crossnetropy
    #Antes estávamos utilizando otimizador fixo, e agora será otimizador do GridSearchCV
    classificador.compile(optimizer=optimizer,
                      loss = loss,
                      metrics= ['binary_accuracy']) #Sempre no formato de lista ou dicionário
    
    #Visualizar a arquitetura total da rede mais seus hiperparâmetros definidos para o treinamento
    classificador.summary()
    
    #O importante é que esta funcao retorne o classificador
    return classificador


#%% Vamos rodar um GridSearch pequeno aqui para teste, mas a máquina não guenta tudo isso não
    
#Na utilização de GridSearchCV, ao instanciar o objeto classificador a partir da classe KerasClassifier, só preciso inserir o build_fn para criar a função 
# e não as épocas e batch size
classificador = KerasClassifier(build_fn=criar_rede)

# %%Hyperparameters
#Hyperparâmetros são inseridos no formato de dicionário. 
#Chaves com o nome do hyperparâmetro e valor na forma de lista

#Observa-se que eu selecionei os piores parametros só para testar mesmo
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



# %% # Criação do GridSearchCV
# - No próprio GridSearchCV está implementado a Validação Cruzada 

gridsearch = GridSearchCV(estimator = classificador, 
                          param_grid = parameters, 
                          scoring = 'accuracy',
                          cv = 5)

gridsearch_2 = GridSearchCV(estimator = classificador, 
                          param_grid = parameters_2, 
                          scoring = 'accuracy',
                          cv = 5)

# %% O treinamento do modelo é realizado a partir do objeto gridsearch instanciado a partir da classe GridSearchCV, logo a esta classe possui a ação, o método, a função fit()
gridsearch = gridsearch.fit(X = X, 
                            y = y)

# %%Parâmetros decentes:
gridsearch_2 = gridsearch_2.fit(X = X, 
                                y = y)

# %%Leitura dos melhores parâmetros
best_params = gridsearch.best_params_
best_accuracy = gridsearch.best_score_  

# %%
best_params_2 = gridsearch_2.best_params_
best_accuracy_2 = gridsearch_2.best_score_