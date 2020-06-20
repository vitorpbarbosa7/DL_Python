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
    #Como essas duas camadas bem grandesx
    return classificador


#%% Vamos rodar um GridSearch pequeno aqui para teste, mas a m�quina n�o guenta tudo isso n�o
    
#Na utiliza��o de GridSearchCV, ao instanciar o objeto classificador a partir da classe KerasClassifier, s� preciso inserir o build_fn para criar a fun��o 
# e n�o as �pocas e batch size
classificador = KerasClassifier(build_fn=criar_rede)

# %%Hyperparameters
#Hyperpar�metros s�o inseridos no formato de dicion�rio. 
#Chaves com o nome do hyperpar�metro e valor na forma de lista

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
gridsearch_2 = GridSearchCV(estimator = classificador, 
                          param_grid = parameters_2, 
                          scoring = 'accuracy',
                          cv = 5)

# %%Par�metros decentes:
gridsearch_2 = gridsearch_2.fit(X = X, 
                                y = y)

# %%
best_params_2 = gridsearch_2.best_params_
best_accuracy_2 = gridsearch_2.best_score_

# %%Ap�s a defini��o dos melhores par�metros: (Eu peguei os melhores do resultado do Jones, porque demoraria demais para rodar)
#Se fosse para pegar do meu pr�prio modelo, poder�amos utilizar os par�metros armazenados
# no dicion�rio atributo do objeto gridsearch_2
opt = keras.optimizers.Adam(lr = 0.001, 
                            decay = 0.001,
                            clipvalue = 0.5)

rede = criar_rede(optimizer = opt,
                  loss = 'binary_crossentropy', 
                  kernel_initializer = 'random_normal', 
                  activation = 'relu', 
                  neurons = 16)

# %% Treinamento do modelo apos o tuning dos parametros
rede.fit(x = X, 
         y = y, 
         batch_size = 16,
         epochs = 100)

# %%Salvar a rede treinada:
#Aqui na realidade ele gera uma string
rede_json = rede.to_json()

#Salvar a estrutura da rede neural realmente em objeto do tipo Json
# JavaScript Object Notation, arquivo do tipo atributo valor. Um dicionario gigante com todas
# Definicoes da arquitetura da rede, exceto os seus pesos
with open('output/rede_json.json', 'w') as json_file:
    json_file.write(rede_json)
    
#Salvar pesos da rede:
rede.save_weights('output/weights.h5')

# %%

















#%%

classificador = KerasClassifier(build_fn = rede)
