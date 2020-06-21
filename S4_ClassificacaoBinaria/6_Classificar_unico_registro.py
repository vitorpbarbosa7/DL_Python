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
        
    #Adição da camada de dropout para zerar neurônios da camada anteriormente definida? 
    #Geralmente eh importante se preocupar em reduzir o overfitting de camadas maiores, 
    #Como essas duas camadas bem grandesx
    return classificador

# %%Após a definição dos melhores parâmetros: (Eu peguei os melhores do resultado do Jones Granatyr, porque demoraria demais para rodar)
#Se fosse para pegar do meu próprio modelo, poderíamos utilizar os parâmetros armazenados
# no dicionário atributo do objeto gridsearch_2
opt = keras.optimizers.Adam(lr = 0.001, 
                            decay = 0.001,
                            clipvalue = 0.5)

rede = criar_rede(optimizer = opt,
                  loss = 'binary_crossentropy', 
                  kernel_initializer = 'random_normal', 
                  activation = 'relu', 
                  neurons = 16)

classificador = KerasClassifier(build_fn=rede)

# %% Treinamento do modelo apos o tuning dos parametros
# Fit do modelo
resultados = cross_val_score(estimator = classificador,
                             X = X,
                             y = y, 
                             cv=10,
                             scoring = 'accuracy')

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
