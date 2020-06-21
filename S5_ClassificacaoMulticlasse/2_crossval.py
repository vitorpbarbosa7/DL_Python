# Importacao dos pacotes
import pandas as pd
import numpy as np

#Rede neural
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout

# Dummy data keras:
from keras.utils import np_utils

#%% Leitura dos dados 
data = pd.read_csv('iris.csv')

#%% Divisao X e y
X = data.drop('class', axis  = 1).values
y = data['class'].values

# %%Label enconder para a classe
from sklearn.preprocessing import LabelEncoder
label_y = LabelEncoder()
y = label_y.fit_transform(y)
y_dummy = np_utils.to_categorical(y)
# %% Estrutura da rede neural
def criar_rede():
    classificador = Sequential()
    
    classificador.add(Dense(units = round((4 + 3)/2),
                            activation = 'relu',
                            kernel_initializer = 'random_uniform',
                            input_dim = 4))
    
    #classificador.add(Dropout(rate = 0.2))
    
    classificador.add(Dense(units = round((4 + 3)/2),
                            activation = 'relu',
                            kernel_initializer = 'random_uniform'))
    
    #classificador.add(Dropout(rate = 0.2))
    
    classificador.add(Dense(units = 3, #Numero de possiveis saidas
                            activation = 'softmax')) #Gera uma probabilidade para cada um dos rotulos, para cada neuronio
    
    #Compilacao
    
    #opt = keras.optimizers.Adam(lr = 0.01, decay = 0.001, clipvalue = 1)
    
    classificador.compile(optimizer = 'adam', 
                          loss = 'categorical_crossentropy', 
                          metrics = ['categorical_accuracy'])
    
    #Querido, nao estava dando certo antes porque voce nao estava retornando o classificador
    return classificador
# %% 
from keras.wrappers.scikit_learn import KerasClassifier

classificador = KerasClassifier(build_fn=criar_rede, 
                                epochs = 500, 
                                batch_size = 8)       
# %%Validacao cruzada
from sklearn.model_selection import cross_val_score
acuracias = cross_val_score(estimator = classificador, 
                             X = X, 
                             y = y, 
                             cv = 10, 
                             scoring ='accuracy')

acuracia_media = acuracias.mean()
acuracia_desvio = acuracias.std()


# %% Metricas de avaliacao
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#Este reshape foi necessario devido ao treinamento da rede ser realizado com input_dim  = 4
predicao = classificador.predict_proba(X_test[5].reshape(1,4))

previsoes = classificador.predict(X_test)
#Se eu quissese definir um threshold para a classe dois, poderia ser dessa maneira
Classe_Dois = (previsoes[:,2] > 0.99)
"""Tem-se que ha apenas 6 deles que a probabilidade eh maior que 2"""

resultado  = classificador.evaluate(X_test, y_test)

#Confusion matrix
y_test_label = tf.argmax(y_test, axis = 1)

matriz = confusion_matrix(y_test_label, tf.argmax(classificador.predict(X_test), axis = 1))



# %% Outra maneira de realizar a analise
previsoes = (previsoes > 0.8)

#Numero de linhas em que nao foi possivel ob    ter probabilidade maior que 0.8 (ou outro valor)
previsoes.sum() - previsoes.shape[0]

"""Em quais linhas nao ha probabilidades maiores que 0.8?"""
i=0
listalinhas =[]
for line in previsoes[:,:]:
    if line.sum() ==0:
        listalinhas.append(i)    
    i = i +1


"""Super comentario, nossa, nao lembrava que dava para fazer assim, legal"""

