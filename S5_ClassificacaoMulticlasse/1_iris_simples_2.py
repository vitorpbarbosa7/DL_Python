# Importacao dos pacotes
import pandas as pd

#Rede neural
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense

#%% Leitura dos dados 
data = pd.read_csv('iris.csv')

#%% Divisao X e y
X = data.drop('class', axis  = 1).values
y = data['class'].values

# %%Label enconder para a classe
from sklearn.preprocessing import LabelEncoder
label_y = LabelEncoder()
y = label_y.fit_transform(y)

# %% Divisao treinamento e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
# %% Estrutura da rede neural

#Rede

def criar_rede(neurons, activation, kernel_initializer):
    classificador = Sequential()
    
    classificador.add(Dense(units = neurons,
                            activation = activation,
                            kernel_initializer = kernel_initializer,
                            input_dim = 4))
    
    classificador.add(Dense(units = neurons,
                            activation = activation,
                            kernel_initializer = kernel_initializer))
        
    classificador.add(Dense(units = 3, #Numero de possiveis saidas
                            activation = 'softmax')) #Gera uma probabilidade para cada um dos rotulos, para cada neuronio
    
    #Compilacao
    
    opt = keras.optimizers.Adam(lr = 0.001, decay = 0.001, clipvalue = 1)
    
    classificador.compile(optimizer = opt, 
                          loss = 'categorical_crossentropy', 
                          metrics = ['categorical_accuracy'])
    
    #Querido, nao estava dando certo antes porque voce nao estava retornando o classificador
    return classificador
# %%        
from keras.wrappers.scikit_learn import KerasClassifier

rede = criar_rede(4, 'relu', 'random_uniform')

classificador = KerasClassifier(build_fn = rede, 
                                epochs = 100, 
                                batch_size = 8)
# %%
classificador.fit(x = X_train, 
                  y = y_train)


# %%






