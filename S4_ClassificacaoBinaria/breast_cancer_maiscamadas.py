#Base de dados de câncer de mama

# %%Importações
import pandas as pd
import numpy as np
import math
import seaborn as sea
import matplotlib.pyplot as plt
# %%Carregar os dados
df_entradas = pd.read_csv('entradas.csv')
df_entradas.info()

df_saidas = pd.read_csv('saidas.csv')

#Finalmente consegui juntar dois dataframes e renomea-los corretamente ok
df_merged = df_entradas.join(df_saidas)
df_merged.rename(columns={'0':'classe'},inplace= True)

# %%Análise exploratória
corr = df_merged.corr()

ax = plt.subplots(figsize=(10,10))
sea.heatmap(corr, annot=True)


# %%Dividir os dados
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(df_entradas, 
                                                                                              df_saidas,
                                                                                              test_size = 0.25)

# %% Inicialização da rede
import keras

#Arquitetura da rede neural
from keras.models import Sequential

#Rede neural fully connected
from keras.layers import Dense

#Criação da rede:
round(previsores_treinamento.shape[1] + 2)

classificador = Sequential()

#1ª Camada 
classificador.add(Dense(units = round((previsores_treinamento.shape[1]+2)/2),
                        activation='relu', 
                        kernel_initializer='random_uniform',
                        input_dim=previsores_treinamento.shape[1]))
#2ª camada
classificador.add(Dense(units = round((previsores_treinamento.shape[1]+2)/2),
                        activation='relu', 
                        kernel_initializer='random_uniform'))

#Camada de saída
classificador.add(Dense(units=1, 
                  activation='sigmoid'))

#Especificar mais detalhadamente o otimizador
#Parece que antes havia até mesmo uma taxa de aprendizado que pode variar
otimizador = keras.optimizers.Adam(learning_rate=0.0001,
                                   decay = 0.001,
                                   clipvalue=0.5)

#Compilção da rede neural:
classificador.compile(optimizer=otimizador,
                      loss = 'binary_crossentropy',
                      metrics= ['binary_accuracy'])

# %%Fit a rede
classificador.fit(previsores_treinamento, 
                  classe_treinamento,
                  batch_size = 10,
                  epochs=100)

# %%Get weights

pesos0 = classificador.layers[0].get_weights()  
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

pesos0_df = pd.DataFrame(pesos0[0])
pesos1_df = pd.DataFrame(pesos1[0])
pesos2_df = pd.DataFrame(pesos2[0])

ax = plt.subplots(figsize=(10,10))
sea.heatmap(pesos0_df, annot=True, cmap="YlGnBu")

ax = plt.subplots(figsize=(10,10))x
sea.heatmap(pesos1_df, annot=True, cmap="YlGnBu")

ax = plt.subplots(figsize=(10,10))
sea.heatmap(pesos2_df, annot=True, cmap="YlGnBu")

# %%Acuracia

previsoes = np.around(classificador.predict(previsores_teste))

resultado = classificador.evaluate(previsores_teste, classe_teste)

from sklearn.metrics import accuracy_score, confusion_matrix

acuracia = accuracy_score(classe_teste, previsoes)
matriz_confusao = confusion_matrix(classe_teste,previsoes)

# %%End




































