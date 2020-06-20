import pandas as pd
import numpy as np

#%%Ler o modelo
from keras.models import model_from_json

arquivo = open('output/rede_json.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close

# %% Instanciar a rede a partir do objeto json carregado como string

classificador = model_from_json(estrutura_rede)

#Atribuir os pesos a rede carregada
# O objeto classificador possui a funcao, metodo, acao de carregar os pesos a partir de um objeto h5
# Arquivos h5 sao estruturadas de dados, neste caso composta por numeros
classificador.load_weights('output/weights.h5')

# %% Classificar apos ter carregado o modelo

novo = np.array([[20.00, 8.34, 118, 2000, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 50, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 500, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

predicao = classificador.predict_proba(novo)

# %% 

entradas = pd.read_csv('entradas.csv')
saidas = pd.read_csv('saidas.csv')

saidas.rename(columns = {'0':'classe'}, inplace = True)

#Essa base esta desbalanceada?
saidas.classe.value_counts() 
#nao

# %% Predicao
predict = classificador.predict_proba(entradas.values[20].reshape(1,30))


# %%Avaliar o desempenho da rede carregada nos dados( no caso sao os mesmos de treinamento)
#Mas poderiam ser outros dados, de qualquer forma seria supervisionado

#Compile novamente
classificador.compile(optimizer = 'adam', 
                      loss = 'binary_crossentropy', 
                      metrics = ['accuracy'])

score = classificador.evaluate(entradas, saidas)

#Loss value:
score[0]

#Acuracia:
score[1]