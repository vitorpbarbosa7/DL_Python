import numpy as np

from keras.models import model_from_json

#Abrir o arquivo da rede neural salva
arquivo = open('classificador_breast_cancer.json', 'r')

#Estrutura da rede:
estrutura_rede = arquivo.read()

#Definindo o classificador a partir da rede neural salva:
classificador = model_from_json(estrutura_rede)

#Carregar os pesos da rede neural:
classificador.load_weights('classificador_breaast.h5')

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

previsao = classificador.predict(novo)

#Para avaliar se é verdadeiro ou não
previsao = (previsao >0.5)

#a partir dos dados online

#Avaliação da rede pronta com dados
previsores = pd.read_csv('https://raw.githubusercontent.com/vitorpbarbosa7/Udemy-Redes-Neurais/master/entradas.csv')
classe = pd.read_csv('https://raw.githubusercontent.com/vitorpbarbosa7/Udemy-Redes-Neurais/master/saidas.csv')

classificador.compile(loss = 'binary_crossentropy',
                      optimizers='adam',
                      metrics = ['binary_accuracy'])

resultado = classificador.evaluate(previsores, classe)