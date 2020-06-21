# Importacao dos pacotes
import pandas as pd

#%% Leitura dos dados 
data = pd.read_csv('iris.csv')

#%% Divisao X e y
X = data.drop('class', axis  = 1).values
y = data['class']

# %% Divisao treinamento e teste
from sklearn.model_selection import train_test_split

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.25)
# %% Estrutura da rede neural
