import numpy as np
import math

#Funcao de ativacao degrau:
#Transfer function) 
#APenas problemas linearmente separaveis, jamais em deep learning
def stepFunction(soma):
    if (soma >=1):
        return 1
    return 0

sinapse = stepFunction(20)

#Funcao sigmoide:
def sigmoidFunction(x):
    y = 1/(1+math.exp(-x))
    return y

sinapse_sig = sigmoidFunction(-5)

#Boa para quando ha valores negativos e positivos na rede
#Logo quando est normaliada, possui valores positivos e negativos
#E a media de uma base normalizada e 0, legal, muito bom
def tanhFunction(x):
    y = (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))
    return y

sinapse_tanh = tanhFunction(-4)

#FUncao relu
#Muito utilziadas em redes neurais convolucionais e 
#em muitas camadas de redes neurais
def relu(soma):
    if soma>=0:
        return soma
    return 0

sinapse_relu = relu(-3)

#Funcao linear (identity) - adequada para regressoes
def linearFunction(soma):
    return soma

sinapse_linearFunction = linearFunction(5)

#Funcao softmax:
def softmaxFunction(vetor):
    ex = np.exp(vetor)
    return ex/ex.sum()

valores_softmax = [5.0,2.0,1.3]
print(softmaxFunction(valores_softmax))
