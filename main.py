import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

csv = pd.read_csv('planos.csv', sep=',')
csv = csv.drop(columns=['nome'])
dados = csv.values

#Ajusta
le = LabelEncoder()
dados[:,1] = le.fit_transform(dados[:,1])

#Separa
atributos = dados[:,0:3] #Estado Civil (0-casado|1-solteiro|2-viuvo)
dados[:,2] = le.fit_transform(dados[:,2]) #Gênero (0-feminino|1-masculino)
dados[:,3] = le.fit_transform(dados[:,3]) #Risco (0-alto|1-baixo|2-medio)
classificadores = dados[:,3]
classificadores = np_utils.to_categorical(classificadores)

#Ajusta os atributos para classificações binários
ct = ColumnTransformer([('binarios', OneHotEncoder(), [2])], remainder='passthrough')
atributos = ct.fit_transform(atributos)

#Criando o modelo
modelo = Sequential()
modelo.add(Dense(units=5, activation='relu'))
modelo.add(Dense(units=5, activation='relu'))
modelo.add(Dense(units=3, activation='softmax')) #A soma de todos não ultrapassa 1

modelo.compile(optimizer='adam',  loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
modelo.fit(atributos, classificadores, batch_size=50, epochs=500)

#Identificando a classificação de um novo usuário
#Estado Civil (0-casado|1-solteiro|2-viuvo)
#Gênero (0-feminino|1-masculino)
novos = np.array([
    [80, 2, 1],
    [27, 0, 0],
    [35, 1, 1]
])
novos = ct.transform(novos)

resultado = modelo.predict(novos)

#Ordem Alfabética - Risco (0-alto|1-baixo|2-medio)
print(resultado)
