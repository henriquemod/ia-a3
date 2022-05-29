import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from DataAnalises import X,y
from Data import indexConvert

X = indexConvert(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y)

neighbors = np.arange(1, 15)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    # Confifurando um classificador knn com k vizinhos
    knn = KNeighborsClassifier(n_neighbors=k)

    # Utilizando o classificador para treinar os dados
    knn.fit(X_train, y_train)

    # Calculando a acurácia do classificador em treino
    train_accuracy[i] = knn.score(X_train, y_train)

    # Calculando a acurácia do classificador em teste
    test_accuracy[i] = knn.score(X_test, y_test)

# Gerar gráfico
plt.title('k-NN Variação de numeros de vizinhos')
plt.plot(neighbors, test_accuracy, label='Testando acurácia')
plt.plot(neighbors, train_accuracy, label='Treinando acurácia')
plt.legend()
plt.xlabel('Numeros de vizinhos')
plt.ylabel('Acurácia')
# plt.show()

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

formated = "{:.2f}".format(knn.score(X_test, y_test) * 100)
accuracy = formated.__str__()

''' 
Testando o KNN com dados de uma pessoa
Idade: 35
Flng-Wgt: 106967
Workclass: Private
Education: Bachelors
Education-Num: 11
Marital Status: Married-civ-spouse
Occupation: Prof-specialty
Relationship: Husband
Race: White
Sex: Male
Captail-gain: 0
Capitasl-los: 0
Hours-per-week: 40
Native-country: United-States
'''
example = [35, 106967, 0, 0, 11, 0, 5, 2, 0, 1, 0, 0, 40, 0]
person_data = knn.predict([example])

print('Acuracia: ' + accuracy + '%')
print('Predição de income:' + person_data[0].__str__() + ' dolares /ano')
