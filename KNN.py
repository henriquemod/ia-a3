import numpy as np
import matplotlib.pyplot as knn_plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from DataAnalises import X_non_treated, X_treated, y_non_treated, y_treated
from Data import indexConvert
from TreatedData import indexConvert as indexConvert_treated


def formatAccuracy(knn, X_t, y_t):
    return "{:.2f}".format(knn.score(X_t, y_t) * 100)


X_non_treated = indexConvert(X_non_treated)
X_treated = indexConvert_treated(X_treated)

# NOTE - Treinando o modelo sem tratamento
X_train_non_treated, X_test_non_treated, y_train_non_treated, y_test_non_treated = train_test_split(
    X_non_treated, y_non_treated, test_size=0.4, random_state=42, stratify=y_non_treated)

# NOTE - Treinando o modelo com tratamento
X_train_treated, X_test_treated, y_train_treated, y_test_treated = train_test_split(
    X_treated, y_treated, test_size=0.4, random_state=42, stratify=y_treated)

neighbors = np.arange(1, 15)
train_accuracy_non_treated = np.empty(len(neighbors))
test_accuracy_non_treated = np.empty(len(neighbors))
train_accuracy_treated = np.empty(len(neighbors))
test_accuracy_treated = np.empty(len(neighbors))

# for i, k in enumerate(neighbors):
#     # Confifurando um classificador knn com k vizinhos
#     knn_non_treated = KNeighborsClassifier(n_neighbors=k)
#     knn_treated = KNeighborsClassifier(n_neighbors=k)

#     # Utilizando o classificador para treinar os dados
#     knn_non_treated.fit(X_train_non_treated, y_train_non_treated)
#     knn_treated.fit(X_train_treated, y_train_treated)

#     # Calculando a acurácia do classificador em treino
#     train_accuracy_non_treated[i] = knn_non_treated.score(
#         X_train_non_treated, y_train_non_treated)
#     train_accuracy_treated[i] = knn_treated.score(
#         X_train_treated, y_train_treated)

#     # Calculando a acurácia do classificador em teste
#     test_accuracy_non_treated[i] = knn_non_treated.score(
#         X_test_non_treated, y_test_non_treated)
#     test_accuracy_treated[i] = knn_treated.score(
#         X_test_treated, y_test_treated)

# # Gerar gráfico
# knn_plt.title('k-NN Variação de numeros de vizinhos')
# knn_plt.plot(neighbors, test_accuracy_non_treated,
#              label='Testando acurácia sem tratamento')
# knn_plt.plot(neighbors, train_accuracy_non_treated,
#              label='Treinando acurácia sem tratamento')
# knn_plt.legend()
# knn_plt.xlabel('Numeros de vizinhos')
# knn_plt.ylabel('Acurácia')
# # knn_plt.show()

# # Gerar gráfico
# knn_plt.title('k-NN Variação de numeros de vizinhos')
# knn_plt.plot(neighbors, test_accuracy_treated,
#              label='Testando acurácia com tratamento')
# knn_plt.plot(neighbors, train_accuracy_treated,
#              label='Treinando acurácia com tratamento')
# knn_plt.legend()
# knn_plt.xlabel('Numeros de vizinhos')
# knn_plt.ylabel('Acurácia')
# knn_plt.show()

knn_non_treated = KNeighborsClassifier(n_neighbors=10)
knn_non_treated.fit(X_train_non_treated, y_train_non_treated)

knn_treated = KNeighborsClassifier(n_neighbors=8)
knn_treated.fit(X_train_treated, y_train_treated)

# formated = "{:.2f}".format(knn.score(X_test, y_test) * 100)
formated_non_treated = formatAccuracy(knn_non_treated, X_test_non_treated, y_test_non_treated)
formated_treated = formatAccuracy(knn_treated, X_test_treated, y_test_treated)
accuracy_non_treated = formated_non_treated.__str__()
accuracy_treated = formated_treated.__str__()

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
example_non_treated = [35, 106967, 0, 0, 11, 0, 5, 2, 0, 1, 0, 0, 40, 0]
example_treated = [35, 0, 0, 0, 5, 2, 0, 1, 0, 0, 40, 0]
person_data_non_treated = knn_non_treated.predict([example_non_treated])
person_data_treated = knn_treated.predict([example_treated])

print('Acuracia sem tratamento: ' + accuracy_non_treated + '%')
print('Acuracia com tratamento: ' + accuracy_treated + '%')
print('Predição de income sem tratamento:' + person_data_non_treated[0].__str__() + ' dolares /ano')
print('Predição de income com tratamento:' + person_data_treated[0].__str__() + ' dolares /ano')
