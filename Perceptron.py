from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from DataAnalises import X_treated, y_treated, X_non_treated, y_non_treated
from Data import indexConvert
from TreatedData import indexConvert as indexConvert_treated


def formatAccuracy(y_t, y_p):
    return "{:.2f}".format(accuracy_score(y_t, y_p) * 100)


# NOTE - Treinando o modelo sem tratamento
X_non_treated = indexConvert(X_non_treated)
x_train_non_treated, x_test_non_treated, y_train_non_treated, y_test_non_treated = train_test_split(
    X_non_treated, y_non_treated, test_size=0.3, random_state=47)

clf_non_treated = MLPClassifier(hidden_layer_sizes=(
    100, 100, 100), max_iter=500, alpha=0.0001, solver='sgd', verbose=10,  random_state=21, tol=0.000000001)

clf_non_treated.fit(x_train_non_treated, y_train_non_treated)
y_pred_non_treated = clf_non_treated.predict(x_test_non_treated)
accuracy_non_treated = formatAccuracy(
    y_test_non_treated, y_pred_non_treated).__str__()

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
person_data_non_treated = clf_non_treated.predict([example_non_treated])


# NOTE - Treinando o modelo com tratamento
X_treated = indexConvert_treated(X_treated)
x_train_treated, x_test_treated, y_train_treated, y_test_treated = train_test_split(
    X_treated, y_treated, test_size=0.3, random_state=21)

clf_treated = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=500,
                            alpha=0.0001, solver='sgd', verbose=10,  random_state=21, tol=0.000000001)

clf_treated.fit(x_train_treated, y_train_treated)
y_pred_treated = clf_treated.predict(x_test_treated)
accuracy_treated = formatAccuracy(y_test_treated, y_pred_treated).__str__()

# example_treated = [35, 0, 0, 0, 5, 2, 0, 1, 0, 0, 40, 0]
example_treated = [1, 0, 0, 0, 5, 2, 0, 1, 0, 0, 1]
person_data_treated = clf_treated.predict([example_treated])

# NOTE - Printando os resultados
print('Acurácia: \n' + 'Antes: ' + accuracy_non_treated +
      '%\n' + 'Depois: ' + accuracy_treated + '%')
print('Predição de income:\n' + 'Antes: ' + person_data_non_treated[0].__str__(
) + ' dólares /ano\n' + 'Depois: ' + person_data_treated[0].__str__() + ' dólares /ano')
