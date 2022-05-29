from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from DataAnalises import X, y
from Data import indexConvert

X = indexConvert(X)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=27)

clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=500, alpha=0.0001,
                    solver='sgd', verbose=10,  random_state=21, tol=0.000000001)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

formated = "{:.2f}".format(accuracy_score(y_test, y_pred) * 100)
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
person_data = clf.predict([example])

print('Acuracia: ' + accuracy + '%')
print('Predição de income:' + person_data[0].__str__() + ' dolares /ano')
