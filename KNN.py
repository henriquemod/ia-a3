import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
             'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc',
             '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
maritalStatus = ['Married-civ-spouse', 'Divorced', 'Never-married',
                 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
              'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
relationship = ['Wife', 'Own-child', 'Husband',
                'Not-in-family', 'Other-relative', 'Unmarried']
race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
sex = ['Female', 'Male']
nativeCountry = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico',
                 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']


def getIndex(array, value):
    for i, name in enumerate(array):
        if name == value.strip():
            return i
    return -1


df = pd.read_csv('./census.csv', names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'], header=None)

X = df.drop('income', axis=1).values
y = df['income'].values

for i, values in enumerate(X):
    X[i][1] = getIndex(workclass, values[1])  # Workclass
    X[i][3] = getIndex(education, values[3])  # Education
    X[i][5] = getIndex(maritalStatus, values[5])  # Marital Status
    X[i][6] = getIndex(occupation, values[6])  # Occupation
    X[i][7] = getIndex(relationship, values[7])  # Relationship
    X[i][8] = getIndex(race, values[8])  # Race
    X[i][9] = getIndex(sex, values[9])  # Sex
    X[i][13] = getIndex(nativeCountry, values[13])  # native country

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
print(knn.score(X_test, y_test))


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

print(knn.predict([[35, 106967, 0, 0, 11, 0, 5, 2, 0, 1, 0, 0, 40, 0]]))


'''
age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
'''
