import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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

df = df.replace(' ?', np.nan)

df['occupation'] = df['occupation'].fillna('Prof-specialty')
df['workclass'] = df['workclass'].fillna('Private')
df['native-country'] = df['native-country'].fillna('United-States')

y = df['income']
X = df.drop('income', axis=1).values

for i, values in enumerate(X):
    X[i][1] = getIndex(workclass, values[1])  # Workclass
    X[i][3] = getIndex(education, values[3])  # Education
    X[i][5] = getIndex(maritalStatus, values[5])  # Marital Status
    X[i][6] = getIndex(occupation, values[6])  # Occupation
    X[i][7] = getIndex(relationship, values[7])  # Relationship
    X[i][8] = getIndex(race, values[8])  # Race
    X[i][9] = getIndex(sex, values[9])  # Sex
    X[i][13] = getIndex(nativeCountry, values[13])  # native country

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
