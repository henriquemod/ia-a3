import numpy as np
import pandas as pd

df = pd.read_csv('./data/census.csv', names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'], header=None)
df = df.replace(' ?', np.nan)


# Occupation - Como a maioria é Prof-specialty, substituímos os valores nulos com ela
df['occupation'] = df['occupation'].fillna('Prof-specialty')
# Workclass - Como a maioria é Private, substituímos os valores nulos com ele
df['workclass'] = df['workclass'].fillna('Private')
# Nativa Country - Como a maioria é United-States, substituímos os valores nulos com ele
df['native-country'] = df['native-country'].fillna('United-States')

y_non_treated = df['income']
X_non_treated = df.drop('income', axis=1).values

dataset = df.copy()

dataset.drop(['fnlwgt'], axis=1, inplace=True)


# NOTE - Distribuindo a coluna de idade em 3 partes significativas e plotando ela correspondente ao atributo de saída(income)
dataset['age'] = pd.cut(dataset['age'], bins=[0, 25, 50, 100], labels=[
                        'Jovem', 'Adulto', 'Idoso'])

# NOTE - Ganho e perda de capital podem ser combinados e transformados em um atributo de diferença de capital.
# Plotando o novo atributo correspondente ao atributo de saída(income)
dataset['Capital Diff'] = dataset['capital-gain'] - dataset['capital-loss']
dataset.drop(['capital-gain'], axis=1, inplace=True)
dataset.drop(['capital-loss'], axis=1, inplace=True)

dataset['Capital Diff'] = pd.cut(
    dataset['Capital Diff'], bins=[-5000, 5000, 100000], labels=['Menor', 'Maior'])

# NOTE -  Dividindo as horas de trabalho em 3 grandes intervalos e plotando-as correspondentes ao atributo de saída(income)
dataset['Horas por semana'] = pd.cut(dataset['hours-per-week'],
                                     bins=[0, 30, 40, 100],
                                     labels=['Poucas horas', 'Horas normais', 'Muitas horas'])

dataset.drop('hours-per-week', axis=1, inplace=True)

# NOTE - Combinando as escolas de ensino mais baixas
dataset.drop(['education-num'], axis=1, inplace=True)
dataset['education'].replace([' 11th', ' 9th', ' 7th-8th', ' 5th-6th', ' 10th', ' 1st-4th', ' Preschool', ' 12th'],
                             ' School', inplace=True)

# NOTE - Como a maioria da raça é branca, o restante pode ser combinado junto formando um novo grupo
dataset['race'].replace([' Black', ' Asian-Pac-Islander',
                         ' Amer-Indian-Eskimo', ' Other'], ' Other', inplace=True)

count = dataset['native-country'].value_counts()

# NOTE -  Como a maioria dos países é americano, combinamos os outros países em um único grupo
countries = np.array(dataset['native-country'].unique())
countries = np.delete(countries, 0)
dataset['native-country'].replace(countries, 'Other', inplace=True)

y_treated = dataset['income']
X_treated = dataset.drop('income', axis=1).values
