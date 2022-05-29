import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

df = pd.read_csv('./census.csv', names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'], header=None)
# df['income']=df['income'].map({' <=50K': 0, ' >50K': 1})
df = df.replace(' ?', np.nan)


# NOTE - Achando o percentual de dados faltando no conjunto de dados
total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
# print(missing_data.head(15))

# NOTE - Como a quantidade de dados faltando é muito pequena,
# podemos substituir os valores nulos com o moda de cada coluna
# print(df['occupation'].describe())
# print('-----------------')
# print(df['workclass'].describe())
# print('-----------------')
# print(df['native-country'].describe())


# Ocupation - Como a maioria é Prof-specialty, substituimos os valores nulos com ela
df['occupation'] = df['occupation'].fillna('Prof-specialty')
# Workclass - Como a maioria é Private, substituimos os valores nulos com ele
df['workclass'] = df['workclass'].fillna('Private')
# Nativa Country - Como a maioria é United-States, substituimos os valores nulos com ele
df['native-country'] = df['native-country'].fillna('United-States')

df.describe(include=["O"])

y_non_treated = df['income']
X_non_treated = df.drop('income', axis=1).values


# NOTE - Visualizando as caracteristicas numéricas do dataset utilizando histogramas
# para analisar a distribuição dessas caracteristicas no dataset
# SECTION
# rcParams['figure.figsize'] = 12, 12
# df[['age', 'fnlwgt', 'education-num', 'capital-gain',
#     'capital-loss', 'hours-per-week']].hist()
# SECTION

# NOTE - Plotando a correlação entre a saida(income) e as caracteristicas
# SECTION
# plt.matshow(df.corr())
# plt.colorbar()
# plt.xticks(np.arange(len(df.corr().columns)),
#            df.corr().columns.values, rotation=45)
# plt.yticks(np.arange(len(df.corr().columns)), df.corr().columns.values)
# for (i, j), corr in np.ndenumerate(df.corr()):
#     plt.text(j, i, '{:0.1f}'.format(corr), ha='center',
#              va='center', color='white', fontsize=14)
# SECTION

# NOTE - Como a correlação entre a saida(income) e fnlwgt é 0, podemos remover essa coluna
df.drop(['fnlwgt'], axis=1, inplace=True)

dataset = df.copy()

# NOTE - Distribuindo a coluna de idade em 3 partes significativas e plotando ela correspondente ao atributo de saida(income)
dataset['age'] = pd.cut(dataset['age'], bins=[0, 25, 50, 100], labels=[
                        'Young', 'Adult', 'Old'])

# sns.catplot(x='income', hue='age', data=dataset, kind='count')

# NOTE - Ganho e perda de capital podem ser combinados e transformados em um atributo de diferença de capital.
# Plotando o novo atributo correspondente ao atributo de saida(income)
dataset['Capital Diff'] = dataset['capital-gain'] - dataset['capital-loss']
dataset.drop(['capital-gain'], axis=1, inplace=True)
dataset.drop(['capital-loss'], axis=1, inplace=True)

dataset['Capital Diff'] = pd.cut(
    dataset['Capital Diff'], bins=[-5000, 5000, 100000], labels=['Minor', 'Major'])
# sns.catplot(x='income', hue='Capital Diff', data=dataset, kind='count')

# Dividindo as horas de trabalho em 3 grandes intervalos e plotando-as correspondentes ao atributo de saida(income)
dataset['Horas por semana'] = pd.cut(dataset['hours-per-week'],
                                     bins=[0, 30, 40, 100],
                                     labels=['Poucas horas', 'Horas normais', 'Muitas horas'])
# sns.catplot(x='income', hue='Horas por semana', data=dataset, kind='count')

# Plotando a distribuição de trabalho
# sns.catplot(x = 'income', hue = 'workclass', data = dataset, kind='count')

# Plot of education corresponding to income
# sns.catplot(x = 'income', hue = 'education', data = dataset, kind='count')

# Combinando as escolas de ensino mais baixas
# SECTION
df.drop(['education-num'], axis = 1, inplace = True)
df['education'].replace([' 11th', ' 9th', ' 7th-8th', ' 5th-6th', ' 10th', ' 1st-4th', ' Preschool', ' 12th'],
                             ' School', inplace = True)
# sns.catplot(x = 'income', hue = 'education', data = df, kind='count')
# SECTION

# Plotando a distribuição de trabalho
# SECTION
# plt.xticks(rotation = 45)
# sns.catplot(x = 'income', hue = 'occupation', data = dataset, kind='count')
# SECTION

# Plotando a distribuição por etnia
# sns.catplot(x='income', hue='race', data=dataset, kind='count')

# Como a maioria da raça é branca, o restante pode ser combinado junto formando um novo grupo
df['race'].unique()
df['race'].replace(['Black', 'Asian-Pac-Islander',
                   'Amer-Indian-Eskimo', 'Other'], ' Other', inplace=True)

# Plotando a distribuição por sexo
# sns.catplot(x='income', hue='sex', data=dataset, kind='count')

count = dataset['native-country'].value_counts()
# print(count)
# Plot of Country corresponding to income
# Prot de paises e seus income
# SECTION
# plt.bar(count.index, count.values)
# plt.xlabel('Countries')
# plt.ylabel('Count')
# plt.title('Count from each Country')
# SECTION

# Como a maioria dos paises é americano, combinamos os outros paises em um unico grupo
countries = np.array(dataset['native-country'].unique())
countries = np.delete(countries, 0)
dataset['native-country'].replace(countries, 'Other', inplace=True)
df['native-country'].replace(countries, 'Other', inplace=True)

# SECTION
# sns.catplot(x='native-country', hue='income', data=dataset, kind='count')
# SECTION

y_treated = df['income']
X_treated = df.drop('income', axis=1).values
# print(df.drop('income', axis=1).head(5))

# plt.show()
