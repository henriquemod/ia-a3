import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

df = pd.read_csv('./census.csv', names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'], header=None)

df = df.replace(' ?', np.nan)


#NOTE - Achando o percentual de dados faltando no conjunto de dados
total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
# print(missing_data.head(15))

#NOTE - Como a quantidade de dados faltando é muito pequena, 
#podemos substituir os valores nulos com o moda de cada coluna
# print(df['occupation'].describe())
# print('-----------------')
# print(df['workclass'].describe())
# print('-----------------')
# print(df['native-country'].describe())


#Ocupation - Como a maioria é Prof-specialty, substituimos os valores nulos com ela
df['occupation'] = df['occupation'].fillna('Prof-specialty')
#Workclass - Como a maioria é Private, substituimos os valores nulos com ele
df['workclass'] = df['workclass'].fillna('Private')
#Nativa Country - Como a maioria é United-States, substituimos os valores nulos com ele
df['native-country'] = df['native-country'].fillna('United-States')

y = df['income']
X = df.drop('income', axis=1).values


#NOTE - Visualizando as caracteristicas numéricas do dataset utilizando histogramas 
#para analisar a distribuição dessas caracteristicas no dataset
# rcParams['figure.figsize'] = 12, 12
# df[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].hist()
# plt.show()

#NOTE - Plotando a correlação entre a saida(income) e as caracteristicas
plt.matshow(df.corr())
plt.colorbar()
plt.xticks(np.arange(len(df.corr().columns)), df.corr().columns.values, rotation = 45) 
plt.yticks(np.arange(len(df.corr().columns)), df.corr().columns.values) 
for (i, j), corr in np.ndenumerate(df.corr()):
    plt.text(j, i, '{:0.1f}'.format(corr), ha='center', va='center', color='white', fontsize=14)
# plt.show()