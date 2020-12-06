# Script criado para resolver o Desafio 01 do Bootcamp Analista de Machine Learning - IGTI
# Autor: Júlio César da Silva Zeferino
# Última atualização do script: 05-12-2020

# Download do arquivo em https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

'''
A ideia deste script é encontrar um modelo para prever o total de bikes que seriam alugadas a partir do grau de humidade no dia e do número de usuários casuais.
'''

# Última atualização do arquivo: 20-12-2013


# importando os pacotes
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

# importando os dados
dados = pd.read_csv('comp_bikes_mod.csv')
dados.head()

# características do dataset
print('Dimensão do Dataset: {}'.format(dados)
dados.info()
print('Tipo de dados existentes no Dataset: \n{}'.format(
    dados.dtypes.value_counts()))
print('Qtde de tipos diferentes de dados no Dataset: {}'.format(
    dados.dtypes.nunique()))
print('Proporção de valores nulos para a variável "temp": {}%'.format(
    round((dados.isnull().sum()['temp'] / len(dados['temp']) * 100), 2)))

# contando valores nulos
dados.isnull().sum()

# retirando as linhas que contém o campo de data nulo
dados.dropna(subset=['dteday'], inplace=True)
dados.info()  # verificando o dataset

# analisando as estatísticas do dataset
dados.describe()

# transformando a coluna season (estações) em dados categóricos
dados['season']=dados['season'].astype('category')
print('Existem {} estações diferentes no dataset'.format(
    dados['season'].nunique()))
# A modificação é feita para que o modelo tenha um comportamento mais adequado

# convertendo o campo dtday de string para o tipo data
dados['dteday']=pd.to_datetime(dados.dteday)
dados.dtypes  # verificando o tipo da coluna

print('A maior data no dataset é {}'.format(dados['dteday'].max()))

# analisando a velocidade do vento (windspeed)
dados.boxplot(['windspeed'])
plt.show()

# selecionando as colunas 'season', 'temp', 'atemp', 'hum', 'windspeed', 'cnt'
dados_filter=dados[['season', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']]

# matriz de correlação
plt.figure(figsize=(16, 10))
matrizCor=dados_filter.corr()  # construindo a matriz de correlação
# plotando a matriz de correlação do seaborn
sn.heatmap(matrizCor, annot=True, vmin=-1, vmax=1, center=0)
plt.show()

# Tratando nulos: substituindo pelo valor médio nas colunas 'hum', 'cnt' e 'casual'
dados_reg=dados[['hum', 'cnt', 'casual']]  # filtrando as colunas
# substituindo os nulos pela média da coluna
dados_reg.fillna(dados_reg.mean(), inplace=True)
dados_reg.isnull().sum()  # avaliando se os valores foram preenchidos

# selecionando as variaveis para o modelo
x=dados_reg[['hum', 'casual']]  # variaveis independentes
y=dados_reg['cnt']  # variavel dependente (target)

##################################################################################################
############################# REGRESSÃO LINEAR ###################################################
##################################################################################################

from sklearn.linear_model import LinearRegression

# construindo o modelo de regressão
reg=LinearRegression()
modelo1=reg.fit(x, y)
print('Y = {} + {}X1 + {}X2'.format(round(reg.intercept_, 2),
      round(reg.coef_[0], 2), round(reg.coef_[1], 2)))

# realizando a previsão
previsao=reg.predict(x)

# analise do modelo
# método para o cálculo do R2 (coeficiente de determinação)
from sklearn.metrics import r2_score

R_2=r2_score(y, previsao)
print('Coeficiente de determinação (R2): ', R_2)

##################################################################################################
############################# ÁRVORES DE DECISÃO #################################################
##################################################################################################

from sklearn.tree import DecisionTreeRegressor

# construindo o modelo
arvore=DecisionTreeRegressor()
modelo2=arvore.fit(x, y)

# realizando a previsão
previsao2=arvore.predict(x)

# analise do modelo
R_2=r2_score(y, previsao2)
print('Coeficiente de determinação (R2): ', R_2)

# plotando as decisões tomadas pela árvore de decisão
from sklearn import tree  # importando a bilioteca para as árvores de decisão
import pydotplus  # biblioteca utilizada como interface para plotas as decisões

dot_dataset=tree.export_graphviz(arvore, out_file=None)
grafico=pydotplus.graph_from_dot_data(dot_dataset)
grafico.write_pdf('bikeshare.pdf')
