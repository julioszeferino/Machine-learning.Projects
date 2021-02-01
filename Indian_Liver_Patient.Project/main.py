# Script criado para resolver o Desafio 03 do Bootcamp Analista de Machine Learning - IGTI
# Autor: Júlio César da Silva Zeferino
# Última atualização do script: 31-01-2021

# Download do arquivo em https://www.openml.org/d/1480

'''
A ideia deste script é encontrar os melhores hiperparâmetros para os modelos testados.
'''

# Última atualização do arquivo: 22-05-2015

# importando as bibliotecas

import pandas as pd  # biblioteca para manipulação de dados
import numpy as np  # biblioteca para utilizacao de vetores e matrizes
import matplotlib.pyplot as plt  # bibloteca para plotar graficos
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.metrics import f1_score, make_scorer

# lendo o csv que contem a base de dados e armazanando em um df
df = pd.read_csv(
    r'D:\User\Documents\LOCAL_GITHUB\Machine-learning.Projects\Indian_Liver_Patient.Project\ilpd.csv')

# imprimindo as 5 primeiras linhas do df para confirmação
df.head(5)

# Verificando o numero de amostras (linhas) e features (colunas) do dataset.
print('Amostras e Features:', df.shape)

# Verificando quais são os tipos das features
df.columns

# validando se ha dados nulos
df.isnull().sum()

# observando as caracteristicas do dataset
df.describe()

# tratando dados categoricos
df = pd.get_dummies(df)

# mapeando a classe para indexar em 0
# criando um dicionario de dados para o mapeamento
name_to_class = {
    1: 0,
    2: 1
}

# substituindo os valores categóricos pelo mapeamento
df['Class'] = df['Class'].map(name_to_class)

# check
df.head(5)

# armazenando os labels em um array
labels = np.array(df['Class'])

# salvando a ordem das features
feature_list = list(df.columns)

# removendo a coluna de labels do df original
df = df.drop('Class', axis=1)

# check
df.columns

# convertendo df para array
data = np.array(df)

####### SVC #######
# definindo o tipo de validacao cruzada e o numero de folds
cv_strat = StratifiedKFold(n_splits=10)

# definindo a estrategia de score a partir da metrica f1
f1 = make_scorer(f1_score)

# definindo hiperparâmetros
distributions = dict(kernel=['sigmoid', 'rbf'],
                     C=uniform(loc=0, scale=10))  # distribuicao uniforme variando entre 0 e 10

# instânciando meu classificador
classifier = SVC()

# instanciando e modelando o grid search com os hiperparametros e a validação definidas.
random_cv = RandomizedSearchCV(
    classifier, distributions, cv=cv_strat, scoring=f1, random_state=42, n_iter=5)
random_cv.fit(data, labels)

# Avaliando os melhores resultados encontrados pelo Random Search
print('Melhor resultado f1:', random_cv.best_score_)
print('\n\nMelhor configuração de hiperparâmetros:', random_cv.best_params_)
print('\n\nConfigurações de todos os hiperparâmetros do melhor estimado encontrado pelo GridSearch: \n',
      random_cv.best_estimator_)

####### Random Forest Classifier #######
# definindo o tipo de validacao cruzada e o numero de folds
cv_strat = StratifiedKFold(n_splits=10)

# definindo a estrategia de score a partir da metrica f1
f1 = make_scorer(f1_score)

# definindo hiperparâmetros
distributions1 = dict(n_estimators=randint(100, 200),
                      bootstrap=[True, False],
                      criterion=['gini', 'entropy'])

# instânciando meu classificador
classifier1 = RandomForestClassifier(random_state=42)

# instanciando e modelando o grid search com os hiperparametros e a validação definidas.
random_cv1 = RandomizedSearchCV(
    classifier1, distributions1, cv=cv_strat, scoring=f1, random_state=42, n_iter=5)
random_cv1.fit(data, labels)

# Olhando para os melhores resultados encontrados pelo Random Search
print('Melhor resultado f1:', random_cv1.best_score_)
print('\n\nMelhor configuração de hiperparâmetros:', random_cv1.best_params_)
print('\n\nConfigurações de todos os hiperparâmetros do melhor estimado encontrado pelo GridSearch: \n',
      random_cv1.best_estimator_)
