#Script for reproducing results of MORO et al (2020)
#Author: Júlio Zeferino (julioszeferino@gmail.com)
#Last script update: 2020-04-13
#
#file downloaded from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
#The description of the data goes here
#
#Last file update: 2012-02-14

library(e1071)
    
#Importando dados--------------------------------------------------------------

    bank = read.csv(file.choose(),sep=';',header=T)
    head(bank) #ver os primeiros registros do arquivo, o último atributo é a classe
    dim(bank) #ver o tamanho do crédito
    fix(bank)

#Criando as amostras de treino e teste-----------------------------------------
    
    set.seed(2523)
    
    amostra = sample(2,45211,replace=T, prob=c(0.7,0.3))
    
    banktreino = bank[amostra==1,] #gerando a amostra de treino (70% dos dados)
    bankteste = bank[amostra==2,] #gerando a amostra de teste

    dim(banktreino) #31715 dados
    dim(bankteste) #13496 dados

#Estimando o Modelo Naive Bayes ----------------------------------------------
    
    modelo <- naiveBayes(y ~., banktreino)
    modelo
    #A chance de dizer não é de 88,27%
    #A chance de dizer sim é de 11,72%

    #Avaliando modelo
    predicao <- predict(modelo,bankteste)
    predicao

    #gerando a matriz de confusao
    confusao = table(bankteste$y,predicao)
    confusao #11779 acertos
    
    #Calculando a taxa de erro
    taxaerro = (confusao[2] + confusao[3]) / sum(confusao)
    taxaerro #12,72%

#Report results----------------------------------------------------------------
    
#O modelo criado para este artigo acertou 1064 vezes quando os clientes 
#aceitaram o depósito a prazo e 74 vezes quando não aceitaram. Ao todo, o 
#modelo acertou 1138 vezes em 1333 dados (tamanho da amostra de teste gerada), 
#i. e., a taxa de acerto do modelo foi de 85,37% das vezes.