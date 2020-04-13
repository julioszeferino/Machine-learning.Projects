#Script for reproducing results of Fernando Amaral (2020)
#Author: Júlio Zeferino (julioszeferino@gmail.com)
#Last script update: 2020-04-13
#
#file downloaded from www.dados.com.br
#The description of the data goes here
#
#Last file update: 2020-04-13

library(e1071)
library(rpart)
library(randomForest)

#Importando dados--------------------------------------------------------------

        credito = read.csv(file.choose(), sep = ";") #importa a planilha
        fix(credito)
        dim(credito) #A planilha tem 20 variáveis e 1000 linhas

#Criando as amostras de treino e teste-----------------------------------------
        
        set.seed(12)
        
        #gerando uma amostra aleatória
        amostra = sample(2,1000,replace = T, prob = c(0.7,0.3))
        amostra
        
        #separando os dados de treino e teste
        creditotreino = credito[amostra==1,] #gerando a amostra de treino
        dim(creditotreino) #70% dos dados
        
        creditoteste = credito[amostra==2,] #gerando a amostra de teste
        dim(creditoteste) #30% dos dados
        
#Estimando o Modelo Naive Bayes ----------------------------------------------
        
        
        modelonaive = naiveBayes(CLASSE ~ ., creditotreino)
        modelonaive
        
        #Avaliando o modelo
        predicaonaive = predict(modelonaive,creditoteste)
        predicaonaive
        
        #Gerando a matriz de confusão
        confusaonaive = table(creditoteste$CLASSE,predicaonaive)
        confusaonaive #220 acertos
        
        #Cálculo taxa de erro
        taxaerronaive = (confusaonaive[2] + confusaonaive[3]) / sum(confusaonaive)
        taxaerronaive #24,39%
        
#Estimando o Modelo de Árvores de Decisão--------------------------------------
        
        modeloarvore = rpart(CLASSE ~ ., data=creditotreino,  method="class") 
        
        #plotando a arvore
        plot(modeloarvore)
        text(modeloarvore, use.n=TRUE, all=TRUE, cex=.4)
        
        #Avaliando o modelo
        predicaoarvore = predict(modeloarvore,newdata=creditoteste)
        predicaoarvore
        
        #Para que seja possível mensurar o desempenho do modelo, será necessário
        #binarizar os resultados.
        
        #Binarizando as predições
        
        #combinando a tabela de teste com as predições
        TabelaCred = cbind(creditoteste,predicaoarvore)
        fix(TabelaCred)
        
        #criando a coluna de resultado
        TabelaCred["Resultado"] = ifelse(TabelaCred$ruim>=0.5,"ruim","bom")
        fix(TabelaCred)
        
        #Matriz de confusao
        confusaoarvore = table(creditoteste$CLASSE,TabelaCred$Resultado)
        confusaoarvore #197 acertos
        
        #Cálculo taxa de erro
        taxaerroarvore = (confusaoarvore[2] + confusaoarvore[3]) / sum(confusaoarvore)
        taxaerroarvore #32,30%
        
#Estimando o Modelo Random Forest----------------------------------------------
        
        modelofloresta = randomForest(CLASSE ~ .,data=creditotreino, ntree=100,importance=T)
        
        #Avaliando o modelo
        predicaofloresta = predict(modelofloresta,creditoteste)
        predicaofloresta
        
        #Matriz de confusao
        confusaofloresta = table(creditoteste$CLASSE,predicaofloresta)
        confusaofloresta #224 acertos
        
        #Cálculo taxa de erro
        taxaerrofloresta = (confusaofloresta[2] + confusaofloresta[3]) / sum(confusaofloresta)
        taxaerrofloresta #23,02%
        
#Report results----------------------------------------------------------------
        
        #Utilizando o algoritmo random forest, foi possível reduzir o erro para
        #aproximadamente 23%, logo, ao fazer a prospecção de novos clientes, há apenas
        #23% de chances de se escolher um cliente que seja inadimplente.
        