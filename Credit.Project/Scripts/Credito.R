####################################################################################
############Aplicação algoritmo de classificação para análise de crédito############
####################################################################################


####Carregando pacotes####

  library(e1071)
  library(rpart)
  library(randomForest)

####Importando dados####

  credito = read.csv(file.choose(), sep = ";") #importa a planilha
  fix(credito)
  dim(credito) #A planilha tem 20 variáveis e 1000 linhas


###Criando as amostras de treino e teste####

  set.seed(12)
  
  #gerando uma amostra aleatória
  amostra = sample(2,1000,replace = T, prob = c(0.7,0.3))
  amostra
  
  #separando os dados de treino e teste
  creditotreino = credito[amostra==1,] #gerando a amostra de treino
  dim(creditotreino) #70% dos dados
  
  creditoteste = credito[amostra==2,] #gerando a amostra de teste
  dim(creditoteste) #30% dos dados

  
#########Algoritmo Naive Bayes#########
  
  modelonaive = naiveBayes(CLASSE ~ ., creditotreino)
  modelonaive
  
###Avaliando o modelo
  
  predicaonaive = predict(modelonaive,creditoteste)
  predicaonaive

###Gerando a matriz de confusão
  
  confusaonaive = table(creditoteste$CLASSE,predicaonaive)
  confusaonaive #220 acertos
  
  #Calculando a taxa de erro
  taxaerronaive = (confusaonaive[2] + confusaonaive[3]) / sum(confusaonaive)
  taxaerronaive #24,39%
  
  #Com este algoritmo conseguimos reduzir a taxa de erro para 24,39%.
  
  
#########Algoritmo Arvore de Decisao#########
  
  modeloarvore = rpart(CLASSE ~ ., data=creditotreino,  method="class") 
  
  #plotando a arvore
  plot(modeloarvore)
  text(modeloarvore, use.n=TRUE, all=TRUE, cex=.4)

###Avaliando o modelo
  
  predicaoarvore = predict(modeloarvore,newdata=creditoteste)
  predicaoarvore

  #Para que seja possível mensurar o desempenho do modelo, será necessário
  #binarizar os resultados.
  
###Binarizando as predições
  
  #combinando a tabela de teste com as predições
  TabelaCred = cbind(creditoteste,predicaoarvore)
  fix(TabelaCred)
  
  #criando a coluna de resultado
  TabelaCred["Resultado"] = ifelse(TabelaCred$ruim>=0.5,"ruim","bom")
  fix(TabelaCred)
  
###Matriz de confusao
  
  confusaoarvore = table(creditoteste$CLASSE,TabelaCred$Resultado)
  confusaoarvore #197 acertos
  
  taxaerroarvore = (confusaoarvore[2] + confusaoarvore[3]) / sum(confusaoarvore)
  taxaerroarvore #32,30%
  
  #Com este algoritmo, não foi possível chegar a taxa de erro menor que 25%.
  

#########Algoritmo Random Forest#########
  
  modelofloresta = randomForest(CLASSE ~ .,data=creditotreino, ntree=100,importance=T)
  
###Avaliando o modelo
  
  predicaofloresta = predict(modelofloresta,creditoteste)
  predicaofloresta
  
###Matriz de confusao
  
  confusaofloresta = table(creditoteste$CLASSE,predicaofloresta)
  confusaofloresta #224 acertos
  
  
  taxaerrofloresta = (confusaofloresta[2] + confusaofloresta[3]) / sum(confusaofloresta)
  taxaerrofloresta #23,02%
  
  #Com o algoritmo random forest foi possível reduzir a taxa de erro para 23,02%.
  
  
#########Conclusão#########
  
  #Utilizando o algoritmo random forest, foi possível reduzir o erro para
  #aproximadamente 23%, logo, ao fazer a prospecção de novos clientes, há apenas
  #23% de chances de se escolher um cliente que seja inadimplente.
  
####################################################################################
  
