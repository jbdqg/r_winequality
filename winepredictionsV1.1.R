### wine-quality.R file ###
#P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
#Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

library(rminer) # load rminer package
library(kernlab) # load svm functions used by rminer
library(mco) # load mco package

models = c("MR", "NN", "SVM") #modelos de DM

##################################################################################################
#Ficheiro com os dados é carregado para a variável dataset
##################################################################################################
file="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" #carregamento é feito diretamente do repositório da UCI
dataset=read.table(file=file,sep=";",header=TRUE) #define-se o separador dos valores e indica-se que a primeira linha é o header (nome das variáveis)
output=ncol(dataset) #devolve o índice da coluna de output. ncol devolve o nr de colunas
maxinputs=output-1 #devolve o número de inputs. como o output é o índice 12, o número de inputs são 11

for (modeltype in models){
  
  ##################################################################################################
  #Para acelerar o processo de desenvolvimento pode-se definir qual a % de dados do dataset que
  # são usados
  #Imprime-se o sumário dos valores estatísticos dos atributos do dataset
  #Imprime-se a tabela de distribuição do número de exemplos em função do output
  ##################################################################################################
  n=nrow(dataset) #número total de exemplos
  sfactor=1 #percentagem de exemplos a considerar
  ns=round(n*sfactor) #número de exemplos considerados
  set.seed(12345) #estabelece que o número gerado é pseudo aleatório para que pedidos para seeds iguais devolvam os mesmos resultados
                  # permite que a execução seja replicada quando é invocada para os mesmos parâmetros
  ALL=sample(1:n,ns) #contém os índices dos exemplos que vão ser considerados
  print(summary(dataset[ALL,])) #imprime dados estatísticos para as variáveis que forem indicadas
  cat("output class distribuition (", sfactor * 100 ,"% samples):\n")
  print(table(dataset[ALL,]$quality)) #imprime a distribuição dos exemplos considerados em função dos valores do campo quality (output)
  
  ##################################################################################################
  #Divide-se o dataset em dois, 2/3 para treino (para fazer o fit do modelo) e 1/3 para 
  # teste (para estimar a capacidade de generalização)
  ##################################################################################################
  H=holdout(dataset[ALL,]$quality,ratio=2/3) #holdout separa os dados na proporção indicada em ratio, em que ratio se refere ao 
                                             #training (tr) e 1-ratio é a parte para teste (ts)
  cat("nr. training samples:",length(H$tr),"\n")
  cat("nr. test samples:",length(H$ts),"\n")
  
  ##################################################################################################
  #Definem-se os parametros globais que vão servir de base para obter o melhor modelo e as 
  # variáveis mais relevantes
  ##################################################################################################
  pformula=quality~. #fórmula que significa que o target é a variável quality e que todos os outros dados são inputs
  
  ##################################################################################################
  #Definição dos parametros para a função mining
  ##################################################################################################
  mruns=4 #número de execuções para a função de mining que vai afinar o hiperparâmetro (NN e SVM)
  mmethod=c("holdout",2/3) #método de validação a considerar para o mining de treino e teste
  mMRmodel="mr" #indica que o modelo de DM é Multiple Regression 
  mNNmodel="mlpe"  #indica que o modelo de DM é Neural Network (multilayer perceptron ensemble)
  mSVMmodel="ksvm" #indica que o modelo de DM é Suupport Vector Machine
  mtask="reg" #a tarefa de DM será regressão
  mconvex=1 #corresponde ao número de pesquisas que são feitas para encontrar o hiperparâmetro depois de se encontrar um mínimo local
  mimethod=c("holdout",2/3) #método de validação interna a considerar para a afinação do hiperparâmetro (NN e SVM)
  
  #parâmetro search usado para a afinação do hiperparâmetro da NN
  #smethod indica o tipo de pesquisa, que como é grid indica que vão ser pesquisados todos os valores entre 0 e 11, com incremento 1 
  mNNsearchlist=list(smethod="grid",search=list(size=seq(0,11,1)),convex=mconvex,method=mimethod)
  
  #parâmetro search usado para a afinação do hiperparâmetro da SVM
  #smethod indica o tipo de pesquisa, que como é grid indica que vão ser pesquisados todos os valores entre -15 e 3, com incremento 2
  mSVMsearchlist=list(smethod="grid",search=list(sigma=2^seq(-15,3,2),C=3),convex=mconvex,method=mimethod)
  
  #TODO feature não é necessário, pois só após o fit é que se valida a importância dos inputs e se remove o menos relevante (em cada iteração)
  pfeaturelist=c(fmethod="sabs",deletions=-1,Runs=2,vmethod="holdout",vpar=2/3)
  
  mscale="all" #indica que vai ser feito o scale (0 mean and 1 standard deviation) dos inputs e do output
  mfdebug=FALSE
  
  ##################################################################################################
  #Definição dos parametros para a função metric
  ##################################################################################################
  mpmetric="ALL" #sendo igual a ALL são devolvidas todas as estatísticas quando for invocada a função mmetric
  
  ##################################################################################################
  #Definição dos parametros para a função fit
  ##################################################################################################
  
  #TODO ver se na search list é necessário passar uma métrica em metric -> terá a ver com a feature selection? se sim, passar a MAD
  
  #TODO feature não é necessário, pois só após o fit é que se valida a importância dos inputs e se remove o menos relevante (em cada iteração)
  ffeaturelist=c(fmethod="sabs",deletions=-1,Runs=2,vmethod="holdout",vpar=2/3)
  
  fscale="all"  #indica que vai ser feito o scale (0 mean and 1 standard deviation) dos inputs e do output
  ftransform="none"
  ffdebug=FALSE
  
  ##################################################################################################
  #Definição dos parametros para a função importance
  ##################################################################################################
  ireall=5 #parâmetro da sensibilidade da análise (Sensitive Analysis)
  imethod="sensv" #indica que a medida é a variância
  imeasure="AAD" #indica que a métrica que vai ser considerada para a análise de sensibilidade das variáveis é a MAD (Average Absolute Deviation)
  isampling="regular" 
  ibaseline="mean" #considera um vetor com as médias de cada atributo
  iresponses=TRUE
  iinteractions=NULL
  ilrandom=-1
  ilfactor=TRUE
  
  #inicializa variáveis para o modelo
  pattributes = c(1,2,3,4,5,6,7,8,9,10,11,output) #variável que armazena quais os atributos que vão ser considerados para a função 
                                                  # em que for usada (fit, mining ou Importance)
  stopcriteria = 0;
  generalizationestimativemae=1 #variável que armazena a qualidade do modelo em cada iteração
  PTM=proc.time() # inicia relógio
  
  while (stopcriteria != 2){
    
    ##################################################################################################
    #Obtenção do melhor modelo (hiperparâmetro) através da função de mining
    #Validação da qualidade do modelo encontrado através da métrica Mean Absolute Error
    ##################################################################################################
    if (modeltype == "MR") {
      modelMining=mining(quality~.,data=dataset[H$tr,pattributes],Runs=mruns,method=mmethod,model=mMRmodel,task=mtask,scale=mscale,fdebug=mfdebug)  
    }else if (modeltype == "NN"){
      modelMining=mining(quality~.,data=dataset[H$tr,pattributes],Runs=mruns,method=mmethod,model=mNNmodel,task=mtask,search=mNNsearchlist,scale=mscale,fdebug=mfdebug)
    }else if (modeltype == "SVM"){
      modelMining=mining(quality~.,data=dataset[H$tr,pattributes],Runs=mruns,method=mmethod,model=mSVMmodel,task=mtask,search=mSVMsearchlist,scale=mscale,fdebug=mfdebug)
    }
    
    modelMetric=mmetric(modelMining, metric=mpmetric)
    
    ##################################################################################################
    #Validação da qualidade do modelo encontrado através da métrica Mean Absolute Error
    #Se a Mean Absolute Error aumentar na iteração seguinte incrementa-se o critério de paragem
    #Se a Mean Absolute Error diminuir na iteração seguinte fica registada como a melhor até ao momento
    ##################################################################################################
    if(modelMetric$MAE[which.min(modelMetric$MAE)] > generalizationestimativemae){
      stopcriteria = stopcriteria + 1
    }else{
      
      ##################################################################################################
      #Obtém-se a execução do mining que teve o menor erro
      ##################################################################################################
      generalizationestimativemae = modelMetric$MAE[which.min(modelMetric$MAE)]
      bestrunindex = which.min(modelMining$error)
      
      ##################################################################################################
      #Registam-se os valores dos parametros da NN/SVM obtidos na execução que teve o menor erro
      #Invoca-se o fit com o modelo encontrado para obter um modelo
      #Invoca-se a função Importance para fazer a análise sensitiva
      #Pelo retorno da função Importance encontra-se a variável menos relevante que deixa de ser
      # considerada na próxima iteração
      ##################################################################################################    
      if (modeltype == "MR") {
        modelFit=fit(pformula,data=dataset[H$tr,pattributes],model=mMRmodel,task=mtask,scale=fscale,fdebug=ffdebug)
      }else if (modeltype == "NN"){
        fitsize = modelMining$mpar[[bestrunindex]]$size
        modelFit=fit(pformula,data=dataset[H$tr,pattributes],model=mNNmodel,task=mtask,search=list(smethod="none",search=list(size=fitsize)),scale=fscale,fdebug=ffdebug)
      }else if (modeltype == "SVM"){
        fitsigma = modelMining$mpar[[bestrunindex]]$kpar$sigma
        fitC = modelMining$mpar[[bestrunindex]]$C
        fitepsilon = modelMining$mpar[[bestrunindex]]$epsilon
        modelFit=fit(pformula,data=dataset[H$tr,pattributes],model=mSVMmodel,task=mtask,search=list(smethod="none",search=list(sigma=fitsigma,C=fitC,epsilon=fitepsilon)),
                     scale=fscale,fdebug=ffdebug)
      }
      
      modelImp=Importance(modelFit, data=dataset[H$tr,pattributes], RealL=ireall, method=imethod, measure=imeasure, sampling=isampling, baseline=ibaseline, 
                          responses=iresponses, interactions=iinteractions, LRandom=ilrandom, Lfactor=ilfactor)
      indexr = which(modelImp$imp %in% sort(modelImp$imp[1:length(pattributes)-1])[1:1]) #indica o índice do input menos relevante
      
      pattributes = pattributes[-indexr]  
      
    }    
  }
  
  ##################################################################################################
  #Definição dos parametros para a função mining que vai validar o modelo encontrado
  ##################################################################################################
  maruns=20 #número de execuções para a função de mining que vai avaliar o modelo encontrado para cada técnica de DM
  mamethod=c("kfold",5) #método de validação do modelo que será por validação cruzada (dataset é dividido em 5 e cada um é usado
                        # como teste numa iteração e o ajuste é feito com os outros 4)
  matask="reg"  #a tarefa de DM será regressão
  mascale="all" #indica que vai ser feito o scale (0 mean and 1 standard deviation) dos inputs e do output
  mafdebug=FALSE
  
  maconvex=0  #corresponde ao número de pesquisas que são feitas para encontrar o hiperparâmetro depois de se encontrar um mínimo local
              #Como para esta execução já se vai usar o melhor hiperparâmetro encontrado, o valor é 0
  maimethod=c("holdout",2/3)  #método de validação interna a considerar para o hiperparâmetro (NN e SVM)
  
  ##################################################################################################
  #Usa-se como dataset todos os dados
  #É invocada a função e mining com o melhor modelo e variáveis selecionadas pelo procedimento
  # de afinação de parâmetros e seleção de variáveis
  ##################################################################################################
  if (modeltype == "MR") {
    modelAllmining=mining(quality~.,data=dataset[ALL,pattributes],Runs=maruns,method=mamethod,model=mMRmodel,task=matask,scale=mascale,fdebug=mafdebug)
    savemining(modelAllmining,"mrmining");
  }else if (modeltype == "NN"){
    masearchlist=list(smethod="none",search=list(ssize=modelFit@mpar$kpar$size),convex=maconvex,method=maimethod)
    modelAllmining=mining(quality~.,data=dataset[ALL,pattributes],Runs=maruns,method=mamethod,model=mNNmodel,task=matask,search=masearchlist,scale=mascale,fdebug=mafdebug)
    savemining(modelAllmining,"nnmining");
  }else if (modeltype == "SVM"){
    masearchlist=list(smethod="none",search=list(sigma=modelFit@mpar$kpar$sigma,C=modelFit@mpar$C,epsilon=modelFit@mpar$epsilon),convex=maconvex,method=maimethod)
    modelAllmining=mining(quality~.,data=dataset[ALL,pattributes],Runs=maruns,method=mamethod,model=mSVMmodel,task=matask,search=masearchlist,scale=mascale,fdebug=mafdebug)
    savemining(modelAllmining,"svmmining");
  }
  
  ##################################################################################################
  #Obtêm-se os valores médios das previsões da execução do modelo com as melhores variáveis
  #Constrói-se a confusion matrix considerando o retorno da função de mining, usando os valores
  # de target e os valores previstos apra cada exemplo
  #Embora a Confusion Matrix seja adequada para problemas de classificação, é possível usá-la neste
  # contexto se os targets forem definidos como labels e forem arredondados os valores previstos
  # devolvidos pelo modelo
  #Calcula-se o Kappa value que reflete a precisão quando comparado com um classificador aleatório 
  # (que tem Kappa = 0%). Quanto maior o valor do Kappa, maior a precisão dos resultados
  ##################################################################################################
  mtestaveraged = .colMeans(do.call(rbind,modelAllmining$test), length(modelAllmining$test), length(modelAllmining$test[[1]]), na.rm = FALSE)
  mpredaveraged = .colMeans(do.call(rbind,modelAllmining$pred), length(modelAllmining$pred), length(modelAllmining$pred[[1]]), na.rm = FALSE)
  
  confusionmatrix = mmetric(factor(c(mtestaveraged)), lapply(mpredaveraged,round), metric="CONF")
  
  kappavalue = mmetric(factor(c(mtestaveraged)), lapply(mpredaveraged,round), metric="KAPPA")
  
  cat(modeltype,  "Mean Absolute Error",generalizationestimativemae,"\n")
  cat(modeltype,  "Kappa:",kappavalue,"\n")
  cat(modeltype,  "Inputs:",length(pattributes),"\n")
  if (modeltype == "NN"){
    cat(modeltype,  "Model:",fitsize,"\n")
  }else if (modeltype == "SVM"){
    cat(modeltype,  "Model:",fitsigma,"\n")
  }
  cat(modeltype,  "Inputs:",length(pattributes),"\n")
  cat(modeltype,  "Confusion Matrix","\n")
  print(confusionmatrix$conf)
  sec=(proc.time()-PTM)[3] # para relógio e obtém segundos
  cat("Tempo decorrido (s):",sec,"\n")
    
}

##################################################################################################
#Colocam-se os melhores modelos encontrados numa lista para gerar o gráfico com a
#curva REC de cada um dos modelos
#A curva REC (Regression Error Characteristic) é usada para comparar a performance dos
# modelos de regressão, em que um modelo ideial teria uma área de 1.
##################################################################################################
mrmining=loadmining("mrmining")
nnmining=loadmining("nnmining")
svmmining=loadmining("svmmining")

miningmodels=vector("list",3); # vector list of mining
miningmodels[[1]]=svmmining
miningmodels[[2]]=nnmining
miningmodels[[3]]=mrmining
mgraph(miningmodels,graph="REC",leg=c("SVM","NN","MR"),xval=2,PDF="reccomparison")
