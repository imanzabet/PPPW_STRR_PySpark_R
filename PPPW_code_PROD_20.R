library(Matrix)
library(dplyr)
library(gee)
library(readr)

library(geeM)

library(reticulate)

library(foreach)
library(doParallel)
library(pryr)

rm(list=setdiff(ls(), "MEM_LOG"))
invisible(gc())
mem_used()

#Using number of available cores to parallelize model

numCores <- detectCores()
cat("number of cores=", numCores)

registerDoParallel(cores=numCores)

ONEY<-TRUE

options(warn=1)
options(digits = 8)

#install_miniconda()

if (.Platform$OS.type=='windows'){
  Win_Flag <-TRUE
} else {Win_Flag <-TRUE}

if (1==0){

  #reticulate::use_python("/usr/bin/python")
  #reticulate::py_install('pandas')
  #reticulate::py_install('s3fs')
  #reticulate::py_install('pyarrow')


  #SFR_sort<-read.csv("SFR_sort.csv")

  #reticulate::use_python("/usr/bin/python")

  #import pyarrow.parquet as pq

  reticulate::py_run_string("import pandas as pd;
      import os;
      data=pd.read_parquet('s3://eqrs-ngmc-datascience-sbx/VAT/SFR_model_input_ADO.parquet')")


  #reticulate::py_run_string("import pandas as pd; import os;data=pd.read_parquet('s3://eqrs-ngmc-sbx/PY2020/FullDeIdentifiedDataSet.parquet')")

  #reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('eqrs-ngmc-sbx/PY2020/FullDeIdentifiedDataSet/output/VAT-Kulbir-5/rInputSFR-375/cache',filesystem=s3).read_pandas().to_pandas()")

  #reticulate::py_run_string("import pyarrow.parquet as pq; import os;pq.write_table(py$data, 's3://eqrs-ngmc-datascience-sbx/SFR_model_input_ADO2.parquet')")

  #reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('eqrs-ngmc-sbx/PY2020/FullDeIdentifiedDataSet/output/VAT/rInputSFRFilterByYear-381/cache',filesystem=s3).read_pandas().to_pandas()")

  #reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('eqrs-ngmc-sbx/PY2020/FullDeIdentifiedDataSet/output/VAT/rInputSFRFilterByYear-382/cache',filesystem=s3).read_pandas().to_pandas()")

  #reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('ado1-eqrs-prodpreview2-ngmc-data-store-pii-bucket/py2021/output/VAT20200510/rInputSFRFilterByYear-382/cache',filesystem=s3).read_pandas().to_pandas()")

  #reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('ado1-eqrs-prodpreview2-ngmc-data-store-pii-bucket/py2021/output/VAT20200518/rInputSFRFilterByYear-382/cache',filesystem=s3).read_pandas().to_pandas()")

  #reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('ado1-eqrs-prodpreview2-ngmc-data-store-pii-bucket/py2021/output/VAT20200529/rInputSFRFilterByYear-382/cache',filesystem=s3).read_pandas().to_pandas()")

  #reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('ado1-eqrs-prodpreview2-ngmc-data-store-pii-bucket/py2021/output/VAT20200604-SFR-1636/rInputSFRFilterByYear-382/cache',filesystem=s3).read_pandas().to_pandas()")

  #reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('ado1-eqrs-prodpreview2-ngmc-data-store-pii-bucket/py2021/output/VAT20200604-SFR-1636/rInputSFRFilterByYear-382/cache',filesystem=s3).read_pandas().to_pandas()")

  #reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('ado1-eqrs-prodpreview2-ngmc/py2021/calc/output/VAT20200618/rInputSFRFilterByYear-381/cache',filesystem=s3).read_pandas().to_pandas()")

  #reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('ado1-eqrs-prodpreview2-ngmc/py2021/calc/output/VAT20200618/rInputSFRFilterByYear-382/cache',filesystem=s3).read_pandas().to_pandas()")

  #read SFR_modelado1-eqrs-prodpreview2-ngmc/py2021/calc/output/TPS20200710/rInputSFRFilterByYear-381

  reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('ado1-eqrs-prodpreview2-ngmc/py2021/calc/output/TPS20200710/rInputSFRFilterByYear-381/cache',filesystem=s3).read_pandas().to_pandas()")

  reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('ado1-eqrs-prodpreview2-ngmc/py2021/calc/output/PY20210001/rInputSFRFilterByYear-381/cache',filesystem=s3).read_pandas().to_pandas()")

  reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('ado1-eqrs-prodpreview2-ngmc/py2021/calc/output/PY20210001/rInputSFRFilterByYear-382/cache',filesystem=s3).read_pandas().to_pandas()")


  #SFR_model_input_ADO_20191220 <- read_delim("SFR_model_input_ADO_20191220.txt", "|", escape_double = FALSE, trim_ws = TRUE)

  #SFR_model_input <- read_delim("ARBOR_zbetas_allgrps_20200211.txt", "|", escape_double = FALSE, trim_ws = TRUE)

  #SFR_model_input<-read.csv("~/SFR_model_input_ADO_20191220.csv")

  #SFR_model_input<-data.frame(SFR_model_input)

  #http://s3//ado1-eqrs-prodpreview2-arbor-data/prodpy21/ARBOR_VATF_RSLTS

  reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('ado1-eqrs-prodpreview2-arbor-data/prodpy21/ARBOR_SFR_DTLS',filesystem=s3).read_pandas().to_pandas()")


  Data<- py$data
  SFR_model_input<-Data
  mem_used()

  #SFR_model_input$PROVFS<-as.character(SFR_model_input$PROVFS)

  SFR_model_input$X1<-NULL

  #Flipping boolean flag for medcov_6m to TRUE for > 6 mos coverage

  #SFR_model_input$medcov_6m<-as.numeric(!SFR_model_input$medcov_6m)


  colnames(SFR_model_input)[4]<-"PROVFS"
  SFR_model_input$PROVFS<-as.character(SFR_model_input$PROVFS)



  #Sort Data

  SFR_sort<-arrange(SFR_model_input,PROVFS,patientId,year)

  #SFR_sort<-arrange(SFR_model_input_ADO_20191220,PROVFS,patientId,year)

  print(str(SFR_sort))

  ##################################
  #Select Desired Calendar Year
  #########################################


  # 2017

  #print(y1)
  #print(str(SFR_sort))
  #print(head(SFR_sort))

  y1<-2019

  #y1<-2018

  SFR_sort17<-filter(SFR_sort,year==y1)



  SFR_sort17FG<-filter(SFR_sort17,Fistpatients11==1)

  #SFR_tails17F<-filter(SFR_sort17F, AVF_f==1 | AVF_f==0)

  #SFR_sort17F<-filter(SFR_sort17F, AVF_f>0 & AVF_f<1)

} else{
  if (Win_Flag==TRUE){
    SFR_sort17FG <- read.csv("C:/Project_Data/PPPW/SFR_sort17FG_sample.csv")
    #SFR_sort17FG = SFR_sort17FG[1:100000,]
  } else {
    install_miniconda()

    reticulate::use_python("/usr/bin/python")
    reticulate::py_install('pandas')
    reticulate::py_install('s3fs=0.3.0')
    reticulate::py_install('pyarrow')
    reticulate::use_python("/usr/bin/python")
    #import pyarrow.parquet as pq

    #reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('s3://eqrs-ngmc-datascience-sbx/PPPW/SFR_sort17FG_parquet',filesystem=s3).read_pandas().to_pandas()")
    reticulate::py_run_string("import pyarrow.parquet as pq;import s3fs;import os;s3 = s3fs.S3FileSystem();data=pq.ParquetDataset('s3://datascience-dataset/SFR_sort17FG_parquet',filesystem=s3).read_pandas().to_pandas()")
    SFR_sort17FG<- py$data
  }
}
#################################################################################
#####    USE DATA SET CREATING CODE HERE
#############################################################################

#SFR_sort17Fppw<-SFR_sort17Pf[,c(1:6,12,16:18)]

#SFR_sort17Fppw<-SFR_sort17Pf[,c(1:5,9:12)]

#SFR_sort17Fppw<-SFR_sort17Pf[,c(1:4,12,16:19)]


#cat("Ran year",y1)

#############################################################
# Filter Code for facilities with all 0s or 1s
############################################################

SFR_sort17Pf<-SFR_sort17FG
if (Win_Flag==1){
  SFR_sort17Pf$'X'<-NULL
  SFR_sort17Pf$'Unnamed..0'<-NULL
}else{
  SFR_sort17Pf$'_c0'<-NULL
}
#### Parallel Filter Code

FAC_f<-SFR_sort17Pf$PROVFS

FAC_f<-data.frame(FAC_f)

FAC_fd<-distinct(FAC_f)

FAC_fd$FAC_f<-as.character(FAC_fd$FAC_f)

NCCN<-nrow(FAC_fd)

#SFR_sort17Pf$pppw_f<-0

SFR_tailsp<-SFR_sort17Pf[0,]


#system.time({
#  SFR_tailsp<-foreach(i=1:(NCCN), .combine=rbind) %dopar% {
#    try({
#      SFR_test<-filter(SFR_sort17Pf,PROVFS==FAC_fd$FAC_f[[i]])
#      MAVF<-mean(SFR_test$pppw)
#      SFR_test$pppw_f<-MAVF
#      if(MAVF==1) {
#        #      cat("WARNING - CCN:",FAC_fd$FAC_f[[i]], "**PROBLEM** - Average PPPW =",MAVF, "\n")
#        #      cat("THIS FACILITY WILL BE FILTERED FROM THE MODEL")
#        SFR_tailsp<-rbind(SFR_tailsp,SFR_test)
#        #      SFR_sort17Pf<-filter(SFR_sort17Pf, PROVFS!=FAC_fd$FAC_f[[i]])
#      }
#      if(MAVF==0) {
#        #      cat("WARNING - CCN:",FAC_fd$FAC_f[[i]], "**PROBLEM** - Average PPPW =",MAVF, "\n")
#        #      cat("THIS FACILITY WILL BE FILTERED FROM THE MODEL")
#        SFR_tailsp<-rbind(SFR_tailsp,SFR_test)
#        #      SFR_sort17Pf<-filter(SFR_sort17Pf, PROVFS!=FAC_fd$FAC_f[[i]])
#      }
#    })
#    SFR_tailsp
#  }
#})

SFR_sort17PFF<-SFR_sort17Pf[!(SFR_sort17Pf$PROVFS %in% SFR_tailsp$PROVFS),]
SFR_sort17PFF<-data.frame(SFR_sort17PFF)
SFR_sort17Fppw<-SFR_sort17PFF

####################################
#Find Distinct Providers
####################################

# 2017
FAC17<-SFR_sort17Fppw$PROVFS
FAC17<-data.frame(FAC17)
FACD17<-distinct(FAC17)

#####################################
#Generate Random Groupings
######################################

set.seed(25)

# 2017
FACD17$groups <- sample(1:20, size=nrow(FACD17), replace=TRUE)
colnames(FACD17)<-c("PROVFS","Group")

######################################
# Merge with large original Data Frame
######################################
# 2017
SFR_sort17FG<-left_join(SFR_sort17Fppw, FACD17, by = "PROVFS")
SFR_sort17FG<-data.frame(SFR_sort17FG)
#SFR_sort17FG$AVF<-as.integer(SFR_sort17FG$AVF)
SFR_sort17FG$PROVFS<-as.character(SFR_sort17FG$PROVFS)
SFR_sort17FG$patientId<-as.integer(SFR_sort17FG$patientId)
#for(i in 7:11) {
#  SFR_sort17FG[,i] <- as.integer(SFR_sort17FG[,i])
#}

########################################################
# Setting up Groups for each calender year Data File
########################################################
# 2017
Grps17<-list()

for(i in 1:20) {
  Grps17[[i]] <- filter(SFR_sort17FG, Group==i)
  Grps17[[i]] <- data.frame(Grps17[[i]])
  Grps17[[i]] <- arrange(Grps17[[i]],PROVFS,patientId)
  Grps17[[i]]$PROVFS <- as.factor(Grps17[[i]]$PROVFS)
}


#Setup Model Formula

mf <- formula(pppw ~ age + a15 + a55 + a70 +PROVFS-1)


###################################
# Run Model for Each Group
#######################################


####################
## GEEM model
####################
#2017
#GeeMALL17<-data.frame(c(rep(0,22)))
#colnames(GeeMAll)<-"GeemMod.beta.1.22."

system.time({
  GeeMALL17_1<-foreach(i=1:20, .combine=cbind) %dopar% {
    {
      print(i)
      xx<-nlevels(Grps17[[i]]$PROVFS)
      GeemMod17<-geem(mf, data=Grps17[[i]], id=Grps17[[i]]$patientId, family=binomial(link ="logit"),init.beta = as.vector(rep(0,xx+4)), corstr = "independence")
      EstimatesGM17<-data.frame(GeemMod17$beta[1:4])
      EstimatesGM17
    }
  }
})

#system.time({
#  GeeMALL17_2<-foreach(i=6:10, .combine=cbind) %dopar% {
#    {
#      xx<-nlevels(Grps17[[i]]$PROVFS)
#      GeemMod17<-geem(mf, data=Grps17[[i]], id=Grps17[[i]]$patientId, family=binomial(link ="logit"),init.beta = as.vector(rep(0,xx+4)), corstr = "independence")
#      EstimatesGM17<-data.frame(GeemMod17$beta[1:4])
#      EstimatesGM17
#    }
#  }
#})

#GeeMALL17<-cbind(GeeMALL17_1, GeeMALL17_2)

GeeMALL17<-GeeMALL17_1

###################################
# Zebata calculation from GeeM model
###################################

#2017

GeeMALL217<-GeeMALL17

m17<-as.matrix(GeeMALL217)

AVG17<-as.matrix(rowMeans(m17))


################################################
# Calculate zbeta values for second stage model
###################################################


####################################################
#  Run matrix calculation over all ten groups
###############################################





#nc<-c(5:8)

nc<-c(5,7:9)

#nc<-nc_o-1

# 2017

Mcoef17<-AVG17
MMc17<-as.matrix(as.numeric(Mcoef17))
DVar17<-as.matrix(SFR_sort17FG[,nc])
ZbetaP17<-DVar17%*%MMc17


#############################
# Run Stage 2 of Model
##################################

#2017

SFRA17<-data.frame(SFR_sort17FG)

FACA17<-FACD17

FACA17$PROVFS<-as.character(FACA17$PROVFS)



################
## Model Stage 2
#############################

#2017

colnames(SFRA17)[2]<-"Provfs"

SFRAB17<-cbind(SFRA17,ZbetaP17)
SFRAB17$'X1'<-NULL

GeeAll17<-data.frame(c(0))
colnames(GeeAll17)<-"GeeS217.coefficients"

# clean up workspace

#remove(SFRA17)
#remove(Data)
#remove(Grps17)
#remove(SFR_model_input)
#remove(SFR_sort)
#remove(ZbetaP17)
#remove(DVar17)
#remove(SFR_sort17)
#remove(SFR_sort17F)
gc()

################################################

NF17<-nrow(FACA17)
Nten17<-floor(NF17/10)
Nten17h<-round(Nten17/2)
Nten17h2<-Nten17-Nten17h

Nrem17<-NF17%%10

system.time({

GeeAll17a<-foreach(i=1:Nten17h, .combine=rbind) %dopar% {
  try({SFR_A17<-filter(SFRAB17,Provfs==FACA17$PROVFS[[1+(i-1)*10]] | Provfs==FACA17$PROVFS[[2+(i-1)*10]] | Provfs==FACA17$PROVFS[[3+(i-1)*10]]| Provfs==FACA17$PROVFS[[4+(i-1)*10]]| Provfs==FACA17$PROVFS[[5+(i-1)*10]]| Provfs==FACA17$PROVFS[[6+(i-1)*10]]| Provfs==FACA17$PROVFS[[7+(i-1)*10]]| Provfs==FACA17$PROVFS[[8+(i-1)*10]]| Provfs==FACA17$PROVFS[[9+(i-1)*10]]| Provfs==FACA17$PROVFS[[10+(i-1)*10]])
  mfa<- formula(pppw ~ 0 + Provfs + offset(SFR_A17$ZbetaP17))
  GeeS217<-gee(mfa, data=SFR_A17, id=SFR_A17$patientId, family=binomial(link ="logit"), scale.fix = FALSE, tol = 0.001, maxiter = 25,corstr = "independence")
  Estimates17<-data.frame(GeeS217$coefficients)
  Estimates17
  })
}


GeeAll17b<-foreach(i=(Nten17h+1):Nten17, .combine=rbind) %dopar% {
  SFR_A17<-filter(SFRAB17,Provfs==FACA17$PROVFS[[1+(i-1)*10]] | Provfs==FACA17$PROVFS[[2+(i-1)*10]] | Provfs==FACA17$PROVFS[[3+(i-1)*10]]| Provfs==FACA17$PROVFS[[4+(i-1)*10]]| Provfs==FACA17$PROVFS[[5+(i-1)*10]]| Provfs==FACA17$PROVFS[[6+(i-1)*10]]| Provfs==FACA17$PROVFS[[7+(i-1)*10]]| Provfs==FACA17$PROVFS[[8+(i-1)*10]]| Provfs==FACA17$PROVFS[[9+(i-1)*10]]| Provfs==FACA17$PROVFS[[10+(i-1)*10]])
  mfa<- formula(pppw ~ 0 + Provfs + offset(SFR_A17$ZbetaP17))
  GeeS217<-gee(mfa, data=SFR_A17, id=SFR_A17$patientId, family=binomial(link ="logit"), scale.fix = FALSE, tol = 0.001, maxiter = 25,corstr = "independence")
  Estimates17<-data.frame(GeeS217$coefficients)
  Estimates17
}

GeeAll17<-rbind(GeeAll17a,GeeAll17b)

if(Nrem17>0) {
  for(i in 1:Nrem17) {
    try({
      SFR_A17<-filter(SFRAB17,Provfs==FACA17$PROVFS[[i+(Nten17*10)]] | Provfs==FACA17$PROVFS[[1]])
      mfa<- formula(pppw ~ 0 + Provfs + offset(SFR_A17$ZbetaP17))
      GeeS217<-gee(mfa, data=SFR_A17, id=SFR_A17$patientId, family=binomial(link ="logit"), scale.fix = FALSE, tol = 0.001, maxiter = 25,corstr = "independence")
      Estimates17<-data.frame(GeeS217$coefficients[2])
      colnames(Estimates17)<-"GeeS217.coefficients"
      GeeAll17<-rbind(GeeAll17, Estimates17)
    })
  }
}
})
###################################
#Write Output Results
###################################

#2017

#try({GeeAllC17<-data.frame(cbind(FACA17[,1], GeeAll17))})

GeeAllC17 <- cbind(rownames(GeeAll17), data.frame(GeeAll17, row.names=NULL))

colnames(GeeAllC17)<-c("Provfs","Estimates")

GeeAllC17$Provfs<-gsub( "Provfs", "", GeeAllC17$Provfs)

GeeAllC17<-as.data.frame(GeeAllC17)

#### End Fix for loss of facilities

colnames(GeeAllC17)<-c("Provfs", "Estimates")

M17<-merge(SFRAB17,GeeAllC17)

SFR_tails17F<-data.frame(SFR_tailsp)

########
#   ADD in removed rows for AVF = 0 and 1
###############################
ntail<-nrow(SFR_tails17F)
SFR_tails17F$Groups<-11
SFR_tails17F$ZbetaP17<-(-30)
#SFR_tails17F$Estimates<--17

SFR_tails17F$Estimates<-(-17+SFR_tails17F$pppw_f*34)
SFR_tails17F$pppw_f<-NULL

SFR_tails17F$pppw<-as.numeric(SFR_tails17F$pppw)



#SFR_tails17F$pppw<-as.numeric(SFR_tails17F$pppw)
#for(i in 1:ntail) {
#  if(SFR_tails17F$pppw[[i]]==1){SFR_tails17F$Estimates[[i]]<-17}
#  if(SFR_tails17F$pppw[[i]]==0){SFR_tails17F$Estimates[[i]]<--17}
#}
#SFR_tails17F$Estimates<-NULL
#SFR_tails17F$Groups<-NULL
#SFR_tails17F$Zbetas<-NULL

#########################################
# Merge excluded facilities with M matrix scores calculated by model and sort
########################################

names(M17)[11]<-"Zbetas"
names(SFR_tails17F)[2]<-"Provfs"
names(SFR_tails17F)[11]<-"Zbetas"

npr<-nrow(M17)

M2<-M17[,c(1,2,3,4,11,12)]
T2<-SFR_tails17F[,c(2,1,3,4,11,12)]

M5<-rbind(M2,T2)

M5_sort<-arrange(M5,Provfs,patientId,year)



####################################
##  Calculate exponentials
####################################


M5_sort$e_alpha<-exp(M5_sort$Estimates)
M5_sort$e_Zbetas<-exp(M5_sort$Zbetas)
M5_sort$e_log<-(M5_sort$e_alpha*M5_sort$e_Zbetas)/(1+M5_sort$e_alpha*M5_sort$e_Zbetas)

############################################
# Calculate Numerator scores
##########################################

Mrows<-nrow(M5_sort)
GCrows<-nrow(GeeAllC17)

#####################################
#Set up matrices
############################
M5_fac<-data.frame(M5_sort$Provfs,M5_sort$Estimates,M5_sort$e_alpha)
M5_faca<-distinct(M5_fac)
M5_facam<-as.matrix(as.numeric(M5_faca[,3]))

M10<-as.matrix(data.frame(M5_sort$e_alpha,M5_sort$e_Zbetas))

#mfunL<-function(m1,m2,i)
#{
#  x1<-m1[i,1]
##### Compute the logistic function
#  x1*m2[,2]/(1+x1*m2[,2])
#}

mfunL<-function(m)
{
  # Compute the logistic function
  m[,1]*m[,2]/(1+m[,1]*m[,2])
}

########################################
# Run loop
###############################
nfac<-nrow(M5_faca)
Mrows<-nrow(M5_sort)
NumTest<-vector()

###Clean memory
mem_used()
#remove(FAC17)
#remove(M2)
#remove(M5_fac)
#remove(T2)
#remove(SFR_sort17FG)
#remove(M17)
#remove(M5_sort)
#remove(SFRAB17)
#remove(GeeS217)
#remove(M5)
gc()

mem_used()


#system.time({
#  NumTest<-foreach(i=1:750, .combine=rbind) %dopar% {
#    x<-M5_facam[i,1]
#    M10m<-matrix(rep(x,Mrows),ncol = 1)
#    M10m6<-matrix(c(M10m,M10[,2]), ncol = 2)
#    eLOG<-mfunL(M10m6)
#    sumLog<-sum(eLOG)
#  }
#})

##########split loop into easy pieces

N5<-floor(nfac/10)

N5rem<-nfac-(10*N5)

Slog<-vector()
N5remA<-0
system.time({
  for(j in 1:10) {
    if(j==10) {N5remA<-N5rem}
    s1<-(1+(j-1)*N5)
    f1<-(N5*j+N5remA)
    NumTest<-foreach(i=s1:f1, .combine=rbind) %dopar% {
      x<-M5_facam[i,1]
      M10m<-matrix(rep(x,Mrows),ncol = 1)
      M10m6<-matrix(c(M10m,M10[,2]), ncol = 2)
      eLOG<-mfunL(M10m6)
      sumLog<-sum(eLOG)
    }
    Slog<-c(Slog,NumTest)
  }
})

mem_used()
#Divide by total number of patient months to get SRR score per facility

SFR<-NumTest/npr

SFR2<-Slog/npr

M5_SFR<-cbind(M5_faca,SFR2)
M5_SFR$X1<-NULL
colnames(M5_SFR)<-c("Provfs","Estimates","e_alpha","PPPW")
M5_PPPW<-M5_SFR
M5_PPPW$PPPW<-M5_PPPW$PPPW*100

#####################################################################
# WRITE Output Files
#######################################################################

write.csv(M5_PPPW, "~/M5_PPPW.csv")


write.csv(GeeAllC17, "~/GeeAllC17.csv")

write.csv(GeeAll17, "~/GeeAll17.csv")

write.csv(SFRAB17, "~/SFRAB17.csv")
write.csv(FACA17, "~/FACA17.csv")
write.csv(ZbetaP17, "~/ZbetaP17.csv")

write.csv(m17, "~/m17.csv")


write.csv(M17, "~/M17.csv")