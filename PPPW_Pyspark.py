import pandas as pd
import numpy as np
import pandas as df

SFR_sort17FG = pd.read_csv("C:/Project_Data/PPPW/SFR_sort17FG_sample.csv")
# SFR_sort17FG = SFR_sort17FG[:100000]
#############################################################
# Filter Code for facilities with all 0s or 1s
############################################################

SFR_sort17Pf=SFR_sort17FG
SFR_sort17Pf=SFR_sort17Pf.drop(columns=['Unnamed: 0.1'])
#### Parallel Filter Code

FAC_f=SFR_sort17Pf['PROVFS']
FAC_fd = pd.DataFrame(FAC_f.unique(), columns=["FAC_f"])
FAC_fd['FAC_f'] = FAC_fd['FAC_f'].astype(str)
NCCN=len(FAC_fd)
SFR_tailsp=SFR_sort17Pf.columns

'''
FAC_f<-data.frame(FAC_f)

FAC_fd<-distinct(FAC_f)

FAC_fd$FAC_f<-as.character(FAC_fd$FAC_f)

NCCN<-nrow(FAC_fd)

#SFR_sort17Pf$pppw_f<-0

SFR_tailsp<-SFR_sort17Pf[0,]
'''

if(False): ### Parallelizing
    from joblib import Parallel, delayed
    import multiprocessing

    # what are your inputs, and what operation do you want to
    # perform on each input. For example...
    inputs = range(10)


    def processInput(i):
        return i * i

    num_cores = multiprocessing.cpu_count()

    results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)


'''
SFR_sort17PFF<-SFR_sort17Pf[!(SFR_sort17Pf$PROVFS %in% SFR_tailsp$PROVFS),]
SFR_sort17PFF<-data.frame(SFR_sort17PFF)
SFR_sort17Fppw<-SFR_sort17PFF
'''
SFR_sort17PFF=SFR_sort17Pf #################### NAIVE TR.
SFR_sort17Fppw=df.DataFrame(SFR_sort17PFF)


####################################
#Find Distinct Providers
####################################

# 2017
'''
FAC17<-SFR_sort17Fppw$PROVFS
FAC17<-data.frame(FAC17)
FACD17<-distinct(FAC17)
https://gist.github.com/conormm/fd8b1980c28dd21cfaf6975c86c74d07#distinct
FAC17.drop_duplicates().reset_index()   OR   FAC17.unique()
'''
FAC17=SFR_sort17Fppw['PROVFS']
FACD17=FAC17.drop_duplicates()
FACD17.reset_index(drop=True)
FACD17 = pd.DataFrame(FACD17)

#####################################
#Generate Random Groupings
######################################
'''
set.seed(25)
# 2017
FACD17$groups <- sample(1:20, size=nrow(FACD17), replace=TRUE)
colnames(FACD17)<-c("PROVFS","Group")
'''
FACD17['groups']=np.random.choice(range(1,20+1), size=len(FACD17), replace=True)
FACD17.columns=['PROVFS', 'Group']


######################################
# Merge with large original Data Frame
##########################################
'''
# 2017
SFR_sort17FG<-left_join(SFR_sort17Fppw, FACD17, by = "PROVFS")
SFR_sort17FG<-data.frame(SFR_sort17FG)
    #SFR_sort17FG$AVF<-as.integer(SFR_sort17FG$AVF)
SFR_sort17FG$PROVFS<-as.character(SFR_sort17FG$PROVFS)
SFR_sort17FG$patientId<-as.integer(SFR_sort17FG$patientId)
    #for(i in 7:11) {
    #  SFR_sort17FG[,i] <- as.integer(SFR_sort17FG[,i])
    #}
'''
SFR_sort17FG=SFR_sort17Fppw.merge(FACD17, on='PROVFS', how='left')
SFR_sort17FG=df.DataFrame(SFR_sort17FG)
SFR_sort17FG['PROVFS'] = SFR_sort17FG['PROVFS'].astype(str)
SFR_sort17FG['patientId'] = SFR_sort17FG['patientId'].astype('int16')


########################################################
# Setting up Groups for each calender year Data File
########################################################
'''
# 2017
Grps17<-list()
for(i in 1:20) {
  Grps17[[i]] <- filter(SFR_sort17FG, Group==i)
  Grps17[[i]] <- data.frame(Grps17[[i]])
  Grps17[[i]] <- arrange(Grps17[[i]],PROVFS,patientId)
  Grps17[[i]]$PROVFS <- as.factor(Grps17[[i]]$PROVFS)
}
'''
Grps17 = [None] *20
for i in range(20):
  Grps17[i] = SFR_sort17FG[SFR_sort17FG['Group']==i+1]  # Group 1..20
  Grps17[i] = Grps17[i].sort_values(by=['PROVFS', 'patientId']) #df.sort(order=["provfs", "patientId"])
  Grps17[i]['PROVFS'] = pd.Categorical(Grps17[i]['PROVFS'], categories=Grps17[i]['PROVFS'].unique(), ordered=False)

'''
#Setup Model Formula
mf <- formula(pppw ~ age + a15 + a55 + a70 +PROVFS-1)
'''

##########################
# Run Model for Each Group
##########################
#############
## GEEM model
#############
'''
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
'''

# print(i)
# xx < -nlevels(Grps17[[i]]$PROVFS)
# xx =
# GeemMod17 < -geem(mf, data=Grps17[[i]], id=Grps17[[i]]$patientId, family = binomial(
#     link="logit"), init.beta = as.vector(rep(0, xx + 4)), corstr = "independence")
# EstimatesGM17 < -data.frame(GeemMod17$beta[1:4])
# EstimatesGM17