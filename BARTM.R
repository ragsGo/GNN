phendat<-read.csv("SNPphen.csv",header=T,sep=",",na.strings = ".")
#Trait 3, change to 2
#print(phendat)
t3naid<-which(is.na(phendat$t3))

#print(phent3<-phendat$t3)
#phent3<-phendat$t3[-t3naid]
phent3<-phendat$t3
#print(phent3)

# 52844
gendat <- matrix(scan("SNPgen.csv",sep=",",skip=1),ncol=9724,byrow=T)
indid<-gendat[,1]

# gendat<-gendat[-t3naid,-1]
dim<-dim(gendat)
allfreq<-colSums(gendat)/(dim[1]*2)
freqdel1<-as.numeric(which(allfreq<0.01))
freqdel2<-as.numeric(which(allfreq>0.99))
ngendatred<-gendat #matrix(gendat[,-c(freqdel1,freqdel2)],nrow=dim[1])
print(ngendatred)
dimred<-dim(ngendatred)
print(dimred)

rm(gendat)

set.seed(1)
randtestid1<-sort(sample.int(dim[1],round(dim[1]*0.3),replace=F))
xtrain1<-ngendatred[-randtestid1,]
xtest1<-ngendatred[randtestid1,]
ytrain1<-phent3[-randtestid1]
ytest1<-phent3[randtestid1]

set.seed(2)
randtestid2<-sort(sample.int(dim[1],round(dim[1]*0.3),replace=F))
xtrain2<-ngendatred[-randtestid2,]
xtest2<-ngendatred[randtestid2,]
ytrain2<-phent3[-randtestid2]
ytest2<-phent3[randtestid2]

set.seed(3)
randtestid3<-sort(sample.int(dim[1],round(dim[1]*0.3),replace=F))
xtrain3<-ngendatred[-randtestid3,]
xtest3<-ngendatred[randtestid3,]
ytrain3<-phent3[-randtestid3]
ytest3<-phent3[randtestid3]

set.seed(4)
randtestid4<-sort(sample.int(dim[1],round(dim[1]*0.3),replace=F))
xtrain4<-ngendatred[-randtestid4,]
xtest4<-ngendatred[randtestid4,]
ytrain4<-phent3[-randtestid4]
ytest4<-phent3[randtestid4]

set.seed(5)
randtestid5<-sort(sample.int(dim[1],round(dim[1]*0.3),replace=F))
xtrain5<-ngendatred[-randtestid5,]
xtest5<-ngendatred[randtestid5,]
ytrain5<-phent3[-randtestid5]
ytest5<-phent3[randtestid5]

rm(ngendatred)


library(BayesTree)
bartFit1 <- bart(xtrain1,ytrain1,xtest1,ntree=100,k=6,nskip=25000,ndpost=50000,
               keepevery=10,sigest=sd(ytrain1))
save(bartFit1,file="BartoutK6N1001.Rdata")
RMSEP1<-sum((ytest1-bartFit1$yhat.test.mean)^2)/length(ytest1)
rm(bartFit1)

bartFit2 <- bart(xtrain2,ytrain2,xtest2,ntree=100,k=6,nskip=25000,ndpost=50000,
               keepevery=10,sigest=sd(ytrain2))
save(bartFit2,file="BartoutK6N1002.Rdata")
RMSEP2<-sum((ytest2-bartFit2$yhat.test.mean)^2)/length(ytest2)
rm(bartFit2)

bartFit3 <- bart(xtrain3,ytrain3,xtest3,ntree=100,k=6,nskip=25000,ndpost=50000,
               keepevery=10,sigest=sd(ytrain3))
save(bartFit3,file="BartoutK6N1003.Rdata")
RMSEP3<-sum((ytest3-bartFit3$yhat.test.mean)^2)/length(ytest3)
rm(bartFit3)

bartFit4 <- bart(xtrain4,ytrain4,xtest4,ntree=100,k=6,nskip=25000,ndpost=50000,
               keepevery=10,sigest=sd(ytrain4))
save(bartFit4,file="BartoutK6N1004.Rdata")
RMSEP4<-sum((ytest4-bartFit4$yhat.test.mean)^2)/length(ytest4)
rm(bartFit4)

bartFit5 <- bart(xtrain5,ytrain5,xtest5,ntree=100,k=6,nskip=25000,ndpost=50000,
               keepevery=10,sigest=sd(ytrain5))
save(bartFit5,file="BartoutK6N1005.Rdata")
RMSEP5<-sum((ytest5-bartFit5$yhat.test.mean)^2)/length(ytest5)
rm(bartFit5)


print(RMSEP1)
print(RMSEP2)
print(RMSEP3)
print(RMSEP4)
print(RMSEP5)


