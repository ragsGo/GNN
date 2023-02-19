phendat<-read.csv("MiceBLphen.csv",header=T,sep=",",na.strings = "..")
#Trait 3, change to 2
# print(phendat)
t3naid<-which(is.na(phendat$t3))

#print(phent3<-phendat$t3)
#phent3<-phendat$t3[-t3naid]
phent3<-phendat$t3
# print(phent3)

# 52844
nIter <- 1000
burnIn <- 500

                     # WHEAT # MiceBL # SNP  # MiceBLHot # SNPHot # WHEAT1Hot
# split_point <- 1451   # 479   # 1451   # 2326 # 1451      # 2326   # 479  # 3620 # 4071 # 8140 # 8140
# last_point  <- 1814   # 599   # 1814   # 3226 # 1814      # 3226   # 599  # 4520 # 9040 # 9040
# ncol        <- 20693  # 1280  # 10347  # 9724 # 20693     # 19447  # 2559 # 9809 # 9809 # 9724
gendat <- matrix(scan("MiceBLgen.csv",sep=",",skip=1), ncol=10347,byrow=T)
indid<-gendat[,1]

# gendat<-gendat[-t3naid,-1]
dim<-dim(gendat)
allfreq<-colSums(gendat)/(dim[1]*2)
freqdel1<-as.numeric(which(allfreq<0.01))
freqdel2<-as.numeric(which(allfreq>0.99))
ngendatred<-gendat #matrix(gendat[,-c(freqdel1,freqdel2)],nrow=dim[1])
#print(ngendatred)
dimred<-dim(ngendatred)

rm(gendat)

set.seed(1)
randtestid1<-sort(sample.int(dim[1],round(dim[1]*0.3),replace=F))
xtrain1<-ngendatred[-randtestid1,]
xtest1<-ngendatred[randtestid1,]
ytrain1<- phent3
print(randtestid1)
ytrain1[randtestid1]<-NA
ytest1<-phent3[randtestid1]
print(dim(xtrain1))
print((ytrain1))
set.seed(2)
randtestid2<-sort(sample.int(dim[1],round(dim[1]*0.3),replace=F))
xtrain2<-ngendatred[-randtestid2,]
xtest2<-ngendatred[randtestid2,]
ytrain2<- phent3
ytrain2[randtestid2]<-NA
ytest2<-phent3[randtestid2]

set.seed(3)
randtestid3<-sort(sample.int(dim[1],round(dim[1]*0.3),replace=F))
xtrain3<-ngendatred[-randtestid3,]
xtest3<-ngendatred[randtestid3,]
ytrain3<-phent3
ytrain3[randtestid3]<-NA
ytest3<-phent3[randtestid3]
set.seed(4)
randtestid4<-sort(sample.int(dim[1],round(dim[1]*0.3),replace=F))
xtrain4<-ngendatred[-randtestid4,]
xtest4<-ngendatred[randtestid4,]
ytrain4<- phent3
ytrain4[randtestid4]<-NA
ytest4<-phent3[randtestid4]


set.seed(5)
randtestid5<-sort(sample.int(dim[1],round(dim[1]*0.3),replace=F))
xtrain5<-ngendatred[-randtestid5,]
xtest5<-ngendatred[randtestid5,]

ytrain5<- phent3
ytrain5[randtestid1]<-NA
ytest5<-phent3[randtestid5]

#rm(ngendatred)
n_iter <- 1000
burn_in <- 500
library(BGLR)
#2# Setting the linear predictor
G <- tcrossprod(ngendatred)
#EVD <- eigen(G)
fm <- BGLR(y=ytrain1, ETA=list(G=list(K=G, model='RKHS')),
             nIter=nIter, burnIn=burnIn, saveAt='eig_'	 )
#fm <- BGLR(y=ytrain1, ETA=list(G=list(V=EVD$vectors, d=EVD$values, model='RKHS')),nIter=nIter, burnIn=burnIn, saveAt='eigb_')
yHat1<-fm$yHat
yhat1vals = yHat1[randtestid1]
RMSEP1<-sum((ytest1-yhat1vals)^2)/length(ytest1)

G <- tcrossprod(ngendatred)
fm <- BGLR(y=ytrain2, ETA=list(G=list(K=G, model='RKHS')),nIter=nIter, burnIn=burnIn, saveAt='eig_'	 )
# EVD <- eigen(G)
# fm <- BGLR(y=ytrain2, ETA=list(G=list(V=EVD$vectors, d=EVD$values, model='RKHS')),nIter=nIter, burnIn=burnIn, saveAt='eigb_')
yHat2<-fm$yHat
yhat2vals = yHat2[randtestid2]
RMSEP2<-sum((ytest2-yhat2vals)^2)/length(ytest2)

G <- tcrossprod(ngendatred)
# EVD <- eigen(G)
# fm <- BGLR(y=ytrain3, ETA=list(G=list(V=EVD$vectors, d=EVD$values, model='RKHS')),nIter=nIter, burnIn=burnIn, saveAt='eigb_')
fm <- BGLR(y=ytrain3, ETA=list(G=list(K=G, model='RKHS')),nIter=nIter, burnIn=burnIn, saveAt='eig_'	 )
yHat3<-fm$yHat
yhat3vals = yHat3[randtestid3]
RMSEP3<-sum((ytest3-yhat3vals)^2)/length(ytest3)

G <- tcrossprod(ngendatred)
# EVD <- eigen(G)
# fm <- BGLR(y=ytrain4, ETA=list(G=list(V=EVD$vectors, d=EVD$values, model='RKHS')),nIter=nIter, burnIn=burnIn, saveAt='eigb_')
fm <- BGLR(y=ytrain4, ETA=list(G=list(K=G, model='RKHS')),nIter=nIter, burnIn=burnIn, saveAt='eig_'	 )
yHat4<-fm$yHat
yhat4vals = yHat4[randtestid4]
RMSEP4<-sum((ytest4-yhat4vals)^2)/length(ytest4)

G <- tcrossprod(ngendatred)
# EVD <- eigen(G)
# fm <- BGLR(y=ytrain5, ETA=list(G=list(V=EVD$vectors, d=EVD$values, model='RKHS')),nIter=nIter, burnIn=burnIn, saveAt='eigb_')
fm <- BGLR(y=ytrain5, ETA=list(G=list(K=G, model='RKHS')),nIter=nIter, burnIn=burnIn, saveAt='eig_'	 )
yHat5<-fm$yHat
yhat5vals = yHat5[randtestid5]
RMSEP5<-sum((ytest5-yhat1vals)^2)/length(ytest5)

print(RMSEP1)
print(RMSEP2)
print(RMSEP3)
print(RMSEP4)
print(RMSEP5)