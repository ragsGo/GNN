                     # Wheat # MiceBL # SNP  # MiceBLHot # SNPHot # WHEAT1Hot
split_point <- 479   # 479   # 1451   # 2326 # 1451      # 2326   # 479  # 3620 # 4071 # 8140 # 8140
last_point  <- 599   # 599   # 1814   # 3226 # 1814      # 3226   # 599  # 4520 # 9040 # 9040
ncol        <- 1280  # 1280  # 10347  # 9724 # 20693     # 19447  # 2559 # 9809 # 9809 # 9724

n_iter <- 1000
burn_in <- 500


dat <- matrix(scan("/home/rags/gnn/gnn/csv-data/WHEAT_combined.csv",sep=","), ncol=ncol, byrow=T)

phen <- dat[,1]-mean(dat[,1])
x <- dat[,3:ncol]

dim<-dim(x)

ysc <- dat[,1]-mean(dat[,1])
# print(ysc)
#Scaled allele frequencies
 scale_apply <- function(x) {
    apply(x, 2, function(y) (y - mean(y))/sd(y))
}
xsc<-scale_apply(x)
xsc<-x

xtrain<-xsc[1:split_point,]
xtest<-xsc[split_point:last_point,]
rm(x)

#Bayesian prediction treated as imputation of missing data
ysctest<-ysc[split_point:last_point]
ysc[split_point:last_point]<-NA

#1# Loading and preparing the input data
library(BGLR)
#2# Setting the linear predictor
ETA<-list(list(X=xsc, model='BL'))
#set.seed(1)
#3# Fitting the model

fm<-BGLR(y=ysc, ETA=ETA, nIter=n_iter, burnIn=burn_in, thin=10)
#save(fm,file='BLASSOQTLMAS.Rdata')

yHat<-fm$yHat

RMSEP<-sum((ysctest-yHat[split_point:last_point])^2)/length(yHat[split_point:last_point])

RMSAEP<-sum(abs(ysctest-yHat[split_point:last_point]))/length(yHat[split_point:last_point])
print(RMSEP)
#print(cor(ysctest, yHat[split_point:last_point]))
varU=scan('ETA_1_lambda.dat')
varE=scan('varE.dat')
print(mean(varE))
print(mean(varU))
h2=varU/(varU+varE)

print(mean(h2))
