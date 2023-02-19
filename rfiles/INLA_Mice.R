library(Matrix)

#Corrected data
dat = read.table("/home/rags/gnn/gnn/csv-data/WHEAT_combined.csv",sep=",")
n = dim(dat)[1]
print(n)
p = dim(dat)[2]
print(p)
X = Matrix(as.matrix(dat[,3:p]),sparse=TRUE) #Skip col 2
y = dat[,1]
#y = yBLcorr
split = 479
ytest <- y[c(split+1):n]
y[c(split+1):n] <- NA

dim<-dim(X)
allfreq<-colSums(X)/(dim[1]*2)
#Only if not already pruned for MAF
#freqdel1<-which(allfreq<0.01)
#freqdel2<-which(allfreq>0.99)
#X<-X[,-c(freqdel1,freqdel2)]
#allfreq<-allfreq[-c(freqdel1,freqdel2)]
m<-length(allfreq)

#Calculate G matrix and its inverse
M<-X-1
Z<-sweep(M,2,2*(allfreq-0.5),"-")
G<-(Z%*%t(Z))/(2*sum(allfreq*(1-allfreq)))
library(corpcor)
library(INLA)
#inla.setOption(pardiso.license="CA2D45D5A831F004F5853AC0C1A97028F7F416981DE83F43BF2F8153") 

#inla.pardiso.check()
G<-make.positive.definite(G)
Ginv<-as(as(as(solve(G), "dMatrix"), "generalMatrix"), "TsparseMatrix")
Ginv[ abs(Ginv) < sqrt(.Machine$double.eps) * max(Ginv) ] = 0

#Set up data frame for INLA
ID<-matrix(1:n,n,1)
dataf<-data.frame(cbind(ID,y))
names(dataf) <- c("ID","y")

#Fit INLA
model = y ~ 1 + f(ID, model="generic0", Cmatrix=Ginv)

fit = inla(model, data=dataf, verbose=TRUE,
           control.compute=list(config=T,dic=T,mlik=T,cpo=T))

summary(fit)
print('here')
#Heritability
#(1/16.27)/((1/16.27)+(1/4.60))
VarG <- 1/fit$summary.hyperpar$mean[2]
VarE <- 1/fit$summary.hyperpar$mean[1]
print('Her--')
print(her <- VarG/(VarG + VarE))
print('here1')
MSEtest <- sum((fit$summary.fitted.values$mean[c(split+1):n] - ytest)^2)/length(ytest)
print('MSE')
print(MSEtest)
#print('corr_kendal==')
#print(cor.test(fit$summary.fitted.values$mean[c(split+1):n],ytest, method = 'kendall'))
print('corr==')
print(cor(fit$summary.fitted.values$mean[c(split+1):n],ytest))

#Sample from posterior, if we want to calculate 95% CI of MSE and her
fit.samples = inla.posterior.sample(100, fit)