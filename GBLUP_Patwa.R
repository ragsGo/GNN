library(BGLR)

#data(mice)
#dim(mice.A)
#dim(mice.pheno)

#phendat = mice.pheno
#phendat$Date.Month = as.factor(phendat$Obesity.Date.Month)
#phendat$Date.Year = as.factor(phendat$Obesity.Date.Year)

#Phenotype needs to be corrected for mean and fixed effects
#lmBL1 = lm(Obesity.BodyLength ~ Date.Month+Date.Year+GENDER+CoatColour+CageDensity+
#          Litter,data=phendat)
#summary(lmBL1)
#anova(lmBL1)
#yBLcorr = resid(lmBL1)
#hist(yBLcorr,25)

#Corrected data
dat = read.table("/home/rags/gnn/gnn/csv-data/pig4.csv",sep=",")
n = dim(dat)[1]
print(n)
Xp = dim(dat)[2]
print(Xp)
X = dat[,2:Xp] #Skip col 2
y = dat[,1]

X = scale(X)

MSEfold = c(1:8)

for (fold in 1:8){
set.seed(fold) #Needs to be changed for the different folds
randind = sample(c(1:n),replace = FALSE)
split = 479 #2175 #2521 3152 #2400
Xperm = X[randind,]
ncolXperm = dim(X)[2]
 xtrain<-X[1:split,]
xtest<-X[c(split+1):n,]

yperm = y[randind]
ytest <- y[c(split+1):n]
ytrain <- yperm
#ytrain <- y
ytrain[c(split+1):n] <- NA

G <- tcrossprod(X)/ncolXperm
#EVD <- eigen(G)
niter = 1000
burnin = 500

 fm1 <- BGLR(y=ytrain, ETA=list(G=list(K=G, model='RKHS')),
            nIter=niter, burnIn=burnin, saveAt='default_')

yHat<-fm1$yHat

RMSEP<-sum((ytest-yHat[c(split+1):n])^2)/length(yHat[c(split+1):n])
MSEfold[fold] = RMSEP



#Spearman rank correlation
#print(cor.test(ytest, yHat[c(split+1):n], method = 'spearman'))
#Distance correlation
library('energy')
print(dcor(ytest, yHat[c(split+1):n]))
}
print(RMSEP)
print(cor(ytest, yHat[c(split+1):n]))
 #print(mean(MSEfold))

#Heritability
varU=scan('default_ETA_G_varU.dat')
varE=scan('default_varE.dat')
h2=varU/(varU+varE)
her <- fm1$ETA[[1]]$varU/(fm1$varE+fm1$ETA[[1]]$varU)
#her - cov(predval)/cov(predval)+cov(trueval)
print(her)
print(mean(h2))
