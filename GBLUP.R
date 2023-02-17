library(BGLR)
# data(mice)
#                    # Wheat# Mice # SNP #
# split_point <- 1451  # 479 # 1451 # 2326 # 3620 # 4071 # 8140 # 8140
# last_point <-1814  # 599 # 1814 # 3226# 4520 # 9040 # 9040
# ncol <- 10347 # 1280 # 10347 # 9724 # 9900 # 9809 # 9809

                     # WHEAT # MiceBL # SNP  # MiceBLHot # SNPHot # WHEAT1Hot
split_point <- 479   # 479   # 1451   # 2326 # 1451      # 2326   # 479  # 3620 # 4071 # 8140 # 8140
last_point  <- 599   # 599   # 1814   # 3226 # 1814      # 3226   # 599  # 4520 # 9040 # 9040
ncol        <- 1280  # 1280  # 10347  # 9724 # 20693     # 19447  # 2559 # 9809 # 9809 # 9724

nIter <- 1000
burnIn <- 500

dat <- matrix(scan("csv-data/WHEAT_combined.csv", sep=","), ncol=ncol, byrow=T)
#dat = read.table("MiceBL.txt",sep=",")

# last_point = dim(dat)[1]
# ncol = dim(dat)[2]
# print(ncol)
# print(last_point)
X <- dat[,2:ncol]
y <- dat[,1] #-mean(dat[,1])

X <- scale(X) #/sqrt(ncol(X))

xtrain<-X[1:split_point,]
xtest<-X[split_point:last_point,]

ysctest <- y[split_point:last_point]
ytrain <- y
ytrain[split_point:last_point] <- NA

G <- tcrossprod(X)
#EVD <- eigen(G)

fm3 <- BGLR(y=ytrain, ETA=list(G=list(K=G, model='RKHS')),
            nIter=nIter, burnIn=burnIn,)
# fm3 <- BGLR(y=ytrain, ETA=list(G=list(V=EVD$vectors, d=EVD$values, model='RKHS')),
#             nIter=nIter, burnIn=burnIn, saveAt='eigb_')
# varE <- scan('eigb_varE.dat')
# varU <- scan('eigb_ETA_G_varU.dat')
# h2_3 <- varU/(varU+varE)

# print(h2_3) are you done?

yHat<-fm3$yHat

RMSEP<-sum((ysctest-yHat[split_point:last_point])^2)/length(yHat[split_point:last_point])
print(RMSEP)
print(cor(ysctest, yHat[split_point:last_point]))