#Read phenotype data incl inb coef
phendat = read.table("p1_data_001.txt",header=T)
inbcoef = phendat$F[22663:24960]

#Read marker data as characters without header
mrkdat = scan("p1_mrk_001.txt",what="character")
inds = seq(1,length(mrkdat),2)
mrks = seq(2,length(mrkdat),2)
allinds = as.numeric(mrkdat[inds])
allmrks = mrkdat[mrks]

markmat = matrix(0,ncol=5010,nrow=length(allinds))
for (i in 1:length(allinds)){ 
markmat[i,] = as.numeric((unlist(strsplit(allmrks[i],NULL))))
}

#Check for no missing data
which(markmat==5)

#Convert SNPs to homo/het data
markmat[which(markmat==3)] = 1
markmat[which(markmat==4)] = 1
markmat[which(markmat==2)] = 0

#Remove data to free up memory
rm(mrkdat)

library(torch)

#Perform Discrete Fast Fourier Transformation of SNP data
numfft = length(as_array(torch_abs(torch_fft_rfft(markmat[1,])$real)))
ftmat = matrix(0,ncol=numfft,nrow=length(allinds))
for (i in 1:length(allinds)){ 
ftmat[i,] = as_array(torch_abs(torch_fft_rfft(markmat[i,])$real))
}

#Remove data to free up memory
rm(markmat)

#Set up train and validation data
library(luz)
inb_dataset <- dataset(
  
  name = "inb_dataset",
  
  initialize = function(inb) { 
    
    # continuous input data (x_cont)   
    x_cont <- ftmat %>%
      as.matrix()
    self$x_cont <- torch_tensor(x_cont)

    # target data (y)
    F <- as.numeric(inbcoef)
    self$y <- torch_tensor(F)
    
  },
  
  .getitem = function(i) {
     list(x_cont = self$x_cont[i, ], y = self$y[i])
    
  },
  
  .length = function() {
    self$y$size()[[1]]
  }
 
)

ntrain = 1500
nvalid = 400
train_indices <- sample(1:nrow(ftmat), ntrain)

train_ds <- inb_dataset(inb[train_indices, ])
tot_ind <- 1:nrow(ftmat)
validtest_ind <- tot_ind[-train_indices]
valid_ind <- sample(validtest_ind, nvalid)
not_test_ind <- c(train_indices,valid_ind)
valid_ds <- inb_dataset(inb[valid_ind, ])
testftmat <- ftmat[-not_test_ind,]
testF <- inbcoef[-not_test_ind]

train_dl <- train_ds %>% dataloader(batch_size = 250, shuffle = TRUE)
valid_dl <- valid_ds %>% dataloader(batch_size = 400, shuffle = FALSE)

#The Mulit Layer Perceptron
model <- nn_module(
  initialize = function() {
    self$features <- nn_sequential(
  nn_linear(2506, 32),
  nn_relu(),
  nn_linear(32, 16),
  nn_relu()
  )
    self$classifier <- nn_sequential(
      nn_linear(16,1)
    )
  },
  forward = function(x) {
    x <- self$features(x)
    x <- self$classifier(x)$squeeze(2)
    x
  }
)


model <- model %>% 
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_mse())
  )


fitted <- model %>%
  fit(train_dl,
    epochs = 1000, valid_data = valid_dl,
    callbacks = list(
      luz_callback_keep_best_model(monitor = "valid_loss"),
      luz_callback_early_stopping(monitor = "valid_loss",patience = 8),
      luz_callback_lr_scheduler(
        lr_one_cycle,max_lr = 0.001,epochs = 100,
        steps_per_epoch = length(train_dl)),
      luz_callback_model_checkpoint(path = "models_complex/")
    ),
    verbose = TRUE
  )

plot(fitted)

#Test predictions
preds <- predict(fitted,testftmat)

#Test correlation
cor(as_array(preds),testF)
#Distance test correlation
library("energy")
dcor(as_array(preds),testF)

