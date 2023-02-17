import pandas as pd
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense,Conv1D,MaxPooling1D, Flatten 
from tensorflow.keras import regularizers
import tensorflow as tf
from bayes_opt import BayesianOptimization
filename = "SNP.csv"
with open(filename) as fp:
                line = fp.readline()
                column_count = len(line.split(","))
value_columns = [str((i+1)) for i in range(column_count-1)]
labels = ["value"] + value_columns
df_whole = pd.read_csv(filename, names=labels)

train_set = 2326
valid_set = 100



test_set = df_whole.shape[0] - valid_set

df_xtrain = df_whole.iloc[0:train_set,1:]
df_ytrain = df_whole['value'][0:train_set]  
df_ytrain = df_ytrain - df_ytrain.mean()

df_xtest = df_whole.iloc[train_set:test_set, 1:]
df_ytest = df_whole['value'][train_set:test_set]
df_ytest = df_ytest - df_ytest.mean()
#print(df_xtrain)
#print(df_ytrain)
#prints



def train(lr = 0.0025, fts = 30, kr_sz=30, pl_sz=2, reg= 0.1):

    #lr = params['lr']
    #fts = params['fts']
    #kr_sz = params['kr_sz']
    #pl_sz = params['pl_sz']
    #reg = params['reg']
    print('shape===',df_xtrain.shape[1])
    prints
    model = Sequential()
    model.add(Conv1D(filters=fts, kernel_size= int(kr_sz) ,strides= 2,padding = "same", activation='linear',input_shape=(df_xtrain.shape[1],1)))
    model.add(MaxPooling1D(pool_size=int(pl_sz)))
    model.add(Flatten())
    model.add(Dense(units=1,kernel_regularizer=regularizers.l1(l1=reg), activation='linear'))
    
    model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=lr), metrics=['mse'])
    model.fit(df_xtrain.values, df_ytrain.values, epochs=150,validation_split =0.1, batch_size=48)
    score = model.evaluate(df_xtest.values, df_ytest.values, steps=10, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return -score[1]

pbounds = {
    'reg': (0.1, 1), 
    'lr': (0.0001, 0.0003 ), 
    'fts': (20,100),
    'pl_sz':(1, 3),
    'kr_sz':(10,50)
    
    }


optimizer = BayesianOptimization(
    f=train,
    pbounds=pbounds,
    verbose=2, 
    random_state=1,
)

optimizer.maximize( n_iter=10,)


for x, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(x, res))

print(optimizer.max)
#print('results ===', model.evaluate(df_xtest, df_ytest))
