import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D

# based on https://www.tutorialspoint.com/keras/keras_convolution_neural_network.htm
# and https://github.com/patwa67/CNNGWP/blob/master/CNNGWPBOQTLMAS2010.R
def load_data(filename):
    with open(filename) as fp:
        line = fp.readline()
        column_count = len(line.split(","))
    value_columns = [str((i + 1)) for i in range(column_count - 1)]
    labels = ["value"] + value_columns
    df = pd.read_csv(filename, names=labels)
    df2 = pd.DataFrame()
    df2["x"] = [np.asarray(x, dtype=np.float32) for x in df[value_columns].values]
    df2["y"] = df["value"]
    return df2

losses = []
corrs = []
test_case = "keras-patwa"


class LossAndErrorPrintingCallback(keras.callbacks.Callback):
   def on_epoch_end(self, epoch, logs=None):
      losses.append(logs["val_loss"])
      pred = np.amax(model.predict(np.expand_dims(np.asarray([np.asarray(x, dtype=np.float32) for x in x_test]), -1)), 1)
      corrs.append(np.corrcoef(pred, y_test)[0][1])
      if epoch in [0, 249]:

         plt.scatter(range(len(pred[:100])), pred[:100], label="Predicted")
         plt.scatter(range(len(y_test[:100])), y_test[:100], label="Correct")
         plt.title(f"Predicted and correct (first 100 subjects) - Epoch {'Final' if epoch != 1 else 0}")
         plt.xlabel("Subject")
         plt.ylabel("Value")
         plt.legend(loc='upper right')
         plt.savefig(f"images/cnn-1-pred-vs-correct-epoch-final-{test_case}.png")
         plt.close()


data = load_data("SNP.csv")

x_train, x_test, y_train, y_test = train_test_split(data["x"].to_numpy(), data["y"].to_numpy())

model = Sequential()
model.add(Conv1D(filters=60, kernel_size=35, padding="same", strides=2,
                 activation='linear', input_shape=(len(x_train[0]), 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='linear'))
model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.Adam(lr=0.00025), metrics=['mse'])
model.fit(
   np.asarray(np.expand_dims(np.asarray([np.asarray(x, dtype=np.float32) for x in x_train]), -1), dtype=np.float32), np.asarray(y_train, dtype=np.float32),
   batch_size=48,
   epochs=250,
   verbose=1,
   callbacks=[LossAndErrorPrintingCallback()],
   validation_data=(np.expand_dims(np.asarray([np.asarray(x, dtype=np.float32) for x in x_test]), -1), np.asarray(y_test, dtype=np.float32))
)

score = model.evaluate(np.expand_dims(np.asarray([np.asarray(x, dtype=np.float32) for x in x_test]), -1), np.asarray(y_test, dtype=np.float32), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(losses, label="Loss")
plt.title("Loss per epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc='upper right')

plt.savefig(f"images/cnn-1-loss-{test_case}.png")
plt.close()

plt.plot(corrs, label="Correlation")
plt.title("Correlation between prediction and correct per epoch")
plt.xlabel("Epoch")
plt.ylabel("Correlation")
plt.legend(loc='upper right')
plt.savefig(f"images/cnn-1-correlation-{test_case}.png")
plt.close()
