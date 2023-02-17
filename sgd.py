# Adapted from https://colab.research.google.com/drive/1_WxDPLGkJY3qJ-PK0J1YjATaZz35efmk#scrollTo=n5NEwC7tYvSo
from math import sqrt

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data(filename, bits=None):
    with open(filename) as fp:
        line = fp.readline()
        column_count = len(line.split(","))
    value_columns = [str((i + 1)) for i in range(column_count - 1)]
    labels = ["value"] + value_columns
    df_whole = pd.read_csv(filename, names=labels)
    if bits is not None:
        df_whole = df_whole[["value"] + bits]

    train_set = 2326
    valid_set = 0
    test_set = df_whole.shape[0] - valid_set

    df_xtrain = df_whole.iloc[0:train_set, 1:]
    df_ytrain = df_whole['value'][0:train_set]
    df_ytrain = df_ytrain - df_ytrain.mean()

    df_xtest = df_whole.iloc[train_set:test_set, 1:]
    df_ytest = df_whole['value'][train_set:test_set]
    df_ytest = df_ytest - df_ytest.mean()

    return df_xtrain, df_ytrain, df_xtest, df_ytest
#
#
# def rmse(prediction, ground_truth):
#     prediction = prediction[ground_truth.nonzero()].flatten()
#     ground_truth = ground_truth[ground_truth.nonzero()].flatten()
#     return sqrt(mean_squared_error(prediction, ground_truth))
#
#
# class Recommender:
#     def __init__(self, n_epochs=200, n_latent_features=3, lmbda=0.1, learning_rate=0.001):
#         self.n_epochs = n_epochs
#         self.n_latent_features = n_latent_features
#         self.lmbda = lmbda
#         self.learning_rate = learning_rate
#
#     def predictions(self, P, Q):
#         return np.dot(P.T, Q)
#
#     def fit(self, train, test):
#         m, n = train.shape
#
#         self.P = 3 * np.random.rand(self.n_latent_features, m)
#         self.Q = 3 * np.random.rand(self.n_latent_features, n)
#
#         self.train_error = []
#         self.val_error = []
#
#         users, items = train
#
#         for epoch in range(self.n_epochs):
#             for u, i in zip(users, items):
#                 error = train[u, i] - self.predictions(self.P[:, u], self.Q[:, i])
#                 self.P[:, u] += self.learning_rate * (error * self.Q[:, i] - self.lmbda * self.P[:, u])
#                 self.Q[:, i] += self.learning_rate * (error * self.P[:, u] - self.lmbda * self.Q[:, i])
#
#             train_rmse = rmse(self.predictions(self.P, self.Q), train)
#             val_rmse = rmse(self.predictions(self.P, self.Q), test)
#             self.train_error.append(train_rmse)
#             self.val_error.append(val_rmse)
#
#         return self
#
#     def predict(self, train, user_index):
#         y_hat = self.predictions(self.P, self.Q)
#         predictions_index = np.where(train[user_index, :] == 0)[0]
#         return y_hat[user_index, predictions_index].flatten()

def create_sgd(filename, epochs=1000, eta0=(0.0000001, 0.0001), alpha=(0.00001, 0.001), learning_rate=('constant', 'optimal', 'invscaling', 'adaptive')):
    xtrain, ytrain, xtest, ytest = load_data(filename)
    # reg = make_pipeline(StandardScaler(),
    #                  SGDRegressor(max_iter=1000, eta0=0.00001))
    def run_sgd(trial):
        _eta = trial.suggest_float("eta0", *eta0)
        _alpha = trial.suggest_float("alpha", *alpha)
        _learning_rate = trial.suggest_categorical("learning_rate", list(learning_rate))
        _fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        reg = SGDRegressor(max_iter=epochs, eta0=_eta, learning_rate=_learning_rate, alpha=_alpha, fit_intercept=_fit_intercept)
        # clf = Recommender()
        reg.fit(xtrain, ytrain)
        train_pred = reg.predict(xtrain)
        test_pred = reg.predict(xtest)
        print(f"Train loss: {mean_squared_error(ytrain, train_pred)}; Test loss: {mean_squared_error(ytest, test_pred)}", )
        # trial.report(mean_squared_error(ytest, test_pred), epochs)
        return mean_squared_error(ytest, test_pred)
    return run_sgd


if __name__ == "__main__":
    # clf = SGDRegressor(max_iter=2000)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        create_sgd(
            "SNP.csv",
        ),
        n_trials=1000,
        n_jobs=-1
    )
