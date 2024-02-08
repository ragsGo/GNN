from sklearn.linear_model import LinearRegression, Lasso,LassoCV,MultiTaskLassoCV
# import statsmodels.api as sm
from regressors import stats
# from scipy import stats

import statsmodels.api as sm
import numpy as np
import pandas as pd
from os import path
import os
import pathlib


from sklearn.model_selection import RepeatedKFold,GridSearchCV

import seaborn as sns

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns

from gnn.loaders.load import load_data
from gnn.networks.networks import create_network_conv1D
# from gnn.trainers.con1d_train import train_conv1D as train

def param_tuning(model, X, y, params, n):
    '''use grid search and K-Fold cross validation to find the best parameters for the regularisation models
    where n is the number of folds'''
    cv = RepeatedKFold(n_splits=n, n_repeats=3, random_state=1)
#     cv = KFold(n_splits=n, shuffle=True)
    gs_r2 = GridSearchCV(model,
                      params,
                      scoring="r2",
                      cv=cv,
                      n_jobs=-1,
                      return_train_score=True)

    gs_mse = GridSearchCV(model,
                      params,
                      scoring="neg_mean_squared_error",
                      cv=cv,
                      n_jobs=-1,
                      return_train_score=True)
    results_r2 = gs_r2.fit(X, y)
    results_mse = gs_mse.fit(X, y)
    return results_r2, results_mse

def stats_result(model_input, X_input, y_input):

    model = model_input
    X = X_input
    y = y_input
    print('stats res')
    model.fit(X, y)

    # Store the coefficients (regression intercept and coefficients) and predictions
    # coefficients = np.append(model.intercept_, model.coef_)
    coefficients = model.coef_
    mse_path = model.score(X_input, y_input)

    print('after')


    return coefficients, mse_path

def get_data(dataset, target):
    dataset = dataset
    X = dataset.drop(columns=[target])
    y = dataset[target]
    sc = StandardScaler()
    dataset_scaled = sc.fit_transform(dataset)
    dataset_scaled = pd.DataFrame(dataset_scaled, columns=dataset.columns)
    X_scaled = dataset_scaled.drop(columns=[target])
    y_scaled = dataset_scaled[target]
    return X_scaled, y_scaled

def create_data(loader,filename):
    filename = str(pathlib.Path("csv-data") / filename)
    dataset = loader(filename)
    print('dataset ==', dataset)

    if len(dataset) > 1:
        data = [x for x in dataset]#.to(device)
    else:
        data = dataset[0]#.to(device)

    return data, dataset.num_features
loader = load_data
dataset = "Values.csv"
data = create_data(loader,dataset)
data, inp_size = data

#
# df = pd.DataFrame({
# 'pos'  : np.arange(data.x.shape[1]),
# 'pval'  : pvals})
# #
df_data = pd.DataFrame(data.x.detach().numpy())
df_data["y"] = data.y.detach().numpy()


lasso = Lasso(alpha = 10)
# alphas = np.arange(1, 2, 0.01)
# params = {'alpha': alphas}
# for i in range(0, data.x.shape[1]):
#     result_r2, result_mse = param_tuning(lasso, data.x.T[i].reshape(-1, 1), data.y, params, n=5)
# lasso = LassoCV(cv=5,max_iter=1000, random_state=0)
reg_coeffs = []
# for i in range(0, data.x.shape[1]):
#     lasso_result = stats_result(lasso, data.x.T[i].reshape(-1, 1), data.y.ravel())
#
# print(lasso.alpha_)
#
# lasso_best = Lasso(alpha=lasso.alpha_)

pvals = []
mse_paths = []
print(data.x)
print(data.y)
# for i in range(0, data.x.shape[1]):
# alphas, mse_path = stats_result(lasso, data.x.reshape(-1, 1), data.y.ravel())
_, score = stats_result(lasso, data.x, data.y.ravel())

print("Train --", score)

_,score = stats_result(lasso, data.test.x, data.test.y.ravel())

print("Text --", score)
    # mod = sm.OLS(data.y.detach().numpy(),data.x.T[i].detach().numpy())
    # fii = mod.fit()
    # p_values = fii.summary2().tables[1]['P>|t|']
    # pvals.append(-np.log10(p_values.values))
    # coeffs, path = lasso_result
    # reg_coeffs.append(coeffs)
    # mse_paths.append()
#
# df = pd.DataFrame({
# 'pos'  : np.arange(data.x.shape[1]),
# 'pval_lasso'  : reg_coeffs})
# # ,
# # 'pvals' : pvals})
# #
# blues = ['teal' for x in df['pos']]
# pinks = ['orange' for x in df['pos']]
# palette = sns.color_palette(None, len(pvals))
# # df["colours_pvals"] = colours
# # df["colours_lasso"]
# plt.figure(figsize=(10, 5))
#
# # Create scatterplot
# # plt.scatter(df['pos'], df['pvals'], c=blues, alpha=0.5)
# plt.scatter(df['pos'], df['pval_lasso'], c=pinks, alpha=0.5)
#
# # Set title and labels
# plt.title('Manhattan Plot')
# plt.legend(["Lasso regression coefficients"])
# plt.xlabel('Marker data')
# plt.ylabel('regression coefficients')
#
# # Show the plot
# plt.show()
