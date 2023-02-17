import os
import pickle
import sys

import numpy as np
import asgl
import pandas as pd

import time
start_time = time.time()
print("Starting")
print("Starting", file=sys.stderr)

filename = "SNP.csv"

with open(filename) as fp:
    line = fp.readline()
    column_count = len(line.split(","))
value_columns = [str((i + 1)) for i in range(column_count - 1)]
labels = ["value"] + value_columns
df_whole = pd.read_csv(filename, names=labels)

train_set = 1215
valid_set = 0
test_set = df_whole.shape[0] - valid_set

df_xtrain = df_whole.iloc[0:train_set, 1:].values
df_ytrain = df_whole['value'][0:train_set].values

df_xtest = df_whole.iloc[train_set:test_set, 1:].values
df_ytest = df_whole['value'][train_set:test_set].values
x = df_whole.iloc[:, 1:].values
y = df_whole['value'].values

group_index = np.array(sum(zip(range(9723//3), range(9723//3), range(9723//3)), tuple()))

# Define parameters grid
lambda1 = (10.0 ** np.arange(-3, 1.51, 0.8)) #  possible values for lambda
alpha = np.arange(0, 1, 0.2) # 5 possible values for alpha

# Define model parameters
model = 'lm'  # linear model
penalization = 'sgl'  # sparse group lasso penalization
parallel = False  # Code executed in parallel
error_type = 'MSE'  # Error measuremente considered. MSE stands for Mean Squared Error.
#
cv_class = asgl.CV(model=model, penalization=penalization, lambda1=lambda1, alpha=alpha,
                   nfolds=5, error_type=error_type, parallel=parallel, random_state=99, intercept=False)

error = cv_class.cross_validation(x=x, y=y, group_index=group_index)

print("--- Cross Validation: %s seconds ---" % (time.time() - start_time))
print("--- Cross Validation: %s seconds ---" % (time.time() - start_time), file=sys.stderr)

# # Select the minimum error
minimum_error_idx = np.argmin(error)

# Select the parameters associated to mininum error values
optimal_parameters = cv_class.retrieve_parameters_value(minimum_error_idx)
optimal_lambda = optimal_parameters.get('lambda1')
optimal_alpha = optimal_parameters.get('alpha')

print(f"optimal: l {optimal_lambda}, a {optimal_alpha}")
print(f"optimal: l {optimal_lambda}, a {optimal_alpha}", file=sys.stderr)

# Split data into train / test
train_idx, test_idx = asgl.train_test_split(nrows=x.shape[0], train_pct=0.7, random_state=1)
# Define asgl class using optimal values
asgl_model = asgl.ASGL(model=model, penalization=penalization, lambda1=optimal_lambda, alpha=optimal_alpha, intercept=False)
# Solve the model
asgl_model.fit(x=df_xtrain, y=df_ytrain, group_index=group_index)

# Obtain betas
final_beta_solution = asgl_model.coef_[0]

np.round(final_beta_solution, 1)
# Obtain predictions
final_prediction = asgl_model.predict(x_new=df_xtest)

# Obtain final errors
final_error = asgl.error_calculator(y_true=df_ytest,
                                    prediction_list=final_prediction,
                                    error_type=error_type)
with open("asgl_model", "wb") as fp:
    pickle.dump(asgl_model, fp)

print(f'Final error is {np.round(final_error[0], 2)}')
print(f'Final error is {np.round(final_error[0], 2)}', file=sys.stderr)
print("--- All: %s seconds ---" % (time.time() - start_time))
print("--- All: %s seconds ---" % (time.time() - start_time), file=sys.stderr)
