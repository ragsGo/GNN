# Adapted from https://colab.research.google.com/drive/1_WxDPLGkJY3qJ-PK0J1YjATaZz35efmk#scrollTo=n5NEwC7tYvSo


import optuna
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


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


def create_sgd(
        filename,
        epochs=1000,
        eta0=(0.0000001, 0.0001),
        alpha=(0.00001, 0.001),
        learning_rate=('constant', 'optimal', 'invscaling', 'adaptive')
):
    xtrain, ytrain, xtest, ytest = load_data(filename)

    def run_sgd(trial):
        _eta = trial.suggest_float("eta0", *eta0)
        _alpha = trial.suggest_float("alpha", *alpha)
        _learning_rate = trial.suggest_categorical("learning_rate", list(learning_rate))
        _fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        reg = SGDRegressor(
            max_iter=epochs,
            eta0=_eta,
            learning_rate=_learning_rate,
            alpha=_alpha,
            fit_intercept=_fit_intercept
        )
        reg.fit(xtrain, ytrain)
        train_pred = reg.predict(xtrain)
        test_pred = reg.predict(xtest)
        print(
            f"Train loss: {mean_squared_error(ytrain, train_pred)}; "
            f"Test loss: {mean_squared_error(ytest, test_pred)}",
        )
        return mean_squared_error(ytest, test_pred)
    return run_sgd


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(
        create_sgd(
            "SNP.csv",
        ),
        n_trials=1000,
        n_jobs=-1
    )
