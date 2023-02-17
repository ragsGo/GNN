import optuna

from sgd import create_sgd

# Before the command is run, ensure following is set:
#   $ mysql -u root -e "CREATE DATABASE IF NOT EXISTS optuna" -p
#   $ optuna create-study --study-name "distributed-example" --storage "mysql://root:1234qwer@localhost/optuna"
#
if __name__ == "__main__":
    study = optuna.load_study(
        study_name="distributed-example1", storage="mysql://root:1234qwer@localhost/optuna"
    )
    study.optimize(
        create_sgd(
            "SNP.csv",
        ),
        n_trials=1000,
    )
