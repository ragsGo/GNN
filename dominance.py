from dominance_analysis import Dominance_Datasets
from dominance_analysis import Dominance
import pandas as pd

with open("SNP.csv") as fp:
    line = fp.readline()
    column_count = len(line.split(","))

value_columns = [str((i + 1)) for i in range(column_count - 1)]
labels = ["value"] + value_columns

data = pd.read_csv("SNP.csv", names=labels)

dominance_regression = Dominance(data=data, target='value', objective=1, pseudo_r2="mcfadden")
incr_variable_rsquare = dominance_regression.incremental_rsquare()
dominance_regression.plot_incremental_rsquare()
print(dominance_regression.dominance_stats())
print(dominance_regression.dominance_level())
