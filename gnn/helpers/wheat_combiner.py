import os.path

import pandas as pd
import numpy as np
import sys

if len(sys.argv) == 1:
    files = ["WHEAT1.csv", "WHEAT2.csv", "WHEAT4.csv", "WHEAT5.csv"]
else:
    files = sys.argv[1:]

with open(files[0]) as fp:
    line = fp.readline()
    column_count = len(line.split(","))
value_columns = [str((i+1)) for i in range(column_count-1)]
labels = ["value"] + value_columns

value_df = None
gen_df = None
for filename in files:
    df = pd.read_csv(filename, names=labels)
    if value_df is None:
        value_df = df["value"]/len(files)
    else:
        value_df += df["value"]/len(files)
    if gen_df is None:
        gen_df = df[value_columns]
gen_df["value"] = value_df
gen_df[labels].to_csv(os.path.splitext(files[0])[0].strip("1") + "_combined.csv", index=False, header=False)

