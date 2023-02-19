#!/usr/bin/env python3
import sys

import pandas as pd

infile = sys.argv[1]
outfile = sys.argv[2]

with open(infile) as fp:
    line = fp.readline()
    column_count = len(line.split(","))
value_columns = [str((i+1)) for i in range(column_count-1)]
labels = ["value"] + value_columns

df_whole = pd.read_csv(infile, names=labels)
data_len = df_whole.shape[0]
df_new = pd.DataFrame()
df_new["value"] = df_whole["value"]

for i in range(1, len(df_whole.columns)):
    df_new[i] = (df_whole[str(i)] == 1).astype(int)

for i in range(1, len(df_whole.columns)):
    df_new[i + len(df_whole.columns) - 1] = (df_whole[str(i)] == 2).astype(int)

df_new.to_csv(outfile, header=False, index=False)
print("Max column:", len(df_new.columns))