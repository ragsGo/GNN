import os.path
import sys

import pandas as pd

infile = sys.argv[1]
split_file = os.path.splitext(infile)
outphen = f"{split_file[0]}phen{split_file[1]}"
outgen = f"{split_file[0]}gen{split_file[1]}"


df = pd.read_csv(infile, header=None)
gen = df[df.columns[1:]]
phen = df[[df.columns[0]]]
gen.index.name = "ID"
gen.to_csv(outgen)
phen.index.name = "ID"
phen.columns = ["t3"]
phen.to_csv(outphen)
