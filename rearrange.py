import sys
from pathlib import Path

import pandas as pd
import numpy as np


def do_the_thing(path):
    data = []
    with open(Path(path)/'p1_mrk_001.txt') as fp:
        fp.readline()
        for line in fp.readlines():
            line = line[8:]
            data.append({x+1: y if y not in ['3', '4'] else '1' for x, y in enumerate(line) if y != '\n'})
    genome = pd.DataFrame(data)
    data = []
    with open(Path(path)/'p1_data_001.txt') as fp:
        headers = fp.readline().split()
        for line in fp.readlines():
            data.append({x: y for x, y in zip(headers, line.split())})
    phen = pd.DataFrame(data)
    genome['value'] = phen['Phen']
    return genome


if __name__ == "__main__":
    df = do_the_thing(sys.argv[1])
    columns = ['value'] + list(range(1, len(df.columns)))
    df.to_csv(sys.argv[2], columns=columns, header=False, index=False)