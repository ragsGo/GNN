import os
import pickle
import time
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error

def pandas_to_dict(df):
    result = defaultdict(dict)
    for i, row in df.iterrows():
        item = result[i]
        # item['father'] = result[int(row['father'])]
        # item['mum'] = result[int(row['mum'])]
        if int(row['father']) > 0:
            item['father'] = (int(row['father']) - 1, result[int(row['father']) - 1])
            result[int(row['father']) - 1][i] = item
        else:
            item['father'] = (-1, {})
        if int(row['mum']) > 0:
            item['mum'] = (int(row['mum']) - 1, result[int(row['mum']) - 1])
            result[int(row['mum']) - 1][i] = item
        else:
            item['mum'] = (-1, {})
    return result


def rels(pedigree, d, i=1):
    results = {d: 1}
    parents = {pedigree[d]["mum"][0]: 0.5, pedigree[d]["father"][0]: 0.5}
    cousins = {}
    cousin_front = {x: 0.5 for x in pedigree[d].keys() if x not in ["mum", "father"]}
    for n in range(i):
        next_parent = defaultdict(list)
        next_cousins = defaultdict(list)
        for p in parents.keys():
            if p < 0:
                continue
            if pedigree[p]["mum"][0] >= 0:
                next_parent[pedigree[p]["mum"][0]].append(parents[p] / 2)
            if pedigree[p]["father"][0] >= 0:
                next_parent[pedigree[p]["father"][0]].append(parents[p] / 2)

        for p in parents.keys():
            results[p] = parents[p] + results.get(p, 0)

        for p in cousin_front.keys():
            cousins[p] = cousin_front[p]
            for child in pedigree[p].keys():
                if child not in ['father', 'mum']:
                    if child in results:
                        continue
                    next_cousins[child].append(cousin_front[p] / 2)
        cousin_front = {}
        for p in next_cousins.keys():
            cousin_front[p] = sum(next_cousins[p] + [cousins.get(p, 0)])

        for p in parents.keys():
            for child in pedigree[p].keys():
                if child not in ['father', 'mum']:
                    if child in results:
                        continue
                    cousin_front[child] = cousin_front.get(child, 0) + (results[p] / 2)
        parents = {}

        for p in next_parent.keys():
            parents[p] = sum(next_parent[p] + [results.get(p, 0)])
    return sorted([(x, y) for x, y in {**cousins, **results}.items()], key=lambda x: x[1], reverse=True)


def mean(vals, weights):
    size = len(vals[0])
    [sum([x[i][n] for i in vals])]

if __name__ == "__main__":
    df = pd.read_csv("Pedigree_Data.csv")
    pedigree = pandas_to_dict(df)
    variance = []
    phenotype = []
    arrays = []
    for i, row in df.iterrows():
        if i >= 1215:
            break
        vals = rels(pedigree, i, 2)
        arrays.append(sum([df.iloc[x[0]][1:9724].tolist() for x in vals[1:5]], []))
        variance.append(vals)
        phenotype.append(row["value"])
    knn = neighbors.KNeighborsRegressor(5, weights="distance")
    y_ = knn.fit(arrays, phenotype)
    tests = []
    for i, row in df.iloc[1215:].iterrows():
        vals = rels(pedigree, i, 2)
        pred = knn.predict([sum([df.iloc[x[0]][1:9724].tolist() for x in vals[1:5]], [])])
        # arrays.append([df.iloc[x[0]]["value"] for x in vals[1:3]])
        tests.append((row["value"], pred[0]))
    print(mean_squared_error(*zip(*tests)))
    # print([x for x in pedigree[203].keys()])
    # # print(rels(pedigree, 203, 2))
    # knn = neighbors.KNeighborsRegressor(2, weights="distance")
    # vals = rels(pedigree, 203, 2)
    # print(vals)
    # X = [[x[1]] for x in rels(pedigree, 203, 2)]
    # y = [df.iloc[x[0]]["value"] for x in vals]
    # print(y)
    # y_ = knn.fit(X, y)
    # print(y_)

# def rels(pedigree, d, s=None, i=1):
#     if s is None:
#         s = set()
#     result = {}
#     for n in range(i):
#
#     for k, item in pedigree[d].items():
#         if isinstance(item, tuple):
#             k = item[0]
#         result[k] = 0.5
#     if i > 1:
#         father_sibs = rels(pedigree[d]['father'][1], s, i-1)
#         mother_sibs = rels(pedigree[d]["mum"][1], s, i-1)
#         combined = {x: ((father_sibs[x] if x in father_sibs else 0)+(mother_sibs[x] if x in mother_sibs else 0))/2
#                     for x in set(father_sibs.keys()) | set(mother_sibs.keys())}
#
#     return result
#         # if k not in ["father", "mum"] and k not in s:
#         #     same_father = 1 if id(item["father"]) == id(d["father"]) else 0
#         #     same_mum = 1 if id(item["mum"]) == id(d["mum"]) else 0
#         #     result[k] = (same_mum + same_father)/4
