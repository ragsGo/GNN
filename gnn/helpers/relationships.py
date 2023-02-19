import csv
import collections
import itertools

grid = collections.Counter()

with open("Pedigree_Data.csv", "r", newline="") as fp:
    reader = csv.reader(fp)
    for line in reader:
        # clean empty names
        line = [name.strip() for name in line if name.strip()]
        # count single works
        if len(line) == 1:
            grid[line[0], line[0]] += 1
        # do pairwise counts
        for pair in itertools.combinations(line, 2):
            grid[pair] += 1
            grid[pair[::-1]] += 1

actors = sorted(set(pair[0] for pair in grid))

with open("connection_grid.csv", "w", newline="") as fp:
    writer = csv.writer(fp)
    writer.writerow([''] + actors)
    for actor in actors:
        line = [actor,] + [grid[actor, other] for other in actors]
        writer.writerow(line)
