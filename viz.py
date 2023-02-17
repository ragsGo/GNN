import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def function(x, y):
    return 4+3*x + 4*y

x1=np.random.rand(100)
x2=np.random.rand(100)+1.2
X=np.concatenate([x1, x2])
Y=0.5+2*X

Z=function(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X, Y, Z, c='r', marker='o')
plt.show()

data=[]
for x, y, z in zip (X, Y, Z):
    data.append([x, y, z])

data_embedded = TSNE(n_components=2).fit_transform(data)


plt.scatter([x for x, y in data_embedded], [y for x, y in data_embedded], color='r')
plt.show()