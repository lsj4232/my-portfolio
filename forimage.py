import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

# Iris: 산점도 (꽃받침 길이 vs 너비)
iris = datasets.load_iris()
X = iris.data[:, :2]   # sepal length, sepal width
y = iris.target

plt.figure()
for c in np.unique(y):
    plt.scatter(X[y == c, 0], X[y == c, 1], label=iris.target_names[c], s=30)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Iris — Sepal length vs width")
plt.legend()
plt.tight_layout()
plt.show()
