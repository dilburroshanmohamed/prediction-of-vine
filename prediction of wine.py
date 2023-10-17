from sklearn.datasets import load_wine
data = load_wine()
X = data.data
y = data.target
print(X.shape)
print(y.shape)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
print(knn)
knn.fit(X, y)
knn.predict([[12.85,3.27,2.58,22,106,1.65,.6,.6,.96,5.58,.87,2.11,570]])