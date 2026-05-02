import pickle
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

model = KNeighborsClassifier()

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train,y_train)
print(model.score(X_test,y_test))


with open('sava_model.pkl','wb') as f:
    pickle.dumps(model)
