from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = [[10],[20],[30],[40],[10],[20],[30],[40],[10],[20],[30],[40]]
y = [30,50,70,90,30,50,70,90,30,50,70,90]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = LinearRegression()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
a = model.predict([[60]])
print(a)
