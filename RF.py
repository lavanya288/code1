from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris=load_iris()
x=iris.data
y=iris.target
rf=RandomForestClassifier(n_estimators=5)
rf.fit(x,y)
predicted=rf.predict(x)
print(predicted)