import bentoml

from sklearn import svm
from sklearn import datasets

#load the training dataset
iris=datasets.load_iris()
X,y=iris.data,iris.target

#trainthe model
clf=svm.SVC(gamma='scale')
clf.fit(X,y)

# save the model to the bentoml local model store

saved_model =bentoml.sklearn.save_model("iris_clf",clf)
print(f"Model saved: {saved_model}")

# tag="iris_clf:rmjpuektkg6zl6eu"  Keeps track of the model

# bentoml models list