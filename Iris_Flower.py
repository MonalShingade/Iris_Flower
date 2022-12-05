import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"E:\Class notes\08_03_Logistic_Regression_Model\Iris.csv")
df["Species"].replace({"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2},inplace=True)

x = df.drop(["Species","Id"],axis=1)
y = df["Species"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=19,stratify=y)

norm_scalar = MinMaxScaler()
norm_scalar.fit(x_train)
array_1 = norm_scalar.transform(x_train)

dt_clf = DecisionTreeClassifier(max_depth=6,random_state=20)
dt_clf.fit(x_train, y_train)

import pickle
with open("Iris_flower.pkl","wb") as file:
    pickle.dump(dt_clf,file)

file = open("object.obj","wb")
pickle.dump(norm_scalar,file)
file.close()