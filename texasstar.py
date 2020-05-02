import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import preprocessing


kf = KFold(n_splits=147) #essentially the same as LeaveOneOut bc there are 147 subjects and 147 folds
kf.get_n_splits(X)
print(kf)

data = pd.read_csv("psych texas star.csv")
data = data.replace(to_replace = np.nan, value = 0)  
X_og = data.drop('Matched', axis=1)
y = data['Matched']
 
enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
enc.fit(X_og.astype(str))
OneHotEncoder(categorical_features=None, categories=None, drop=None,
              dtype=<class 'numpy.float64'>, handle_unknown='ignore',
              n_values=None, sparse=True)
X = enc.transform(X_og).toarray()


bsvm = SVC(kernel='linear', class_weight='balanced')
y_preddata=[]

for train_index, test_index in kf.split(X):
      print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      model = bsvm.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      model.predict(X_train)
      y_preddata.append(y_pred)
      confusion_matrix(y_train, model.predict(X_train))
      confusion_matrix(y_test, y_pred)
      print(classification_report(y_test, y_pred))
      confusion_matrix(y_test, model.predict(X_test))
      
      
print(np.asarray(y_preddata))
