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

load = pd.read_csv("psych texas star no_nan.csv")
#data = load.replace(to_replace = np.nan, value = 0)  

X_og = data.drop(['Matched', 'Step 2 CK', 'AOA', 'Cumulative Quartile', '# Ho0red Clerkships', 'Ho0rs-A This Specialty', 'GHHS', 'Other Degrees', 'Research Year', '# Research Experiences', '# Abstracts, Pres, Posters', '# Peer-Rev Publications', '# Volunteer Experiences', '# Leadership Positions', 'Required to Remediate', 'Pass Attempt - Step 1', 'Pass Attempt - Step 2 CK', 'Pass Attempt - Step 2 CS', '# Programs Applied', '# Interviews Attended', 'Home State', 'Release - Step 2 CK', 'Release - Step 2 CS', 'Majority of Interview Offers', 'Majority of Interviews Attended'],  axis=1)

X_og = X_og.astype('category')
y = data['Matched']
 

enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
enc.fit(X_og)
enc_df = pd.DataFrame(enc.fit_transform(X_og).toarray())
X = enc.transform(X_og).toarray()


kf = KFold(n_splits=147) #essentially the same as LeaveOneOut bc there are 147 subjects and 147 folds
kf.get_n_splits(X)
print(kf)


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


#X_og = data.drop('Matched', 'Step 1', 'Step 2 CK', 'AOA', 'Cumulative Quartile', '# Ho0red Clerkships', 'Ho0rs-A This Specialty', 'GHHS', 'Other Degrees', 'Research Year', '# Research Experiences', '# Abstracts, Pres, Posters', '# Peer-Rev Publications', '# Volunteer Experiences', '# Leadership Positions', 'Required to Remediate', 'Pass Attempt - Step 1', 'Pass Attempt - Step 2 CK', 'Pass Attempt - Step 2 CS', '# Programs Applied', '# Interviews Attended', 'Home State', 'Release - Step 2 CK', 'Release - Step 2 CS', 'Majority of Interview Offers', 'Majority of Interviews Attended',  axis=1)
