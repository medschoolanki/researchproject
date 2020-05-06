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
data = load.replace(to_replace = np.nan, value = 0)  

X = data.drop(['Matched', 'OtherDegrees', 'Release-Step2CK', 'Release-Step2CS', 'MajorityofInterviewOffers', 'MajorityofInterviewsAttended', 'HomeState'],  axis=1).values
y = data['Matched'].values
 

#enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
#enc.fit(X_og)
#enc_df = pd.DataFrame(enc.fit_transform(X_og).toarray())
#X = enc.transform(X_og).toarray()


kf = KFold(n_splits=622) #essentially the same as LeaveOneOut bc there are 147 subjects and 147 folds
kf.get_n_splits(X)
print(kf)

loo = LeaveOneOut()
loo.get_n_splits(X)
print(loo)


#bsvm = SVC(kernel='linear', class_weight='balanced') #DEFAULT HYPERPARAMETERS, BELOW IS GRID SEARCHED PARAMETERS
bsvm = SVC(C=1e-06, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
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
      


y_p = np.stack(y_preddata,axis=1)
y_p = np.reshape(y_p,(622,))



def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1
    print("TP:",TP,"FP:",FP,"TN:",TN,"FN:",FN)
    Sensitivity = TP/(TP+FN)
    Specificity = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(FN+TN)
    print("Accuracy:",metrics.accuracy_score(np.array(y_p),np.array(y)))
    print("Sensitivity:", Sensitivity, "Specificity:", Specificity, "PPV:", PPV, "NPV:", NPV)
    

perf_measure(y,y_p)


#FEATURE IMPORTANCE
features_names = ['Step1','Step2CK','AOA','CumulativeQuartile','#Ho0redClerkships','Ho0rs-AThisSpecialty','GHHS','ResearchYear','#ResearchExperiences','#Abstracts,Pres,Posters','#Peer-RevPublications','#VolunteerExperiences','#LeadershipPositions','RequiredtoRemediate','PassAttempt-Step1','PassAttempt-Step2CK','PassAttempt-Step2CS','#ProgramsApplied','#InterviewsAttended']
pd.Series(abs(svm.coef_[0]), index=features_names).nlargest(19).plot(kind='barh')
plt.show()

#X_og = data.drop('Matched', 'Step 1', 'Step 2 CK', 'AOA', 'Cumulative Quartile', '# Ho0red Clerkships', 'Ho0rs-A This Specialty', 'GHHS', 'Other Degrees', 'Research Year', '# Research Experiences', '# Abstracts, Pres, Posters', '# Peer-Rev Publications', '# Volunteer Experiences', '# Leadership Positions', 'Required to Remediate', 'Pass Attempt - Step 1', 'Pass Attempt - Step 2 CK', 'Pass Attempt - Step 2 CS', '# Programs Applied', '# Interviews Attended', 'Home State', 'Release - Step 2 CK', 'Release - Step 2 CS', 'Majority of Interview Offers', 'Majority of Interviews Attended',  axis=1)
