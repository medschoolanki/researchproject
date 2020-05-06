import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, f1_score
from sklearn.model_selection import validation_curve

#### CODE ASSUMES YOU HAVE ALREADY DEFINED X and y AFTER IMPORTING YOUR DATASET #####
X_train, X_test, y_train, y_test = train_test_split(reducedX, y, test_size = 0.2, random_state = 0)

print("Training set size: %.0f" % len(X_train))
print("Testing set size: %.0f" % len(X_test))

############# RANDOM FOREST FINE TUNING ###############
clf = RandomForestClassifier(class_weight='balanced')
n_estimators = [5, 10, 15, 20, 22, 25, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100]
max_depth = [2, 5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 
scorers = {
	'precision_score': make_scorer(precision_score),
	'recall_score': make_scorer(recall_score),
	'accuracy_score': make_scorer(accuracy_score),
	'f1_score': make_scorer(f1_score)
}

###### VALIDATION CURVE FOR n_estimators ########
param_range = [10, 15, 20, 22, 25, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 150, 200, 300, 500, 1000, 1200]
train_scoreNum, test_scoreNum = validation_curve(
                                RandomForestClassifier(),
                                X = X_train, y = y_train, 
                                param_name = 'n_estimators', 
                                param_range = param_range, cv = 5, scoring='balanced_accuracy')
# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scoreNum, axis=1)
train_std = np.std(train_scoreNum, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scoreNum, axis=1)
test_std = np.std(test_scoreNum, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(param_range, train_mean, label="Training score", color="black")
plt.plot(param_range, test_mean, label="Cross-validation score", color="red")

# Plot accurancy bands for training and test sets
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

# Create plot
plt.title("Validation Curve With Random Forest")
plt.xlabel("Number Of Trees")
plt.ylabel("Balanced Accuracy")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(clf, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1, scoring='recall_weighted')
bestF = gridF.fit(X_train, y_train)
bestF.best_params_
bestF.best_estimator_


#SVC recall finetuning
svc = SVC(kernel = 'rbf', gamma = 1e-2, C = 10, probability = True)
param_grid = {'C': [0.000001, 0.0001, 0.001, 0.01, .1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear']}
scorers = {
	'precision_score': make_scorer(precision_score),
	'recall_score': make_scorer(recall_score),
	'accuracy_score': make_scorer(accuracy_score),
	'f1_score': make_scorer(f1_score)
}


def grid_search_wrapper(refit_score='recall_score'):
	skf = StratifiedKFold(n_splits=10)
	grid_search = GridSearchCV(svc, param_grid, scoring=scorers, refit=refit_score, cv=skf, return_train_score=True, n_jobs=-1)
	grid_search.fit(X_train, y_train)
	y_pred = grid_search.predict(X_test)
	print('Best params for {}'.format(refit_score))
	print(grid_search.best_params_)
	print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
	print(pd.DataFrame(confusion_matrix(y_test, y_pred),
		columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
	return grid_search


grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
grid.fit(X_train, y_train) 
print(grid.best_params_) 
print(grid.best_estimator_) 


grid_search_clf = grid_search_wrapper(refit_score='recall_score')


#fine tuning of parameters
y_scores = grid_search_clf.decision_function(X_test)
p, r, thresholds = precision_recall_curve(y_test, y_scores)


def adjusted_classes(y_scores, t):
	return [1 if y >= t else 0 for y in y_scores]


def precision_recall_threshold(p, r, thresholds, t=0.5):
	y_pred_adj = adjusted_classes(y_scores, t)
	print(pd.DataFrame(confusion_matrix(y_test, y_pred_adj),
		columns=['pred_neg', 'pred_pos'],
		index=['neg', 'pos']))
	plt.figure(figsize=(8,8))
	plt.title("Precision and Recall curve ^ = current threshold")
	plt.step(r, p, color='b', alpha=0.2,where='post')
	plt.fill_between(r, p, step='post', alpha=0.2, color='b')
	plt.ylim([0.5, 1.01]);
	plt.xlim([0.5, 1.01]);
	plt.xlabel('Recall');
	plt.ylabel('Precision');
	close_default_clf = np.argmin(np.abs(thresholds - t))
	plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k', markersize=15)





bsvm = SVC(kernel='rbf', C=0.01, gamma=0.0001, probability=True, class_weight='balanced')
for train_index, test_index in kf.split(X):
     print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]

model = bsvm.fit(X_train, y_train)
predictions = bsvm.predict(X_test)
accuracy = cross_val_score(model, X, y, cv=LeaveOneOut(), scoring='accuracy')
print("Cross-validated accuracy:", accuracy)
print("Accuracy mean:",accuracy.mean())

precision = cross_val_score(model, X, y, cv=LeaveOneOut(), scoring='precision_weighted')
print("Cross-validated precision:", precision)
print("Precision mean:",precision.mean())

recall = cross_val_score(model, X, y, cv=LeaveOneOut(), scoring='recall_weighted')
print("Cross-validated recall:", recall)
print("Recall mean:",recall.mean())

