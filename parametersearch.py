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



data = pd.read_csv("RD+Controls_pairwise_corrs.csv")
X = data.drop('RD', axis=1).values
y = data['RD'].values
kf = KFold(n_splits=147)
kf.get_n_splits(X)
print(kf)
bsvm = SVC(kernel='linear', class_weight='balanced')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print("Training set size: %.0f" % len(X_train))
print("Testing set size: %.0f" % len(X_test))


#gridsearch
#define scores
#def scores(model):
#	model.fit(xtrain, ytrain.ravel())
#	y_pred = model.predict(xtest)
#   	print("Accuracy score: %.3f" % metrics.accuracy_score(ytest, y_pred))
#	print("Recall: %.3f" % metrics.recall_score(ytest, y_pred))
#	print("Precision: %.3f" % metrics.precision_score(ytest, y_pred))
#	print("F1: %.3f" % metrics.f1_score(ytest, y_pred))
#	proba = model.predict_proba(xtest)
#	print("Log loss: %.3f" % metrics.log_loss(ytest, proba))
#	pos_prob = proba[:, 1]
#	print("Area under ROC curve: %.3f" % metrics.roc_auc_score(ytest, pos_prob))
#	cv = cross_val_score(model, xtest, ytest.ravel(), cv = 3, scoring = 'accuracy')
#	print("Accuracy (cross validation score): %0.3f (+/- %0.3f)" % (cv.mean(), cv.std() * 2))
#	return y_pred

#cv = StratifiedKFold(n_splits = 3, random_state = 0)
#define gridsearch
#def grid_search(model, grid):
#	clf = GridSearchCV(model, grid, cv = cv, n_jobs = -1, verbose = 2, iid = False)
#	scores(clf)    
#	print(clf.best_params_)


#dummy svc variable for parameter search
#y_svc = scores(svc)

svc = SVC(kernel = 'rbf', gamma = 1e-2, C = 10, probability = True)

#values for above dummy svc
#gamma = [x for x in np.logspace(-4, 1, num = 6, endpoint=10)]
#C = [x for x in np.logspace(-2, 2, num = 5, endpoint=100)]
#kernel = ['rbf', 'sigmoid', 'linear']
#probability = [True]

#grid = {'gamma': gamma,
#        'C': C,
#        'kernel': kernel,
#        'probability': probability}

#grid_search(svc, grid)


#recall finetuning
param_grid = {'C': [0.000001, 0.0001, 0.001, 0.01, .1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf','linear','poly','sigmoid']}
scorers = {
	'precision_score': make_scorer(precision_score),
	'recall_score': make_scorer(recall_score),
	'accuracy_score': make_scorer(accuracy_score),
	'f1_score': make_scorer(f1_score)
}


def grid_search_wrapper(refit_score='f1_score'):
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

