### ---- Non-NN Learners ---- ###

# SVM

def svm(x_train, y_train, x_test, y_test):
	from sklearn import svm
	lin_clf = svm.SVC(kernel = 'rbf')
	lin_clf.fit(x_train, y_train)
	print("Fit")
	print("training accuracy", clf.score(x_train, y_train))
	print("test accuracy", lin_clf.score(x_test, y_test))

# Random Forest

def randomForest(x_train, y_train, x_test, y_test):
	from sklearn.ensemble import RandomForestClassifier
	print("Starting")
	clf = RandomForestClassifier(n_estimators = 50, max_depth = 10,
		class_weight = 'balanced')
	clf.fit(x_train, y_train)
	print("Fit")
	print("training accuracy", clf.score(x_train, y_train))
	print("test accuracy", clf.score(x_test, y_test))


# AdaBoost with Decision Trees
def adaBoost(x_train, y_train, x_test, y_test):
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import AdaBoostClassifier

	abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 6),
		n_estimators = 10)
	abc.fit(x_train, y_train)
	print("Trained")
	print("Training accuracy", abc.score(x_train, y_train))
	print("Test accuracy", abc.score(x_test, y_test))

