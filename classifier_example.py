from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
# load the iris datasets
dataset = datasets.load_iris()
# fit a Naive Bayes model to the data
model = GaussianNB()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))










	# SVM looks much better in validation

	# print "training SVM..."
	
	# # although one needs to choose these hyperparams
	# C = 173
	# gamma = 1.31e-5
	# shrinking = True

	# probability = True
	# verbose = True

	# svc = SVC( C = C, gamma = gamma, shrinking = shrinking, probability = probability, verbose = verbose )
	# svc.fit( x_train, y_train )
	# p = svc.predict_proba( x_test )	
	
	# auc = AUC( y_test, p[:,1] )
	# print "SVM AUC", auc	
	

	# print "training random forest..."

	# n_trees = 100
	# max_features = int( round( sqrt( x_train.shape[1] ) * 2 ))		# try more features at each split
	# max_features = 'auto'
	# verbose = 1
	# n_jobs = 1

	# rf = RF( n_estimators = n_trees, max_features = max_features, verbose = verbose, n_jobs = n_jobs )
	# rf.fit( x_train, y_train )

	# p = rf.predict_proba( x_test )

	# auc = AUC( y_test, p[:,1] )
	# print "RF AUC", auc

	# # AUC 0.701579086548
	# # AUC 0.676126704696

	# # max_features * 2
	# # AUC 0.710060065732
	# # AUC 0.706282346719


