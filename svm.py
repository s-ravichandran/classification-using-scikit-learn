import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import random
import sys

# args : svm.py float(sampleRatio) 

data_dir = './input/'	# needs trailing slash

# validation split, both files with headers and the Happy column
train_file = data_dir + 'census.data.v5.csv'
test_file = data_dir + 'census.test.v5.csv'

print "Before reading files"

large_train = pd.read_csv( train_file )
large_test = pd.read_csv( test_file )

# Stratified Sampling
sample_ratio = float(sys.argv[1])
include_cols = ['Label']
print "Large Test Stats\n", large_test.Label.describe(), "\n"

##StratifiedShuffleSplit(y, n_iter=10, test_size=0.1, train_size=None, random_state=None)
sss = StratifiedShuffleSplit(large_train.Label, n_iter=1, test_size = sample_ratio, train_size=None, random_state=0)
for train_split_ind, test_split_ind in sss:
	print "Sampled Train :", test_split_ind
	train = large_train.iloc[test_split_ind]
	
sss = StratifiedShuffleSplit(large_test.Label, n_iter=1, test_size = sample_ratio, train_size=None, random_state=0)
for train_split_ind, test_split_ind in sss:
	print "Sampled Test :", test_split_ind
	test = large_test.iloc[test_split_ind]

	
del(large_train)
del(large_test)
	
""""
train = 
test = large_test.sample(int(sys.argv[1]))
"""
# y
y_train = train.Label
y_test = test.Label
#print type(y_test) #should be series
# populate lists of numeric and cat attributes

all_cols = list(train.columns.values)
numeric_cols = ['age','wage per hour','capital gains','capital losses','dividends from stocks','num persons worked for employer','weeks worked in year',]
# remove_cols = ['Label','instance weight','migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt']
remove_cols = ['Label','instance weight']
cat_cols =  [x for x in all_cols if x not in numeric_cols and x not in remove_cols]

# handle numerical features
x_num_train = train[ numeric_cols ].as_matrix()
x_num_test = test[ numeric_cols ].as_matrix()

x_train_count = x_num_train.shape[0]
x_test_count = x_num_test.shape[0]

x_num_combined = np.concatenate((x_num_train,x_num_test), axis=0) # 0 -row 1 - col

# print "\nTRAIN NUM dims: ", x_num_train.shape, ", num rows: ", x_train_count
# print "TEST NUM dims: ", x_num_test.shape, ", num rows: ", x_test_count
# print "COMBINED NUM dims: ", x_num_combined.shape	

# scale numeric features to <0,1>
max_num = np.amax( x_num_combined, 0 )

x_num_combined = np.true_divide(x_num_combined, max_num) # scale by max. truedivide needed for decimals
x_num_train = x_num_combined[0:x_train_count]
x_num_test = x_num_combined[x_train_count:]

# print "\nTRAIN NUM dims: ", x_num_train.shape, ", expected num rows: ", x_train_count
# print "TEST NUM dims: ", x_num_test.shape, ", expected num rows: ", x_test_count

# categorical

x_cat_train = train.drop( numeric_cols + remove_cols, axis = 1 )
x_cat_test = test.drop( numeric_cols + remove_cols, axis = 1 )

x_cat_train.fillna( 'NA', inplace = True )
x_cat_test.fillna( 'NA', inplace = True )

x_cat_combined = pd.concat((x_cat_train, x_cat_test), axis=0)

# print "\nTRAIN CAT dims: ", x_cat_train.shape, ", num rows: ", x_train_count
# print "TEST CAT dims: ", x_cat_test.shape, ", num rows: ", x_test_count
# print "COMBINED CAT dims: ", x_cat_combined.shape	

# print "\nTYPES\nx_cat_train: ", type(x_cat_train)
# print "x_cat_combined: ", type(x_cat_combined)

# one-of-k handling for categorical features
# pandas.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False)
print "Before getting cat -> dummies"
vec_x_cat_combined = pd.get_dummies(x_cat_combined, columns=cat_cols, sparse=False)#.as_matrix()

vec_x_cat_train = vec_x_cat_combined[0:x_train_count]
vec_x_cat_test = vec_x_cat_combined[x_train_count:]

# print "\nExpanded TRAIN CAT dims: ", vec_x_cat_train.shape, ", expected num rows: ", x_train_count
# print "Expanded TEST CAT dims: ", vec_x_cat_test.shape, ", expected num rows: ", x_test_count


# combine numerical and categorical
del(vec_x_cat_combined)
x_train = np.hstack(( x_num_train, vec_x_cat_train.ix[:,:200] )) # returns ndarray
del(x_num_train)
x_train = np.hstack(( x_train, vec_x_cat_train.ix[:,200:] )) # returns ndarray
del(vec_x_cat_train)
x_test = np.hstack(( x_num_test, vec_x_cat_test.ix[:,:200] ))
del(x_num_test)
x_test = np.hstack(( x_test, vec_x_cat_test.ix[:,200:] ))
del(vec_x_cat_test)
# print "\nx_train: ", x_train.shape, ", ", type(x_train)
# print "x_test: ", x_test.shape, ", ", type(x_test)



# working fine upto here - Data Processing
# below - classifier specific logic

classifier_alg = "SVM"

svm_classifier = svm.SVC(probability=True)

print "Before fit"

svm_classifier.fit( x_train, y_train , sample_weight=train['instance weight'].values)

print "Before predict"

predicted = svm_classifier.predict( x_test )
predicted_train = svm_classifier.predict( x_train )
print "Train Predictions: \n" + (metrics.classification_report(y_train, predicted_train)) 
print "Test Predictions: \n" + metrics.classification_report(y_test, predicted)

# Data for plotting

probs = svm_classifier.predict_proba(x_test)
y_conf=[]
y_test_num=[]
y_train_conf=[]
y_train_num=[]
probs_train = svm_classifier.predict_proba(x_train)

print "Before changing y_label to num"
print "y_test", len(y_test), "\t", type(y_test)
y_test = y_test.as_matrix()
# print y_test
for i in range(len(y_test)):
	# print y_test[i]
	# print i
	if y_test[i] == 'Pos':
		y_test_num.append(1)
	else:
		y_test_num.append(0)

y_train = y_train.as_matrix()
for i in range(len(y_train)):
	if(y_train[i] == 'Pos'):
		y_train_num.append(1)
	else:
		y_train_num.append(0)

#Neg is 0 in probs and 0 in y_test
#pos is 1 in probs and 1 in y_test

print "Going to plot ",classifier_alg

for class_to_plot in [0,1]:
	y_conf = [] # Test Set
	for i in range(len(y_test)):
		y_conf.append(probs[i][class_to_plot])
	precision, recall, thresholds = precision_recall_curve(y_test_num, y_conf, pos_label=class_to_plot)
	plt.plot(recall,precision)
	
	y_train_conf=[] # Train Set
	for i in range(len(y_train)):
		y_train_conf.append(probs_train[i][class_to_plot])
	precision, recall, thresholds = precision_recall_curve(y_train_num, y_train_conf, pos_label=class_to_plot)
	plt.plot(recall, precision)
	
	if(class_to_plot == 0):
		plt.axis([0,1,0.8,1])
		plt.yticks(np.arange(0.8, 1.05, 0.1))
	else:
		plt.axis([0,1,0,1])
		plt.yticks(np.arange(0, 1.1, 0.1))
	
	print "Checking class ",class_to_plot
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.grid(b=True, which='major', axis='both', color='black', linestyle='-', alpha=0.3)
	plt.xticks(np.arange(0, 1.1, 0.1))
	
	plt.title(classifier_alg+': ' + str(class_to_plot)) #+' stratified sample '+str(sample_ratio))
	filename = "./plots/"+ classifier_alg + "_"+str(class_to_plot)+"_default_"+str(sample_ratio)+".png"#+str(num_estimators)+"_"
	plt.savefig(filename)
	plt.clf()



"""
#Learn without Instance weights

svm_classifier = svm.SVC(probability=True)
svm_classifier.fit( x_train, y_train )

predicted = svm_classifier.predict( x_test )

print "Going to plot unweighted"


for class_to_plot in [0,1]:
	y_conf = []
	for i in range(len(y_test)):
		y_conf.append(probs[i][class_to_plot])
	precision, recall, thresholds = precision_recall_curve(y_test_num, y_conf, pos_label=class_to_plot)
	plt.plot(recall,precision)
	if(class_to_plot == 0):
		plt.axis([0,1,0.8,1])
	else:
		plt.axis([0,1,0,1])
	print "Checking class",class_to_plot

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('SVM without instance weights for class ' + str(class_to_plot))
	filename = "./plots/svm_unweighted_"+str(class_to_plot)+".png"
	plt.savefig(filename)
	plt.clf()
"""
