import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import svm
from sklearn import metrics
from sklearn import neighbors, datasets
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import sys


# args : knn.py float(sampleRatio) 

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

classifier_alg = "KNN"

y_test_num=[]
print "Before for reset"
print len(y_test)
print type(y_test)
y_test = y_test.as_matrix()
# print y_test
for i in range(len(y_test)):
	# print y_test[i]
	# print i
	if y_test[i] == 'Pos':
		y_test_num.append(1)
	else:
		y_test_num.append(0)
expected = y_test

pos_precision_weighted=[]
pos_precision_uniform=[]
pos_recall_weighted=[]
pos_recall_uniform=[]

neg_precision_weighted=[]
neg_precision_uniform=[]
neg_recall_weighted=[]
neg_recall_uniform=[]

print x_train.size
print y_train.size

for weights in ['uniform']:
	print(weights)
	for n_neighbors in [1, 2, 3, 5, 10]:
		print("Num neighbors: ")
		print(n_neighbors)	
		knn_classifier = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
		print "Before fit"
		knn_classifier.fit(x_train, y_train)
		print "Before predict"
		predicted = knn_classifier.predict(x_test)
		precision, recall, fscore, support = precision_recall_fscore_support(expected, predicted)
		print "Precision: ",precision
		print "Recall: ",recall
		print(metrics.classification_report(expected, predicted))
		if weights=='distance':
			pos_precision_weighted.append(precision[1])
			pos_recall_weighted.append(recall[1])
			neg_precision_weighted.append(precision[0])
			neg_recall_weighted.append(recall[0])
		else:
			pos_precision_uniform.append(precision[1])
			pos_recall_uniform.append(recall[1])
			neg_precision_uniform.append(precision[0])
			neg_recall_uniform.append(recall[0])
				
colors = ['brown', 'red', 'yellow', 'green', 'blue']
numbers = [1, 2, 3, 5, 10]

print "Going to plot Pos Uniform"
plt.axis([0,1,0,1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(b=True, which='major', axis='both', color='black', linestyle='-', alpha=0.3)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title('K-NN - varying \'K\' ')
for p,r, c, n in zip(pos_precision_uniform, pos_recall_uniform, colors, numbers):
	# print p
	# print r
	plt.scatter(r,p, color = c, label=str(n)+'-NN')
	
plt.legend(loc='upper right')
plt.savefig("./plots/knn_uniform_pos_"+str(sample_ratio)+"_variousK.png")
plt.clf()

"""
for p,r in zip(neg_precision_uniform, neg_recall_uniform):
	# print p
	# print r
	plt.scatter(r,p)
	plt.axis([0,1,0,1])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.grid(b=True, which='major', axis='both', color='black', linestyle='-', alpha=0.3)
	plt.xticks(np.arange(0, 1.1, 0.1))
	plt.yticks(np.arange(0, 1.1, 0.1))
	plt.title('K-NN with Uniform weights for Neg class')
	plt.savefig("./plots/knn_uniform_neg_"+str(sample_ratio)+".png")
	
plt.clf()



for p,r in zip(neg_precision_weighted, neg_recall_weighted):
	# print p
	# print r
	plt.scatter(r,p)
	plt.axis([0,1,0,1])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.grid(b=True, which='major', axis='both', color='black', linestyle='-', alpha=0.3)
	plt.xticks(np.arange(0, 1.1, 0.1))
	plt.yticks(np.arange(0, 1.1, 0.1))
	plt.title('K-NN with weights for Neg class')
	plt.savefig("./plots/knn_weighted_neg_"+str(sample_ratio)+".png")
	
plt.clf()

print "Going to plot Pos Weighted"
plt.axis([0,1,0,1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(b=True, which='major', axis='both', color='black', linestyle='-', alpha=0.3)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title('K-NN with weights for Pos class')
for p,r,c in zip(pos_precision_weighted, pos_recall_weighted, colors):
	# print p
	# print r
	plt.scatter(r,p, color = c)
	plt.savefig("./plots/knn_weighted_pos_"+str(sample_ratio)+".png")
	
plt.clf()
"""
