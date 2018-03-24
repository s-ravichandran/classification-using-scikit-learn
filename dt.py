import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import sys


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

data_dir = './input/'	# needs trailing slash

# validation split, both files with headers and the Happy column
train_file = data_dir + 'census.data.v5.csv'
test_file = data_dir + 'census.test.v5.csv'

###

enc_cols = []

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

y_train=train.Label
x_train = train.drop(['Label', 'instance weight'], axis=1)


# print enc_cols

y_test = test.Label
x_test = test.drop(['Label', 'instance weight'], axis=1)

y_train_num=[]
for i in range(len(y_train)):
	if y_train[i] == 'Pos':
		y_train_num.append(1)
	else:
		y_train_num.append(0)

all_cols = list(train.columns.values)
numeric_cols = ['age','wage per hour','capital gains','capital losses','dividends from stocks','num persons worked for employer','weeks worked in year',]
# remove_cols = ['Label','instance weight','migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt']
remove_cols = ['Label','instance weight']
cat_cols =  [x for x in all_cols if x not in numeric_cols and x not in remove_cols]

used_cols = [x for x in all_cols if x not in remove_cols]

# handle numerical features
x_num_train = train[ numeric_cols ].as_matrix()
x_num_test = test[ numeric_cols ].as_matrix()

x_train_count = x_num_train.shape[0]
x_test_count = x_num_test.shape[0]

x_num_combined = np.concatenate((x_num_train,x_num_test), axis=0) # 0 -row 1 - col


# scale numeric features to <0,1>
max_num = np.amax( x_num_combined, 0 )

x_num_combined = np.true_divide(x_num_combined, max_num) # scale by max. truedivide needed for decimals
x_num_train = x_num_combined[0:x_train_count]
x_num_test = x_num_combined[x_train_count:]


# categorical

x_cat_train = train.drop( numeric_cols + remove_cols, axis = 1 )
x_cat_test = test.drop( numeric_cols + remove_cols, axis = 1 )

x_cat_train.fillna( 'NA', inplace = True )
x_cat_test.fillna( 'NA', inplace = True )

# print x_cat_train.size
print len(x_cat_train)
print type(x_cat_train)

vec_x_cat_train = []

vec_x_cat_test=[]

set_of_values =[]

# for i in range(x_cat_train.size/len(x_cat_train)):
	
#ADDED NOW
# for i in range(x_cat_train.size/len(x_cat_train)):
# 	le = preprocessing.LabelEncoder()
# 	le = le.fit(x_cat_train.loc[i].values)
# 	print le.classes_
# 	vec_x_cat_train.append(le.transform(x_cat_train))

# for i in range(x_cat_test.size/len(x_cat_test)):
# 	le = preprocessing.LabelEncoder()
# 	le = le.fit(x_cat_test.loc[i].values)
# 	print le.classes_
# 	vec_x_cat_test.append(le.transform(x_cat_test))


vec_x_cat_train = MultiColumnLabelEncoder(columns = cat_cols).fit_transform(x_cat_train)
vec_x_cat_test = MultiColumnLabelEncoder(columns = cat_cols).fit_transform(x_cat_test)

# print vec_x_cat_train

# for item in x_cat_train:
# 	le = preprocessing.LabelEncoder()
# 	le = le.fit(x_cat_train[0])
# 	print le.classes_
# 	vec_x_cat_train = le.transform(x_cat_train) 

# test_le = preprocessing.LabelEncoder()
# test_le = le.fit(x_cat_test)
# print test_le.classes_
# vec_x_cat_test = test_le.transform(x_cat_test) 

# x_cat_combined = pd.concat((x_cat_train, x_cat_test), axis=0)

# print "Before getting cat -> dummies"
# vec_x_cat_combined = pd.get_dummies(x_cat_combined, columns=cat_cols, sparse=False)#.as_matrix()

# le = preprocessing.LabelEncoder()
# le.fit(x_cat_combined)
# vec_x_cat_combined = le.transform(x_cat_combined)

# vec_x_cat_train = vec_x_cat_combined[0:x_train_count]
# vec_x_cat_test = vec_x_cat_combined[x_train_count:]

# combine numerical and categorical

x_train = np.hstack(( x_num_train, vec_x_cat_train )) # returns ndarray
x_test = np.hstack(( x_num_test, vec_x_cat_test ))


# print x_train.head()

dt_classifier = DecisionTreeClassifier(min_samples_split=300,max_leaf_nodes=30)
dt_classifier.fit(x_train,y_train_num)
tree.export_graphviz(dt_classifier, feature_names=used_cols,out_file="./weighted_tree.dot")
predicted = dt_classifier.predict( x_test )

# # print(metrics.classification_report(expected, predicted))

probs = dt_classifier.predict_proba(x_test)
y_conf=[]
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

# #Neg is 0 in probs and 0 in y_test
# #pos is 1 in probs and 1 in y_test
# print "Going to plot"

for class_to_plot in [0,1]:
	y_conf = []
	for i in range(len(y_test)):
		y_conf.append(probs[i][class_to_plot])
	precision, recall, thresholds = precision_recall_curve(y_test_num, y_conf, pos_label=class_to_plot, sample_weight=test['instance weight'].values)
	plt.plot(recall,precision)
	plt.axis([0,1,0,1])
	plt.yticks(np.arange(0, 1.1, 0.1))

	print "Checking class",class_to_plot
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('DT with instance weights for class ' + str(class_to_plot))
	plt.grid(b=True, which='major', axis='both', color='black', linestyle='-', alpha=0.3)
	plt.xticks(np.arange(0, 1.1, 0.1))
	filename = "./plots/dt_weighted_"+str(class_to_plot)+".png"
	plt.savefig(filename)
	# plt.clf()


# #Learn without Instance weights
plt.clf()

numLeaves = int(sys.argv[1])
dt_classifier = DecisionTreeClassifier(max_leaf_nodes=numLeaves)
dt_classifier.fit( x_train, y_train )
tree.export_graphviz(dt_classifier, feature_names=all_cols,out_file="./tree.dot")

predicted = dt_classifier.predict( x_test )

print "Going to plot unweighted"


for class_to_plot in [0,1]:
	y_conf = []
	for i in range(len(y_test)):
		y_conf.append(probs[i][class_to_plot])
	precision, recall, thresholds = precision_recall_curve(y_test_num, y_conf, pos_label=class_to_plot)
	plt.plot(recall,precision)
	if(class_to_plot == 0):
		plt.axis([0,1,0.8,1])
		plt.yticks(np.arange(0.8, 1.05, 0.1))
	else:
		plt.axis([0,1,0,1])
		plt.yticks(np.arange(0, 1.1, 0.1))
	print "Checking class",class_to_plot

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('DT without instance weights for class ' + str(class_to_plot))
	plt.grid(b=True, which='major', axis='both', color='black', linestyle='-', alpha=0.3)
	plt.xticks(np.arange(0, 1.1, 0.1))
	filename = "./plots/dt_unweighted_"+str(class_to_plot)+"maxLeaf_"+str(numLeaves)+".png"
	plt.savefig(filename)
	# plt.clf()




temp = dt_classifier.feature_importances_
print "NumCols : "+str(len(all_cols))
print "Feautre Importances - length"+str(len(temp))
print temp

print "Feats"



for i in range(0, len(temp)):
	print all_cols[i+2]," ",temp[i]