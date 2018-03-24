import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt



data_dir = './input/'	# needs trailing slash

# validation split, both files with headers and the Happy column
train_file = data_dir + 'census.data.v3.csv'
test_file = data_dir + 'census.test.v3.csv'

###

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

# print train['instance weight'].values

# numeric x

numeric_cols = ['age','wage per hour','capital gains','capital losses','dividends from stocks','num persons worked for employer','weeks worked in year',]
x_num_train = train[ numeric_cols ].as_matrix()
x_num_test = test[ numeric_cols ].as_matrix()

# scale to <0,1>

max_train = np.amax( x_num_train, 0 )
max_test = np.amax( x_num_test, 0 )		# not really needed

x_num_train = x_num_train / max_train
x_num_test = x_num_test / max_train		# scale test by max_train

# y

y_train = train.Label
# print y_train
y_test = test.Label

# categorical

cat_train = train.drop( numeric_cols + ['Label','instance weight','migration code-change in msa','migration code-change in reg','migration code-move within reg','migration prev res in sunbelt'], axis = 1 )
cat_test = test.drop( numeric_cols + ['Label','instance weight'], axis = 1 )

cat_train.fillna( 'NA', inplace = True )
cat_test.fillna( 'NA', inplace = True )

x_cat_train = cat_train.T.to_dict().values()
x_cat_test = cat_test.T.to_dict().values()

# vectorize

vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( x_cat_train )
vec_x_cat_test = vectorizer.transform( x_cat_test )

print(type(vec_x_cat_train))
"""
# complete x

x_train = np.hstack(( x_num_train, vec_x_cat_train ))
x_test = np.hstack(( x_num_test, vec_x_cat_test ))

svm_classifier = svm.SVC(probability=True)
svm_classifier.fit( x_train, y_train , sample_weight=train['instance weight'].values)

predicted = svm_classifier.predict( x_test )
expected = y_test

print(metrics.classification_report(expected, predicted))

probs = svm_classifier.predict_proba(x_test)
y_conf=[]
for i in range(len(y_test)):
	if y_test[i] == 'Pos':
		y_test[i] = 1
	else:
		y_test[i] = -1

for i in range(len(expected)):
	y_conf.append(probs[i][1])

precision, recall, thresholds = precision_recall_curve(expected, y_conf, pos_label=-1, sample_weight=test['instance weight'].values)
print precision
print recall
print thresholds
plt.plot(recall,precision)
plt.axis([0,1,0,1])
plt.show()
"""
