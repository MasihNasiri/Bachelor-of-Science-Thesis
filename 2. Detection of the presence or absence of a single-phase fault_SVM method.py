import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
zf = pd.read_csv('finalExcel2.csv')
zf = shuffle(zf, random_state=0)
X = zf[['t', 'ia', 'ib', 'ic','isq','isd', 'vsd', 'vsq', 'ha', 'hb', 'hc', 'speed', 'angle', 'torque']].values
Y = zf['output'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=4)
print('Train Set :', X_train.shape, Y_train.shape)
print('Test set :', X_test.shape, Y_test.shape)
clf = svm.SVC(kernel='poly')
#clf = svm.SVC(kernel='sigmoid')
#clf = svm.SVC(kernel='rbf')
clf.fit(X_train, Y_train)
yhat = clf.predict(X_test)
print('Train Set acc: ', metrics.accuracy_score(Y_train, clf.predict(X_train)))
print('Test Set acc: ', metrics.accuracy_score(Y_test, yhat))