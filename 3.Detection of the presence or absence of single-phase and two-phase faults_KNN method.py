import pandas as pd
from sklearn import  preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from  sklearn import  metrics
df = pd.read_csv('finalExcel3.csv')
df=shuffle(df,random_state=0)
X = df[['t', 'ia', 'ib', 'ic','isq','isd', 'vsd', 'vsq', 'ha', 'hb', 'hc', 'speed', 'angle', 'torque']].values
Y = df['output'].values
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 ,random_state=4)
print('Train Set :',X_train.shape,Y_train.shape)
print('Test set :',X_test.shape,Y_test.shape)
k=9
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train,Y_train)
yhat = neigh.predict(X_test)
print('Train Set acc: ' , metrics.accuracy_score(Y_train,neigh.predict(X_train)) )
print('Test Set acc: ' , metrics.accuracy_score(Y_test,yhat))