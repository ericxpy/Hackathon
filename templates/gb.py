from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

depth = 5

data = pd.read_csv("tr.csv",header=0)
data= data.drop([data.columns[0]],axis=1)

label= data['prod']
feat= data[data.columns[1:]]

xtrain, xval, ytrain, yval = train_test_split(feat, label)

gb1=GradientBoostingRegressor(max_depth=depth,n_estimators=120,learning_rate=0.1)
gb1.fit(xtrain,ytrain)

'''
testd = pd.read_csv("td.csv",header=0)
testd = testd.drop([testd.columns[0]],axis=1)

test = testd[data.columns[1:]]

yval = testd['prod']'''

errors= [mean_squared_error(yval,ypred) for ypred in gb1.staged_predict(xval)]
plt.plot(np.array(range(120)),errors)
bestn = np.argmin(errors)

gb2=GradientBoostingRegressor(max_depth=depth,n_estimators=bestn,learning_rate=0.1)
gb2.fit(feat,label)

final = pd.read_csv("corn.csv",header=0)

test = pd.read_csv("td.csv",header=0)
test = test.drop([test.columns[0]],axis=1)
tlabel= test['prod']
tfeat= test[test.columns[1:]]
y2 = gb2.predict(final[final.columns[1:5]])
    

final['prediction'] = pd.Series(y2, index=final.index)
final.to_csv('final.csv')