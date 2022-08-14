import pandas as pd
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle

df = pd.read_csv('iris.csv')


array= df.values
x= array[:,0:4]
y= array[:,4]
validation_size=0.20
seed=6
x_train, x_test,y_train, y_test=model_selection.train_test_split(x,y,test_size=validation_size, random_state=seed)
seed=6
scoring='accuracy'

lda = LinearDiscriminantAnalysis()
lda.fit(x, y)

pickle.dump(lda,open('irisflower_pred_model','wb'))