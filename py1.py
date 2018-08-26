from data import *
from datalabels import *
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
import matplotlib.pyplot as plt
import seaborn as sns

C1Data=pd.read_csv(io='data_classifier1.csv')
#C1Data.drop(['District','Month','Year'],axis=1,inplace=True)
Districts=['Nagpur','Pune','Nashik','Aurangabad','Amravati','Solapur','Yavatmal','Latur']
train=pd.DataFrame()
test=pd.DataFrame()
for district in Districts:
	train=train.append(C1Data[C1Data['District']==district][:150])
	test=test.append(C1Data[C1Data['District']==district][150:])
	
train.drop(['District','Month','Year'],axis=1,inplace=True)
test.drop(['District','Month','Year'],axis=1,inplace=True)

randomforest=RandomForestClassifier(n_estimators=100)
randomforest.fit(train.drop(['Drought Classification'],axis=1),train['Drought Classification'])
C1Output=randomforest.predict(test.drop(['Drought Classification'],axis=1))
accuracy=accuracy_score(test['Drought Classification'],C1Output)
precision=precision_score(test['Drought Classification'],C1Output,average='macro')
recall=recall_score(test['Drought Classification'],C1Output,average='macro')
cm=confusion_matrix(test['Drought Classification'],C1Output)
dfcm=pd.DataFrame(cm)
plt.figure(1)
sns.heatmap(dfcm,annot=True)
plt.title('Classifier 1 Random Forest Classifier Confusion Matrix, Accuracy=%f, Precision=%f, Recall=%f'%(accuracy,precision,recall))
plt.show()