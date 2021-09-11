import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve

df=pd.read_csv(r'C:\Users\RAJAT DEVARAKONDA\Desktop\Data_Science\PROJECTS DEPLOYMENTS\telecome_chrun\data.csv')
print(df.shape)
X = df.drop('Churn',axis=1)
Y = df['Churn']
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3,random_state=60,stratify=Y)
print("Shape of Training Data",xtrain.shape)
print("Shape of Testing Data",xtest.shape)
print("Expenses Rate in Training Data",ytrain.mean())
print("Expenses Rate in Testing Data",ytest.mean())
#scaling
scl = StandardScaler() 
#scl the feature based on mean and std ...i have preferd standrisation coz features can take form of normal distribution
#this will make models to learn weights easily and also will make less sentive to outliers 
scl.fit(xtrain)
xtrain_scl = pd.DataFrame(scl.transform(xtrain))
xtest_scl = pd.DataFrame(scl.fit_transform(xtest))
xtest_scl.columns = xtest.columns 
xtrain_scl.columns = xtrain.columns 

#To evaluate performances of all the models
def performance(p,ytest,m,xtest,s):
    print('------------------------------------',m,'------------------------------------')
    print('Accuracy',np.round(accuracy_score(p,ytest),4))
    print('----------------------------------------------------------')
    print('Mean of Cross Validation Score',np.round(s.mean(),4))
    print('----------------------------------------------------------')
    print('AUC_ROC Score',np.round(roc_auc_score(ytest,m.predict_proba(xtest)[:,1]),4))
    print('----------------------------------------------------------')
    print('Confusion Matrix')
    print(confusion_matrix(p,ytest))
    print('----------------------------------------------------------')
    print('Classification Report')
    print(classification_report(p,ytest))
model = GradientBoostingClassifier(learning_rate=0.75,max_depth=1,n_estimators=300)
model.fit(xtrain, ytrain)
import joblib
joblib.dump(model,'Churn.pkl')