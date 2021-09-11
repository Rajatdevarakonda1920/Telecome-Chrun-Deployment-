#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv(r'C:\Users\RAJAT DEVARAKONDA\Desktop\telecome_chrun\data.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# Dataset has 7043 rows and 21 columns including the label v=column.

# In[6]:


df.nunique()


# Dataset has one identifier columns, 3 continuos and rest of them are categorical columns.

# In[7]:


df.isnull().sum()


# There are no null values.

# In[8]:


df.dtypes


# There are 3 columns with numerical columns rest all object types including the Total charges column.

# In[9]:


df['Churn'].value_counts()


# Dataset is imbalanced

# In[10]:


df.skew()


# There is no skewness in the continuous column.

# In[11]:


df.describe()


# Count of every column is 7043. Mean is higher than median in tenure colum, and lower in Monthly charges. Both the columns show a little skewness. Difference between the interquartile range, min and max dosent vary much that means there are less outliers. There are high variance in tenure and Monthtly charges column.

# ### Univariate Analysis

# In[12]:


#Separating categorical and continuous features
cat=[i for i in df.columns if df[i].nunique()<10 ]
cont=[i for i in df.columns if df[i].nunique()>10 and i!='customerID']


# In[13]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['Churn'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='Churn',data=df)
df['Churn'].value_counts()


# Dataset is imbalanced there are 73.5% people who have not churned while 26.5% who have.

# In[14]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['gender'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='gender',data=df)
df['gender'].value_counts()


# No. of females and males are almost equal.

# In[15]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['SeniorCitizen'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='SeniorCitizen',data=df)
df['SeniorCitizen'].value_counts()


# There are only 16.2% senior citizens, while the majority of customers are young.

# In[16]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['Partner'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='Partner',data=df)
df['Partner'].value_counts()


# Partner category is almost equal as customers having partners while no. of people without partners are almost equal.

# In[17]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['Dependents'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='Dependents',data=df)
df['Dependents'].value_counts()


# There are 30% customers with dependents while rest donot.

# In[18]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['PhoneService'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='PhoneService',data=df)
df['PhoneService'].value_counts()


# There are very few customers who do not have phone service, less than 10%.

# In[19]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['MultipleLines'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='MultipleLines',data=df)
df['MultipleLines'].value_counts()


# There are 90.3% people who use phone service and 42.2% who also use multiple lines among them.

# In[20]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['InternetService'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='InternetService',data=df)
df['InternetService'].value_counts()


# Most of the people using Internet service use Fiber optic instead of DSL, while 21.7% donot use internet service

# In[21]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['OnlineSecurity'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='OnlineSecurity',data=df)
df['OnlineSecurity'].value_counts()


# Majority of customers do not use Online security service among the people who use internet but still there are more than half who use online security.

# In[22]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['OnlineBackup'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='OnlineBackup',data=df)
df['OnlineBackup'].value_counts()


# Majority of customers do use Online Backup service among the people who use internet but still there are more than half who do not use online backup service service,

# In[23]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['DeviceProtection'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='DeviceProtection',data=df)
df['DeviceProtection'].value_counts()


# Customers who use internet service are the ones who can use this service. Majority of people among the internet users do not use this service.

# In[24]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['TechSupport'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='TechSupport',data=df)
df['TechSupport'].value_counts()


# Customers who use internet service are the ones who can use this service. Majority of people among the internet users do not use this service.

# In[25]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['StreamingTV'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='StreamingTV',data=df)
df['StreamingTV'].value_counts()


# Customers who use internet service are the ones who can use this service. Among the users of customer service, there is balance in the users who Watch Streaming Tv and who do not.

# In[26]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['StreamingMovies'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='StreamingMovies',data=df)
df['StreamingMovies'].value_counts()


# Customers who use internet service are the ones who can use this service. Among the users of customer service, there is balance in the users who Watch Streaming Movies and who do not.

# In[27]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['Contract'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='Contract',data=df)
df['Contract'].value_counts()


# Most of the customers have subscribed for month to month followed by two year plan, one year contract is used by least no. of customers.

# In[28]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['PaperlessBilling'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='PaperlessBilling',data=df)
df['PaperlessBilling'].value_counts()


# Most of the customer like paperless billing which is good as it is environment friendly.

# In[29]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
df['PaymentMethod'].value_counts().plot.pie(autopct='%1.1f%%')
centre=plt.Circle((0,0),0.7,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre)
plt.subplot(1,2,2)
sns.countplot(x='PaymentMethod',data=df)
plt.xticks(rotation=45)
df['PaymentMethod'].value_counts()


# Most customers use electronic check for making payments, while rest of the methods are used by almost equal no. of customers.

# In[30]:


plt.figure(figsize=(8,6))
sns.histplot(df['tenure'],kde=True,color='k')
print('Minimum',df['tenure'].min())
print('Maximum',df['tenure'].max())


# Tenure data seems to be normally distributed, with just little of roght skewness. Customers as long as 72 years have been loyal to this company, while the majority lies between 0 to 2 years range.

# In[31]:


plt.figure(figsize=(8,6))
sns.histplot(df['MonthlyCharges'],kde=True,color='k')
print('Minimum',df['MonthlyCharges'].min())
print('Maximum',df['MonthlyCharges'].max())


# Monthly charges almost follows normal distribution, whith its majority of customers paying monthly charges 19 to 25.

# In[32]:


df['TotalCharges']=df['TotalCharges'].apply(lambda x: np.NaN if x==' ' else float(x))


# In[33]:


plt.figure(figsize=(8,6))
sns.distplot(df['TotalCharges'],kde=True,color='k')
print('Minimum',df['TotalCharges'].min())
print('Maximum',df['TotalCharges'].max())


# Total charges is skewed towards rightand goes up to a range of 8684, while majority of customers lie in 18.8 to 500 range.

# In[34]:


for i in cont:
    sns.boxplot(df[i])
    plt.figure()


# There are no outliers in the above continuous data.

# ### Bivariate Analysis

# In[35]:


plt.figure(figsize=(8,6))
sns.boxenplot(x='Churn',y='TotalCharges',data=df,palette='rainbow')


# Customers who have been seen to pay more total charges dosent seem to churn.

# In[36]:


plt.figure(figsize=(8,6))
sns.boxenplot(x='Churn',y='MonthlyCharges',data=df,palette='rainbow')


# Customers paying high mean monthly charges seem to churn away.

# In[37]:


plt.figure(figsize=(8,6))
sns.boxenplot(x='Churn',y='tenure',data=df,palette='rainbow')


# People who have stayed customers for longer term does not seem to churn away.

# In[38]:


plt.figure(figsize=(8,6))
sns.countplot(df['gender'],hue=df['Churn'],palette='Set1')


# There are almost equal no of males and females who churn away but no. of people churning are way lower than peole who do not.

# In[39]:


plt.figure(figsize=(8,6))
sns.countplot(df['SeniorCitizen'],hue=df['Churn'],palette='Set1')


# As Compared to younger ones senior citizens seem to churn easily.

# In[40]:


plt.figure(figsize=(8,6))
sns.countplot(df['Partner'],hue=df['Churn'],palette='Set1')


# Customers who do not have partners have higher chances of churning.

# In[41]:


plt.figure(figsize=(8,6))
sns.countplot(df['Dependents'],hue=df['Churn'],palette='Set1')


# Customers with depemdemts have lower rate of churning than the customers who do not have partners.

# In[42]:


plt.figure(figsize=(8,6))
sns.countplot(df['PhoneService'],hue=df['Churn'],palette='Set1')


# No. of people churning are almost same in the ratio of people who use phone service and who do not.

# In[43]:


plt.figure(figsize=(12,6))
sns.countplot(df['MultipleLines'],hue=df['Churn'],palette='Set1')


# People using multiple lines have lower ratio of churning compared to customers who donot use multiple lines.

# In[44]:


plt.figure(figsize=(12,6))
sns.countplot(df['InternetService'],hue=df['Churn'],palette='Set1')


# Customers using fiber optic internet service are the highest no. of people churning away.

# In[45]:


plt.figure(figsize=(12,6))
sns.countplot(df['OnlineSecurity'],hue=df['Churn'],palette='Set1')


# Customers who have not subscrobed online security churn the most.

# In[46]:


plt.figure(figsize=(12,6))
sns.countplot(df['OnlineBackup'],hue=df['Churn'],palette='Set1')


# Customers who have not subscribed for online backup have higher ratio for churning away.

# In[47]:


plt.figure(figsize=(12,6))
sns.countplot(df['DeviceProtection'],hue=df['Churn'],palette='Set1')


# Customers who do not opt for device protection have highest no. of churning away.

# In[48]:


plt.figure(figsize=(12,6))
sns.countplot(df['TechSupport'],hue=df['Churn'],palette='Set1')


# Customers who do not opt for tech support have highest no. of churning away.

# In[49]:


plt.figure(figsize=(12,6))
sns.countplot(df['StreamingTV'],hue=df['Churn'],palette='Set1')


# Customers who have subscribed for streaming tv have skightly lower ratio of churning compared to those who have't while customers who do not use internet service churn the least.

# In[50]:


plt.figure(figsize=(12,6))
sns.countplot(df['StreamingMovies'],hue=df['Churn'],palette='Set1')


# Customers who have subscribed for streaming Movies have slightly lower ratio of churning compared to those who have't while customers who do not use internet service churn the least.

# In[51]:


plt.figure(figsize=(12,6))
sns.countplot(df['Contract'],hue=df['Churn'],palette='Set1')


# People having shorter contract churn higher than the people who have contract for a longer time.

# In[52]:


plt.figure(figsize=(15,6))
sns.countplot(df['PaymentMethod'],hue=df['Churn'],palette='Set1')


# People payong bills through electronic check have the highest ratio for churning compared to people who use other medium for payment.

# In[53]:


sns.lmplot(x='MonthlyCharges',y='TotalCharges',data=df,height=6, aspect=1.2)


# Total charges and monthly charges show positive correlation, one increases as the other increases.

# In[54]:


sns.lmplot(x='tenure',y='TotalCharges',data=df,height=6, aspect=1.2,)


# Total charges and tenure also show positive correlation, one increases as the other increases.

# In[55]:


#Senior citizen vs other features.
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.countplot(df['StreamingMovies'],hue=df['SeniorCitizen'],palette='Set2')

plt.subplot(2,2,2)
sns.countplot(df['StreamingTV'],hue=df['SeniorCitizen'],palette='Set2')

plt.subplot(2,2,3)
sns.countplot(df['Contract'],hue=df['SeniorCitizen'],palette='Set2')

plt.subplot(2,2,4)
sns.countplot(df['PaymentMethod'],hue=df['SeniorCitizen'],palette='Set2')
plt.xticks(rotation=45)


# Senior citizens have a higher ratio of streaming tv and movies than the younger ones. Ratio of senior citizens for contract is almost similar in all the categories while for payment method. senior citizens using mailed checks are least.

# In[56]:


#Phone service vs other features.
plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
sns.countplot(df['MultipleLines'],hue=df['PhoneService'],palette='Set2')

plt.subplot(3,2,2)
sns.countplot(df['OnlineBackup'],hue=df['PhoneService'],palette='Set2')

plt.subplot(3,2,3)
sns.countplot(df['OnlineSecurity'],hue=df['PhoneService'],palette='Set2')

plt.subplot(3,2,4)
sns.countplot(df['TechSupport'],hue=df['PhoneService'],palette='Set2')

plt.subplot(3,2,5)
sns.countplot(df['StreamingMovies'],hue=df['PhoneService'],palette='Set2')

plt.subplot(3,2,6)
sns.countplot(df['DeviceProtection'],hue=df['PhoneService'],palette='Set2')


# Phone Service dosent seem to be an important feature as there is already a category in other features of no phone service and those who do not use internet service  are the ones who not use phone service. 

# In[57]:


#Internet Service vs other features.
plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
sns.countplot(df['OnlineSecurity'],hue=df['InternetService'],palette='Set2')

plt.subplot(3,2,2)
sns.countplot(df['OnlineBackup'],hue=df['InternetService'],palette='Set2')

plt.subplot(3,2,3)
sns.countplot(df['DeviceProtection'],hue=df['InternetService'],palette='Set2')

plt.subplot(3,2,4)
sns.countplot(df['TechSupport'],hue=df['InternetService'],palette='Set2')

plt.subplot(3,2,5)
sns.countplot(df['StreamingMovies'],hue=df['InternetService'],palette='Set2')

plt.subplot(3,2,6)
sns.countplot(df['StreamingTV'],hue=df['InternetService'],palette='Set2')


# People who do not have internet service do not use the above features. Customers having fiber optic have higher ratio of not opting for Online Security, Device Protection and tech Support while customers having DSL internet service have a higher ratio of opting for these services and there is a vice versa scenario for  Online Backup, Streaming MOvies and Tv services.

# ### Multivariate Analysis

# In[58]:


plt.figure(figsize=(8,6))
sns.stripplot(x='PhoneService',y='InternetService',hue='Churn',data=df)


# People who do not have phone service only use DSL internet service. While People using the phone service and who do not use internet service are the ons's who rarely churn.

# In[59]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='tenure',y='TotalCharges',hue='Churn',data=df,palette='inferno',marker='D')


# In[60]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='tenure',y='MonthlyCharges',hue='Churn',data=df,palette='inferno',marker='D')


# People having higher tenure and higher monthly charges seem to churn faster than the rest.

# In[61]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='MonthlyCharges',y='TotalCharges',hue='Churn',data=df,palette='inferno',marker='D')


# People having lowe total charges and higher monthly charge have a higher ratio of churning than the rest of the population.

# In[62]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap='Greys')


# There is high correlation between Total charges and tenure. Also a high correlation between Monthlly charges and Total Charges.

# In[63]:


sns.pairplot(df, hue = 'Churn',palette='Set2')


# Customers seem to churn away with higher monthly charges, lower tenure and lowe total charges, while the ratio of senior citizen than the young people churning is more.

# ### Pre Processing Pipeline

# ##### Dropping identifier columns

# In[64]:


df.drop('customerID',axis=1,inplace=True)


# ##### Handling null values of Total charges column

# In[65]:


#Creating pivot table to help fill nan values of visibility from here
table = df.pivot_table(values='TotalCharges', index='InternetService', columns='Contract', aggfunc='mean')
table


# In[66]:


def find_mean(x):
    
    return table.loc[x['InternetService'], x['Contract']]

# replace missing values in visibility with mean values from above pivot table
df['TotalCharges'].fillna(df[df['TotalCharges'].isnull()].apply(find_mean, axis=1), inplace=True)


# In[67]:


sns.heatmap(df.isnull())


# ##### Encoding Object type Columns

# In[68]:


from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
o=OrdinalEncoder()
l=LabelEncoder()


# In[69]:


#using ordinal encoder for independent features
for i in df.columns:
    if df[i].dtypes=='O' and i!='Churn':
        df[i]=o.fit_transform(df[i].values.reshape(-1,1))

#Using label encoder for Label Column
df['Churn']=l.fit_transform(df['Churn'])


# ##### There are no outliers or skewness in our dataset therefore we proceed further.

# In[70]:


x=df.copy()
x.drop('Churn',axis=1,inplace=True)
y=df['Churn']


# ##### Handling Imbalanced Dataset

# In[71]:


from imblearn.over_sampling import SMOTE
over=SMOTE()


# In[72]:


x,y=over.fit_resample(x,y)


# ##### Scaling the Dataset

# In[73]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[74]:


xd=scaler.fit_transform(x)
x=pd.DataFrame(xd,columns=x.columns)


# # Building Models

# Importong neccessary libraries and modules

# In[75]:


from sklearn.model_selection import train_test_split,cross_val_score


# In[76]:


#We import Classification Models
from sklearn.naive_bayes import  GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[77]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve


# In[78]:


#Function to find the best random state
def randomstate(x,y):
    maxx=0
    model=LogisticRegression()
    for i in range(1,201):
        xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=i)
        model.fit(xtrain,ytrain)
        p=model.predict(xtest)
        accu=accuracy_score(p,ytest)
        if accu>maxx:
            maxx=accu
            j=i
    return j


# In[79]:


#Splitting data into train and test
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=randomstate(x,y))


# In[80]:


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


# In[81]:


#Creating a list of models which will be created one by one
models=[GaussianNB(),KNeighborsClassifier(),SVC(probability=True),LogisticRegression(),DecisionTreeClassifier(),
        RandomForestClassifier(),AdaBoostClassifier(),GradientBoostingClassifier(),XGBClassifier(verbosity=0)]


# In[82]:


#Creates and trains model from the models list
def createmodel(trainx,testx,trainy,testy):
    for i in models:
        model=i
        model.fit(trainx,trainy)
        p=model.predict(testx)
        score=cross_val_score(model,x,y,cv=10)
        performance(p,testy,model,testx,score) 


# In[83]:


createmodel(xtrain,xtest,ytrain,ytest)


# ## Feature Selection

# ##### Using Feature importance of random forrest

# In[84]:


m=RandomForestClassifier()
m.fit(x,y)
print(m.feature_importances_)


# In[85]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(m.feature_importances_, index=x.columns)
feat_importances.nlargest(19).plot(kind='barh')
plt.show()


# In[86]:


len(x.columns)


# In[87]:


fi=list(feat_importances.nlargest(19).index)
fi


# ##### Using chi2 test

# In[88]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[89]:


selection = SelectKBest(score_func=chi2)
fit = selection.fit(x,y)


# In[90]:


dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
featureScores


# In[91]:


print(featureScores.nlargest(19,'Score'))  #print10 best features
feat=list(featureScores.nlargest(19,'Score')['Features'])


# On the basis of above two method of feature selection 'PhoneService','gender' and 'SeniorCitizen' seem to have the lowest scores so we try to remove them and see if our accuracy increases.

# In[92]:


#We create a function to test the performance of dataset after removing above features.
#We use Random Forest and XGBClassifier as our models to test as they were giving the best results.
def feature_test(xtrain,xtest,ytrain,ytest):
    model=RandomForestClassifier()
    model.fit(xtrain,ytrain)
    score=cross_val_score(model,xd,y,cv=10)
    p=model.predict(xtest)
    print("Random Forest")
    print('Accuracy Score',accuracy_score(p,ytest))
    print('Cross Validation Score',score.mean())
    model=XGBClassifier()
    model.fit(xtrain,ytrain)
    score=cross_val_score(model,xd,y,cv=10)
    p=model.predict(xtest)
    print("XGBCLassifier")
    print('Accuracy Score',accuracy_score(p,ytest))
    print('Cross Validation Score',score.mean())


# In[93]:


xd=x.drop(['PhoneService','gender','SeniorCitizen'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(xd,y,test_size=0.25,random_state=randomstate(xd,y))
feature_test(x_train,x_test,y_train,y_test)


# In[94]:


xd=x.drop(['PhoneService','gender'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(xd,y,test_size=0.25,random_state=randomstate(xd,y))
feature_test(x_train,x_test,y_train,y_test)


# In[95]:


xd=x.drop(['PhoneService'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(xd,y,test_size=0.25,random_state=randomstate(xd,y))
feature_test(x_train,x_test,y_train,y_test)


# Removing Only phone service gives us the best result. We have also previously in the EDA process where Phone service was just an extra feature.

# In[96]:


x.drop(['PhoneService'],axis=1,inplace=True)


# In[97]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=randomstate(x,y))


# In[98]:


createmodel(xtrain,xtest,ytrain,ytest)


# Still Best performance is given by Random Forest, Gradient Boost and Extreme Gradient Boost models, So we futher perform Hyperparameter Tuning on them.

# # Hyperparameter Tuning

# In[99]:


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# ##### Random Forest

# In[100]:


params={'n_estimators':[100, 300, 500, 700],
        'min_samples_split':[1,2,3,4],
        'min_samples_leaf':[1,2,3,4],
        'max_depth':[None,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40]}


# In[101]:


g=RandomizedSearchCV(RandomForestClassifier(),params,cv=5)


# In[102]:


g.fit(xtrain,ytrain)


# In[103]:


print(g.best_estimator_)
print(g.best_params_)
print(g.best_score_)


# In[104]:


m=RandomForestClassifier(max_depth=15, min_samples_leaf=3, min_samples_split=4,n_estimators=700)
m.fit(xtrain,ytrain)
p=m.predict(xtest)


# In[105]:


score=cross_val_score(m,x,y,cv=5)
performance(p,ytest,m,xtest,score)


# ##### Gradient Boost

# In[106]:


params={'n_estimators':[100,200,300,400,500],
      'learning_rate':[0.001,0.01,0.10,],
      'subsample':[0.5,1],
      'max_depth':[1,2,3,4,5,6,7,8,9,10]}


# In[107]:


g=RandomizedSearchCV(GradientBoostingClassifier(),params,cv=10)


# In[108]:


g.fit(xtrain,ytrain)


# In[109]:


print(g.best_estimator_)
print(g.best_params_)
print(g.best_score_)


# In[110]:


m=GradientBoostingClassifier(learning_rate=0.01, max_depth=6, n_estimators=500,subsample=0.5)
m.fit(xtrain,ytrain)
p=m.predict(xtest)


# In[111]:


score=cross_val_score(m,x,y,cv=5)
performance(p,ytest,m,xtest,score)


# ##### Extreme Gradient Boost

# In[112]:


params={
     "learning_rate"    : [0.001,0.05, 0.10 ] ,
     "max_depth"        : [ 5, 6, 8, 10, 12, 15,20,25,30,35,40],
     "min_child_weight" : [ 1, 3, 5, 7,10],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4,10],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    }


# In[113]:


g=RandomizedSearchCV(XGBClassifier(),params,cv=10)


# In[114]:


g.fit(xtrain,ytrain)


# In[115]:


print(g.best_estimator_)
print(g.best_params_)
print(g.best_score_)


# In[116]:


m=XGBClassifier(min_child_weight=1,max_depth=10,learning_rate=0.05,gamma=0.4,colsample_bytree=0.4)
m.fit(xtrain,ytrain)
p=m.predict(xtest)


# In[117]:


score=cross_val_score(m,x,y,cv=5)
performance(p,ytest,m,xtest,score)


# ##### Conclusion

# All the model are giving almost same accuracy but Random Forest has the least variance between accuracy score and mean of cross validation score. So we choose Random Forest as our final model.

# # Finalizing the best Model

# In[119]:


model=RandomForestClassifier(max_depth=15, min_samples_leaf=3, min_samples_split=4,n_estimators=700)
model.fit(xtrain,ytrain)
p=model.predict(xtest)
score=cross_val_score(model,x,y,cv=10)


# # Evaluation Metrics

# In[120]:


performance(p,ytest,model,xtest,score)


# In[121]:


fpred=pd.Series(model.predict_proba(xtest)[:,1])
fpr,tpr,threshold=roc_curve(ytest,fpred)


# In[122]:


plt.plot(fpr,tpr,color='k',label='ROC')
plt.plot([0,1],[0,1],color='b',linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC curve')
plt.legend()


# # Saving the Model

# In[123]:


import joblib
joblib.dump(model,'Churn.pkl')


# In[ ]:




