#Importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#Importing the data_set
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
#extracting feature from df_train
y=df_train.iloc[:,11]
df_train=df_train.iloc[:,0:11]
#df_train = df_train.drop(['Cabin','Ticket','Name'], axis = 1) 
#deciding continuous and category variables
X_cat=df_train.iloc[:,[1,3,5,6,10]]
X_cont=df_train.iloc[:,[0,2,4,7,8,9]]
#assigning vacant values to null
X_cont=X_cont.values
X_cat=X_cat.values

X_cont[pd.isnull(X_cont)]='NaN'
X_cat[pd.isnull(X_cat)]='NaN'
#Applying label encoder to the categorical values that are character
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,Imputer,StandardScaler
label_x=LabelEncoder()
X_cat[:,1]=label_x.fit_transform(X_cat[:,1])
X_cat[:,4]=label_x.fit_transform(X_cat[:,4])
#Applying one hot encoding
hot_x=OneHotEncoder(categorical_features=[0,1,2,3,4])
X_cat=hot_x.fit_transform(X_cat).toarray()
#Applying label encoder to the continuous values that are character

X_cont[:,1]=label_x.fit_transform(X_cont[:,1])
X_cont[:,3]=label_x.fit_transform(X_cont[:,3])
X_cont[:,5]=label_x.fit_transform(X_cont[:,5])
#Dealing with missing values of continuous variables 
imp=Imputer(missing_values='NaN',axis=0)
X_cont=imp.fit_transform(X_cont)
#Applying StandardScalar to the model
sc_x=StandardScaler()
X_cont=sc_x.fit_transform(X_cont)
#Now concatenating categorical variable and continuous variable
x=np.zeros((891,29))

x[:,0:23]=X_cat
x[:,23:29]=X_cont
#Splitting the training set into test set and train set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
#Running model
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators = 250, random_state = 42)
clf.fit(X_train,y_train)
#Checking accuracy of the model
y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)
#Drawing confusion matrix 
from sklearn.metrics import confusion_matrix
conf=confusion_matrix(y_pred,y_test)


