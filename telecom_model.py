import pandas as pd
from sklearn.preprocessing import StandardScaler
df=pd.read_csv(r'C:\Users\ADMIN\Desktop\ml\supervised\classification\telecom_dt')
print(df.head())
y_column=df['Customer Status']
y=df['Customer Status'].values
cd=['Customer Status','Unnamed: 0']
x=df.drop(cd,axis=1)
print(x.head())
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(model.score(x_test,y_test))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))#0.8119233498935415
from imblearn.combine import SMOTEENN
smt=SMOTEENN()
xs,ys=smt.fit_resample(x,y)
xs_train,xs_test,ys_train,ys_test=train_test_split(xs,ys,test_size=0.2,random_state=42)
scalers=StandardScaler()
xs_train=scalers.fit_transform(xs_train)
xs_test=scalers.transform(xs_test)
model_s=DecisionTreeClassifier(criterion='gini',random_state=42,max_depth=6,min_samples_leaf=8)
model_s.fit(xs_train,ys_train)
ys_pred=model_s.predict(xs_test)
print(model_s.score(xs_test,ys_test))#0.8860103626943006
print(confusion_matrix(ys_test,ys_pred))
print(classification_report(ys_test,ys_pred))
from sklearn.ensemble import RandomForestClassifier
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scalerr=StandardScaler()
x_train=scalerr.fit_transform(x_train)
x_test=scalerr.transform(x_test)
model_rf=RandomForestClassifier()
model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)
model_rf.fit(x_train,y_train)
y_pred=model_rf.predict(x_test)
print(model_rf.score(x_test,y_test))#0.7054648687012065
print(classification_report(y_test, y_pred, labels=[0,1]))
smtrt=SMOTEENN()
xsrt,ysrt=smtrt.fit_resample(x,y)
xsrt_train,xsrt_test,ysrt_train,ysrt_test=train_test_split(xsrt,ysrt,test_size=0.2,random_state=42)
scalersrt=StandardScaler()
xsrt_train=scalersrt.fit_transform(xsrt_train)
xsrt_test=scalersrt.transform(xsrt_test)
model_srt=DecisionTreeClassifier(criterion='gini',random_state=42,max_depth=6,min_samples_leaf=8)
model_srt.fit(xsrt_train,ysrt_train)
ysrt_pred=model_srt.predict(xsrt_test)
print(model_srt.score(xsrt_test,ysrt_test))#0.8887722980062959
print(confusion_matrix(ysrt_test,ysrt_pred))
print(classification_report(ysrt_test,ysrt_pred))
from sklearn.neighbors import KNeighborsClassifier
modelknn=KNeighborsClassifier(n_neighbors=1)
#from sklearn.model_selection import GridSearchCV
#import numpy as np
#param_grid=({'n_neighbors':np.arange(1,12)})
#gscv=GridSearchCV(modelknn,param_grid=param_grid,cv=10)
#gscv.fit(x,y)
#print(gscv.best_params_)     #n=1
from sklearn.model_selection import train_test_split
xk_train,xk_test,yk_train,yk_test=train_test_split(x,y,test_size=0.2,random_state=42)
scalerk=StandardScaler()
xk_train=scalerk.fit_transform(xk_train)
xk_test=scalerk.transform(xk_test)
modelknn.fit(xk_train,yk_train)
yk_pred=modelknn.predict(xk_test)
print(modelknn.score(xk_test,yk_test))#0.74
print(confusion_matrix(yk_test,yk_pred))
print(classification_report(yk_test,yk_pred))
import pickle
pickle.dump(model_srt,open("telecom_model.pkl",'wb'))
telecom_churn=pickle.load(open("telecom_model.pkl",'rb'))
print(telecom_churn.predict(xsrt_test[0:]))


