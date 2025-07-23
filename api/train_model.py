import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
data={
    'math':[50,60,70,20,90,50,90],
    'science':[40,30,50,23,90,60,80],
    'english':[30,20,40,60,90,30,80],
    'result':['Fail','Fail','Fail','Fail','pass','Fail','pass']
}
df=pd.DataFrame(data)
df['result']=df['result'].map({'pass':1,'Fail':0})
#train
x=df[['math','science','english']]
y=df['result']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(x_test)
#70%ip  30%ip,70%op  ,30%   
#call the model 
model=LogisticRegression()
model.fit(x_train,y_train)
joblib.dump(model,'model.pkl')



