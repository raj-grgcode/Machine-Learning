from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
data={
    'study_hr':[2,3,4,5,6,7,8,9],
    'passed':[0,0,0,0,1,1,1,1]
}

df=pd.DataFrame(data)
print(df.head(3))
print(df)
x=df[['study_hr']]
y=df[['passed']]
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(max_depth=2)
model.fit(xTrain,yTrain)
predictions=model.predict(xTest)
accuracy=accuracy_score(yTest,predictions)
print("Accuracy",accuracy*100)
print("Predictions",predictions)
d2=pd.DataFrame({'study_hr':[6]})
print("If student studied for 6 hrs",model.predict(d2))

