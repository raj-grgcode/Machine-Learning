from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
data = {
    'age': [22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60, 26, 33, 44, 51],
    'income': [25, 30, 35, 45, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 32, 55, 78, 92],
    'will_buy': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
}
df=pd.DataFrame(data)
x=df[['age','income']]
y=df['will_buy']

xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(xTrain,yTrain)
prediction=model.predict(xTest)
accuracy=accuracy_score(yTest,prediction)
print("accuracy",accuracy*100)
print("prediction",prediction)