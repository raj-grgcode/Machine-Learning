from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

data = {
    'study_hours': [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 3, 5, 7, 9, 2, 4],
    'attendance': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85, 88, 90, 92, 95, 55, 68, 82, 93, 42, 62],
    'passed': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
}
df=pd.DataFrame(data)
print(df)

x=df[['study_hours','attendance']]
y=df['passed']
#split the data
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(xTrain,yTrain)
prediction=model.predict(xTest)
accuracy=accuracy_score(yTest,prediction)
print(accuracy*100,"accuracte")
print(prediction,"predicted")