from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd



#Creating a data dictionary with actual data...
data={
    'exclamation':[5,0,8,1,10,2,7,0,9,1],
    'spam':[10,1,15,0,20,12,1,18,0,1],
    'length':[200,500,160,600,100,550,180,620,129,580],
    'is_spam':[1,0,1,0,1,0,1,0,1,0]
}
df=pd.DataFrame(data)

#Separating features and label(the actual thing to be determined)
x=df[['exclamation','spam','length']]
y=df['is_spam']

#spliting them based on training and testing 
#where test_size=0.2 i.e 80 percent training and 20 percent testing

xTrain, xTest, yTrain,yTest=train_test_split(x,y,test_size=0.2,random_state=40)
#(so x,y has been splitted into 4 parts ...80%training 20%testing from features and labels)
#random_state to have same shuffle pattern for everytime
#Create the model
model=LogisticRegression()
#Train the model where all the math happens automatically
model.fit(xTrain,yTrain)
predictions=model.predict(xTest)

#checkingh accuracy
accuracy=accuracy_score(yTest,predictions)
print(accuracy*100,"accuracy")
print(predictions,"predictions")