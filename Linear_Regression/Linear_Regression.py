import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
sample = 100
size = np.random.randint(800, 3000, sample)
bed = np.random.randint(1, 6, sample)
age = np.random.randint(0, 50, sample)
price = (50000 + 200*size + 30000*bed - 1000*age + np.random.randn(sample)*10000)

df = pd.DataFrame({
    'size': size,
    'bed': bed,
    'age': age,
    'price': price
})

x = df[['size', 'bed', 'age']]
y = df['price']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(xTrain, yTrain)  
predict = model.predict(xTest)

print(f"Price = {model.intercept_:,.0f} + "
      f"({model.coef_[0]:.0f} × Size) + "
      f"({model.coef_[1]:,.0f} × Bedrooms) + "
      f"({model.coef_[2]:.0f} × Age)")

print("\nSample predictions vs actual:")
print("\nSize | Beds | Age |  Actual Price | Predicted Price | Error")
print("-" * 70)

for i in range(5):

    actual = yTest.iloc[i]
    predicted = predict[i]
    error = abs(actual - predicted)  
    
    print(f"{xTest.iloc[i, 0]:4.0f} | {xTest.iloc[i, 1]:4.0f} | {xTest.iloc[i, 2]:3.0f} | "
          f"${actual:13,.0f} | ${predicted:15,.0f} | ${error:,.0f}")

r2 = r2_score(yTest ,predict)
mae = mean_absolute_error(yTest,predict)
mse = mean_squared_error(yTest, predict)
rmse = np.sqrt(mean_squared_error(yTest, predict))

print(r2)
print(mae)
print(mse)
print(rmse)