"""
COMPLETE LINEAR REGRESSION TUTORIAL
This single program teaches you EVERYTHING you need to know about Linear Regression.
Run this step-by-step and read the comments carefully!
"""

# ============================================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("LINEAR REGRESSION: COMPLETE TUTORIAL")
print("=" * 60)

# ============================================================================
# STEP 2: CREATE SAMPLE DATA
# ============================================================================
print("\n📊 STEP 2: Creating Sample Dataset")
print("-" * 60)

# We'll predict house prices based on:
# - Size (sqft)
# - Bedrooms
# - Age (years)

np.random.seed(42)  # For reproducibility

# Generate 100 houses
n_samples = 100

size = np.random.randint(800, 3000, n_samples)  # 800-3000 sqft
bedrooms = np.random.randint(1, 6, n_samples)   # 1-5 bedrooms
age = np.random.randint(0, 50, n_samples)       # 0-50 years old

# TRUE RELATIONSHIP (this is what the model will try to learn):
# Price = 50,000 + (200 × Size) + (30,000 × Bedrooms) - (1,000 × Age) + noise
true_price = (50000 + 
              200 * size + 
              30000 * bedrooms - 
              1000 * age + 
              np.random.randn(n_samples) * 10000)  # Add some random noise

# Create DataFrame
df = pd.DataFrame({
    'Size': size,
    'Bedrooms': bedrooms,
    'Age': age,
    'Price': true_price
})

print("Dataset created!")
print(f"Total houses: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset statistics:")
print(df.describe())

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n🔍 STEP 3: Exploratory Data Analysis")
print("-" * 60)

# Check for missing values (ALWAYS DO THIS!)
print(f"Missing values:\n{df.isnull().sum()}")
print("✅ No missing values - we're good!")

# Visualize relationships
print("\nVisualizing relationships between features and price...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Size vs Price
axes[0].scatter(df['Size'], df['Price'], alpha=0.5)
axes[0].set_xlabel('Size (sqft)')
axes[0].set_ylabel('Price ($)')
axes[0].set_title('Size vs Price')

# Bedrooms vs Price
axes[1].scatter(df['Bedrooms'], df['Price'], alpha=0.5)
axes[1].set_xlabel('Bedrooms')
axes[1].set_ylabel('Price ($)')
axes[1].set_title('Bedrooms vs Price')

# Age vs Price
axes[2].scatter(df['Age'], df['Price'], alpha=0.5)
axes[2].set_xlabel('Age (years)')
axes[2].set_ylabel('Price ($)')
axes[2].set_title('Age vs Price')

plt.tight_layout()
plt.savefig('/home/claude/eda_plots.png')
print("✅ Plots saved! (Check eda_plots.png)")
plt.close()

# Check correlation (detect multicollinearity)
print("\nCorrelation Matrix:")
correlation = df.corr()
print(correlation)

# Visualize correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.savefig('/home/claude/correlation_matrix.png')
print("✅ Correlation heatmap saved!")
plt.close()

# INTERPRETATION:
# - High correlation between features = multicollinearity (BAD)
# - In our case, features aren't too correlated (< 0.7), so we're good!

# ============================================================================
# STEP 4: PREPARE DATA FOR MODELING
# ============================================================================
print("\n🛠️ STEP 4: Preparing Data")
print("-" * 60)

# Separate features (X) and target (y)
X = df[['Size', 'Bedrooms', 'Age']].values  # Features (inputs)
y = df['Price'].values  # Target (output)

print(f"Features shape: {X.shape}")  # (100, 3) = 100 samples, 3 features
print(f"Target shape: {y.shape}")    # (100,) = 100 samples

# Reshape y to column vector for consistency
y = y.reshape(-1, 1)
print(f"Target shape after reshape: {y.shape}")  # (100, 1)

# ============================================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================================
print("\n✂️ STEP 5: Splitting Data into Train and Test Sets")
print("-" * 60)

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42     # For reproducibility
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# WHY SPLIT?
# - Train on 80% of data
# - Test on unseen 20% to measure REAL performance
# - This detects overfitting!

# ============================================================================
# STEP 6: TRAIN THE MODEL
# ============================================================================
print("\n🎯 STEP 6: Training Linear Regression Model")
print("-" * 60)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)  # This is where gradient descent happens!

print("✅ Model trained!")

# Check the learned parameters
print("\nLearned Parameters:")
print(f"Intercept (β₀): ${model.intercept_[0]:,.2f}")
print(f"Coefficients (β₁, β₂, β₃):")
print(f"  - Size: ${model.coef_[0][0]:,.2f} per sqft")
print(f"  - Bedrooms: ${model.coef_[0][1]:,.2f} per bedroom")
print(f"  - Age: ${model.coef_[0][2]:,.2f} per year")

# INTERPRETATION:
# The equation our model learned:
# Price = Intercept + (Coef_Size × Size) + (Coef_Bedrooms × Bedrooms) + (Coef_Age × Age)

print("\nOur model's equation:")
print(f"Price = {model.intercept_[0]:,.0f} + "
      f"({model.coef_[0][0]:.0f} × Size) + "
      f"({model.coef_[0][1]:,.0f} × Bedrooms) + "
      f"({model.coef_[0][2]:.0f} × Age)")

# Compare to TRUE relationship we used to generate data:
print("\nTRUE relationship (what we used to generate data):")
print("Price = 50,000 + (200 × Size) + (30,000 × Bedrooms) - (1,000 × Age)")
print("👆 See how close the model got?")

# ============================================================================
# STEP 7: MAKE PREDICTIONS
# ============================================================================
print("\n🔮 STEP 7: Making Predictions")
print("-" * 60)

# Predict on test set
y_pred = model.predict(X_test)

# Show some examples
print("Sample predictions vs actual:")
print("\nSize | Beds | Age |  Actual Price | Predicted Price | Error")
print("-" * 70)
for i in range(5):
    actual = y_test[i][0]
    predicted = y_pred[i][0]
    error = abs(actual - predicted)
    print(f"{X_test[i][0]:4.0f} | {X_test[i][1]:4.0f} | {X_test[i][2]:3.0f} | "
          f"${actual:13,.0f} | ${predicted:15,.0f} | ${error:,.0f}")

# ============================================================================
# STEP 8: EVALUATE THE MODEL
# ============================================================================
print("\n📈 STEP 8: Evaluating Model Performance")
print("-" * 60)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Performance Metrics:")
print(f"  R² Score: {r2:.4f}")
print(f"    👉 Interpretation: Model explains {r2*100:.1f}% of price variance")
print(f"    👉 {r2:.2f} > 0.7 means GOOD model!")

print(f"\n  MAE (Mean Absolute Error): ${mae:,.2f}")
print(f"    👉 On average, predictions are off by ${mae:,.0f}")

print(f"\n  RMSE (Root Mean Squared Error): ${rmse:,.2f}")
print(f"    👉 Similar to MAE, but penalizes large errors more")

print(f"\n  MSE (Mean Squared Error): ${mse:,.2f}")
print(f"    👉 Rarely used alone - just for reference")

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'Actual vs Predicted Prices (R² = {r2:.3f})')
plt.legend()
plt.savefig('/home/claude/predictions_vs_actual.png')
print("\n✅ Prediction plot saved! (predictions_vs_actual.png)")
plt.close()

# ============================================================================
# STEP 9: CHECK ASSUMPTIONS
# ============================================================================
print("\n✅ STEP 9: Checking Linear Regression Assumptions")
print("-" * 60)

# Calculate residuals (errors)
residuals = y_test - y_pred

# 1. Check for homoscedasticity (constant error variance)
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price ($)')
plt.ylabel('Residuals (Errors) ($)')
plt.title('Residual Plot - Check for Homoscedasticity')
plt.savefig('/home/claude/residual_plot.png')
print("✅ Residual plot saved!")
plt.close()

print("\nResidual Plot Interpretation:")
print("  ✅ GOOD: Points randomly scattered around zero")
print("  ❌ BAD: Fan shape (variance increases/decreases)")

# 2. Check if residuals are normally distributed
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=20, edgecolor='black')
plt.xlabel('Residual ($)')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.savefig('/home/claude/residual_distribution.png')
print("✅ Residual distribution plot saved!")
plt.close()

print("\nResidual Distribution Interpretation:")
print("  ✅ GOOD: Bell-shaped (normal distribution)")
print("  ❌ BAD: Heavily skewed or multiple peaks")

# ============================================================================
# STEP 10: MAKE A NEW PREDICTION
# ============================================================================
print("\n🏠 STEP 10: Predict Price for a New House")
print("-" * 60)

# New house: 2000 sqft, 3 bedrooms, 10 years old
new_house = np.array([[2000, 3, 10]])

predicted_price = model.predict(new_house)[0][0]

print(f"New House Details:")
print(f"  Size: 2000 sqft")
print(f"  Bedrooms: 3")
print(f"  Age: 10 years")
print(f"\n💰 Predicted Price: ${predicted_price:,.2f}")

# Manual calculation to verify
manual_prediction = (model.intercept_[0] + 
                    model.coef_[0][0] * 2000 + 
                    model.coef_[0][1] * 3 + 
                    model.coef_[0][2] * 10)
print(f"Manual Calculation: ${manual_prediction:,.2f}")
print("✅ Both match!")

# ============================================================================
# STEP 11: BONUS - GRADIENT DESCENT FROM SCRATCH
# ============================================================================
print("\n🎓 STEP 11: BONUS - How Gradient Descent Works (From Scratch)")
print("-" * 60)

def gradient_descent_demo(X, y, learning_rate=0.00000001, iterations=100):
    """
    Simplified gradient descent for demonstration.
    In practice, ALWAYS use sklearn!
    """
    m, n = X.shape  # m = samples, n = features
    theta = np.zeros((n, 1))  # Start with zeros
    
    costs = []  # Track cost over time
    
    for i in range(iterations):
        # 1. Make predictions with current theta
        predictions = X.dot(theta)
        
        # 2. Calculate error
        errors = predictions - y
        
        # 3. Calculate gradient (direction to improve)
        gradient = (1/m) * X.T.dot(errors)
        
        # 4. Update theta
        theta = theta - learning_rate * gradient
        
        # 5. Calculate cost (MSE)
        cost = (1/(2*m)) * np.sum(errors**2)
        costs.append(cost)
        
        if i % 20 == 0:
            print(f"  Iteration {i}: Cost = ${cost:,.2f}")
    
    return theta, costs

print("Running gradient descent from scratch...")
print("(This is what sklearn does internally!)")

# Add intercept column to X_train for manual calculation
X_train_with_intercept = np.c_[np.ones((X_train.shape[0], 1)), X_train]

theta_manual, costs = gradient_descent_demo(
    X_train_with_intercept, 
    y_train, 
    learning_rate=0.00000001,  # Very small for stability
    iterations=100
)

print(f"\nManual Gradient Descent Results:")
print(f"  Intercept: ${theta_manual[0][0]:,.2f}")
print(f"  Coefficients: {theta_manual[1:].flatten()}")

# Plot cost over iterations
plt.figure(figsize=(10, 6))
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function Decrease Over Iterations')
plt.savefig('/home/claude/gradient_descent_cost.png')
print("✅ Gradient descent cost plot saved!")
plt.close()

print("\nGradient Descent Visualization:")
print("  👉 Cost decreases with each iteration")
print("  👉 Eventually plateaus (converges)")
print("  👉 This is the 'walking down the hill' process!")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("🎉 TUTORIAL COMPLETE!")
print("=" * 60)

print("\n📚 What You Learned:")
print("  1. ✅ How to load and explore data (EDA)")
print("  2. ✅ Check for missing values and correlations")
print("  3. ✅ Prepare features and target")
print("  4. ✅ Split into train/test sets")
print("  5. ✅ Train a Linear Regression model")
print("  6. ✅ Make predictions")
print("  7. ✅ Evaluate with R², MAE, RMSE")
print("  8. ✅ Check assumptions (residual plots)")
print("  9. ✅ Predict on new data")
print(" 10. ✅ Understand gradient descent")

print("\n💡 Key Takeaways:")
print("  • Always visualize data BEFORE modeling")
print("  • Always use train/test split")
print("  • R² > 0.7 = good model")
print("  • Check residual plots for assumptions")
print("  • In practice, just use sklearn!")

print("\n📁 Files Generated:")
print("  • eda_plots.png")
print("  • correlation_matrix.png")
print("  • predictions_vs_actual.png")
print("  • residual_plot.png")
print("  • residual_distribution.png")
print("  • gradient_descent_cost.png")

print("\n🚀 Next Steps:")
print("  1. Try this on a real dataset (Kaggle)")
print("  2. Practice writing this from memory")
print("  3. Experiment with different features")
print("  4. Try adding polynomial features")

print("\n" + "=" * 60)
print("Good luck with your ML journey! 🎯")
print("=" * 60)