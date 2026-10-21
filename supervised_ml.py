import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Dataset
size = [500, 700, 800, 1000, 1200, 1500, 1800, 2000]
price = [100, 150, 180, 200, 250, 300, 350, 400]

# Convert to DataFrame
df = pd.DataFrame({'size': size, 'price': price})
X = df[['size']]
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")

# Visualize & SAVE
plt.scatter(X, y, color='blue', label='Real Data')
plt.plot(X, model.predict(X), color='red', label='Predicted Line')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($1000)')
plt.legend()
plt.savefig('supervised_result.png')
print("Graph saved as supervised_result.png")