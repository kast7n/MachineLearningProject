import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------------------------
# PROJECT OBJECTIVES AND METHODOLOGY:
# ------------------------------------------------------------------------------
# 1. Identify a supervised problem task:
#    - We are solving a regression task: predicting the continuous value (stock's 
#      closing price) based on date.
#
# 2. Select a labeled dataset:
#    - We use 'goog.csv', a dataset available on Kaggle that includes historical
#      stock data (dates, open, high, low, close, volume, etc.) for Google.
#
# 3. Overview of the data:
#    - Size: 1,258 rows and 14 columns.
#    - Features: date, open, high, low, close, volume, and others.
#    - Output Label: 'close' (the stock’s closing price).
#
# 4. Task's Nature:
#    - This is a regression problem.
#
# 5. Model Architecture and Loss Function:
#    - We use a Linear Regression model.
#    - The model minimizes the Mean Squared Error (MSE) during training.
#
# 6. Training Process Implementation:
#    - Data is preprocessed by converting dates to a numerical (ordinal) format.
#    - The dataset is split into training (80%) and testing (20%) sets.
#    - The Linear Regression model is trained with default hyperparameters.
#
# 7. Model Evaluation:
#    - The trained model is evaluated using RMSE (Root Mean Squared Error).
#
# 8. Additional Requirement:
#    - The model accepts an input date (target date) for prediction.
#    - It then compares the predicted price on that target date with the predicted
#      price for your current date (here taken as the latest date in the dataset)
#      and decides if it is worth investing.
# ------------------------------------------------------------------------------

# 1. Load the dataset (ensure 'goog.csv' is in the same directory)
df = pd.read_csv("goog.csv")

# 2. Data Overview: Display structure and sample rows
print("Data Information:")
print(df.info())
print("\nSample Data:")
print(df.head())

# 3. Preprocess the Data
# Convert the 'date' column to a datetime object.
df['date'] = pd.to_datetime(df['date'])

# For our regression, we transform the date into a numerical feature using its ordinal.
df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())

# 4. Define features and target variable.
# Here, we use the numerical representation of the date as our single feature.
X = df[['date_ordinal']]
y = df['close']

# 5. Split the dataset into training and testing sets (80% training, 20% testing).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Initialize and train the Linear Regression model.
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Evaluate the model on the test set.
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# 8. Function to predict the stock price for a given date using the trained model.
def predict_stock_price(input_date):
    """
    Predicts the closing stock price for a given date.
    
    Parameters:
        input_date (str or datetime): The date for which to predict the price.
        
    Returns:
        float: The predicted closing price.
    """
    # Ensure the input_date is in datetime format.
    if isinstance(input_date, str):
        input_date = pd.to_datetime(input_date)
        
    # Convert date to its ordinal representation.
    date_ordinal = np.array([[input_date.toordinal()]])
    
    # Predict the closing price.
    predicted_price = model.predict(date_ordinal)[0]
    return predicted_price

# 9. Get the target date input from the user.
# This is the date for which you want a price prediction.
input_date_str = input("Enter the target date (YYYY-MM-DD) for prediction: ")
target_date = pd.to_datetime(input_date_str)

# 10. Define the 'current date' for comparison.
# For consistency, we use the latest date available in the dataset as your current date.
current_date = df['date'].max()

# 11. Predict prices for both the target date and current date.
predicted_target_price = predict_stock_price(target_date)
predicted_current_price = predict_stock_price(current_date)

print("\nPredicted Prices:")
print(f"Predicted Price for your target date ({target_date.date()}): ${predicted_target_price:.2f}")
print(f"Predicted Price for current date ({current_date.date()}): ${predicted_current_price:.2f}")

# 12. Investment Decision Logic:
#    - Invest if the predicted price for the target date is higher than that for your current date.
if predicted_target_price > predicted_current_price:
    print("Investment Decision: Invest")
else:
    print("Investment Decision: Do not invest")
