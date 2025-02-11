import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt

# Load the dataset
file_path = "world_gdp_data.csv"
df = pd.read_csv(file_path, encoding="latin1")

# Trim spaces from column names
df.columns = df.columns.str.strip()

# Remove the "indicator" column if it exists
if "indicator" in df.columns:
    df.drop(columns=["indicator"], inplace=True)

# Convert dataset from wide to long format
df_long = pd.melt(df, id_vars=["country_name"], var_name="Year", value_name="GDP_Growth")

# Convert Year & GDP Growth to numeric
df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
df_long["GDP_Growth"] = pd.to_numeric(df_long["GDP_Growth"], errors="coerce")

# Drop rows with missing values
df_long.dropna(subset=["GDP_Growth", "Year"], inplace=True)

# Convert Year to 10-Year Groups
df_long["Year_Grouped"] = (df_long["Year"] // 10) * 10

# Average GDP Growth for each 10-Year period
df_grouped = df_long.groupby(["country_name", "Year_Grouped"]) ["GDP_Growth"].mean().reset_index()

def train_model_for_country(country: str):
    """
    Trains a model for a specific country.
    """
    df_country = df_grouped[df_grouped["country_name"] == country]
    X = df_country[["Year_Grouped"]]
    y = df_country["GDP_Growth"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline(steps=[("regressor", LinearRegression())])
    pipeline.fit(X_train, y_train)
    mse = mean_squared_error(y_test, pipeline.predict(X_test))
    print(f"Mean Squared Error for {country}: {mse:.4f}")
    return pipeline, X_test, y_test

def predict_gdp_growth(pipeline, year: int) -> float:
    """
    Predicts the GDP growth rate for a given year.
    """
    year_grouped = (year // 10) * 10
    input_df = pd.DataFrame({"Year_Grouped": [year_grouped]})
    prediction = pipeline.predict(input_df)
    return round(prediction[0], 2)

def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluates the model using Accuracy, Precision, Recall, and MSE.
    """
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    y_test_binary = (y_test > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    report = classification_report(y_test_binary, y_pred_binary, target_names=["Negative Growth", "Positive Growth"], labels=[0, 1], zero_division=0)
    print("\nClassification Report:\n", report)
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "MSE": mse}

# Example Usage
example_country = "Germany"
example_year = 2015
pipeline, X_test, y_test = train_model_for_country(example_country)
predicted_growth = predict_gdp_growth(pipeline, example_year)
print(f"\nPredicted GDP Growth for {example_country} in {example_year}: {predicted_growth}%")

# Evaluate Model
metrics = evaluate_model(pipeline, X_test, y_test)
print("\nEvaluation Metrics:", metrics)

# Plotting Results
years = list(range(1980, 2040, 10))
predicted_gdp_growth = [predict_gdp_growth(pipeline, year) for year in years]
actual_gdp_growth = df_grouped[df_grouped["country_name"] == example_country].set_index("Year_Grouped").reindex(years)["GDP_Growth"]
plt.figure(figsize=(10, 6))
plt.plot(years, predicted_gdp_growth, marker='o', label='Predicted GDP Growth')
plt.plot(years, actual_gdp_growth, marker='x', label='Actual Average GDP Growth', linestyle='--')
plt.xlabel('Year')
plt.ylabel('GDP Growth')
plt.title(f'Predicted vs Actual Average GDP Growth for {example_country} (10-Year Intervals)')
plt.legend()
plt.grid(True)
plt.show()
