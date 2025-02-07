import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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
df_grouped = df_long.groupby(["country_name", "Year_Grouped"])["GDP_Growth"].mean().reset_index()

def train_model_for_country(country: str):
    """
    Trains a model for a specific country.

    Parameters:
        country (str): The country name.

    Returns:
        pipeline: Trained model pipeline.
        X_test: Test features.
        y_test: Test target.
    """
    # Filter the dataset for the specific country
    df_country = df_grouped[df_grouped["country_name"] == country]

    # Select features (10-Year Grouped Year) and target (GDP Growth)
    X = df_country[["Year_Grouped"]]
    y = df_country["GDP_Growth"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build a pipeline with Linear Regression
    pipeline = Pipeline(steps=[
        ("regressor", LinearRegression())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test

# Define a function to predict GDP growth for a given country and year
def predict_gdp_growth(pipeline, year: int) -> float:
    """
    Predicts the GDP growth rate for a given year using the trained model.

    Parameters:
        pipeline: Trained model pipeline.
        year (int): The year for prediction.

    Returns:
        float: Predicted GDP growth rate.
    """
    # Convert input year to 10-year group
    year_grouped = (year // 10) * 10

    # Predict using the model
    input_df = pd.DataFrame({"Year_Grouped": [year_grouped]})
    prediction = pipeline.predict(input_df)
    return round(prediction[0], 2)

# Example: Train model and predict GDP growth for China in 2015
example_country = "Germany"
example_year = 2015
pipeline, X_test, y_test = train_model_for_country(example_country)
predicted_growth = predict_gdp_growth(pipeline, example_year)
print(f"\nPredicted GDP Growth for {example_country} in {example_year}: {predicted_growth}%")

# Plotting the predicted GDP growth for 10-year intervals for a specific country
country_to_plot = example_country
years = list(range(1980, 2040, 10))
predicted_gdp_growth = [predict_gdp_growth(pipeline, year) for year in years]

# Get the actual average GDP growth from the data
actual_gdp_growth = df_grouped[df_grouped["country_name"] == country_to_plot].set_index("Year_Grouped").reindex(years)["GDP_Growth"]

plt.figure(figsize=(10, 6))
plt.plot(years, predicted_gdp_growth, marker='o', label='Predicted GDP Growth')
plt.plot(years, actual_gdp_growth, marker='x', label='Actual Average GDP Growth', linestyle='--')
plt.xlabel('Year')
plt.ylabel('GDP Growth')
plt.title(f'Predicted vs Actual Average GDP Growth for {country_to_plot} (10-Year Intervals)')
plt.legend()
plt.grid(True)
plt.show()