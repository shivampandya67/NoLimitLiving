import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA

# Load your dataset (replace 'your_data.csv' with the actual filename)
data = pd.read_csv('reshaped_counties_by_year.csv')  # Update this line as needed

# Function to train ARIMA models for each county
def train_arima_models(data):
    county_models = {}
    counties = data['GeoName'].unique()

    for county in counties:
        county_data = data[data['GeoName'] == county].set_index('Year')['All industry total']
        
        # Fit ARIMA model
        model = ARIMA(county_data, order=(1,1,1))  # Adjust the order as needed based on your data
        fitted_model = model.fit()
        county_models[county] = fitted_model

    return county_models

# Train the models
models = train_arima_models(data)

# Save the models
def save_models(county_models):
    for county, model in county_models.items():
        joblib.dump(model, f'models/{county}_model.pkl')

save_models(models)
