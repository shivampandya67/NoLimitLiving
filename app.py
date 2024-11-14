from flask import Flask, request, render_template
import joblib
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the CSV data once to avoid repeated I/O operations
data = pd.read_csv('reshaped_counties_by_year.csv')  # Update with your actual CSV path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the county name from the form input and strip '_model' suffix
    county_input = request.form.get('county')
    county_name = county_input.replace('_model', '')

    try:
        # Load the model for the specified county
        model_file_path = os.path.join('models', f'{county_input}.pkl')
        model = joblib.load(model_file_path)
        
        # Get the forecasted values for 2022 (step 12) and 2025 (step 47)
        forecast = model.forecast(steps=48)  # Forecasting 4 years ahead (48 steps)
        
        # Extract predictions for January 2022 and January 2025
        selected_forecasts = {
            '2022': forecast[12],
            '2025': forecast[47]
        }

        # Fetch the 2021 value for "All Industry Total" for the specified county
        filtered_2021 = data[(data['GeoName'] == county_name) & (data['Year'] == 2021)]
        value_2021 = filtered_2021['All industry total']
        
        # Extract the actual 2021 value from the Series
        if isinstance(value_2021, pd.Series) and not value_2021.empty:
            base_value = value_2021.iloc[0]
            
            # Calculate percentage growth for 2022 and 2025, formatted to 5 decimal places
            growth_percentages = {
                year: round(((forecast_value - base_value) / base_value) * 100, 5) //
                for year, forecast_value in selected_forecasts.items()
            }

            return render_template(
                'index.html',
                county=county_name,
                growth_percentages=growth_percentages
            )
        else:
            # Handle case where no 2021 data is found for the specified county and attribute
            return render_template(
                'index.html',
                prediction="No 2021 data found for the specified county and attribute.",
                county=county_name
            )

    except FileNotFoundError:
        return render_template('index.html', prediction="Model not found for the specified county", county=county_name)
    except Exception as e:
        return render_template('index.html', prediction=str(e), county=county_name)

if __name__ == '__main__':
    app.run(debug=True)
