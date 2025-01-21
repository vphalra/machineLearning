from flask import Flask, request, render_template, redirect, url_for
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load regression model
model = load('RandomForestPricePredictor.joblib')

@app.route('/')
def home():
    # Render the home page with an optional prediction message
    prediction = request.args.get('prediction', None)
    return render_template('index.html', prediction=prediction)

@app.route('/prediction', methods=['POST'])
def predict():
    try:
        # Collect data from form submission
        data = {
            "Lot Area": float(request.form['Lot Area']),
            "TotalSQFT": float(request.form['TotalSQFT']),
            "TotalBaths": float(request.form['TotalBaths']),
            "Garage Cars": float(request.form['Garage Cars']),
            "Year Built": float(request.form['Year Built']),
            "Bedroom AbvGr": float(request.form['Bedroom AbvGr']),
            "Kitchen AbvGr": float(request.form['Kitchen AbvGr']),
            "Overall Qual": float(request.form['Overall Qual'])
        }

        # Prepare the input data for prediction
        input_data = pd.DataFrame([data])
        input_data = input_data.reindex(columns=[
            'Lot Area', 'TotalSQFT', 'TotalBaths', 'Garage Cars',
            'Year Built', 'Bedroom AbvGr', 'Kitchen AbvGr', 'Overall Qual'
        ], fill_value=0)

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Redirect back to the home page with the prediction
        return redirect(url_for('home', prediction=f"Predicted Price: ${prediction:,.2f}"))

    except Exception as e:
        # Redirect back to the home page with the error message
        return redirect(url_for('home', prediction=f"Error: {str(e)}"))

if __name__ == '__main__':
    app.run(debug=True)
