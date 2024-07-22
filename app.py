from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

model = joblib.load('rf_model.pkl')

app = Flask(__name__, template_folder = 'templates')

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
        
    # Get user inputs for the features
    input_features = [float(request.form['Latitude']),
                        float(request.form['Longitude']),
                        float(request.form['HouseMedAge']),
                        float(request.form['TotRooms']),
                        float(request.form['Population']),
                        float(request.form['Households']),
                        float(request.form['MedIncome']),
                        float(request.form['OceanProximity'])]

    # Scale the user inputs
    input_features_array = [np.array (input_features)]
    scaler = StandardScaler()
    input_features_scaled = scaler.transform (input_features_array)

    # Make a prediction using the model
    predicted_price = round (model.predict(input_features_scaled)[0], 2)

    # Concat with text
    prediction = 'Estimated price of home in California is  $ {}'.format(predicted_price)

    return render_template('index.html', prediction_text = prediction)

if __name__ == '__main__':
    app.run(debug = True)
