from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import csv

model = joblib.load("Training/california_house_price_prediction_model.pkl")

app = Flask(__name__, template_folder="Templates")


def read_csv(file_path):
    data = []
    with open(file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


@app.route("/")
def home():
    csv_data = read_csv("Data/cal_cities_lat_long.csv")
    return render_template("index.html", choices=csv_data)


@app.route("/predict", methods=["POST", "GET"])
def predict():
    csv_data = read_csv("Data/cal_cities_lat_long.csv")

    # Get user inputs for the features
    input_features = [float(request.form["Latitude"]), float(request.form["Longitude"])]
    # input_features = [float(request.form["Latitude"]), float(request.form["Longitude"]), float(request.form["HouseMedAge"]), float(request.form["TotRooms"]), float(request.form["Population"]), float(request.form["Households"]), float(request.form["MedIncome"]), float(request.form["OceanProximity"])]

    # Scale the user inputs
    input_features_array = [np.array(input_features)]
    scaler = StandardScaler()
    input_features_scaled = scaler.transform(input_features_array)

    # Make a prediction using the model
    predicted_price = round(model.predict(input_features_scaled)[0], 2)

    # Concat with text
    prediction = "Estimated price of home in California is  $ {}".format(predicted_price)

    return render_template("index.html", prediction_text=prediction, choices=csv_data)


if __name__ == "__main__":
    app.run(debug=True)
