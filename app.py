from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("models/credit_card_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read inputs from HTML form
        Gender = float(request.form["Gender"])
        Age = float(request.form["Age"])
        Debt = float(request.form["Debt"])
        Married = float(request.form["Married"])
        BankCustomer = float(request.form["BankCustomer"])

        industry_map = {
                "IT": 0,
                "Banking": 1,
                "Business": 2,
                "Government": 3,
                "Healthcare": 4,
                "Education": 5,
                "Manufacturing": 6,
                "Retail": 7,
                "Construction": 8,
                "Agriculture": 9,
                "Transport": 10,
                "Telecommunication": 11,
                "Hospitality": 12,
                "Other": 13
            }

        ethnicity_map = {
            "Asian": 0,
            "White": 1,
            "Black": 2,
            "Hispanic": 3,
            "Other": 4
        }

        citizen_map = {
            "Citizen": 0,
            "Permanent Resident": 1,
            "Foreign National": 2
        }

        Industry = industry_map[request.form["Industry"]]
        Ethnicity = ethnicity_map[request.form["Ethnicity"]]
        Citizen = citizen_map[request.form["Citizen"]]
        YearsEmployed = float(request.form["YearsEmployed"])
        PriorDefault = float(request.form["PriorDefault"])
        Employed = float(request.form["Employed"])
        CreditScore = float(request.form["CreditScore"])
        DriversLicense = float(request.form["DriversLicense"])
        ZipCode = float(request.form["ZipCode"])
        Income = float(request.form["Income"])

        # Feature order MUST match the model training order
        features = np.array([[
            Gender,
            Age,
            Debt,
            Married,
            BankCustomer,
            Industry,
            Ethnicity,
            YearsEmployed,
            PriorDefault,
            Employed,
            CreditScore,
            DriversLicense,
            Citizen,
            ZipCode,
            Income
        ]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "✅ Credit Card Approved"
        else:
            result = "❌ Credit Card Rejected"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return render_template(
            "result.html",
            prediction="Prediction Error",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)