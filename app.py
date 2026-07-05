from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# ==========================
# Load Model & Encoders
# ==========================

model = joblib.load("models/credit_card_model.pkl")
encoders = joblib.load("models/encoders.pkl")


# ==========================
# Home Page
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# Prediction
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------- Numeric Features --------

        Gender = int(request.form["Gender"])
        Age = float(request.form["Age"])
        Debt = float(request.form["Debt"])
        Married = int(request.form["Married"])
        BankCustomer = int(request.form["BankCustomer"])

        YearsEmployed = float(request.form["YearsEmployed"])

        PriorDefault = int(request.form["PriorDefault"])
        Employed = int(request.form["Employed"])

        CreditScore = float(request.form["CreditScore"])

        DriversLicense = int(request.form["DriversLicense"])

        ZipCode = float(request.form["ZipCode"])

        Income = float(request.form["Income"])


        # -------- Encode Categorical --------

        Industry = encoders["Industry"].transform(
            [request.form["Industry"]]
        )[0]

        Ethnicity = encoders["Ethnicity"].transform(
            [request.form["Ethnicity"]]
        )[0]

        Citizen = encoders["Citizen"].transform(
            [request.form["Citizen"]]
        )[0]


        # -------- Create DataFrame --------

        data = pd.DataFrame([{

            "Gender": Gender,
            "Age": Age,
            "Debt": Debt,
            "Married": Married,
            "BankCustomer": BankCustomer,
            "Industry": Industry,
            "Ethnicity": Ethnicity,
            "YearsEmployed": YearsEmployed,
            "PriorDefault": PriorDefault,
            "Employed": Employed,
            "CreditScore": CreditScore,
            "DriversLicense": DriversLicense,
            "Citizen": Citizen,
            "ZipCode": ZipCode,
            "Income": Income

        }])


        # -------- Prediction --------

        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0]


        approved_probability = round(probability[1] * 100, 2)
        rejected_probability = round(probability[0] * 100, 2)


        if prediction == 1:

            result = "✅ Credit Card Approved"

        else:

            result = "❌ Credit Card Rejected"


        return render_template(
            "result.html",
            prediction=result,
            approved=approved_probability,
            rejected=rejected_probability
        )

    except Exception as e:

        return render_template(
            "result.html",
            prediction="Prediction Error",
            error=str(e)
        )


# ==========================
# Run App
# ==========================

if __name__ == "__main__":
    app.run(debug=True)