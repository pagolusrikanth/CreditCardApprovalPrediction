# 💳 Credit Card Approval Prediction System

A Machine Learning-based web application that predicts whether a credit card application will be **Approved** or **Rejected** based on an applicant's financial and personal details.

The project uses **Python**, **Flask**, **Scikit-learn**, **Random Forest**, and **XGBoost** to build an intelligent prediction system with a responsive web interface.

---

## 📌 Features

- User-friendly Flask web application
- Predict Credit Card Approval instantly
- Displays Approval & Rejection Probability
- Machine Learning based prediction
- Multiple ML algorithms comparison
- Random Forest selected as the best model
- Clean responsive UI
- Error handling
- Trained model (.pkl) included

---

## 🛠 Technologies Used

- Python 3.x
- Flask
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- HTML5
- CSS3

---

## 📂 Project Structure

```
CreditCardApprovalPrediction/
│
├── dataset/
│   ├── crx.csv
│   └── clean_dataset.csv
│
├── models/
│   ├── credit_card_model.pkl
│   └── encoders.pkl
│
├── screenshots/
│   ├── form.png
│   ├── approve.png
│   └── reject.png
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Machine Learning Models Compared

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 86.23% |
| Decision Tree | 86.23% |
| Random Forest | **88.41%** |
| XGBoost | Tested |

Random Forest achieved the highest accuracy and was selected for deployment.

---

## 📥 Dataset

Dataset used:

**Credit Card Approval Dataset**

Features include:

- Gender
- Age
- Debt
- Married
- Bank Customer
- Industry
- Ethnicity
- Years Employed
- Prior Default
- Employed
- Credit Score
- Driver's License
- Citizen
- Zip Code
- Income

Target:

- Approved (1)
- Rejected (0)

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/pagolusrikanth/CreditCardApprovalPrediction.git
```

Move into the project

```bash
cd CreditCardApprovalPrediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## 🧠 Model Training

To retrain the model using your dataset

```bash
python train_model.py
```

This generates

```
models/
    credit_card_model.pkl
    encoders.pkl
```

---

## 📷 Screenshots

### Home Page

Add:

```
screenshots/form.png
```

### Approved Prediction

Add:

```
screenshots/approve.png
```

### Rejected Prediction

Add:

```
screenshots/reject.png
```

---

## 📈 Future Enhancements

- Deep Learning Model
- LightGBM
- Explainable AI (SHAP)
- User Authentication
- Cloud Deployment
- REST API
- Database Integration

---

## 👨‍💻 Author

**Srikanth Pagolu**

M.Tech (Computer Science)

GitHub:
https://github.com/pagolusrikanth

---

## 📄 License

This project is licensed under the MIT License.