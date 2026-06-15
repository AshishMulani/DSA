from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
amount_model = joblib.load("amount.pkl")
category_model = joblib.load("category.pkl")
subcategory_model = joblib.load("subcategory.pkl")

# Load encoders
encoders = joblib.load("encoders.pkl")
le_category = joblib.load("le_category.pkl")
le_sub = joblib.load("le_sub.pkl")


@app.route('/api/v1/predict-purchase', methods=['POST'])
def predict():

    data = request.json
    
    features = {
        'Customer_Age': data['Customer_Age'],
        'Customer_Gender': encoders['Customer_Gender'].transform(
            [data['Customer_Gender']]
        )[0],
        'City': encoders['City'].transform(
            [data['City']]
        )[0],
        'month': data['month'],
        'Season': encoders['Season'].transform(
            [data['Season']]
        )[0],
        'Membership_Status': encoders['Membership_Status'].transform(
            [data['Membership_Status']]
        )[0],
        'Traffic_Source': encoders['Traffic_Source'].transform(
            [data['Traffic_Source']]
        )[0]
    }

    features = pd.DataFrame([features])

    features = features[
    [
        'Customer_Age',
        'Customer_Gender',
        'City',
        'month',
        'Season',
        'Membership_Status',
        'Traffic_Source'
    ]
    ]

    predicted_amount = amount_model.predict(features)[0]

    predicted_category = category_model.predict(features)[0]

    predicted_subcategory = subcategory_model.predict(features)[0]

    return jsonify({
        "predicted_order_amount":
            round(float(predicted_amount), 2),

        "predicted_category":
            le_category.inverse_transform(
                [predicted_category]
            )[0],

        "predicted_subcategory":
            le_sub.inverse_transform(
                [predicted_subcategory]
            )[0]
    })


if __name__ == '__main__':
    app.run(debug=True)