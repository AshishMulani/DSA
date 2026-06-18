import pandas as pd
from flask import Flask, jsonify, request
import pickle
from sklearn import datasets

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

iris = datasets.load_iris()
target_names = iris.target_names

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    return jsonify({
    "Prediction": target_names[int(prediction[0])]
})

if __name__=='__main__':
    app.run(debug = True)