from flask import Flask, render_template, request
import joblib
import numpy as np


model = joblib.load('model/insurance_premium_predictor.pkl')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        sex = 1 if request.form['sex'] == 'male' else 0
        smoker = 1 if request.form['smoker'] == 'yes' else 0
        region = request.form['region']

        regions = ['southwest', 'southeast', 'northwest', 'northeast']
        region_data = [1 if region == r else 0 for r in regions]

        features = np.array([age, bmi, children, sex, smoker] + region_data).reshape(1, -1)

        prediction = model.predict(features)

        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
