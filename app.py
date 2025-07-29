from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        int(request.form['Pclass']),
        int(request.form['Sex']),
        float(request.form['Age']),
        float(request.form['Fare'])
    ]
    prediction = model.predict([features])[0]
    result = "Survived" if prediction == 1 else "Did not survive"
    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == '__main__':
    app.run(debug=True)
