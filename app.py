import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import pickle
import joblib

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = joblib.load('model_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    # Get the input values from the form
    title = request.form.get("title")
    text = request.form.get("text")
    subject = request.form.get("subject")
    
    # Concatenate title and text to form the input
    input_text = title + " " + text
    
    # Transform the input using the vectorizer
    feature = vectorizer.transform([input_text])
    
    # Predict using the model
    score = model.predict(feature)

    # Render the template with the prediction message
    if score[0] == 1:
        return render_template("index.html", message="Fake")
    elif score[0] == 0:
        return render_template("index.html", message="True")
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
