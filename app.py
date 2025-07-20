from flask import Flask, render_template, request
import pickle
import numpy as np
from keras.models import load_model

app = Flask(__name__)

with open('model/label_encoders.pickle', 'rb') as f:
    label_encoders = pickle.load(f)
with open('model/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
model = load_model('model/sentiment_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and check form data
        age = request.form['age']
        gender = request.form['gender']
        education = request.form['education']
        job = request.form['job']
        experience = request.form['experience']

        if not all([age, gender, education, job, experience]):
            return "Please fill in all fields.", 400

        # Encode categorical fields
        gender_encoded = label_encoders['gender'].transform([gender])[0]
        education_encoded = label_encoders['education'].transform([education])[0]

        # Tokenize and pad job title
        job_seq = tokenizer.texts_to_sequences([job])
        job_padded = np.pad(job_seq[0], (0, max(0, 5-len(job_seq[0]))), 'constant')[:5]

        input_features = [float(age), float(gender_encoded), float(education_encoded)] + list(job_padded) + [float(experience)]
        X_input = np.array(input_features, dtype=np.float32).reshape(1, -1)

        prediction = model.predict(X_input)
        pred_value = float(prediction[0][0])
        if np.isnan(pred_value):
            return "Prediction failed: Output is invalid or input not recognized. Check if all fields are valid and match training data.", 400

        salary = int(pred_value)
        return render_template('result.html',
                               age=age,
                               gender=gender,
                               education=education,
                               job=job,
                               experience=experience,
                               salary=salary)
    except Exception as e:
        return f"⚠️ Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
