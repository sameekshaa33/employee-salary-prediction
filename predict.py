import pickle
import numpy as np
from keras.models import load_model

def load_all():
    with open('model/label_encoders.pickle', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('model/tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    model = load_model('model/sentiment_model.keras')
    return label_encoders, tokenizer, model

def preprocess_input(age, gender, education, job_title, experience, encoders, tokenizer):
    # Encode categorical features
    gender_encoded = encoders['gender'].transform([gender])[0]
    education_encoded = encoders['education'].transform([education])[0]
    
    # Tokenize and pad job title
    job_seq = tokenizer.texts_to_sequences([job_title])
    job_seq_padded = np.pad(job_seq[0], (0, max(0, 5 - len(job_seq[0]))), 'constant')[:5]

    # Assemble final numeric input
    input_list = [float(age), float(gender_encoded), float(education_encoded)] + list(job_seq_padded) + [float(experience)]
    
    X_input = np.array(input_list, dtype=np.float32).reshape(1, -1)# force correct dtype

    return X_input

def predict_salary(age, gender, education, job_title, experience):
    encoders, tokenizer, model = load_all()
    X_input = preprocess_input(age, gender, education, job_title, experience, encoders, tokenizer)
    prediction = model.predict(X_input)
    return int(prediction[0][0])
