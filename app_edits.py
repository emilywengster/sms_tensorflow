from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load the model
model = load_model('spam_detection_model_4.keras')

# Load the tokenizer
with open('tokenizer_4.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Constants
SEQUENCE_LENGTH = 100

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        sequence = tokenizer.texts_to_sequences([message])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
        prediction = model.predict(padded_sequence)[0][0]

        if prediction > 0.5:
            result = 'Spam'
        else:
            result = 'Ham'

        return jsonify({'prediction': result, 'probability': float(prediction)})

if __name__ == '__main__':
    app.run(port=8000)
