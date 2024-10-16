from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.stem import WordNetLemmatizer
import string

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define file paths
MODEL_PATH = 'models/logistic_regression_model.joblib'
RESULTS_DIR = 'results/'
MODEL_DIR = 'models/'

# Tokenization and preprocessing utilities
punkt_param = PunktParameters()
sentence_splitter = PunktSentenceTokenizer(punkt_param)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def preprocess_text(text):
    tokens = [word for sentence in sentence_splitter.tokenize(text.lower()) for word in sentence.split()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


@app.route('/train', methods=['POST'])
def train_model():
    print("call train func")
    file = request.files.get('file')
    print("file received")
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    df = pd.read_csv(file)
    if 'text' not in df.columns or 'label' not in df.columns:
        return jsonify({'error': 'Missing columns: text and/or label'}), 400

    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])

    # Dimensionality reduction (SVD)
    n_components = 100
    svd = TruncatedSVD(n_components=n_components)
    reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(reduced_tfidf_matrix, df['label'], test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    print("Starting training the model..")
    model.fit(X_train, y_train)

    # Save model, vectorizer, and SVD
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.joblib'))
    joblib.dump(svd, os.path.join(MODEL_DIR, 'svd.joblib'))
    return jsonify({'message': 'Model, vectorizer, and SVD trained and saved successfully'})



@app.route('/predict', methods=['POST'])
def predict_text():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Load the pre-trained model, vectorizer, and SVD transformer
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model not found. Train the model first.'}), 400

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.joblib'))
    svd = joblib.load(os.path.join(MODEL_DIR, 'svd.joblib'))

    # Preprocess the input text
    processed_text = preprocess_text(text)

    # Vectorization using the saved TF-IDF vectorizer
    tfidf_matrix = vectorizer.transform([processed_text])

    # Dimensionality reduction using the saved SVD transformer
    reduced_tfidf_matrix = svd.transform(tfidf_matrix)

    # Prediction
    y_pred = model.predict(reduced_tfidf_matrix)
    return jsonify({'prediction': y_pred[0]})

@app.route('/predictfile', methods=['POST'])
def predict_file():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Load the pre-trained model, vectorizer, and SVD transformer
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model not found. Train the model first.'}), 400

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.joblib'))
    svd = joblib.load(os.path.join(MODEL_DIR, 'svd.joblib'))

    # Load the CSV file and preprocess it
    df = pd.read_csv(file)
    if 'text' not in df.columns:
        return jsonify({'error': 'Missing text column'}), 400

    df['processed_text'] = df['text'].apply(preprocess_text)

    # Vectorization using the saved TF-IDF vectorizer
    tfidf_matrix = vectorizer.transform(df['processed_text'])

    # Dimensionality reduction using the saved SVD transformer
    reduced_tfidf_matrix = svd.transform(tfidf_matrix)

    # Prediction
    y_pred = model.predict(reduced_tfidf_matrix)

    # Save the results to a JSON file
    df['prediction'] = y_pred
    result_file = f"{RESULTS_DIR}results_{file.filename}.json"
    df.to_json(result_file, orient='records', lines=True)

    return jsonify({'message': 'Predictions saved', 'result_file': result_file})


@app.route('/results/<filename>', methods=['GET'])
def get_results(filename):
    result_file = f"{RESULTS_DIR}{filename}"
    if not os.path.exists(result_file):
        return jsonify({'error': 'File not found'}), 404

    return send_file(result_file)


if __name__ == '__main__':
    app.run(debug=True)
