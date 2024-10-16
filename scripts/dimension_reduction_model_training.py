import os
import pandas as pd
import string
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.stem import PorterStemmer, WordNetLemmatizer
import joblib

# Define model and report file names
model_filename = 'models/logistic_regression_model.joblib'
report_filename = 'results/classification_report.txt'

# Load the dataset
csvFile = "data/data_jobs.csv"
df = pd.read_csv(csvFile)

# Example column to process (job description)
text_data = df['text']  # Assuming the text column is 'text'

# Initialize the Punkt tokenizer manually
punkt_param = PunktParameters()
sentence_splitter = PunktSentenceTokenizer(punkt_param)

# Step 1: Tokenization
def tokenize(text):
    return [word for sentence in sentence_splitter.tokenize(text.lower()) for word in sentence.split()]

# Step 2: Remove Stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

# Step 3: Stemming and Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def apply_stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

def apply_lemmatization(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

# Step 4: Remove Punctuation
def remove_punctuation(tokens):
    return [word for word in tokens if word not in string.punctuation]

# Step 5: Combining everything
def preprocess_text(text):
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = remove_punctuation(tokens)
    tokens = apply_lemmatization(tokens)  # You can use stemming or lemmatization
    return tokens

# Apply preprocessing to each job description
df['processed_text'] = text_data.apply(preprocess_text)

# Convert processed tokens back to sentences for vectorization
df['processed_text_joined'] = df['processed_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

# Step 6: Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_text_joined'])

# Step 1: Dimensionality Reduction using Truncated SVD (optional)
n_components = 100  # Adjust this number based on your needs
svd = TruncatedSVD(n_components=n_components)
reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)

# Step 2: Train/Test Split
# Assuming there's a 'label' column in the dataset, otherwise generate random labels for testing
if 'label' in df.columns:
    labels = df['label']
else:
    labels = [0 if i < len(df) / 2 else 1 for i in range(len(df))]  # Random binary labels

X_train, X_test, y_train, y_test = train_test_split(reduced_tfidf_matrix, labels, test_size=0.2, random_state=42)

# Check if the model already exists
if os.path.exists(model_filename):
    # Load the model
    print(f"Loading existing model from {model_filename}")
    model = joblib.load(model_filename)
else:
    # Step 3: Train a new Logistic Regression model
    print("Training a new model...")
    model = LogisticRegression()

    # Grid Search for hyperparameter tuning
    param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Update the model with the best found hyperparameters
    model = grid_search.best_estimator_

    # Save the model to a file
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

# Step 4: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Save classification report to a .txt file
report = classification_report(y_test, y_pred)
with open(report_filename, 'w') as f:
    f.write(f"Classification Report:\n{report}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"F1-Score: {f1}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")

print(f"Classification report and additional metrics saved to {report_filename}")

# Cross-validation for performance evaluation
cross_val_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean cross-validation score: {cross_val_scores.mean()}")
