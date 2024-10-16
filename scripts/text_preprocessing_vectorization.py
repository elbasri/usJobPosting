import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# csv
csvFile = "data/data_jobs.csv"

# Load the dataset
df = pd.read_csv(csvFile)

# Example column to process (job description)
text_data = df['text'].head(50)  # Let's take the first 5 rows for demonstration

# Initialize the Punkt tokenizer manually
punkt_param = PunktParameters()
sentence_splitter = PunktSentenceTokenizer(punkt_param)

# Step 1: Tokenization (modified to use the existing Punkt tokenizer)
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

# Step 6: Vectorization using TF-IDF
vectorizer = TfidfVectorizer()

# Convert processed tokens back to sentences for vectorization
# Add a check to ensure only iterable data is processed
df['processed_text_joined'] = df['processed_text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')

tfidf_matrix = vectorizer.fit_transform(df['processed_text_joined'])

# Show the TF-IDF representation
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# Optional: Convert to DataFrame for better readability
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
print(tfidf_df.head())
