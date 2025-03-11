import re 
import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import os

# Download the stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load all models
def load_model(filename):
    with open(os.path.join(BASE_DIR, 'models', filename), 'rb') as file:
        return pickle.load(file)

cv = load_model('count_vectorizer.pkl')
feature = load_model('feature_names.pkl')
tfidf = load_model('tfidf.pkl')

# Preprocessing function
def preprocessing(txt): 
    txt = re.sub('[^a-zA-Z]', ' ', txt).lower()
    txt = nltk.word_tokenize(txt)
    ps = PorterStemmer()
    txt = [ps.stem(word) for word in txt if word not in stop_words and len(word) > 3]
    return ' '.join(txt)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # Try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    except Exception as e:
        raise Exception(f"Error reading file: {file.name} - {str(e)}")    
    return text

#extract keywords 
def extract_keywords(txt): 
    txt = preprocessing(txt)
    txt = cv.transform([txt])
    txt = tfidf.transform(txt)
    sorted_items = sort_coo(txt.tocoo())
    keywords = extract_topn_from_vector(feature, sorted_items, 10)
    return keywords

def process_file(file):
    text = extract_text_from_txt(file)
    keywords = extract_keywords(text)
    return keywords