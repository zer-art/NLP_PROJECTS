import pickle
import PyPDF2  # Extract text from PDF
import re
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords if not already downloaded
nltk.download('stopwords', quiet=True)
english_stopwords = stopwords.words('english')

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
svc_model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'resume_model.pkl'), 'rb'))  # Adjust path as needed
tfidf = pickle.load(open(os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'), 'rb'))  # Adjust path as needed
le = pickle.load(open(os.path.join(BASE_DIR, 'models', 'encoder.pkl'), 'rb'))  # Adjust path as needed

# Function to clean resume text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = [word for word in text.split() if word not in english_stopwords]
    text = ' '.join(text)
    return text

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # Try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text

# Function to predict the category of a resume
def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = clean_text(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = svc_model.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]  # Return the category name
