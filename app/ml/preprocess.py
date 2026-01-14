import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove numbers
    text = re.sub(r"\d+", "", text)
    
    # 3. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # 4. Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # 5. Remove stop words & lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    text = " ".join(words)
    
    return text
