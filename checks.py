"""# More detailed GPU availability check for spaCy
import spacy
from thinc.api import require_gpu

try:
    require_gpu(0)  # This will raise an error if GPU is not available to spaCy
    print("GPU successfully required")
except Exception as e:
    print(f"Error requiring GPU: {e}")

# Now check spaCy's preference
gpu_available = spacy.prefer_gpu()
print(f"spaCy GPU available: {gpu_available}")"""


"""import spacy
try:
    # Try loading Turkish model
    nlp = spacy.load("tr_core_news_trf")
    
    # Test lemmatization with a Turkish sample
    sample = "Öğrenciler matematiği anlamak için çalışıyorlar."
    doc = nlp(sample)
    lemmas = [token.lemma_ for token in doc if not token.is_punct]
    
    print("\nSpaCy lemmatization test:")
    print(f"Original: {sample}")
    print(f"Lemmatized: {' '.join(lemmas)}")
except Exception as e:
    print(f"SpaCy model test failed: {e}")"""