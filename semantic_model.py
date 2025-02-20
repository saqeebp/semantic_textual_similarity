# Part A: Model Training & Similarity Calculation
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticModel:
    def __init__(self, data_path='dataneuron_text_similarity.csv'):
        self.df = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._train()
    
    def _clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower().strip()
        return text
    
    def _train(self):
        # Combine all text for TF-IDF training
        self.df['text1_clean'] = self.df['text1'].apply(self._clean_text)
        self.df['text2_clean'] = self.df['text2'].apply(self._clean_text)
        combined_text = pd.concat([self.df['text1_clean'], self.df['text2_clean']])
        self.vectorizer.fit(combined_text)
    
    def predict(self, text1, text2):
        # Preprocess & vectorize input texts
        text1_clean = self._clean_text(text1)
        text2_clean = self._clean_text(text2)
        vectors = self.vectorizer.transform([text1_clean, text2_clean])
        return float(cosine_similarity(vectors[0], vectors[1])[0][0])
