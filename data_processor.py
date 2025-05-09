import json
import os
import traceback
import re
import requests
import logging
import sys
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logger = logging.getLogger("SHL_Data_Processor")

class DataProcessor:
    """
    Class to handle all data loading, processing, and embedding operations
    for the SHL Assessment Recommendation System.
    """
    
    def __init__(self):
        """
        Initialize the data processor with TF-IDF vectorizer.
        """
        self.assessments = []
        self.embeddings = None
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=5000, 
            ngram_range=(1, 2)
        )
        self.load_data()
        
    def load_data(self) -> None:
        """Load assessment data from JSON file and create embeddings."""
        try:
            # Load assessments from the JSON file
            # Handle both local and deployment paths
            try:
                with open('data/shl_assessments.json', 'r') as f:
                    self.assessments = json.load(f)
            except FileNotFoundError:
                # Try alternative path for Streamlit Cloud
                with open('./data/shl_assessments.json', 'r') as f:
                    self.assessments = json.load(f)
            
            # Create description texts for embedding
            description_texts = []
            for assessment in self.assessments:
                # Combine name, test_type, and description for richer semantic matching
                text = f"{assessment['name']} {assessment['test_type']} {assessment.get('description', '')}"
                description_texts.append(text)
            
            # Generate embeddings for all assessments using TF-IDF
            self.vectorizer.fit(description_texts)
            self.embeddings = self.vectorizer.transform(description_texts)
            
            logger.info(f"Loaded {len(self.assessments)} assessments and created embeddings.")
        except Exception as e:
            logger.error(f"Error loading assessment data: {str(e)}")
            logger.error(traceback.format_exc())
            # Initialize with empty data if loading fails
            self.assessments = []
            self.embeddings = None
    
    def get_embeddings_for_query(self, query: str):
        """
        Generate embeddings for a query string.
        
        Args:
            query: Text query to embed
            
        Returns:
            TF-IDF sparse matrix for the query
        """
        query_vec = self.vectorizer.transform([query])
        return query_vec
    
    def fetch_text_from_url(self, url: str) -> Optional[str]:
        """
        Fetch text content from a URL.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Text content of the URL or None if fetching fails
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Extract text content, removing HTML
            text = re.sub(r'<.*?>', ' ', response.text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            print(f"Error fetching URL content: {str(e)}")
            return None
    
    def get_all_assessments(self) -> List[Dict[str, Any]]:
        """Return all assessment data."""
        return self.assessments
    
    def get_assessment_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find an assessment by its name.
        
        Args:
            name: Name of the assessment to find
            
        Returns:
            Assessment data dictionary or None if not found
        """
        for assessment in self.assessments:
            if assessment['name'].lower() == name.lower():
                return assessment
        return None
