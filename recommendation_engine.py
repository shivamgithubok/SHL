import numpy as np
from typing import List, Dict, Any, Optional
import re
from data_processor import DataProcessor
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    """
    Engine for providing SHL assessment recommendations based on
    job descriptions or queries.
    """
    
    def __init__(self, data_processor: DataProcessor):
        """
        Initialize the recommendation engine with a data processor.
        
        Args:
            data_processor: DataProcessor instance with loaded assessment data
        """
        self.data_processor = data_processor
    
    def get_recommendations(self, query: str, url: Optional[str] = None, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Get assessment recommendations based on a text query or URL content.
        
        Args:
            query: Text query or job description
            url: Optional URL to fetch content from
            max_results: Maximum number of recommendations to return
            
        Returns:
            List of assessment recommendations with similarity scores
        """
        # If URL is provided, fetch and use its content
        if url and url.strip():
            url_content = self.data_processor.fetch_text_from_url(url)
            if url_content:
                query = f"{query} {url_content}"
        
        # Process the query text
        query = self._preprocess_query(query)
        
        # If we don't have any assessment data, return empty list
        if len(self.data_processor.assessments) == 0 or self.data_processor.embeddings is None:
            return []
        
        # Get embedding for the query
        query_embedding = self.data_processor.get_embeddings_for_query(query)
        
        # Calculate cosine similarity between query and all assessments
        similarities = self._calculate_similarities(query_embedding)
        
        # Extract duration constraints from query if any
        max_duration = self._extract_duration_constraint(query)
        
        # Get top k recommendations
        recommendations = self._get_top_recommendations(similarities, max_results, max_duration)
        
        return recommendations
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query text to enhance matching quality.
        
        Args:
            query: Text query to preprocess
            
        Returns:
            Preprocessed query text
        """
        # Convert to lowercase
        query = query.lower()
        
        # Replace multiple spaces with a single space
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def _calculate_similarities(self, query_embedding) -> np.ndarray:
        """
        Calculate cosine similarity between a query embedding and all assessment embeddings.
        
        Args:
            query_embedding: TF-IDF vector for the query
            
        Returns:
            Array of similarity scores
        """
        # Calculate cosine similarity between query and all assessments
        similarities = cosine_similarity(query_embedding, self.data_processor.embeddings).flatten()
        
        return similarities
    
    def _extract_duration_constraint(self, query: str) -> Optional[int]:
        """
        Extract time/duration constraints from the query.
        
        Args:
            query: Text query to analyze
            
        Returns:
            Maximum duration in minutes or None if no constraint found
        """
        # Look for patterns like "30 minutes", "45 mins", "1 hour", etc.
        minute_patterns = [
            r'(\d+)\s*(?:minute|minutes|min|mins)',
            r'(\d+)\s*(?:hour|hours|hr|hrs)'
        ]
        
        for pattern in minute_patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                duration = int(matches[0])
                # Convert hours to minutes if needed
                if 'hour' in pattern or 'hr' in pattern:
                    duration *= 60
                return duration
        
        return None
    
    def _get_top_recommendations(self, 
                              similarities: np.ndarray, 
                              max_results: int, 
                              max_duration: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get top recommendations based on similarity scores and constraints.
        
        Args:
            similarities: Array of similarity scores
            max_results: Maximum number of results to return
            max_duration: Maximum allowed duration in minutes
            
        Returns:
            List of assessment dictionaries with added similarity scores
        """
        # Create a list of (index, similarity) tuples
        indexed_similarities = [(i, sim) for i, sim in enumerate(similarities)]
        
        # Sort by similarity in descending order
        indexed_similarities.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for idx, similarity in indexed_similarities:
            assessment = self.data_processor.assessments[idx]
            
            # Apply duration filter if specified
            if max_duration:
                # Extract numeric value from duration string
                duration_str = assessment.get('duration', '0 minutes')
                duration_match = re.search(r'(\d+)', duration_str)
                duration_value = int(duration_match.group(1)) if duration_match else 0
                
                # Skip if longer than max_duration
                if duration_value > max_duration:
                    continue
            
            # Add recommendation with similarity score
            recommendation = assessment.copy()
            recommendation['similarity'] = round(float(similarity), 4)
            recommendations.append(recommendation)
            
            # Stop once we have enough recommendations
            if len(recommendations) >= max_results:
                break
        
        # Ensure we return at least one recommendation if available
        if not recommendations and len(self.data_processor.assessments) > 0:
            # Get the highest similarity assessment regardless of constraints
            idx = indexed_similarities[0][0]
            assessment = self.data_processor.assessments[idx]
            assessment = assessment.copy()
            assessment['similarity'] = round(float(indexed_similarities[0][1]), 4)
            recommendations.append(assessment)
        
        return recommendations
