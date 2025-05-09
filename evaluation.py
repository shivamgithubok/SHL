from typing import List, Dict, Any, Tuple
import json
import numpy as np

class Evaluator:
    """
    Class for evaluating the performance of the recommendation system
    using standard Information Retrieval metrics.
    """
    
    def __init__(self):
        """Initialize the evaluator with test data."""
        self.test_data = self._load_test_data()
        
    def _load_test_data(self) -> List[Dict[str, Any]]:
        """
        Load test data including queries and their relevant assessments.
        
        Returns:
            List of test cases with queries and relevant assessments
        """
        # This is a simplified version of the test data shown in the PDF
        # In a real scenario, this would be loaded from a file
        test_data = [
            {
                "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
                "relevant_assessments": [
                    "Automata - Fix (New)",
                    "Core Java (Entry Level) (New)", 
                    "Java 8 (New)",
                    "Core Java (Advanced Level) (New)",
                    "Agile Software Development"
                ]
            },
            {
                "query": "I want to hire new graduates for a sales role in my company, the budget is for about an hour for each test. Give me some options",
                "relevant_assessments": [
                    "Entry level Sales 7.1 (International)",
                    "Entry Level Sales Sift Out 7.1",
                    "Entry Level Sales Solution",
                    "Sales Representative Solution",
                    "Sales Support Specialist Solution",
                    "Technical Sales Associate Solution",
                    "SVAR - Spoken English (Indian Accent) (New)",
                    "Sales & Service Phone Solution",
                    "Sales & Service Phone Simulation",
                    "English Comprehension (New)"
                ]
            }
        ]
        return test_data
    
    def compute_recall_at_k(self, recommendations: List[Dict[str, Any]], 
                           relevant_assessments: List[str], k: int = 3) -> float:
        """
        Compute Recall@K for a single query.
        
        Args:
            recommendations: List of recommended assessments
            relevant_assessments: List of relevant assessment names
            k: Cutoff threshold
            
        Returns:
            Recall@K score
        """
        if not relevant_assessments:
            return 1.0  # Perfect recall if there are no relevant items
        
        # Count relevant items in the top-k recommendations
        relevant_in_top_k = 0
        for i, rec in enumerate(recommendations):
            if i >= k:
                break
            if rec['name'] in relevant_assessments:
                relevant_in_top_k += 1
        
        # Calculate recall
        recall = relevant_in_top_k / len(relevant_assessments)
        return recall
    
    def compute_ap_at_k(self, recommendations: List[Dict[str, Any]], 
                       relevant_assessments: List[str], k: int = 3) -> float:
        """
        Compute Average Precision@K for a single query.
        
        Args:
            recommendations: List of recommended assessments
            relevant_assessments: List of relevant assessment names
            k: Cutoff threshold
            
        Returns:
            AP@K score
        """
        if not relevant_assessments:
            return 1.0  # Perfect AP if there are no relevant items
        
        precision_sum = 0.0
        num_relevant_found = 0
        
        # Calculate precision at each relevant position
        for i, rec in enumerate(recommendations[:k]):
            if rec['name'] in relevant_assessments:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        # Calculate average precision
        if num_relevant_found == 0:
            return 0.0
        
        ap = precision_sum / min(len(relevant_assessments), k)
        return ap
    
    def evaluate_recommendations(self, query: str, recommendations: List[Dict[str, Any]], 
                               k: int = 3) -> Dict[str, float]:
        """
        Evaluate recommendations for a given query against test data.
        
        Args:
            query: Query text
            recommendations: List of recommended assessments
            k: Cutoff threshold
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Find the matching test case
        matching_test_case = None
        for test_case in self.test_data:
            # Simple string matching - in a real system, would use better matching
            if query.lower() in test_case['query'].lower() or test_case['query'].lower() in query.lower():
                matching_test_case = test_case
                break
        
        if not matching_test_case:
            return {"recall@k": 0.0, "map@k": 0.0, "has_test_data": False}
        
        # Compute evaluation metrics
        recall = self.compute_recall_at_k(
            recommendations, 
            matching_test_case['relevant_assessments'], 
            k
        )
        
        ap = self.compute_ap_at_k(
            recommendations, 
            matching_test_case['relevant_assessments'], 
            k
        )
        
        return {
            "recall@k": recall,
            "map@k": ap,
            "has_test_data": True
        }
    
    def evaluate_system(self, recommendation_engine, k: int = 3) -> Dict[str, float]:
        """
        Evaluate the entire recommendation system on all test cases.
        
        Args:
            recommendation_engine: The recommendation engine to evaluate
            k: Cutoff threshold
            
        Returns:
            Dictionary with overall evaluation metrics
        """
        recall_scores = []
        ap_scores = []
        
        for test_case in self.test_data:
            query = test_case['query']
            relevant_assessments = test_case['relevant_assessments']
            
            # Get recommendations for the query
            recommendations = recommendation_engine.get_recommendations(query, max_results=10)
            
            # Compute metrics
            recall = self.compute_recall_at_k(recommendations, relevant_assessments, k)
            ap = self.compute_ap_at_k(recommendations, relevant_assessments, k)
            
            recall_scores.append(recall)
            ap_scores.append(ap)
        
        # Calculate mean metrics
        mean_recall = np.mean(recall_scores) if recall_scores else 0.0
        mean_ap = np.mean(ap_scores) if ap_scores else 0.0
        
        return {
            "mean_recall@k": mean_recall,
            "map@k": mean_ap,
            "k": k,
            "num_test_cases": len(self.test_data)
        }
