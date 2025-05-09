import streamlit as st
import pandas as pd
import requests
import json
import os
import sys
import logging
from io import StringIO
from typing import Dict, List, Optional, Any
import traceback

# Configure logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("SHL_Recommendation_App")

# Import the recommendation engine and data processor
from recommendation_engine import RecommendationEngine
from data_processor import DataProcessor
from evaluation import Evaluator

# Initialize the data processor, recommendation engine, and evaluator
data_processor = DataProcessor()
recommendation_engine = RecommendationEngine(data_processor)
evaluator = Evaluator()

def format_url(url: str) -> str:
    """Format URL as clickable markdown link."""
    name = url.split('/')[-2] if url.endswith('/') else url.split('/')[-1]
    return f"[View Details]({url})"

def create_recommendation_table(recommendations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame for displaying recommendations.
    
    Args:
        recommendations: List of assessment recommendation dictionaries
        
    Returns:
        Formatted pandas DataFrame for display
    """
    if not recommendations:
        return pd.DataFrame()
    
    # Extract relevant fields and format for display
    table_data = []
    for rec in recommendations:
        table_data.append({
            "Assessment Name": rec["name"],
            "URL": format_url(rec["url"]),
            "Remote Testing": rec["remote_testing"],
            "Adaptive/IRT Support": rec["adaptive_support"],
            "Duration": rec["duration"],
            "Test Type": rec["test_type"],
            "Relevance Score": f"{rec['similarity']:.4f}"
        })
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    return df

def get_recommendations_from_text(text: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Get recommendations using the recommendation engine.
    
    Args:
        text: Job description or query text
        max_results: Maximum number of recommendations to return
        
    Returns:
        List of assessment recommendations
    """
    try:
        recommendations = recommendation_engine.get_recommendations(
            query=text,
            max_results=max_results
        )
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        traceback.print_exc()
        return []

def evaluate_results(query: str, recommendations: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate the recommendations using the evaluator.
    
    Args:
        query: The query text
        recommendations: List of assessment recommendations
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        metrics = evaluator.evaluate_recommendations(query, recommendations, k=3)
        return metrics
    except Exception as e:
        st.error(f"Error evaluating recommendations: {str(e)}")
        return {"recall@k": 0.0, "map@k": 0.0, "has_test_data": False}

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="SHL Assessment Recommendation System",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 SHL Assessment Recommendation System")
    
    st.markdown("""
    This application helps hiring managers find the right SHL assessments for their job roles.
    Enter a job description or a specific query about what you're looking for, and we'll recommend
    suitable assessments from SHL's catalog.
    """)
    
    # Input section
    st.header("Job Description or Query")
    
    input_option = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload", "Example Queries"],
        horizontal=True
    )
    
    query_text = ""
    
    if input_option == "Text Input":
        query_text = st.text_area(
            "Enter job description or query:",
            height=150,
            placeholder="Example: I am hiring for Java developers who can collaborate effectively with business teams."
        )
    
    elif input_option == "File Upload":
        uploaded_file = st.file_uploader("Upload a job description file", type=["txt", "pdf"])
        if uploaded_file is not None:
            # Handle different file types
            file_extension = uploaded_file.name.split(".")[-1].lower()
            
            if file_extension == "txt":
                # Read text file
                string_data = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                query_text = string_data
                st.success(f"Uploaded and processed text file: {uploaded_file.name}")
            elif file_extension == "pdf":
                # Create uploads directory if it doesn't exist
                import os
                os.makedirs("uploads", exist_ok=True)
                
                # Save the uploaded PDF temporarily
                pdf_path = f"uploads/{uploaded_file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                try:
                    # Save file for user to see
                    st.success(f"PDF file saved: {uploaded_file.name}")
                    st.info("For PDF files, please enter your job description or query manually in the text area below.")
                    
                    # Provide a text area for manual entry
                    query_text = st.text_area(
                        "Enter your query based on the PDF content:",
                        height=150,
                        placeholder="Enter your job description or query here after reviewing the PDF file."
                    )
                except Exception as e:
                    st.error(f"Error processing PDF file: {str(e)}")
                    st.info("Please try uploading a text file or enter your query manually.")
    
    elif input_option == "Example Queries":
        example_queries = [
            "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
            "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
            "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins.",
            "I want to hire new graduates for a sales role in my company, the budget is for about an hour for each test. Give me some options"
        ]
        
        selected_example = st.selectbox(
            "Choose an example query:",
            example_queries
        )
        
        query_text = selected_example
        st.info("Example query selected. You can modify it in the text area below.")
        query_text = st.text_area("Edit query if needed:", value=query_text, height=100)
    
    # Max results slider
    max_results = st.slider("Maximum number of recommendations:", min_value=1, max_value=10, value=5)
    
    # Button to generate recommendations
    generate_button = st.button("Generate Recommendations")
    
    if generate_button and query_text:
        with st.spinner("Generating recommendations..."):
            # Get recommendations
            recommendations = get_recommendations_from_text(query_text, max_results)
            
            if recommendations:
                # Create and display recommendation table
                df = create_recommendation_table(recommendations)
                st.header("Recommended Assessments")
                st.dataframe(df, hide_index=True, use_container_width=True)
                
                # Evaluate recommendations
                metrics = evaluate_results(query_text, recommendations)
                
                # Display evaluation metrics if test data exists
                if metrics["has_test_data"]:
                    st.header("Evaluation Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Recall@3", f"{metrics['recall@k']:.2f}")
                    with col2:
                        st.metric("MAP@3", f"{metrics['map@k']:.2f}")
                    
                    st.info("""
                    **Recall@3**: Proportion of relevant assessments found in the top 3 recommendations.
                    **MAP@3**: Mean Average Precision at 3, considering both relevance and ranking order.
                    """)
            else:
                st.warning("No recommendations found. Please try a different query.")
    
    # API information section
    st.header("API Access")
    st.markdown("""
    You can access this recommendation engine programmatically via our API:
    
    **Health Check Endpoint**:
    ```
    GET /health
    ```
    
    **Recommendation Endpoint**:
    ```
    POST /recommend
    {
        "query": "Your job description or query text",
        "url": "Optional URL to fetch additional content",
        "max_results": 10
    }
    ```
    
    **GET Recommendation Endpoint**:
    ```
    GET /recommend?query=Your+job+description&max_results=10
    ```
    """)
    
    # Footer with information about the system
    st.markdown("---")
    st.markdown("""
    ### About This System
    
    This SHL Assessment Recommendation System uses natural language processing and semantic search to match job descriptions with appropriate assessments.
    
    * **Data**: Assessments from SHL's product catalog
    * **Matching**: Semantic similarity using TF-IDF vectorization
    * **Evaluation**: Measured using Recall@3 and MAP@3 metrics
    
    For more information, see the deployment instructions or contact support.
    """)
    
if __name__ == "__main__":
    main()
    