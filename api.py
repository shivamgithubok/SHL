from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional, Union
import uvicorn
from recommendation_engine import RecommendationEngine
from data_processor import DataProcessor
import os

# Initialize the data processor and recommendation engine
data_processor = DataProcessor()
recommendation_engine = RecommendationEngine(data_processor)

# Initialize FastAPI
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API for recommending SHL assessments based on job descriptions",
    version="1.0.0",
)

class HealthCheck(BaseModel):
    """Response model for health check endpoint"""
    status: str
    version: str

class RecommendationRequest(BaseModel):
    """Request model for recommendation endpoint"""
    query: str
    url: Optional[HttpUrl] = None
    max_results: Optional[int] = 10

class AssessmentResponse(BaseModel):
    """Response model for a single assessment"""
    name: str
    url: str
    remote_testing: str
    adaptive_support: str
    duration: str
    test_type: str
    similarity: float
    description: Optional[str] = None

class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoint"""
    recommendations: List[AssessmentResponse]
    count: int

@app.get("/health", response_model=HealthCheck)
def health_check():
    """Health check endpoint to verify API is running"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    """
    Get assessment recommendations based on a job description or query
    
    - **query**: Job description or natural language query
    - **url**: Optional URL to fetch additional job description content
    - **max_results**: Maximum number of recommendations to return (default: 10)
    """
    try:
        # Validate max_results
        max_results = min(max(request.max_results, 1), 10) if request.max_results else 10
        
        # Get recommendations using the engine
        recommendations = recommendation_engine.get_recommendations(
            query=request.query,
            url=str(request.url) if request.url else None,
            max_results=max_results
        )
        
        return {
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/recommend", response_model=RecommendationResponse)
def recommend_get(
    query: str = Query(..., description="Job description or natural language query"),
    url: Optional[str] = Query(None, description="Optional URL to fetch additional job description content"),
    max_results: Optional[int] = Query(10, description="Maximum number of recommendations to return")
):
    """
    Get assessment recommendations based on a job description or query (GET method)
    
    - **query**: Job description or natural language query
    - **url**: Optional URL to fetch additional job description content
    - **max_results**: Maximum number of recommendations to return (default: 10)
    """
    try:
        # Create a request object and use the POST handler
        request = RecommendationRequest(
            query=query,
            url=url,
            max_results=max_results
        )
        return recommend(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    # Run the API server if executed directly
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
