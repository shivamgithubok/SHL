# SHL Assessment Recommendation System Deployment Instructions

This document provides comprehensive instructions for deploying the SHL Assessment Recommendation System.

## Project Overview

The SHL Assessment Recommendation System is a web application that helps hiring managers find appropriate assessments for job roles. Given a job description or natural language query, the system recommends relevant SHL assessments.

### Key Features
- Natural language job description input
- Text file and PDF upload capability
- URL-based content fetching
- Assessment recommendations with detailed attributes
- API access for programmatic usage
- Evaluation metrics for recommendation quality

## File Structure

```
shl-assessment-recommender/
├── app.py                    # Streamlit web application
├── api.py                    # FastAPI backend
├── data_processor.py         # Data loading and embedding
├── recommendation_engine.py  # Core recommendation logic
├── evaluation.py             # System evaluation metrics
├── data/                     # Contains assessment data
│   └── shl_assessments.json  # Assessment catalog
├── uploads/                  # Directory for uploaded files
├── .streamlit/               # Streamlit configuration
│   └── config.toml           # Server settings
├── deployment_requirements.txt # Package dependencies
└── Procfile                  # Process declarations
```

## Deployment Options

### Option 1: Deployment on Streamlit Cloud

#### Prerequisites
- A GitHub account
- Your project code pushed to a GitHub repository

#### Steps

1. **Push your code to GitHub**
   - Create a new repository on GitHub
   - Push your project code to the repository

2. **Sign up for Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Sign in with your GitHub account

3. **Deploy your app**
   - Click "New app"
   - Select your repository, branch, and the `app.py` file
   - Advanced settings: Set the Python requirements file to `deployment_requirements.txt`
   - Click "Deploy"
   - Your app will be live at a URL like: `https://username-app-name-streamlit-app.streamlit.app`

### Option 2: Deployment on Render

Render is a unified cloud platform that can host both your Streamlit frontend and FastAPI backend.

#### Prerequisites
- A Render account (https://render.com)
- Your project code pushed to a GitHub repository

#### Steps for Deploying the Streamlit App

1. **Sign up for Render**
   - Create an account at https://render.com

2. **Create a new Web Service**
   - Click "New" and select "Web Service"
   - Connect your GitHub repository
   - Choose the repository containing your SHL Assessment Recommendation System

3. **Configure the Streamlit Web Service**
   - Name: `shl-recommendation-frontend`
   - Environment: `Python 3`
   - Build Command: `pip install -r deployment_requirements.txt`
   - Start Command: `streamlit run app.py --server.port 5000 --server.address 0.0.0.0`
   - Plan: Free

4. **Add Environment Variables (if needed)**
   - No special environment variables are required for this project

5. **Deploy the Streamlit App**
   - Click "Create Web Service"
   - Render will deploy your Streamlit app, which will be available at a URL like `https://shl-recommendation-frontend.onrender.com`

#### Steps for Deploying the API

1. **Create another Web Service for the API**
   - Click "New" and select "Web Service"
   - Connect to the same GitHub repository

2. **Configure the API Web Service**
   - Name: `shl-recommendation-api`
   - Environment: `Python 3`
   - Build Command: `pip install -r deployment_requirements.txt`
   - Start Command: `uvicorn api:app --host 0.0.0.0 --port 8000`
   - Plan: Free

3. **Deploy the API**
   - Click "Create Web Service"
   - Your API will be available at a URL like `https://shl-recommendation-api.onrender.com`

4. **Update Your Streamlit App (If Needed)**
   - If your Streamlit app makes direct calls to the API, update the API URL in your code to point to the new Render-hosted API URL

### Option 3: Running Locally

To run the application locally:

1. **Install dependencies**
   ```
   pip install -r deployment_requirements.txt
   ```

2. **Ensure directory structure**
   - Make sure the `data` and `uploads` directories exist
   - Verify that `data/shl_assessments.json` contains assessment data

3. **Run the Streamlit app**
   ```
   streamlit run app.py
   ```

4. **Run the API (in a separate terminal)**
   ```
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

5. Access the Streamlit app at `http://localhost:5000` and the API at `http://localhost:8000`

## Implementation Details

### Data Processing
- Assessment data is stored in JSON format
- Text is processed using TF-IDF vectorization for semantic similarity
- No external API keys or services required

### API Documentation
- The API documentation is available at `/docs` (e.g., `http://localhost:8000/docs` when running locally)
- Two main endpoints: `/health` and `/recommend`
- Both GET and POST methods supported for recommendations

### Evaluation
- System performance is measured using Recall@3 and MAP@3 metrics
- Test cases are included in the evaluation module

## Troubleshooting

If you encounter issues:

1. **Verify package installation**
   - Check that all dependencies in `deployment_requirements.txt` are installed

2. **Check file permissions**
   - Ensure the `uploads` directory is writable

3. **Review logs**
   - Check Streamlit and web server logs for errors

4. **Verify data access**
   - Confirm that `data/shl_assessments.json` is present and properly formatted
