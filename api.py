"""News Summarization API.

This module provides API endpoints for extracting and analyzing news about companies.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

import json
import uvicorn

import utils


app = FastAPI(
    title="News Summarization API",
    description="API for extracting and analyzing news about companies",
    version="1.0.0"
)


class CompanyRequest(BaseModel):
    """Request model for company news queries."""
    
    company_name: str


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint returning API information.
    
    Returns:
        Dict containing welcome message
    """
    return {"message": "News Summarization and Analysis API"}


@app.post("/news")
async def get_company_news(request: CompanyRequest) -> Dict[str, Any]:
    """Get news articles and sentiment analysis for a company.
    
    Args:
        request: CompanyRequest object containing the company name
        
    Returns:
        Dict containing news and sentiment analysis
        
    Raises:
        HTTPException: If news retrieval fails
    """
    try:
        result = utils.get_company_news_with_sentiment(request.company_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/news/{company_name}")
async def get_company_news_get(company_name: str) -> Dict[str, Any]:
    """Get news articles and sentiment analysis for a company (GET method).
    
    Args:
        company_name: Name of the company to get news for
        
    Returns:
        Dict containing news and sentiment analysis
        
    Raises:
        HTTPException: If news retrieval fails
    """
    try:
        result = utils.get_company_news_with_sentiment(company_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)