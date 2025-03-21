"""Sentiment analysis module for processing news articles content."""

import logging
import nltk
from typing import Dict, Any, List
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Set up logging
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")


class SentimentAnalyzer:
    """Class for sentiment analysis and topic extraction."""
    
    def __init__(self):
        """Initialize sentiment analysis tools."""
        self.sia = SentimentIntensityAnalyzer()
        # Use a more advanced sentiment model when needed
        try:
            self.advanced_sentiment = pipeline("sentiment-analysis")
        except Exception as e:
            logger.warning(f"Could not load advanced sentiment model: {e}")
            self.advanced_sentiment = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze the sentiment of a given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and category
        """
        # Use VADER for faster analysis
        scores = self.sia.polarity_scores(text)
        
        # Determine sentiment category
        compound = scores['compound']
        if compound >= 0.05:
            category = "Positive"
        elif compound <= -0.05:
            category = "Negative"
        else:
            category = "Neutral"
        
        return {
            "scores": scores,
            "category": category,
            "compound": compound
        }
    
    def extract_topics(self, keywords: List[str], text: str) -> List[str]:
        """Extract main topics from the article based on keywords and content.
        
        Args:
            keywords: List of keywords from the article
            text: Full article text
            
        Returns:
            List of identified topics
        """
        # This is a simplified version - could be enhanced with entity recognition
        common_business_topics = [
            "Financial Results", "Stock Market", "Earnings", "Revenue",
            "Product Launch", "Innovation", "Research", "Development",
            "Merger", "Acquisition", "Partnership", "Collaboration",
            "Regulation", "Compliance", "Legal", "Lawsuit",
            "Leadership", "Management", "Executive", "CEO",
            "Market Share", "Competition", "Industry", "Sector"
        ]
        
        topics = set()
        
        # Add keywords that match common topics
        for keyword in keywords:
            for topic in common_business_topics:
                if (keyword.lower() in topic.lower() or 
                        topic.lower() in keyword.lower()):
                    topics.add(topic)
        
        # If we have few topics, add some based on text content
        if len(topics) < 3:
            for topic in common_business_topics:
                if topic.lower() in text.lower():
                    topics.add(topic)
        
        return list(topics)[:5]  # Return top 5 topics