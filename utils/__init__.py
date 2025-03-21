"""
Utility functions for extracting and analyzing company news.
"""
from .news_extractor import NewsExtractor
from .sentiment_analyzer import SentimentAnalyzer
from .text_to_speech import TextToSpeechHindi
from .analysis import (
    get_company_news_with_sentiment,
    perform_comparative_analysis,
    get_final_sentiment,
    generate_final_summary
)

# For backward compatibility
def initialize_extractors():
    """Initialize and return extractors and analyzers."""
    news_extractor = NewsExtractor()
    sentiment_analyzer = SentimentAnalyzer()
    tts_engine = TextToSpeechHindi()
    return news_extractor, sentiment_analyzer, tts_engine

# Expose the main functions
__all__ = [
    'NewsExtractor',
    'SentimentAnalyzer',
    'TextToSpeechHindi',
    'initialize_extractors',
    'get_company_news_with_sentiment'
]