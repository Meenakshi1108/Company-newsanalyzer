"""
Analysis functions for processing news articles and generating insights.
"""
import logging
from typing import List, Dict, Any

from .news_extractor import NewsExtractor
from .sentiment_analyzer import SentimentAnalyzer
from .text_to_speech import TextToSpeechHindi

# Set up logging
logger = logging.getLogger(__name__)

def get_company_news_with_sentiment(company_name: str) -> Dict[str, Any]:
    """
    Main function to get company news with sentiment analysis.
    
    Args:
        company_name: Name of the company to analyze
        
    Returns:
        Dictionary with company name, articles with sentiment, and analysis
    """
    news_extractor = NewsExtractor()
    sentiment_analyzer = SentimentAnalyzer()
    tts_engine = TextToSpeechHindi()
    
    # Get articles
    articles = news_extractor.get_articles_for_company(company_name)
    
    # If we couldn't get enough articles
    if len(articles) < 3:  # Lowered threshold to 3 articles
        return {
            "company": company_name,
            "status": "error",
            "message": (f"Could only find {len(articles)} articles. "
                        f"Need at least 3 for meaningful analysis.")
        }
    
    # Process each article with sentiment and topics
    processed_articles = []
    for article in articles:
        sentiment = sentiment_analyzer.analyze_sentiment(article["text"])
        topics = sentiment_analyzer.extract_topics(
            article.get("keywords", []), article["text"]
        )
        
        processed_articles.append({
            "title": article["title"],
            "summary": article["summary"],
            "sentiment": sentiment["category"],
            "sentiment_score": sentiment["compound"],
            "topics": topics,
            "url": article["url"],
            "source": article["source"],
            "date": article.get("date", article.get("publish_date", "Unknown"))
        })
    
    # Add comparative analysis
    comparative_analysis = perform_comparative_analysis(processed_articles)
    
    # Generate final summary for TTS
    final_summary = generate_final_summary(
        company_name, processed_articles, comparative_analysis
    )
    
    # Generate Hindi TTS
    audio_file = None
    try:
        audio_file = tts_engine.generate_summary_audio(final_summary)
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
    
    # Return the processed data
    return {
        "company": company_name,
        "status": "success",
        "articles_count": len(processed_articles),
        "articles": processed_articles,
        "comparative_analysis": comparative_analysis,
        "summary": final_summary,
        "audio_file": audio_file
    }


def perform_comparative_analysis(articles: List[Dict]) -> Dict[str, Any]:
    """
    Perform comparative analysis across articles.
    
    Args:
        articles: List of processed articles
        
    Returns:
        Dictionary with analysis results
    """
    # Calculate sentiment distribution
    sentiment_distribution = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for article in articles:
        sentiment = article.get("sentiment")
        if sentiment in sentiment_distribution:
            sentiment_distribution[sentiment] += 1
    
    # Extract all topics
    all_topics = {}
    for article in articles:
        for topic in article.get("topics", []):
            if topic in all_topics:
                all_topics[topic] += 1
            else:
                all_topics[topic] = 1
    
    # Find common and unique topics
    common_topics = [topic for topic, count in all_topics.items() if count > 1]
    
    # Prepare comparison insights based on sentiments and topics
    comparisons = []
    
    # Compare positive vs negative coverage
    pos_count = sentiment_distribution["Positive"]
    neg_count = sentiment_distribution["Negative"]
    
    if pos_count > neg_count:
        comparisons.append({
            "Comparison": (f"Overall positive sentiment dominates with "
                           f"{pos_count} positive articles vs "
                           f"{neg_count} negative articles."),
            "Impact": "The company is currently receiving favorable media coverage."
        })
    elif neg_count > pos_count:
        comparisons.append({
            "Comparison": (f"Overall negative sentiment dominates with "
                           f"{neg_count} negative articles vs "
                           f"{pos_count} positive articles."),
            "Impact": "The company is currently facing challenging media coverage."
        })
    else:
        comparisons.append({
            "Comparison": "Media coverage shows balanced sentiment.",
            "Impact": "The company has mixed reception in current news."
        })
    
    # Compare topic focus
    if common_topics:
        top_topic = max(all_topics.items(), key=lambda x: x[1])[0]
        comparisons.append({
            "Comparison": f"The dominant topic in coverage is '{top_topic}'.",
            "Impact": (f"Media attention is currently focused on "
                       f"the company's {top_topic.lower()} aspects.")
        })
    
    # Create topic overlap analysis
    unique_topics_by_article = {}
    for i, article in enumerate(articles):
        article_topics = set(article.get("topics", []))
        other_articles_topics = set()
        for j, other_article in enumerate(articles):
            if i != j:
                other_articles_topics.update(other_article.get("topics", []))
        
        unique_to_this_article = article_topics - other_articles_topics
        if unique_to_this_article:
            unique_topics_by_article[article["title"]] = list(unique_to_this_article)
    
    # Return the analysis
    return {
        "sentiment_distribution": sentiment_distribution,
        "coverage_differences": comparisons,
        "topic_overlap": {
            "common_topics": common_topics,
            "unique_topics_by_article": unique_topics_by_article
        },
        "final_sentiment": get_final_sentiment(articles)
    }


def get_final_sentiment(articles: List[Dict]) -> str:
    """
    Generate a final sentiment summary based on all articles.
    
    Args:
        articles: List of processed articles
        
    Returns:
        Summary string describing overall sentiment
    """
    # Count sentiments
    sentiments = [article.get("sentiment") for article in articles]
    positive_count = sentiments.count("Positive")
    negative_count = sentiments.count("Negative")
    neutral_count = sentiments.count("Neutral")
    
    # Calculate average sentiment score
    avg_score = 0
    if articles:
        scores = [article.get("sentiment_score", 0) for article in articles]
        avg_score = sum(scores) / len(scores)
    
    # Generate summary
    article_count = len(articles)
    if avg_score >= 0.25:
        return (f"The company's news coverage is predominantly positive. "
                f"{positive_count} out of {article_count} articles "
                f"show positive sentiment.")
    elif avg_score <= -0.25:
        return (f"The company's news coverage is predominantly negative. "
                f"{negative_count} out of {article_count} articles "
                f"show negative sentiment.")
    else:
        return (f"The company's news coverage is mixed or neutral. "
                f"The sentiment is balanced across {article_count} articles.")


def generate_final_summary(
    company_name: str, articles: List[Dict], analysis: Dict
) -> str:
    """
    Generate a final summary for text-to-speech.
    
    Args:
        company_name: Name of the company
        articles: List of processed articles
        analysis: Comparative analysis results
        
    Returns:
        Summary text for TTS conversion
    """
    # Get overall sentiment
    final_sentiment = analysis.get("final_sentiment", "")
    
    # Get top topics
    common_topics = analysis.get("topic_overlap", {}).get("common_topics", [])
    topics_text = ", ".join(common_topics[:3]) if common_topics else "various topics"
    
    # Create summary
    summary = f"News summary for {company_name}. "
    summary += f"{final_sentiment} "
    summary += f"The company news mainly focuses on {topics_text}. "
    
    # Add brief info about top 3 articles
    summary += "Here are the main highlights: "
    
    for i, article in enumerate(articles[:3]):
        summary += f"Article {i+1}: {article['title']}. "
        summary += f"This article is {article['sentiment'].lower()} in tone. "
        summary += f"{article['summary'][:200]} "
    
    return summary