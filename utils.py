"""
Utility functions for extracting and analyzing company news.
"""
# Standard library imports
import os
import re
import uuid
import logging
import tempfile
import threading
import queue
import asyncio
from typing import List, Dict, Any, Tuple

# Third-party library imports
import requests
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from GoogleNews import GoogleNews
from newspaper import Article, Config
from googletrans import Translator
from gtts import gTTS

# Constants
MAX_CHUNK_SIZE = 1500
TRANSLATION_TIMEOUT = 60
REQUEST_TIMEOUT = 15
MIN_CONTENT_LENGTH = 500
MIN_ARTICLE_TEXT_LENGTH = 200
USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")


class TextToSpeechHindi:
    """Class for translating text to Hindi and generating speech."""
    
    def __init__(self):
        """Initialize translator and temp directory."""
        self.translator = Translator()
        self.temp_dir = tempfile.gettempdir()
        
    async def _translate_chunk(self, chunk: str) -> str:
        """Translate a single chunk of text asynchronously."""
        translation = await self.translator.translate(chunk, src='en', dest='hi')
        return translation.text
        
    def translate_to_hindi(self, text: str) -> str:
        """
        Translate text from English to Hindi.
        
        Args:
            text: English text to translate
            
        Returns:
            Hindi translation or empty string if translation fails
        """
        try:
            if not text:
                return ""
                
            # For short texts, try a simpler approach first
            if len(text) < 1000:
                try:
                    # Use alternative translation approach for shorter texts
                    from googletrans import Translator as SyncTranslator
                    sync_translator = SyncTranslator()
                    result = sync_translator.translate(text, src='en', dest='hi')
                    if hasattr(result, 'text') and result.text:
                        return result.text
                except Exception as e:
                    logger.warning(
                        f"Simple translation failed, falling back to async: {e}"
                    )
            
            # Continue with the chunked async approach for longer texts
            chunks = [
                text[i:i+MAX_CHUNK_SIZE] 
                for i in range(0, len(text), MAX_CHUNK_SIZE)
            ]
            
            # Use threading with explicit timeout
            result_queue = queue.Queue()
            translation_thread = threading.Thread(
                target=self._run_async_translation, 
                args=(chunks, result_queue)
            )
            translation_thread.daemon = True
            translation_thread.start()
            translation_thread.join(timeout=TRANSLATION_TIMEOUT)
            
            if not translation_thread.is_alive() and not result_queue.empty():
                translated_chunks = result_queue.get()
                if translated_chunks:
                    return " ".join(translated_chunks)
                
            logger.error("Translation failed or timed out")
            # Return empty instead of original to prevent English audio
            return ""
            
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return ""
        
    def _run_async_translation(self, chunks, result_queue):
        """Run async translation in a separate thread."""
        try:
            # Create async function for translations
            async def translate_all_chunks():
                tasks = [self._translate_chunk(chunk) for chunk in chunks]
                return await asyncio.gather(*tasks)
            
            # Create and run a new event loop in this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                translated = loop.run_until_complete(translate_all_chunks())
                result_queue.put(translated)
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Thread translation error: {e}")
            result_queue.put([])
        
    def generate_speech(self, text: str, output_file: str = None) -> str:
        """
        Generate speech from text in Hindi.
        
        Args:
            text: Hindi text to convert to speech
            output_file: Optional path for output file
            
        Returns:
            Path to generated audio file or None if failed
        """
        try:
            if not output_file:
                # Create a unique filename
                filename = f"tts_{uuid.uuid4()}.mp3"
                output_file = os.path.join(self.temp_dir, filename)
                
            # Generate speech
            tts = gTTS(text=text, lang='hi', slow=False)
            tts.save(output_file)
            
            return output_file
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return None
            
    def generate_summary_audio(self, summary_text: str) -> str:
        """
        Generate Hindi audio from English summary.
        
        Args:
            summary_text: English text to translate and convert
            
        Returns:
            Path to audio file or None if failed
        """
        try:
            # Translate summary to Hindi
            logger.info("Starting translation to Hindi")
            hindi_text = self.translate_to_hindi(summary_text)
            
            # Verify we have Hindi text (add a debug log)
            is_likely_hindi = any(
                ord(char) > 2304 and ord(char) < 2432 
                for char in hindi_text[:100]
            )
            logger.info(
                f"Translation complete, text appears to be Hindi: {is_likely_hindi}"
            )
            
            # Only generate speech if we have Hindi text
            if hindi_text and is_likely_hindi:
                return self.generate_speech(hindi_text)
            elif hindi_text:
                # If we have text but it's not Hindi (likely English fallback)
                logger.error("Translation failed to produce Hindi text")
                return None
            return None
        except Exception as e:
            logger.error(f"Error in generate_summary_audio: {e}")
            return None


class NewsExtractor:
    """Class for extracting news articles about companies."""
    
    def __init__(self):
        """Initialize news extraction components."""
        self.googlenews = GoogleNews(lang='en', period='7d')
        # Configure newspaper
        self.config = Config()
        self.config.browser_user_agent = USER_AGENT
        self.config.request_timeout = REQUEST_TIMEOUT
        
    def search_news(self, company_name: str, max_results: int = 30) -> List[Dict]:
        """
        Search for news articles related to the company using GoogleNews.
        
        Args:
            company_name: Name of the company to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of news article dictionaries
        """
        try:
            self.googlenews.clear()
            self.googlenews.search(company_name)
            self.googlenews.get_page(1)
            result = self.googlenews.result()
            
            # Try to get more results if needed
            if len(result) < 20:
                self.googlenews.get_page(2)
                result.extend(self.googlenews.result())
            
            logger.info(f"Found {len(result)} news articles for {company_name}")
            # Return more than needed to account for failed scrapes
            return result[:max_results]
        except Exception as e:
            logger.error(f"Error searching news for {company_name}: {e}")
            return []
    
    def _clean_url(self, url: str) -> str:
        """
        Clean Google News URLs by removing tracking parameters.
        
        Args:
            url: URL to clean
            
        Returns:
            Cleaned URL
        """
        if not url:
            return ""
        
        # Remove Google tracking parameters
        clean_url = url.split("&ved=")[0]
        
        # Some URLs still have other parameters we want to keep
        return clean_url
    
    def extract_article_content(self, url: str) -> Dict[str, Any]:
        """
        Extract article content using newspaper3k and BeautifulSoup.
        
        Args:
            url: URL of the article
            
        Returns:
            Dictionary with title, text, and other metadata or None if failed
        """
        try:
            # Clean the URL first
            url = self._clean_url(url)
            
            # First, check if the page is accessible
            response = requests.get(
                url,
                headers={'User-Agent': self.config.browser_user_agent},
                timeout=self.config.request_timeout
            )
            if response.status_code != 200:
                logger.warning(
                    f"Failed to access URL: {url}, "
                    f"Status code: {response.status_code}"
                )
                return None
            
            # Check if the page is likely to be JavaScript-heavy
            soup = BeautifulSoup(response.text, 'html.parser')
            main_text = soup.get_text()
            
            # If page seems empty or has very little text content
            if len(main_text.strip()) < MIN_CONTENT_LENGTH:
                logger.warning(
                    f"URL appears to be JavaScript-heavy or has minimal content: {url}"
                )
                return None
            
            # Parse with newspaper3k
            article = Article(url, config=self.config)
            article.download()
            article.parse()
            
            # Check if we got meaningful content before running NLP
            if not article.title or len(article.text) < MIN_ARTICLE_TEXT_LENGTH:
                logger.warning(f"Insufficient content extracted from {url}")
                return None
                
            # Run NLP for summary and keywords
            try:
                article.nlp()
            except Exception as nlp_err:
                logger.warning(f"Error running NLP on article from {url}: {nlp_err}")
                # Create a basic summary if NLP fails
                if article.text:
                    summary = article.text[:500] + "..."
                    keywords = []
                else:
                    return None
            else:
                summary = article.summary
                keywords = article.keywords
            
            # Extract the article data
            result = {
                "title": article.title,
                "text": article.text,
                "summary": summary,
                "keywords": keywords,
                "publish_date": article.publish_date,
                "url": url,
                "source": self._extract_source_from_url(url)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def _extract_source_from_url(self, url: str) -> str:
        """
        Extract the source name from URL.
        
        Args:
            url: URL to extract source from
            
        Returns:
            Source name or "Unknown" if extraction fails
        """
        try:
            # Extract domain
            domain = re.search(r'https?://(?:www\.)?([^/]+)', url).group(1)
            # Clean up domain to get source name
            parts = domain.split('.')
            if len(parts) > 1:
                return parts[-2].capitalize()
            return domain.capitalize()
        except Exception:
            return "Unknown"
            
    def get_articles_for_company(
        self, company_name: str, min_articles: int = 10
    ) -> List[Dict]:
        """
        Get articles for a company with complete content extraction.
        
        Args:
            company_name: Name of the company
            min_articles: Minimum number of articles to retrieve
            
        Returns:
            List of article dictionaries
        """
        # Search for more articles
        search_results = self.search_news(company_name, max_results=50)
        
        articles = []
        for result in search_results:
            if len(articles) >= min_articles:
                break
                
            url = result.get('link')
            if not url:
                continue
                
            article_content = self.extract_article_content(url)
            if article_content:
                # Add the Google News metadata
                article_content.update({
                    "media": result.get('media', ''),
                    "date": result.get('date', '')
                })
                articles.append(article_content)
        
        logger.info(
            f"Successfully extracted {len(articles)} articles for {company_name}"
        )
        # Return all successful articles, even if fewer than requested
        return articles


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
        """
        Analyze the sentiment of a given text.
        
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
        """
        Extract main topics from the article based on keywords and content.
        
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


def initialize_extractors():
    """Initialize and return extractors and analyzers."""
    news_extractor = NewsExtractor()
    sentiment_analyzer = SentimentAnalyzer()
    tts_engine = TextToSpeechHindi()
    return news_extractor, sentiment_analyzer, tts_engine


def get_company_news_with_sentiment(company_name: str) -> Dict[str, Any]:
    """
    Main function to get company news with sentiment analysis.
    
    Args:
        company_name: Name of the company to analyze
        
    Returns:
        Dictionary with company name, articles with sentiment, and analysis
    """
    news_extractor, sentiment_analyzer, tts_engine = initialize_extractors()
    
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


# Test the extraction - only runs when script is executed directly
if __name__ == "__main__":
    company = "Microsoft"
    result = get_company_news_with_sentiment(company)
    print(f"Retrieved {result['articles_count']} articles for {company}")
    for i, article in enumerate(result.get('articles', [])):
        print(f"\nArticle {i+1}: {article['title']}")
        print(f"Sentiment: {article['sentiment']}")
        print(f"Topics: {article['topics']}")