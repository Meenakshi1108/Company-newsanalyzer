"""News extraction module for fetching and processing company news articles."""

import re
import logging
import requests
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from GoogleNews import GoogleNews
from newspaper import Article, Config

# Constants
REQUEST_TIMEOUT = 15
MIN_CONTENT_LENGTH = 500
MIN_ARTICLE_TEXT_LENGTH = 200
USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
)

# Set up logging
logger = logging.getLogger(__name__)


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
        """Search for news articles related to the company using GoogleNews.
        
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
            
            logger.info(
                f"Found {len(result)} news articles for {company_name}"
            )
            # Return more than needed to account for failed scrapes
            return result[:max_results]
        except Exception as e:
            logger.error(f"Error searching news for {company_name}: {e}")
            return []
    
    def _clean_url(self, url: str) -> str:
        """Clean Google News URLs by removing tracking parameters.
        
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
        """Extract article content using newspaper3k and BeautifulSoup.
        
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
                    f"URL appears to be JavaScript-heavy or has minimal "
                    f"content: {url}"
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
                logger.warning(
                    f"Error running NLP on article from {url}: {nlp_err}"
                )
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
        """Extract the source name from URL.
        
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
        """Get articles for a company with complete content extraction.
        
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
            f"Successfully extracted {len(articles)} articles for "
            f"{company_name}"
        )
        # Return all successful articles, even if fewer than requested
        return articles