"""
NewsScraperAgent is responsible for scraping news articles from Business Insider website.
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsScraperAgent:
    """Agent for scraping financial news articles from Business Insider."""
    
    def __init__(self):
        """Initialize NewsScraperAgent."""
        self.data_path = Path(__file__).parent.parent / "data" / "raw"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self.max_empty_pages = 20  # Increased for better historical coverage
        self.max_pages = 200  # Increased max pages to check
        self.min_coverage = 90  # Reduced minimum coverage requirement

    def _parse_date(self, date_str: str) -> str:
        """
        Parse different date formats from Business Insider.
        Handles both full datetime strings and relative dates (e.g., '287d', '1y').
        
        Args:
            date_str: Date string from the website
            
        Returns:
            Datetime string in YYYY-MM-DD HH:MM:SS format
        """
        try:
            if 'd' in date_str:  # Relative date format (e.g., "287d")
                days_ago = int(date_str.replace('d', ''))
                date = datetime.now() - timedelta(days=days_ago)
                return date.strftime('%Y-%m-%d %H:%M:%S')
            elif 'y' in date_str:  # Year format (e.g., "1y")
                years_ago = int(date_str.replace('y', ''))
                date = datetime.now() - timedelta(days=years_ago*365)
                return date.strftime('%Y-%m-%d %H:%M:%S')
            elif 'mo' in date_str:  # Month format (e.g., "3mo")
                months_ago = int(date_str.replace('mo', ''))
                date = datetime.now() - timedelta(days=months_ago*30)
                return date.strftime('%Y-%m-%d %H:%M:%S')
            else:  # Full datetime format
                return datetime.strptime(date_str, '%m/%d/%Y %I:%M:%S %p').strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logger.debug(f"Error parsing date {date_str}: {str(e)}")
            return None

    def _scrape_page(self, url: str, start_date: str, end_date: str) -> list:
        """
        Scrape a single page of news articles.
        
        Args:
            url: URL to scrape
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of article dictionaries
        """
        articles_data = []
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            logger.debug(f"Response status code: {response.status_code}")
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            latest_news_div = soup.find('div', class_='latest-news')
            
            if not latest_news_div:
                logger.warning(f"Could not find latest-news container on {url}")
                return articles_data
            
            articles = latest_news_div.find_all('div', class_='latest-news__story')
            logger.debug(f"Found {len(articles)} articles on the page")
            
            for article in articles:
                try:
                    meta_div = article.find('div', class_='latest-news__meta')
                    if not meta_div:
                        continue
                        
                    date_element = meta_div.find('time', class_='latest-news__date')
                    if not date_element:
                        continue
                        
                    datetime_str = date_element.get('datetime')
                    if not datetime_str:
                        datetime_str = date_element.text.strip()
                    
                    parsed_date = self._parse_date(datetime_str)
                    if not parsed_date:
                        continue
                        
                    article_date = datetime.strptime(parsed_date.split()[0], '%Y-%m-%d')
                    if start_date <= article_date.strftime('%Y-%m-%d') <= end_date:
                        news_link = article.find('a', class_='news-link')
                        title = news_link.get_text(strip=True)
                        source = meta_div.find('span', class_='latest-news__source').get_text(strip=True)
                        
                        logger.debug(f"Found article: {title} from {parsed_date}")
                        
                        articles_data.append({
                            'datetime': parsed_date,
                            'title': title,
                            'source': source
                        })
                        
                except Exception as e:
                    logger.debug(f"Error processing article: {str(e)}")
                    continue
                    
            return articles_data
            
        except Exception as e:
            logger.error(f"Error scraping page {url}: {str(e)}")
            return articles_data

    def _select_daily_article(self, daily_articles: pd.DataFrame) -> pd.Series:
        """
        Select the most relevant article for a given day based on certain criteria.
        
        Current criteria:
        1. Prefer articles that mention NVDA in the title
        2. Prefer articles from Seeking Alpha (typically more detailed analysis)
        3. If multiple articles meet the same criteria, take the latest one
        
        Args:
            daily_articles: DataFrame containing articles for a single day
            
        Returns:
            Series containing the selected article
        """
        daily_articles['datetime'] = pd.to_datetime(daily_articles['datetime'])
        daily_articles['relevance_score'] = 0
        
        # Check for NVDA mention in title (highest priority)
        daily_articles.loc[daily_articles['title'].str.contains('NVDA|Nvidia', case=False), 'relevance_score'] += 3
        
        # Prefer Seeking Alpha articles (medium priority)
        daily_articles.loc[daily_articles['source'] == 'Seeking Alpha', 'relevance_score'] += 2
        
        # Sort by relevance score and datetime
        daily_articles = daily_articles.sort_values(['relevance_score', 'datetime'], ascending=[False, False])
        
        return daily_articles.iloc[0]

    def fetch_news(self, ticker: str, start_date: str, end_date: Optional[str] = None, max_articles: int = 5000) -> pd.DataFrame:
        """
        Fetch financial news articles for a given stock ticker from Business Insider.
        
        Args:
            ticker: Stock ticker symbol (e.g., NVDA)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
            max_articles: Maximum number of articles to fetch (default 5000)
            
        Returns:
            DataFrame with news articles, one per day
        """
        try:
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')

            logger.info(f"Fetching Business Insider news for {ticker} from {start_date} to {end_date}")
            
            base_url = f'https://markets.businessinsider.com/news/{ticker.lower()}-stock'
            page = 1
            all_articles = []
            consecutive_empty_pages = 0
            dates_found = set()
            reached_start_date = False
            
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            target_days = (end_date_dt - start_date_dt).days + 1

            while len(all_articles) < max_articles and consecutive_empty_pages < self.max_empty_pages:
                url = f"{base_url}?p={page}"
                logger.debug(f"Scraping page {page}: {url}")
                
                articles = self._scrape_page(url, start_date, end_date)
                
                if not articles:
                    consecutive_empty_pages += 1
                    if page > self.max_pages:
                        logger.info(f"Reached maximum page limit ({self.max_pages})")
                        break
                else:
                    consecutive_empty_pages = 0
                    
                    oldest_date = None
                    for article in articles:
                        article['ticker'] = ticker
                        article_date = article['datetime'].split()[0]
                        article_dt = datetime.strptime(article_date, '%Y-%m-%d')
                        
                        if start_date_dt <= article_dt <= end_date_dt:
                            dates_found.add(article_date)
                            if not oldest_date or article_date < oldest_date:
                                oldest_date = article_date
                            if article_dt <= start_date_dt:
                                reached_start_date = True
                    
                    all_articles.extend(articles)
                    logger.debug(f"Total articles collected: {len(all_articles)}, Unique dates: {len(dates_found)}")
                    
                    coverage = len(dates_found) / min(target_days, 365) * 100
                    if coverage >= self.min_coverage and reached_start_date:
                        logger.info(f"Achieved {coverage:.1f}% date coverage")
                        break
                
                page += 1
                delay = min(2 + (page // 10), 5)
                time.sleep(delay)

            # Create DataFrame
            columns = ['datetime', 'title', 'source']
            df = pd.DataFrame(all_articles, columns=columns)
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['date'] = df['datetime'].dt.date
                
                # Group by date and select most relevant article for each day
                daily_articles = []
                for date, group in df.groupby('date'):
                    daily_articles.append(self._select_daily_article(group))
                
                df_daily = pd.DataFrame(daily_articles)
                df_daily = df_daily.sort_values('datetime', ascending=False)
                df_daily = df_daily[columns]
                
                # Save to CSV
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{ticker}_daily_news_{timestamp}.csv"
                filepath = self.data_path / filename
                df_daily.to_csv(filepath, index=False, encoding='utf-8')
                logger.info(f"Saved {len(df_daily)} daily articles to {filepath}")
                
                return df_daily
            else:
                logger.warning("No articles found within the specified date range")
                return pd.DataFrame(columns=columns)
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            raise

if __name__ == "__main__":
    agent = NewsScraperAgent()
    
    # Interactive mode
    print("\n=== Business Insider News Scraper ===")
    print("Type a ticker symbol to fetch news or 'quit' to exit\n")
    while True:
        ticker = input("Enter ticker symbol (or 'quit' to exit): ").strip().upper()
        if ticker == 'QUIT':
            break
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD, leave blank for today): ").strip() or datetime.now().strftime('%Y-%m-%d')
        
        try:
            df = agent.fetch_news(ticker, start_date, end_date)
            if not df.empty:
                print(f"Fetched {len(df)} articles for {ticker}.")
                print(df.head())
            else:
                print("No articles found.")
        except Exception as e:
            print(f"Error: {str(e)}")