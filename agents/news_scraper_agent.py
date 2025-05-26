"""
NewsScraperAgent is responsible for scraping news articles from Business Insider website.
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup
from typing import Optional, List, Dict, Any, Tuple, Set
import time
import json
import hashlib
import os
import concurrent.futures
import functools
import lxml.html
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import re
import gc
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsScraperAgent:
    """Agent for scraping financial news articles from Business Insider."""
    
    def __init__(self, cache_enabled: bool = True, cache_expiry_hours: int = 48):
        """Initialize NewsScraperAgent with caching capabilities."""
        self.data_path = Path(__file__).parent.parent / "data" / "raw"
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration - more aggressive caching
        self.cache_enabled = cache_enabled
        self.cache_expiry_hours = cache_expiry_hours  # Extended to 48 hours
        self.cache_path = Path(__file__).parent.parent / "data" / "cache" / "news_scraper"
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Create optimized session with connection pooling
        self.session = self._create_optimized_session()
        
        # Ultra-aggressive optimizations
        self.max_empty_pages = 3  # Allow more empty pages for long periods
        self.max_pages = 100  # Increased for multi-year coverage
        self.min_coverage = 90  # Increased to 90% for complete coverage
        self.use_threading = True  # Enable parallel scraping
        self.max_threads = 8  # Increased number of concurrent requests
        self.chunk_size = 5  # Number of pages to process in each batch
        self.max_retries = 1  # Minimal retries
        self.batch_sleep = 0.1  # Minimal sleep between batches
        self.timeout = 2  # Even shorter timeout
        self.fill_missing_dates = False  # Disable date filling for speed
        self.last_coverage_percent = 0  # Track the last coverage percentage
        
        # Precompile regex patterns for faster matching
        self.ticker_pattern = None  # Will be set for each ticker
        self.date_patterns = {
            'days': re.compile(r'(\d+)d'),
            'months': re.compile(r'(\d+)mo'),
            'years': re.compile(r'(\d+)y')
        }
        
        # Cache for date parsing to avoid repetitive calculations
        self.date_parse_cache = {}
        
    def _create_optimized_session(self) -> requests.Session:
        """Create an optimized requests session with connection pooling and retries."""
        session = requests.Session()
        
        # Set headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive'  # Keep connections alive for pooling
        })
        
        # Configure retry strategy (minimal to keep it fast)
        retry_strategy = Retry(
            total=1,  # Only 1 retry
            backoff_factor=0.1,  # Minimal backoff
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        # Mount the adapter to the session
        adapter = HTTPAdapter(
            max_retries=retry_strategy, 
            pool_connections=10,  # Keep more connections in the pool
            pool_maxsize=10       # Increase max size of the pool
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
        
    def _get_industry_terms(self, ticker: str) -> List[str]:
        """
        Simplified: Return basic financial terms only - no complex industry mapping.
        """
        return ['stock', 'earnings', 'price']

    def _generate_cache_key(self, ticker: str, start_date: str, end_date: str) -> str:
        """
        Generate a unique cache key for the given parameters.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Unique cache key string
        """
        cache_string = f"{ticker}_{start_date}_{end_date}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _get_cache_filepath(self, cache_key: str) -> Path:
        """Get the filepath for a cache entry."""
        return self.cache_path / f"{cache_key}.json"

    def _is_cache_valid(self, cache_filepath: Path) -> bool:
        """
        Check if cache file exists and is not expired.
        
        Args:
            cache_filepath: Path to cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_filepath.exists():
            return False
            
        # Check if cache has expired
        file_modified_time = datetime.fromtimestamp(cache_filepath.stat().st_mtime)
        expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)
        
        if file_modified_time < expiry_time:
            logger.debug(f"Cache expired for {cache_filepath}")
            return False
            
        return True

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame, metadata: Optional[dict] = None) -> None:
        """
        Save data to cache.
        
        Args:
            cache_key: Unique cache key
            data: DataFrame to cache
            metadata: Optional metadata to store with the cache
        """
        if not self.cache_enabled:
            return
            
        cache_filepath = self._get_cache_filepath(cache_key)
        
        cache_data = {
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'data': data.to_dict('records') if not data.empty else []
        }
        
        try:
            with open(cache_filepath, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, default=str)
            logger.info(f"Saved {len(data)} articles to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {str(e)}")

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            DataFrame if cache hit, None if cache miss
        """
        if not self.cache_enabled:
            return None
            
        cache_filepath = self._get_cache_filepath(cache_key)
        
        if not self._is_cache_valid(cache_filepath):
            return None
            
        try:
            with open(cache_filepath, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if not cache_data['data']:
                # Return empty DataFrame with correct columns
                return pd.DataFrame(columns=['datetime', 'title', 'source'])
            
            df = pd.DataFrame(cache_data['data'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            logger.info(f"Cache hit! Loaded {len(df)} articles from cache: {cache_key}")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {str(e)}")
            return None

    def _clear_expired_cache(self) -> None:
        """Clear expired cache files."""
        if not self.cache_enabled:
            return
            
        try:
            expired_count = 0
            for cache_file in self.cache_path.glob("*.json"):
                if not self._is_cache_valid(cache_file):
                    cache_file.unlink()
                    expired_count += 1
            
            if expired_count > 0:
                logger.info(f"Cleared {expired_count} expired cache files")
                
        except Exception as e:
            logger.warning(f"Failed to clear expired cache: {str(e)}")

    def get_cache_info(self) -> dict:
        """
        Get information about the current cache state.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_enabled:
            return {"cache_enabled": False}
            
        try:
            cache_files = list(self.cache_path.glob("*.json"))
            valid_files = [f for f in cache_files if self._is_cache_valid(f)]
            
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "cache_enabled": True,
                "cache_path": str(self.cache_path),
                "total_files": len(cache_files),
                "valid_files": len(valid_files),
                "expired_files": len(cache_files) - len(valid_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_expiry_hours": self.cache_expiry_hours
            }
        except Exception as e:
            logger.error(f"Error getting cache info: {str(e)}")
            return {"cache_enabled": True, "error": str(e)}

    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """
        Clear cache files. If ticker is provided, only clear cache for that ticker.
        
        Args:
            ticker: Optional ticker to clear cache for specific ticker only
        """
        if not self.cache_enabled:
            logger.info("Cache is disabled")
            return
            
        try:
            if ticker:
                # Clear cache files that contain the ticker
                pattern = f"*{ticker.lower()}*"
                files_to_remove = list(self.cache_path.glob(pattern))
            else:
                # Clear all cache files
                files_to_remove = list(self.cache_path.glob("*.json"))
            
            for cache_file in files_to_remove:
                cache_file.unlink()
            
            logger.info(f"Cleared {len(files_to_remove)} cache files" + 
                       (f" for ticker {ticker}" if ticker else ""))
                       
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

    @functools.lru_cache(maxsize=128)
    def _parse_date(self, date_str: str) -> str:
        """
        Parse different date formats from Business Insider with memoization.
        Handles both full datetime strings and relative dates (e.g., '287d', '1y').
        
        Args:
            date_str: Date string from the website
            
        Returns:
            Datetime string in YYYY-MM-DD HH:MM:SS format
        """
        # Check the cache first for this date string
        if date_str in self.date_parse_cache:
            return self.date_parse_cache[date_str]
            
        try:
            result = None
            days_match = self.date_patterns['days'].search(date_str)
            months_match = self.date_patterns['months'].search(date_str)
            years_match = self.date_patterns['years'].search(date_str)
            
            if days_match:  # Relative date format (e.g., "287d")
                days_ago = int(days_match.group(1))
                date = datetime.now() - timedelta(days=days_ago)
                result = date.strftime('%Y-%m-%d %H:%M:%S')
            elif months_match:  # Month format (e.g., "3mo")
                months_ago = int(months_match.group(1))
                date = datetime.now() - timedelta(days=months_ago*30)
                result = date.strftime('%Y-%m-%d %H:%M:%S')
            elif years_match:  # Year format (e.g., "1y")
                years_ago = int(years_match.group(1))
                date = datetime.now() - timedelta(days=years_ago*365)
                result = date.strftime('%Y-%m-%d %H:%M:%S')
            else:  # Full datetime format
                result = datetime.strptime(date_str, '%m/%d/%Y %I:%M:%S %p').strftime('%Y-%m-%d %H:%M:%S')
                
            # Store in cache
            self.date_parse_cache[date_str] = result
            return result
            
        except Exception:
            # Return None but don't log for speed
            return None

    def _scrape_page(self, url: str, start_date: str, end_date: str) -> list:
        """
        Scrape a single page of news articles using direct lxml for maximum speed.
        
        Args:
            url: URL to scrape
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of article dictionaries
        """
        articles_data = []
        
        try:
            # Use the session for connection pooling
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                return articles_data
            
            # Use lxml for faster parsing (much faster than BeautifulSoup)
            html = lxml.html.fromstring(response.content)
            
            # Use XPath for direct selection (faster than CSS selectors)
            story_elements = html.xpath('//div[contains(@class, "latest-news")]//div[contains(@class, "latest-news__story")]')
            if not story_elements:
                return articles_data
                
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            for story in story_elements:  # Process ALL articles on the page, not just first 30
                try:
                    # Get date with XPath
                    date_elements = story.xpath('.//time[contains(@class, "latest-news__date")]')
                    if not date_elements:
                        continue
                        
                    date_element = date_elements[0]
                    datetime_str = date_element.get('datetime') or date_element.text_content().strip()
                    if not datetime_str:
                        continue
                        
                    # Fast path for already seen date strings
                    parsed_date = self._parse_date(datetime_str)
                    if not parsed_date:
                        continue
                        
                    article_date = datetime.strptime(parsed_date.split()[0], '%Y-%m-%d')
                    if start_date_dt <= article_date <= end_date_dt:
                        # Get title with XPath
                        title_elements = story.xpath('.//a[contains(@class, "news-link")]')
                        if not title_elements:
                            continue
                            
                        title = title_elements[0].text_content().strip()
                        if not title:
                            continue
                            
                        # Enhanced financial relevance filtering for decision-making
                        title_lower = title.lower()
                        
                        # Check if the article directly mentions the ticker (highest relevance)
                        ticker_mentioned = bool(self.ticker_pattern.search(title))
                        
                        # MAXIMUM COVERAGE FILTERING: 90%+ coverage with quality preservation
                        
                        # ENHANCED FINANCIAL KEYWORDS - Comprehensive coverage for investment decisions
                        
                        # Very broad relevance check - comprehensive financial/business content
                        business_financial_terms = [
                            # Core financial metrics & performance
                            'earnings', 'revenue', 'profit', 'stock', 'shares', 'market', 'trading', 'price',
                            'analyst', 'rating', 'target', 'estimate', 'forecast', 'guidance', 'outlook',
                            'quarter', 'fiscal', 'results', 'performance', 'financial', 'business',
                            'eps', 'ebitda', 'cash flow', 'margin', 'dividend', 'yield', 'payout',
                            
                            # Market sentiment & movement
                            'upgrade', 'downgrade', 'buy', 'sell', 'strong', 'weak', 'beat', 'missed',
                            'positive', 'negative', 'growth', 'decline', 'gains', 'losses', 'surge', 'fall',
                            'bullish', 'bearish', 'rally', 'correction', 'volatility', 'momentum',
                            'breakout', 'resistance', 'support', 'trend', 'reversal',
                            
                            # Business operations & strategy
                            'company', 'companies', 'industry', 'sector', 'investment', 'investor',
                            'merger', 'acquisition', 'deal', 'partnership', 'expansion', 'restructuring',
                            'ipo', 'spinoff', 'buyback', 'split', 'launch', 'product', 'service',
                            
                            # Technology & innovation (high relevance for modern stocks)
                            'technology', 'tech', 'ai', 'artificial intelligence', 'machine learning',
                            'chip', 'semiconductor', 'cloud', 'software', 'platform', 'digital',
                            'innovation', 'patent', 'research', 'development', 'automation',
                            
                            # Economic indicators & macro factors
                            'fed', 'federal reserve', 'interest rate', 'inflation', 'gdp', 'employment',
                            'economy', 'economic', 'policy', 'regulation', 'government', 'trade',
                            'tariff', 'supply chain', 'commodity', 'oil', 'energy', 'currency',
                            
                            # Financial institutions & markets
                            'bank', 'banking', 'credit', 'loan', 'mortgage', 'insurance', 'fund',
                            'etf', 'portfolio', 'allocation', 'diversification', 'risk', 'hedge',
                            'nasdaq', 'nyse', 's&p', 'dow', 'index', 'futures', 'options',
                            
                            # Healthcare & biotech (major sector)
                            'drug', 'pharmaceutical', 'biotech', 'clinical', 'trial', 'fda',
                            'approval', 'therapy', 'treatment', 'medical', 'healthcare',
                            
                            # Energy & materials
                            'renewable', 'solar', 'wind', 'battery', 'electric', 'mining',
                            'metal', 'steel', 'copper', 'lithium', 'sustainability', 'green',
                            
                            # Retail & consumer
                            'retail', 'consumer', 'sales', 'store', 'online', 'e-commerce',
                            'brand', 'marketing', 'advertising', 'subscription', 'customer'
                        ]
                        
                        # Only skip completely irrelevant content
                        irrelevant_terms = ['weather forecast', 'sports results', 'celebrity news']
                        
                        # Check relevance with enhanced scoring
                        has_business_relevance = any(term in title_lower for term in business_financial_terms)
                        is_irrelevant = any(term in title_lower for term in irrelevant_terms)
                        
                        # Skip only completely irrelevant content
                        if is_irrelevant:
                            continue
                            
                        # Keep if ticker mentioned OR has any business/financial relevance
                        # This should capture 90%+ of all business news
                        if not (ticker_mentioned or has_business_relevance):
                            continue
                        
                        # Enhanced relevance scoring for better prioritization
                        if ticker_mentioned:
                            # Check if company name is also mentioned for highest relevance
                            company_name = self._get_company_name(ticker) 
                            if company_name and company_name.lower() in title_lower:
                                relevance_score = 15  # Highest: both ticker and company name
                            else:
                                relevance_score = 10  # High: ticker mentioned
                        elif has_business_relevance:
                            # Score based on number of financial terms found
                            term_count = sum(1 for term in business_financial_terms if term in title_lower)
                            if term_count >= 3:
                                relevance_score = 8   # High financial relevance
                            elif term_count >= 2:
                                relevance_score = 6   # Medium-high financial relevance
                            else:
                                relevance_score = 4   # Medium financial relevance
                        else:
                            relevance_score = 2   # Low: minimal relevance
                        
                        # Get source with XPath
                        source_elements = story.xpath('.//span[contains(@class, "latest-news__source")]')
                        source = source_elements[0].text_content().strip() if source_elements else "Business Insider"
                        
                        # Simple duplicate detection - just normalize title
                        normalized_title = ' '.join(title_lower.split())[:50]  # First 50 chars only
                        
                        articles_data.append({
                            'datetime': parsed_date,
                            'title': title,
                            'source': source,
                            'normalized_title': normalized_title,  # For duplicate detection
                            'relevance_score': relevance_score     # For article prioritization
                        })
                        
                except Exception:
                    # Skip errors silently for speed
                    continue
                    
            return articles_data
            
        except Exception:
            # Skip detailed logging for speed
            return articles_data

    def _scrape_additional_sources(self, ticker: str, missing_dates: List[datetime.date]) -> List[Dict[str, Any]]:
        """
        Simplified: Skip additional source scraping to improve performance.
        This was a major bottleneck with complex parsing and multiple HTTP requests.
        """
        return []  # Return empty list to skip this entirely
        
    def _get_company_name(self, ticker: str) -> Optional[str]:
        """
        Get the company name for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Company name or None
        """
        # Enhanced company names for popular tickers - covers major indices
        company_names = {
            # FAANG/Tech Giants
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'GOOG': 'Google',
            'AMZN': 'Amazon',
            'META': 'Meta',
            'TSLA': 'Tesla',
            'NVDA': 'NVIDIA',
            'AMD': 'AMD',
            'INTC': 'Intel',
            'NFLX': 'Netflix',
            'CRM': 'Salesforce',
            'ORCL': 'Oracle',
            'ADBE': 'Adobe',
            'PYPL': 'PayPal',
            'UBER': 'Uber',
            'LYFT': 'Lyft',
            'SNAP': 'Snapchat',
            'TWTR': 'Twitter',
            'SQ': 'Block',
            
            # Finance/Banking
            'JPM': 'JPMorgan',
            'BAC': 'Bank of America',
            'WFC': 'Wells Fargo',
            'GS': 'Goldman Sachs',
            'MS': 'Morgan Stanley',
            'C': 'Citigroup',
            'USB': 'U.S. Bank',
            'PNC': 'PNC Bank',
            'TFC': 'Truist',
            'COF': 'Capital One',
            'AXP': 'American Express',
            'BLK': 'BlackRock',
            'SCHW': 'Charles Schwab',
            
            # Healthcare/Pharma
            'JNJ': 'Johnson & Johnson',
            'PFE': 'Pfizer',
            'UNH': 'UnitedHealth',
            'ABT': 'Abbott',
            'TMO': 'Thermo Fisher',
            'DHR': 'Danaher',
            'BMY': 'Bristol Myers',
            'AMGN': 'Amgen',
            'GILD': 'Gilead',
            'BIIB': 'Biogen',
            'REGN': 'Regeneron',
            'VRTX': 'Vertex',
            'MRNA': 'Moderna',
            
            # Consumer/Retail
            'WMT': 'Walmart',
            'TGT': 'Target',
            'COST': 'Costco',
            'HD': 'Home Depot',
            'LOW': 'Lowe\'s',
            'NKE': 'Nike',
            'SBUX': 'Starbucks',
            'MCD': 'McDonald\'s',
            'DIS': 'Disney',
            'KO': 'Coca-Cola',
            'PEP': 'PepsiCo',
            'PG': 'Procter & Gamble',
            
            # Energy
            'XOM': 'Exxon',
            'CVX': 'Chevron',
            'COP': 'ConocoPhillips',
            'EOG': 'EOG Resources',
            'SLB': 'Schlumberger',
            'OXY': 'Occidental',
            
            # Industrial/Manufacturing
            'BA': 'Boeing',
            'CAT': 'Caterpillar',
            'GE': 'General Electric',
            'MMM': '3M',
            'HON': 'Honeywell',
            'LMT': 'Lockheed Martin',
            'RTX': 'Raytheon',
            'UPS': 'UPS',
            'FDX': 'FedEx',
            
            # Automotive
            'F': 'Ford',
            'GM': 'General Motors',
            'RIVN': 'Rivian',
            'LCID': 'Lucid Motors',
            
            # Communications
            'VZ': 'Verizon',
            'T': 'AT&T',
            'TMUS': 'T-Mobile',
            'CMCSA': 'Comcast',
            
            # Semiconductors
            'TSM': 'Taiwan Semiconductor',
            'ASML': 'ASML',
            'QCOM': 'Qualcomm',
            'AVGO': 'Broadcom',
            'TXN': 'Texas Instruments',
            'AMAT': 'Applied Materials',
            'LRCX': 'Lam Research',
            'KLAC': 'KLA Corporation',
            'MRVL': 'Marvell',
            
            # Real Estate
            'AMT': 'American Tower',
            'PLD': 'Prologis',
            'CCI': 'Crown Castle',
            'EQIX': 'Equinix',
            'SPG': 'Simon Property',
            
            # Utilities
            'NEE': 'NextEra Energy',
            'DUK': 'Duke Energy',
            'SO': 'Southern Company',
            'D': 'Dominion Energy'
        }
        
        return company_names.get(ticker.upper())
        
    def set_performance_profile(self, profile: str = "balanced") -> None:
        """
        Set the performance profile for the scraper.
        
        Args:
            profile: One of "speed", "balanced", "thorough", or "complete"
        """
        if profile == "speed":
            # Ultra-fast but less thorough
            self.max_empty_pages = 1
            self.max_pages = 10
            self.min_coverage = 20
            self.max_threads = 10
            self.chunk_size = 5
            self.timeout = 1.5
            self.batch_sleep = 0.05
            self.fill_missing_dates = False
            logger.info("Set performance profile to 'speed'")
        elif profile == "thorough":
            # More thorough but slower
            self.max_empty_pages = 3
            self.max_pages = 30
            self.min_coverage = 70
            self.max_threads = 4
            self.chunk_size = 4
            self.timeout = 3
            self.batch_sleep = 0.2
            self.fill_missing_dates = True
            logger.info("Set performance profile to 'thorough'")
        elif profile == "complete":
            # Maximum coverage (90%+) with reasonable performance
            self.max_empty_pages = 5
            self.max_pages = 150  # Increased for multi-year coverage
            self.min_coverage = 90
            self.max_threads = 8
            self.chunk_size = 5
            self.timeout = 3
            self.batch_sleep = 0.1
            self.fill_missing_dates = True
            logger.info("Set performance profile to 'complete'")
        else:  # balanced
            # Default balanced profile
            self.max_empty_pages = 2
            self.max_pages = 15
            self.min_coverage = 40
            self.max_threads = 6
            self.chunk_size = 5
            self.timeout = 2
            self.batch_sleep = 0.1
            self.fill_missing_dates = False
            logger.info("Set performance profile to 'balanced'")
            
    def estimate_performance(self, ticker: str = "AAPL", days: int = 30) -> Dict[str, Any]:
        """
        Estimate the scraping performance for a ticker.
        
        Args:
            ticker: Stock ticker to test with
            days: Number of days to include in the test
            
        Returns:
            Dictionary with performance metrics
        """
        # Clear cache for this test
        self.clear_cache(ticker)
        
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Time the operation
        start_time = time.time()
        df = self.fetch_news(ticker, start_date, end_date)
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        articles_count = len(df)
        articles_per_second = articles_count / elapsed_time if elapsed_time > 0 else 0
        days_covered = len(df['datetime'].dt.date.unique()) if not df.empty else 0
        coverage_percent = (days_covered / days) * 100 if days > 0 else 0
        
        # Return performance metrics
        return {
            "ticker": ticker,
            "days_requested": days,
            "days_covered": days_covered,
            "coverage_percent": round(coverage_percent, 2),
            "articles_found": articles_count,
            "elapsed_seconds": round(elapsed_time, 2),
            "articles_per_second": round(articles_per_second, 2),
            "settings": {
                "max_threads": self.max_threads,
                "max_pages": self.max_pages,
                "min_coverage": self.min_coverage
            }
        }

    def fetch_news(self, ticker: str, start_date: str, end_date: Optional[str] = None, max_articles: int = 5000) -> pd.DataFrame:
        """
        Fetch financial news articles for a given stock ticker from Business Insider.
        Uses caching, connection pooling, parallel processing and optimized parsing for maximum speed.
        
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

            # Check cache first - with longer expiry time for better performance
            cache_key = self._generate_cache_key(ticker, start_date, end_date)
            cached_data = self._load_from_cache(cache_key)
            
            if cached_data is not None:
                logger.info(f"Returning cached data for {ticker} from {start_date} to {end_date}")
                return cached_data

            logger.info(f"Fetching Business Insider news for {ticker} from {start_date} to {end_date}")
            
            # Compile the ticker pattern for relevance filtering
            # Include company name if available for better matching
            company_name = self._get_company_name(ticker)
            if company_name:
                # Create pattern that matches ticker OR company name
                pattern_parts = [ticker, ticker.lower(), company_name, company_name.lower()]
                pattern = '|'.join(re.escape(part) for part in pattern_parts)
                self.ticker_pattern = re.compile(pattern, re.IGNORECASE)
            else:
                self.ticker_pattern = re.compile(f"{ticker}|{ticker.lower()}", re.IGNORECASE)
            
            # Base URLs - fetch from both the stock page and the company page for better coverage
            base_urls = [
                f'https://markets.businessinsider.com/news/{ticker.lower()}-stock',
                f'https://markets.businessinsider.com/news/{ticker.lower()}',
                f'https://markets.businessinsider.com/stocks/{ticker.lower()}-stock',  # Additional URL format
                f'https://markets.businessinsider.com/stocks/{ticker.lower()}'         # Additional URL format
            ]
            
            all_articles = []
            dates_found = set()
            
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
            target_days = (end_date_dt - start_date_dt).days + 1
            
            # Process both URLs in parallel batches for maximum speed
            for base_url in base_urls:
                # Early termination if we already have good coverage
                if len(dates_found) / target_days * 100 >= self.min_coverage:
                    logger.info(f"Already achieved sufficient coverage ({len(dates_found)} dates)")
                    break
                
                # Prepare URLs for dynamic batch processing
                all_pages = list(range(1, self.max_pages + 1))
                
                if self.use_threading:
                    # Process in batches with ThreadPoolExecutor
                    for batch_start in range(0, len(all_pages), self.chunk_size):
                        batch_pages = all_pages[batch_start:batch_start+self.chunk_size]
                        urls_to_scrape = [f"{base_url}?p={page}" for page in batch_pages]
                        
                        articles_batch = []
                        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                            future_to_url = {
                                executor.submit(self._scrape_page, url, start_date, end_date): url 
                                for url in urls_to_scrape
                            }
                            
                            for future in concurrent.futures.as_completed(future_to_url):
                                try:
                                    articles = future.result()
                                    if articles:
                                        articles_batch.extend(articles)
                                except Exception:
                                    # Skip errors for speed
                                    pass
                        
                        # Process the batch results
                        if not articles_batch:
                            # No articles in this batch, likely reached the end
                            break
                            
                        # Add ticker and update dates found
                        for article in articles_batch:
                            article['ticker'] = ticker
                            article_date = article['datetime'].split()[0]
                            dates_found.add(article_date)
                        
                        all_articles.extend(articles_batch)
                        
                        # Check if we have enough coverage to stop early
                        coverage = len(dates_found) / target_days * 100
                        if coverage >= self.min_coverage:
                            logger.info(f"Achieved {coverage:.1f}% date coverage, stopping early")
                            break
                            
                        # Minimal sleep between batches
                        time.sleep(self.batch_sleep)
                        
                        # Force garbage collection to free memory
                        if len(all_articles) > 1000:
                            gc.collect()
                else:
                    # Sequential fallback (should rarely be used)
                    page = 1
                    consecutive_empty_pages = 0
                    
                    while consecutive_empty_pages < self.max_empty_pages and page <= self.max_pages:
                        url = f"{base_url}?p={page}"
                        
                        articles = self._scrape_page(url, start_date, end_date)
                        
                        if not articles:
                            consecutive_empty_pages += 1
                        else:
                            consecutive_empty_pages = 0
                            
                            for article in articles:
                                article['ticker'] = ticker
                                article_date = article['datetime'].split()[0]
                                dates_found.add(article_date)
                            
                            all_articles.extend(articles)
                            
                            coverage = len(dates_found) / target_days * 100
                            if coverage >= self.min_coverage:
                                logger.info(f"Achieved {coverage:.1f}% date coverage")
                                break
                        
                        page += 1
                        time.sleep(self.batch_sleep)

            # Create DataFrame - optimized for speed
            if all_articles:
                # Create DataFrame efficiently from records
                df = pd.DataFrame.from_records(all_articles)
                
                # Convert datetime efficiently
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['date'] = df['datetime'].dt.date
                
                # Remove duplicates based on normalized titles (faster duplicate detection)
                if 'normalized_title' in df.columns:
                    initial_count = len(df)
                    df = df.drop_duplicates(subset=['normalized_title'], keep='first')
                    df = df.drop(columns=['normalized_title'])  # Remove helper column
                    removed_count = initial_count - len(df)
                    if removed_count > 0:
                        logger.info(f"Removed {removed_count} duplicate articles")
                
                # Check if we need to fill missing dates for better coverage
                found_dates = set(df['date'].unique())
                all_dates = set([start_date_dt.date() + timedelta(days=x) for x in range(target_days)])
                missing_dates = all_dates - found_dates
                
                # Skip complex date filling - just work with what we have for speed
                coverage_percent = len(found_dates) / target_days * 100
                logger.info(f"Coverage: {coverage_percent:.1f}% ({len(found_dates)} dates found)")
                
                # Group by date and select most relevant article for each day - ENHANCED
                dates = df['date'].unique()
                daily_articles = []
                
                # Process each date with enhanced relevance scoring
                for date in dates:
                    daily_group = df[df['date'] == date]
                    
                    # Enhanced scoring: prioritize by relevance_score, then ticker mention
                    if 'relevance_score' in daily_group.columns and len(daily_group) > 1:
                        # Sort by relevance score (highest first), then by datetime (most recent first)
                        daily_group = daily_group.sort_values(['relevance_score', 'datetime'], ascending=[False, False])
                    elif 'title' in daily_group.columns and len(daily_group) > 1:
                        # Fallback: prefer articles that mention the ticker directly
                        ticker_mentioned = daily_group['title'].str.contains(ticker, case=False, na=False)
                        if ticker_mentioned.any():
                            daily_group = daily_group[ticker_mentioned]
                    
                    # Take the most relevant article for this date
                    daily_articles.append(daily_group.iloc[0])
                
                # Create final dataframe
                columns = ['datetime', 'title', 'source']
                df_daily = pd.DataFrame(daily_articles)
                
                # Select and sort columns
                if not df_daily.empty:
                    df_daily = df_daily.sort_values('datetime', ascending=False)
                    for col in columns:
                        if col not in df_daily.columns:
                            df_daily[col] = None
                    df_daily = df_daily[columns]
                
                    # Save to CSV
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{ticker}_daily_news_{timestamp}.csv"
                    filepath = self.data_path / filename
                    df_daily.to_csv(filepath, index=False, encoding='utf-8')
                    logger.info(f"Saved {len(df_daily)} daily articles to {filepath}")
                    
                    # Save to cache
                    metadata = {
                        'ticker': ticker,
                        'start_date': start_date,
                        'end_date': end_date,
                        'fetched_articles': len(all_articles),
                        'unique_dates': len(dates_found),
                        'coverage_percentage': len(dates_found) / target_days * 100
                    }
                    self._save_to_cache(cache_key, df_daily, metadata)
                    
                    # Clean up memory
                    del all_articles
                    gc.collect()
                    
                    return df_daily
            
            # No articles found
            logger.warning("No articles found within the specified date range")
            return pd.DataFrame(columns=['datetime', 'title', 'source'])
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            # Return empty DataFrame instead of raising exception for robustness
            return pd.DataFrame(columns=['datetime', 'title', 'source'])

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