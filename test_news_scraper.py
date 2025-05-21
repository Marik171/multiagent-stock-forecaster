"""
Test script for NewsScraperAgent functionality with Business Insider
"""
import os
from datetime import datetime, timedelta
from agents.news_scraper_agent import NewsScraperAgent

def test_news_scraper_agent():
    """Test the main functionality of NewsScraperAgent with Business Insider."""
    try:
        print("\n=== Testing Business Insider News Scraper ===")
        agent = NewsScraperAgent()
        
        # Test parameters - use last 2 days for faster testing
        ticker = "NVDA"  # NVIDIA tends to have frequent news coverage
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=1825)).strftime('%Y-%m-%d') # 5 years of data
        max_articles = 10000  # Increased for more comprehensive testing

        print(f"\nConfiguration:")
        print(f"Ticker: {ticker}")
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Max Articles: {max_articles}")
        
        df = agent.fetch_news(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            max_articles=max_articles
        )
        
        print("\n=== Results ===")
        if df is None or df.empty:
            print("❌ No articles found")
            return False
            
        # Verify required columns exist
        required_columns = ['datetime', 'title', 'source']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {', '.join(missing_columns)}")
            return False
            
        print(f"✓ Found {len(df)} articles")
        print(f"✓ All required columns present: {', '.join(df.columns)}")
        
        # Display sample articles
        if not df.empty:
            print("\nLatest articles:")
            print("-" * 80)
            for _, row in df.head(3).iterrows():
                print(f"Date: {row['datetime']}")
                print(f"Title: {row['title']}")
                print(f"Source: {row['source']}")
                print("-" * 80)
        
        # Additional checks
        sources = df['source'].unique()
        print(f"\nSources found: {', '.join(sources)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_news_scraper_agent()
    print(f"\nTest {'✓ succeeded' if success else '❌ failed'}")
