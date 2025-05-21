"""Test the SentimentAnalyzerAgent functionality."""
import logging
from pathlib import Path
import pandas as pd
from agents.sentiment_agent import SentimentAnalyzerAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sentiment_agent():
    """Test the main functionality of SentimentAnalyzerAgent."""
    try:
        # Initialize agent
        agent = SentimentAnalyzerAgent()
        
        # Process sample data
        df = agent.run()
        
        # Validate output
        required_columns = [
            'Date', 'news_text',
            'news_sentiment_score', 'neg_prob_news', 'pos_prob_news'
        ]
                          
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ Missing columns: {missing_cols}")
            return False
            
        # Check sentiment scores are in valid range (-1 to 1)
        score_columns = ['news_sentiment_score']
        for col in score_columns:
            if not df[col].between(-1, 1).all():
                logger.error(f"❌ Invalid sentiment scores in {col}")
                return False
        logger.info("✓ Sentiment scores are within valid range (-1 to 1)")
        
        # Check probability columns are in valid range (0 to 1)
        prob_columns = ['neg_prob_news', 'pos_prob_news']
        for col in prob_columns:
            if not df[col].between(0, 1).all():
                logger.error(f"❌ Invalid probabilities in {col}")
                return False
        logger.info("✓ Probability values are within valid range (0 to 1)")
        
        # Verify sentiment scores are consistent with probability differences
        eps = 1e-6  # Small epsilon for floating point comparison
        news_score_diff = abs(df['news_sentiment_score'] - (df['pos_prob_news'] - df['neg_prob_news']))
        if not (news_score_diff < eps).all():
            logger.error("❌ News sentiment scores don't match probability differences")
            return False
        logger.info("✓ News sentiment scores correctly calculated from probabilities")
        
        # For YouTube, we're using a weighted combination, so just check range and correlation
        yt_scores = df['youtube_sentiment_score']
        yt_prob_diff = df['pos_prob_yt'] - df['neg_prob_yt']
        if not yt_scores.between(-1, 1).all():
            logger.error("❌ YouTube sentiment scores outside valid range")
            return False
        if not yt_prob_diff.between(-1, 1).all():
            logger.error("❌ YouTube probability differences outside valid range")
            return False
            
        # Check correlation between scores and probability differences
        correlation = yt_scores.corr(yt_prob_diff)
        if correlation < 0.5:  # Strong positive correlation expected
            logger.error(f"❌ Weak correlation between YouTube scores and probabilities: {correlation:.2f}")
            return False
        logger.info(f"✓ YouTube sentiment analysis is consistent (correlation: {correlation:.2f})")
        
        # Verify output file exists
        output_path = Path(__file__).parent / "data" / "sentiment_file" / "NVDA_with_sentiment_scored.csv"
        if not output_path.exists():
            logger.error("❌ Output file not created")
            return False
        logger.info("✓ Output file created successfully")
            
        # Print summary statistics
        logger.info("\nProcessed data shape: {df.shape}")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        logger.info("\nSentiment Score Statistics:")
        logger.info("\n" + str(df[score_columns].describe()))
        
        logger.info("\nProbability Statistics:")
        logger.info("\n" + str(df[prob_columns].describe()))
        
        # Sample some results
        logger.info("\nSample Results (first 3 rows):")
        sample_cols = ['Date', 'news_sentiment_score', 'neg_prob_news', 'pos_prob_news', 
                      'youtube_sentiment_score', 'neg_prob_yt', 'pos_prob_yt', 'avg_sentiment_score']
        logger.info("\n" + str(df[sample_cols].head(3)))
        
        logger.info("\n✓ SentimentAnalyzerAgent test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    test_sentiment_agent()