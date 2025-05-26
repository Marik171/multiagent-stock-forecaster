# 📈 Multi-Agent Stock Prediction Framework

---

## 🚀 Project Description

This project is a **Multi-Agent Framework** that predicts stock prices by combining **technical indicators**, **sentiment analysis** from **news articles** and **YouTube comments**, and a **deep learning model** for forecasting.

It features:

- Intelligent agents for data fetching, scraping, preprocessing, prediction, and reasoning.
- Emotion analysis (fear, greed, anxiety, FOMO, etc.) using open-source foundation models.
- Future stock price forecasting using an LSTM deep learning model.
- Streamlit user interface for easy interaction.
- Downloadable PDF reports summarizing insights.
- LLM-powered investment recommendation system.

---

## 🧩 Project Structure

```bash
.
├── agents/
│   ├── stock_fetch_agent.py
│   ├── news_scraper_agent.py
│   ├── youtube_scraper_agent.py
│   ├── preprocessing_agent.py
│   ├── sentiment_agent.py
│   ├── prediction_agent.py
│   └── reasoning_agent.py
├── app/
│   ├── main.py
│   └── llm_recommendation.py
├── data/
│   ├── raw/
│   └── processed/
├── models_saved/
├── reports/
├── utils/
│   └── feature_engineering.py
├── .env
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock-prediction-agents.git
cd stock-prediction-agents
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

Create a `.env` file in the root directory and add your keys:

```env
NEWS_API_KEY=your_newsapi_key
YOUTUBE_API_KEY=your_youtube_api_key
OPENAI_API_KEY=your_openai_or_huggingface_api_key_if_used
```

*(You can get these for free from NewsAPI, YouTube Data API, and Huggingface if needed.)*

### 5. Run the Application

```bash
streamlit run app/main.py
```

---

## 🧠 How It Works

1. **User Input:**  
   Provide the stock ticker, start/end dates, and prediction timeline (e.g., 10 days).

2. **Agents Work Together:**
   - StockFetchAgent fetches historical prices + technical indicators.
   - NewsScraperAgent and YouTubeScraperAgent scrape news and comments.
   - PreprocessingAgent cleans the data.
   - SentimentAgent analyzes emotions (fear, greed, anxiety, etc.).
   - PredictionAgent forecasts future stock prices.
   - ReasoningAgent coordinates agent workflows intelligently.

3. **Visualizations & Reports:**
   - Interactive charts
   - Sentiment and emotion timelines
   - Stock price predictions
   - Downloadable PDF report
   - LLM-generated investment advice

---

## 📊 Example Outputs

- Stock Price Trend Graph
- Sentiment & Emotion Score Graphs
- Predicted Future Prices
- Investment Recommendation Texts
- PDF Downloadable Report

---

## 🧩 Tech Stack

- Python
- Streamlit
- yfinance
- NewsAPI, YouTube Data API
- HuggingFace Transformers (for emotion detection)
- TensorFlow (LSTM model)
- fpdf2 (for PDF reports)

---

## ⚡ Future Improvements

- Add more emotional granularity (e.g., "hope", "panic").
- Integrate Twitter and Reddit scraping.
- Upgrade LSTM to hybrid Transformer+LSTM model.
- Full AutoML agent to optimize prediction models.
- Multilingual emotion analysis for international news.