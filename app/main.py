"""
SentimentAgents Dashboard - A Streamlit interface for the multiagent stock forecaster.

This module provides a web interface to orchestrate the full multi-agent workflow 
for stock price prediction with sentiment analysis.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import logging
import streamlit as st
import pandas as pd
import io
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os
import requests
from requests.exceptions import RequestException, ConnectionError, Timeout
import base64

from agents.orchestrator_agent import OrchestratorAgent
from agents.stock_fetch_agent import StockFetchAgent
from agents.news_scraper_agent import NewsScraperAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.balanced_sentiment_agent import DistilBERTSentimentAgent
from agents.prophet_model_enhanced_simple import PredictorAgent
from agents.visualizer_agent import VisualizerAgent
from agents.reasoning_agent import ReasoningAgent
from utils.knowledge_base_creator import KnowledgeBaseCreator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SentimentAgents Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💹"
)

# Try loading the external CSS file, if it exists
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Add logo image if file exists
logo_path = '/Users/marik/Desktop/multiagent_stock_forecaster/Gemini_Generated_Image_os9c3oos9c3oos9c.png'
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=200, use_container_width=True)
st.sidebar.markdown("### 📊 Analysis Parameters", unsafe_allow_html=False)

# Custom CSS for modern fintech dark theme
st.markdown("""
    <style>
        /* Global styling */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0E0F1D !important;
            color: #E0E0E5 !important;
            font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Headers */
        h1, h2, h3, .css-10trblm, .css-hxt7ib p {
            color: #FFFFFF;
            font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 600;
        }
        
        h1 {
            font-size: 2.4rem;
            letter-spacing: -0.5px;
            margin-bottom: 1.5rem;
            background: linear-gradient(90deg, #00FFE1, #FF00E1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: white;
        }
        
        h2 {
            font-size: 1.6rem;
            letter-spacing: -0.3px;
        }
        
        /* Metric Cards */
        .metric-card {
            background: #1A1C2E;
            padding: 1.4rem;  /* Reduced from 1.8rem */
            border-radius: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, #00FFE1, #FF00E1);
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
            border-color: rgba(255, 255, 255, 0.1);
        }
        
        .metric-card h3 {
            color: #A0A0B2;
            font-size: 0.8rem;  /* Decreased from 0.9rem */
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.4rem;  /* Slightly reduced spacing */
            font-weight: 300;
        }
        
        .metric-card p {
            background: linear-gradient(90deg, #00FFE1, #FF00E1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.6rem;  /* Decreased from 2rem */
            font-weight: 700;
            margin: 0;
            font-feature-settings: "tnum";
            font-variant-numeric: tabular-nums;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #5469d4, #7795f8);
            color: white;
            border: none;
            padding: 0.85rem 2rem;
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 0.3px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(84, 105, 212, 0.25);
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #4a5cc2, #6983e8);
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(84, 105, 212, 0.35);
        }
        
        /* Chart containers */
        .chart-container {
            background-color: #1A1C2E;
            border-radius: 0 0 12px 12px;  /* Rounded only on bottom corners */
            padding: 1.2rem;  /* Reduced padding */
            margin: 0 0 1.5rem 0;  /* Removed top margin */
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-top: none;  /* Remove top border to connect with header */
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
        }
        
        /* Card header */
        .card-header {
            background: linear-gradient(90deg, #00FFE1, #FF00E1);
            padding: 10px 16px;
            border-radius: 8px 8px 0 0;
            font-size: 1.1rem;
            font-weight: bold;
            color: #12121F;
            margin-bottom: 0;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] > div {
            background-color: #111228;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
            padding: 2rem 1rem;
        }
        
        /* Make sidebar text white and more visible */
        section[data-testid="stSidebar"] label, 
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stDateInput,
        section[data-testid="stSidebar"] p {
            color: #FFFFFF !important;
        }
        
        /* Ensure sidebar headings are prominent */
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        
        /* Inputs styling */
        div[data-baseweb="select"] {
            border-radius: 8px;
        }
        
        div[data-baseweb="select"] > div {
            background-color: #1A1C2E;
            color: #FFFFFF;  /* Changed from #E0E0E5 to white for better visibility */
            border-color: rgba(255, 255, 255, 0.2);  /* Slightly more visible border */
            border-radius: 8px;
        }
        
        /* Input field styling */
        .stTextInput > div > div {
            background-color: #1A1C2E;
            color: #FFFFFF;  /* Changed to white for better visibility */
            border-color: rgba(255, 255, 255, 0.2);  /* More visible border */
            border-radius: 8px;
        }
        
        /* Input field text */
        .stTextInput input {
            color: #FFFFFF !important;  /* Make text input value white */
            font-weight: 500 !important;  /* Make it slightly bolder */
        }
        
        /* Date input styling */
        .stDateInput > div > div {
            background-color: #1A1C2E;
            color: #FFFFFF;  /* Changed to white for better visibility */
            border-color: rgba(255, 255, 255, 0.2);  /* More visible border */
            border-radius: 8px;
        }
        
        /* Date input text */
        .stDateInput input {
            color: #FFFFFF !important;  /* Make date input value white */
            font-weight: 500 !important;  /* Make it slightly bolder */
        }
        
        /* Select box text (dropdown values) */
        div[data-baseweb="select"] span {
            color: #FFFFFF !important;  /* Make select box value white */
            font-weight: 500 !important;  /* Make it slightly bolder */
        }
        
        /* Make dataframes more fintech-like */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .stDataFrame div[data-testid="stDataFrame"] {
            background-color: #1A1C2E !important;
        }
        
        .stDataFrame th {
            background-color: #111228 !important;
            color: #E0E0E5 !important;
        }
        
        .stDataFrame td {
            background-color: #1A1C2E !important;
            color: #E0E0E5 !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #1A1C2E;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            color: #E0E0E5;
        }
        
        /* Footer styling */
        footer {
            visibility: hidden;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #5469d4, #7795f8);
        }
        
        /* Main content container */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        
        /* Download button styling */
        .stDownloadButton button {
            background-color: #5469d4 !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }
        
        .stDownloadButton button:hover {
            background-color: #3a4cb1 !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        }
        
        /* AI Query interface styling */
        .card-header {
            background-color: #1e1e2d;
            color: #ffffff;
            padding: 0.75rem;
            border-radius: 8px 8px 0 0;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0;
        }
        
        /* Improved input styling */
        .stTextInput div[data-baseweb="base-input"] {
            background-color: #1A1C2E;
            border-color: #5469d4;
        }
        .stTextInput input {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

def get_predictions_csv_path(ticker):
    # Correct: Use the project root's data_predictions directory
    return os.path.join(project_root, "data_predictions", f"{ticker}_prophet_predictions.csv")

@st.cache_data
def load_predictions_csv(ticker):
    path = get_predictions_csv_path(ticker)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

# Initialize knowledge base creator (singleton for app session)
if 'kb_creator' not in st.session_state:
    st.session_state.kb_creator = KnowledgeBaseCreator()
kb_creator = st.session_state.kb_creator

def build_ollama_prompt(query, ticker, n_results=20):
    """Build a concise, fast prompt for Ollama - simplified for speed."""
    
    # Check if knowledge base exists for this ticker
    kb_dir = Path("knowledge_base")
    latest_kb_file = kb_dir / f"{ticker}_knowledge_base_latest.json"
    
    if not latest_kb_file.exists():
        return f"No data available for {ticker}. Run analysis first."
    
    # Get minimal context from knowledge base
    try:
        context = kb_creator.query_knowledge_base(str(latest_kb_file), query)
        if not context:
            context = "No relevant information found."
    except Exception as e:
        return f"Error accessing data: {str(e)}"
    
    # Create simple, direct prompt
    prompt = f"Stock: {ticker}\nData: {context[:500]}...\nQ: {query}\nA:"
    return prompt

# Hardcode the correct Ollama URL - no environment variables, no overrides
OLLAMA_URL = "http://localhost:11434"  # This should not be used directly, see query_ollama function

# --- Ollama connection constants ---
OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
MAX_RETRIES = 3
RETRY_DELAY = 1

# --- Ollama connection helpers ---
def reload_ollama_model(model_name="llama3"):
    import subprocess
    try:
        result = subprocess.run(['ollama', 'pull', model_name], capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False

def restart_ollama_service():
    import subprocess
    import time
    try:
        subprocess.run(['pkill', 'ollama'], capture_output=True)
        time.sleep(2)
        subprocess.Popen(['ollama', 'serve'])
        time.sleep(5)
        return True
    except Exception:
        return False

def wait_for_ollama():
    import socket
    import time
    for _ in range(MAX_RETRIES):
        try:
            with socket.create_connection((OLLAMA_HOST, OLLAMA_PORT), timeout=5):
                return True
        except (socket.timeout, socket.error):
            time.sleep(RETRY_DELAY)
    return False

def query_ollama(prompt, model_name="llama3"): 
    api_endpoint = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    if not wait_for_ollama():
        if restart_ollama_service() and wait_for_ollama():
            pass
        else:
            return f"[Ollama error] Ollama is not responding on {OLLAMA_HOST}:{OLLAMA_PORT}"
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Connection": "close"
    })
    try:
        # Ultra-fast optimizations for sub-10 second responses
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 256,        # Much smaller context window
                "temperature": 0.1,    # Very focused responses
                "num_predict": 80,     # Much shorter responses
                "top_k": 10,           # Fewer token choices
                "top_p": 0.7,          # More decisive
                "repeat_penalty": 1.1, # Prevent repetition
                "stop": ["\n\n", "Question:", "Query:"]  # Stop early
            }
        }
        last_error = None
        model_reloaded = False
        for attempt in range(MAX_RETRIES):
            try:
                response = session.post(api_endpoint, json=payload, timeout=15)  # Reduced from 30s to 15s
                if response.status_code == 500:
                    if not model_reloaded:
                        if reload_ollama_model(model_name):
                            model_reloaded = True
                            import time; time.sleep(2)
                            continue
                        else:
                            if restart_ollama_service():
                                import time; time.sleep(2)
                                continue
                    if len(prompt) > 500:
                        payload["prompt"] = prompt[:500] + "..."
                        continue
                response.raise_for_status()
                try:
                    data = response.json()
                    result = data.get("response", "[Empty response from Ollama]")
                    if result:
                        return result
                    last_error = "Empty response from model"
                    continue
                except ValueError as e:
                    last_error = f"Could not parse JSON response: {str(e)}"
                    continue
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if "too many open files" in str(e).lower():
                    import time; time.sleep(3)
                elif any(x in str(e).lower() for x in ["connection refused", "connection reset"]):
                    if restart_ollama_service():
                        import time; time.sleep(2)
                        continue
                if attempt < MAX_RETRIES - 1:
                    import time; time.sleep(RETRY_DELAY * (attempt + 1))
                continue
        return f"[Ollama error] Failed after {MAX_RETRIES} attempts: {last_error}"
    except Exception as e:
        return f"[Ollama error] Unexpected error: {str(e)}"
    finally:
        session.close()

def query_ollama_subprocess(prompt, model_name="llama3"):
    """Query Ollama using subprocess as a fallback method."""
    import subprocess
    import json
    import tempfile
    import os
    
    api_endpoint = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    temp_path = None
    
    try:
        # First check if Ollama is responding
        if not wait_for_ollama():
            return f"[Ollama subprocess error] Ollama is not responding on {OLLAMA_HOST}:{OLLAMA_PORT}"
        
        # Create the curl command with explicit connection handling
        cmd = [
            'curl', '-s',
            '-X', 'POST',
            '-H', 'Connection: close',
            '-H', 'Content-Type: application/json',
            api_endpoint,
            '-d', json.dumps({
                "model": model_name,
                "prompt": prompt,
                "stream": False
            })
        ]
        
        # Run with retries
        for attempt in range(MAX_RETRIES):
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0 and result.stdout:
                    try:
                        response_json = json.loads(result.stdout)
                        return response_json.get("response", "[Empty response]")
                    except json.JSONDecodeError as e:
                        if attempt < MAX_RETRIES - 1:
                            import time; time.sleep(RETRY_DELAY)
                            continue
                        return f"[Ollama subprocess error] Could not parse response: {result.stdout[:100]}..."
                else:
                    if attempt < MAX_RETRIES - 1:
                        import time; time.sleep(RETRY_DELAY)
                        continue
                    return f"[Ollama subprocess error] Command failed: {result.stderr}"
                    
            except subprocess.TimeoutExpired:
                if attempt < MAX_RETRIES - 1:
                    import time; time.sleep(RETRY_DELAY)
                    continue
                return "[Ollama subprocess error] Request timed out"
                
    except Exception as e:
        return f"[Ollama subprocess error] {str(e)}"

def initialize_agents():
    """Initialize all required agents and the orchestrator."""
    try:
        # Initialize orchestrator with all agents
        orchestrator = OrchestratorAgent(
            StockFetchAgent,
            NewsScraperAgent,
            PreprocessingAgent,
            DistilBERTSentimentAgent,
            PredictorAgent,
            VisualizerAgent,
            reasoning_agent_cls=ReasoningAgent
        )
        return orchestrator
    except Exception as e:
        st.error(f"Failed to initialize agents: {str(e)}")
        logger.error(f"Agent initialization failed: {str(e)}", exc_info=True)
        st.stop()

def plotly_fig_to_png(fig):
    """Convert a plotly figure to PNG for downloading."""
    img_bytes = pio.to_image(fig, format="png", width=1200, height=800)
    return img_bytes

def run_analysis(orchestrator, ticker, start_date, end_date, forecast_period, strategy='balanced'):
    """Run the full analysis workflow using the orchestrator."""
    try:
        with st.spinner("🔄 Running analysis..."):
            # Create a progress bar
            progress_bar = st.progress(0, text="0%")
            status_text = st.empty()
            import time
            # Update status
            status_text.text("Fetching stock and news data...")
            progress_bar.progress(40, text="40%")
            # Smoothly and slowly animate progress bar from 41% to 99% while orchestrator runs
            for pct in range(41, 100, 1):
                progress_bar.progress(pct, text=f"{pct}%")
                time.sleep(0.025)
            # Run orchestrator (blocking call)
            outputs = orchestrator.run(
                ticker=ticker,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                forecast_period=forecast_period,
                max_news_articles=10000,
                reasoning_strategy=strategy
            )
            # Complete progress bar
            progress_bar.progress(100, text="100%")
            status_text.text("✅ Analysis complete!")
            return outputs
            
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        return None

def display_metrics(outputs):
    """Display key metrics in a visually appealing way."""
    if not outputs:
        return
        
    st.subheader("📊 Key Metrics")
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>Stock Data</h3>
                <p>Records: {}</p>
            </div>
        """.format(len(outputs["stock_df"])), unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>News Articles</h3>
                <p>Articles: {}</p>
            </div>
        """.format(len(outputs["news_df"])), unsafe_allow_html=True)
        
    with col3:
        sentiment_df = outputs.get("sentiment_df")
        if sentiment_df is not None and not sentiment_df.empty:
            # Check if the expected column exists
            if "news_sentiment_score" in sentiment_df.columns:
                avg_sentiment = sentiment_df["news_sentiment_score"].mean()
            elif "sentiment_score" in sentiment_df.columns:
                avg_sentiment = sentiment_df["sentiment_score"].mean()
            else:
                avg_sentiment = 0.0
            
            st.markdown("""
                <div class="metric-card">
                    <h3>Sentiment Score</h3>
                    <p>Average: {:.2f}</p>
                </div>
            """.format(avg_sentiment), 
            unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="metric-card">
                    <h3>Sentiment Score</h3>
                    <p>No sentiment data available</p>
                </div>
            """, unsafe_allow_html=True)

def display_investment_recommendation(outputs):
    """Display professional investment recommendation cards."""
    if not outputs or 'reasoning_analysis' not in outputs or not outputs['reasoning_analysis']:
        return
    
    analysis = outputs['reasoning_analysis']
    recommendation = analysis['recommendation']
    targets = analysis['targets']
    risk = analysis['risk']
    momentum = analysis['momentum']
    sentiment = analysis['sentiment']
    
    st.markdown("### 🎯 Investment Recommendation")
    
    # Main recommendation card
    action_color = {
        "STRONG BUY": "#00FF7F", 
        "BUY": "#32CD32",
        "HOLD": "#FFD700",
        "WEAK SELL": "#FF6347",
        "SELL": "#DC143C"
    }.get(recommendation['action'], "#FFD700")
    
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1A1C2E 0%, #2D1B69 100%); 
                    padding: 2rem; border-radius: 16px; 
                    border: 2px solid {action_color}; 
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                    margin-bottom: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <h2 style="color: white; margin: 0; font-size: 1.8rem;">{analysis['ticker']} Analysis</h2>
                <div style="background: {action_color}; color: #000; padding: 0.75rem 1.5rem; 
                           border-radius: 25px; font-weight: bold; font-size: 1.1rem;">
                    {recommendation['action']}
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem;">
                <div style="text-align: center;">
                    <h4 style="color: #A0A0B2; margin: 0 0 0.5rem 0;">Confidence</h4>
                    <p style="color: {action_color}; font-size: 2rem; font-weight: bold; margin: 0;">
                        {recommendation['confidence']}%
                    </p>
                </div>
                <div style="text-align: center;">
                    <h4 style="color: #A0A0B2; margin: 0 0 0.5rem 0;">Position Size</h4>
                    <p style="color: white; font-size: 1.2rem; font-weight: 600; margin: 0;">
                        {recommendation['position_size']}
                    </p>
                </div>
                <div style="text-align: center;">
                    <h4 style="color: #A0A0B2; margin: 0 0 0.5rem 0;">Risk Score</h4>
                    <p style="color: {'#FF6B6B' if risk['risk_score'] > 70 else '#FFD93D' if risk['risk_score'] > 40 else '#6BCF7F'}; 
                       font-size: 1.5rem; font-weight: bold; margin: 0;">
                        {risk['risk_score']:.1f}/100
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Price targets and returns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Price Targets")
        current_price = targets['current']
        
        st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <h3>Current Price</h3>
                <p>${current_price:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        
        target_data = [
            ("1 Week", targets['target_1w'], recommendation['expected_return_1w']),
            ("1 Month", targets['target_1m'], recommendation['expected_return_1m']),
            ("3 Month", targets['target_3m'], recommendation['expected_return_3m'])
        ]
        
        for period, target, return_pct in target_data:
            color = "#32CD32" if return_pct > 0 else "#DC143C" if return_pct < 0 else "#FFD700"
            st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 1rem;">
                    <h3>{period} Target</h3>
                    <p>${target:.2f} <span style="color: {color};">({return_pct:+.1f}%)</span></p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📈 Technical Analysis")
        
        # Momentum metrics
        momentum_color = "#32CD32" if momentum['momentum_score'] > 5 else "#DC143C" if momentum['momentum_score'] < -5 else "#FFD700"
        st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <h3>Momentum Score</h3>
                <p style="color: {momentum_color};">{momentum['momentum_score']:.1f}</p>
            </div>
        """, unsafe_allow_html=True)
        
        trend_color = "#32CD32" if momentum['trend_strength'] > 60 else "#DC143C" if momentum['trend_strength'] < 40 else "#FFD700"
        st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <h3>Trend Strength</h3>
                <p style="color: {trend_color};">{momentum['trend_strength']:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        sentiment_color = "#32CD32" if sentiment['avg_sentiment'] > 0.1 else "#DC143C" if sentiment['avg_sentiment'] < -0.1 else "#FFD700"
        sentiment_label = "Bullish" if sentiment['avg_sentiment'] > 0.1 else "Bearish" if sentiment['avg_sentiment'] < -0.1 else "Neutral"
        st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <h3>Market Sentiment</h3>
                <p style="color: {sentiment_color};">{sentiment_label}</p>
            </div>
        """, unsafe_allow_html=True)
        
        volatility_color = "#DC143C" if momentum['volatility'] > 50 else "#FFD700" if momentum['volatility'] > 25 else "#32CD32"
        st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <h3>Volatility</h3>
                <p style="color: {volatility_color};">{momentum['volatility']:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Detailed reasoning
    if 'detailed_reasoning' in analysis:
        with st.expander("📋 Detailed Analysis & Reasoning", expanded=False):
            st.markdown(analysis['detailed_reasoning'])

def test_ollama_connection():
    """Test if Ollama is running and accessible on the expected port."""
    import requests
    from requests.exceptions import RequestException
    url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            if "llama3" not in model_names and len(model_names) > 0:
                return True, f"Ollama is running but llama3 model is missing. Available models: {', '.join(model_names)}"
            elif len(model_names) == 0:
                return True, "Ollama is running but no models are available. Please run: ollama pull llama3"
            else:
                return True, f"Ollama is running correctly. Models: {', '.join(model_names)}"
        else:
            return False, f"Ollama responded with error status code {response.status_code}"
    except RequestException as e:
        return False, f"Connection error: {str(e)}. Make sure Ollama is running on port 11434."
    except Exception as e:
        return False, f"Error connecting to Ollama: {str(e)}"

def main():
    """Main application function."""
    
    # Initialize Streamlit session state variables to prevent AttributeError
    if "current_ticker" not in st.session_state:
        st.session_state.current_ticker = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "query_results" not in st.session_state:
        st.session_state.query_results = None
    if "last_ollama_url" not in st.session_state:
        st.session_state.last_ollama_url = "http://localhost:11434"
    if "ollama_connection_status" not in st.session_state:
        st.session_state.ollama_connection_status = None
    
    # Title and description with more professional styling
    st.markdown("<h1 style='text-align: center; margin-bottom: 0.5rem;'>Sentiment Agents Framework</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #5469d4; margin-bottom: 2rem;'>Advanced Stock Forecast Platform with AI-Powered Sentiment Analysis</p>", unsafe_allow_html=True)
    
    # Stock ticker input with automatic uppercase conversion
    ticker = st.sidebar.text_input("Stock Ticker", value="NVDA").upper()
    
    # Date range inputs
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=30)
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start_date,
        max_value=default_end_date
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=default_end_date,
        max_value=default_end_date,
        min_value=start_date
    )
    
    # Forecast horizon selection with extended options up to 365 days
    forecast_options = {
        "7 days": 7, 
        "14 days": 14, 
        "30 days": 30,
        "60 days": 60,
        "90 days": 90,
        "180 days": 180,
        "365 days": 365
    }
    forecast_selection = st.sidebar.selectbox(
        "Forecast Horizon",
        list(forecast_options.keys()),
        index=2
    )
    forecast_period = forecast_options[forecast_selection]

    # Investment strategy selection
    st.sidebar.markdown("### 📈 Investment Strategy")
    strategy_descriptions = {
        "balanced": "Balanced approach weighing all factors equally",
        "growth": "Focus on momentum and growth potential",
        "value": "Emphasis on fundamental value and trends",
        "momentum": "Prioritize price momentum and market sentiment",
        "contrarian": "Seek opportunities against market consensus"
    }
    strategy = st.sidebar.selectbox(
        "Strategy",
        list(strategy_descriptions.keys()),
        format_func=lambda x: x.title(),
        help="Select your investment strategy"
    )
    st.sidebar.caption(strategy_descriptions[strategy])

    # Initialize agents
    orchestrator = initialize_agents()

    # --- Dashboard output caching ---
    if 'analysis_outputs' not in st.session_state:
        st.session_state.analysis_outputs = None
    if 'analysis_params' not in st.session_state:
        st.session_state.analysis_params = {}

    # Helper: check if analysis params changed
    def analysis_params_changed():
        params = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'forecast_period': forecast_period
        }
        return params != st.session_state.analysis_params

    # Run analysis button (should be above LLM chatbot)
    run_analysis_clicked = st.sidebar.button("🚀 Run Analysis")
    # If parameters changed, clear outputs but do not run analysis automatically
    if analysis_params_changed():
        st.session_state.analysis_outputs = None
        st.session_state.analysis_params = {
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'forecast_period': forecast_period
        }
    # Only run analysis if button is clicked
    if run_analysis_clicked:
        outputs = run_analysis(orchestrator, ticker, start_date, end_date, forecast_period, strategy)
        if outputs:
            st.session_state.analysis_outputs = outputs
            st.session_state.analysis_params = {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'forecast_period': forecast_period
            }
            # Invalidate predictions CSV cache for this ticker so chatbot/dashboard see new file
            st.cache_data.clear()
    outputs = st.session_state.analysis_outputs

    # --- Render dashboard if outputs exist ---
    if outputs:
        display_metrics(outputs)
        
        # Display investment recommendation if available
        if 'reasoning_analysis' in outputs:
            display_investment_recommendation(outputs)
        
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        with st.container():
            col = st.columns(1)[0]
            col.markdown('<div class="card-header">📈 Price Prediction</div>', unsafe_allow_html=True)
            col.markdown('<div class="chart-container">', unsafe_allow_html=True)
            col.plotly_chart(outputs["price_fig"], use_container_width=True)
            col.markdown('</div>', unsafe_allow_html=True)
        with st.container():
            col = st.columns(1)[0]
            col.markdown('<div class="card-header">📊 Sentiment Analysis</div>', unsafe_allow_html=True)
            col.markdown('<div class="chart-container">', unsafe_allow_html=True)
            col.plotly_chart(outputs["sentiment_fig"], use_container_width=True)
            col.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top: 2rem;'>📊 Data Tables</h3>", unsafe_allow_html=True)
        with st.expander("🔍 Stock Data Preview"):
            st.markdown('<div style="background-color: #1A1C2E; border-radius: 8px; padding: 1rem;">', unsafe_allow_html=True)
            st.dataframe(outputs["stock_df"])
            st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("📰 News Data Preview"):
            st.markdown('<div style="background-color: #1A1C2E; border-radius: 8px; padding: 1rem;">', unsafe_allow_html=True)
            st.dataframe(outputs["news_df"])
            st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("🎯 Predictions Preview"):
            st.markdown('<div style="background-color: #1A1C2E; border-radius: 8px; padding: 1rem;">', unsafe_allow_html=True)
            st.dataframe(outputs["prediction_df"])
            st.markdown('</div>', unsafe_allow_html=True)
        st.subheader("📥 Download Data and Charts")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Predictions CSV",
                data=outputs["prediction_df"].to_csv().encode("utf-8"),
                file_name=f"{ticker}_predictions.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="📥 Download Full Analysis CSV",
                data=outputs["merged_df"].to_csv().encode("utf-8"),
                file_name=f"{ticker}_full_analysis.csv",
                mime="text/csv"
            )
        col1, col2 = st.columns(2)
        with col1:
            price_chart_png = plotly_fig_to_png(outputs["price_fig"])
            st.download_button(
                label="📥 Download Price Chart (PNG)",
                data=price_chart_png,
                file_name=f"{ticker}_price_chart.png",
                mime="image/png"
            )
        with col2:
            sentiment_chart_png = plotly_fig_to_png(outputs["sentiment_fig"])
            st.download_button(
                label="📥 Download Sentiment Chart (PNG)",
                data=sentiment_chart_png,
                file_name=f"{ticker}_sentiment_chart.png",
                mime="image/png"
            )

    # --- Chatbot in the left sidebar, below analysis parameters and Run Analysis button ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='color:#5469d4;'>🤖 LLM Chatbot</h3>", unsafe_allow_html=True)
    predictions_df = load_predictions_csv(ticker)
    if predictions_df is None:
        st.sidebar.info(f"No predictions CSV found for ticker '{ticker}'. Please run analysis for this ticker first.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "sidebar_chatbot_input" not in st.session_state:
            st.session_state.sidebar_chatbot_input = ""
        chatbot_input = st.sidebar.text_input("Type your message...", value=st.session_state.sidebar_chatbot_input, key="sidebar_chatbot_input")
        send_clicked = st.sidebar.button("Send", key="sidebar_chatbot_send")
        if send_clicked and chatbot_input:
            st.session_state.chat_history.append({"role": "user", "content": chatbot_input})
            spinner_placeholder = st.sidebar.empty()
            spinner_placeholder.info("LLM is thinking...")
            prompt = build_ollama_prompt(chatbot_input, ticker)
            if prompt.startswith("[No stock data found"):
                response = prompt
            else:
                response = query_ollama(prompt)
            spinner_placeholder.empty()
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        # --- Modern chat UI ---
        chat_css = """
        <style>
        .chat-message { display: flex; align-items: flex-start; margin-bottom: 1rem; opacity: 0; animation: fadeInChat 0.7s forwards; }
        .chat-message.user { flex-direction: row-reverse; }
        .chat-avatar {
            width: 38px; height: 38px; border-radius: 50%; background: #23244a;
            display: flex; align-items: center; justify-content: center; margin: 0 0.7rem;
            box-shadow: 0 2px 8px rgba(84, 105, 212, 0.15); transition: box-shadow 0.3s;
        }
        .chat-message.user .chat-avatar {
            background: linear-gradient(135deg, #5469d4 60%, #00FFE1 100%);
            box-shadow: 0 2px 8px rgba(0,255,225,0.15);
        }
        .chat-message.assistant .chat-avatar {
            background: linear-gradient(135deg, #23244a 60%, #FF00E1 100%);
            box-shadow: 0 2px 8px rgba(255,0,225,0.15);
        }
        .chat-bubble {
            background: #23244a; color: #E0E0E5; border-radius: 16px; padding: 0.9rem 1.2rem;
            max-width: 80%; font-size: 1.05rem; box-shadow: 0 2px 8px rgba(84, 105, 212, 0.10);
            transition: background 0.3s, box-shadow 0.3s;
        }
        .chat-message.user .chat-bubble {
            background: linear-gradient(90deg, #5469d4 60%, #00FFE1 100%); color: #fff;
            box-shadow: 0 2px 8px rgba(0,255,225,0.10);
        }
        .chat-message.assistant .chat-bubble {
            background: linear-gradient(90deg, #23244a 60%, #FF00E1 100%); color: #fff;
            box-shadow: 0 2px 8px rgba(255,0,225,0.10);
        }
        @keyframes fadeInChat {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """
        st.sidebar.markdown(chat_css, unsafe_allow_html=True)
        # Llama SVG for assistant, person SVG for user
        llama_icon = '''<svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="16" cy="16" r="16" fill="#FF00E1"/><path d="M10 22c0-2.5 2-4.5 4.5-4.5S19 19.5 19 22" stroke="#fff" stroke-width="2" stroke-linecap="round"/><ellipse cx="16" cy="14" rx="5" ry="7" fill="#fff"/><ellipse cx="14.5" cy="13.5" rx="0.8" ry="1.2" fill="#23244a"/><ellipse cx="17.5" cy="13.5" rx="0.8" ry="1.2" fill="#23244a"/></svg>'''
        person_icon = '''<svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="16" cy="16" r="16" fill="#5469d4"/><ellipse cx="16" cy="13" rx="6" ry="6" fill="#fff"/><ellipse cx="16" cy="25" rx="10" ry="6" fill="#fff"/></svg>'''
        chat_html = ""
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f'''<div class="chat-message user"><div class="chat-avatar">{person_icon}</div><div class="chat-bubble">{msg['content']}</div></div>'''
            else:
                chat_html += f'''<div class="chat-message assistant"><div class="chat-avatar">{llama_icon}</div><div class="chat-bubble">{msg['content']}</div></div>'''
        st.sidebar.markdown(chat_html, unsafe_allow_html=True)
        st.sidebar.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        if st.sidebar.button("🗑️ Clear Chat", key="sidebar_clear_chat"):
            st.session_state.chat_history = []
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='color: white;'>Made with ❤️ by Sentiment Agents Team</div>", 
        unsafe_allow_html=True
    )
    
    # --- Ollama connection test (should be at the top of main, after sidebar setup) ---
    try:
        import subprocess
        result = subprocess.run(['lsof', '-i', ':11434'], capture_output=True, text=True)
        output = result.stdout
        if not output:
            st.sidebar.warning("⚠️ Ollama not detected on port 11434")
            st.sidebar.info("Run: `ollama serve` to start Ollama")
        # Then try API connection
        connection_status, connection_message = test_ollama_connection()
        if connection_status:
            st.sidebar.success(f"✅ Ollama ready: {connection_message}")
        else:
            st.sidebar.error(f"❌ {connection_message}")
            st.sidebar.info("Check Ollama Server Test section for troubleshooting")
    except Exception as e:
        st.sidebar.error(f"Unexpected error: {str(e)}")
        logger.error(f"Unexpected error during connection test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()