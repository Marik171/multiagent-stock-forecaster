# Model Files

## Large files excluded from GitHub

Some large model files have been excluded from this GitHub repository due to GitHub's file size limitations:

- `distilbert_news_NVDA/model.safetensors` (255MB)
- `distilbert_news_NVDA_balanced/model.safetensors` (255MB)

## How to get the model files

### Option 1: Download pre-trained DistilBERT models
You can download the DistilBERT model from Hugging Face directly:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# For financial sentiment analysis
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Save locally
model.save_pretrained("./distilbert_news_NVDA")
tokenizer.save_pretrained("./distilbert_news_NVDA")
```

### Option 2: Run the fine-tuning script
You can run the `fine_tune_distilbert_news.py` script to recreate the fine-tuned models:

```bash
python fine_tune_distilbert_news.py --output_dir="./distilbert_news_NVDA"
python fine_tune_distilbert_news.py --output_dir="./distilbert_news_NVDA_balanced" --class_balanced=True
```

### Option 3: Download from alternative storage
The model files may be available from an alternative storage location. Contact the repository owner for access.
