#!/usr/bin/env python3
"""
Visualize the results of the DistilBERT fine-tuning on NVDA news headlines.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the confusion matrix
cm = np.load("./distilbert_news_NVDA/confusion_matrix.npy")

# Class labels
class_labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.title("Confusion Matrix - DistilBERT on NVDA News")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("./distilbert_news_NVDA/confusion_matrix.png")
plt.close()

# Calculate class-wise metrics
precision = np.zeros(3)
recall = np.zeros(3)
f1 = np.zeros(3)

for i in range(3):
    # Precision = TP / (TP + FP)
    precision[i] = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
    # Recall = TP / (TP + FN)
    recall[i] = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

# Plot metrics by class
plt.figure(figsize=(12, 6))
x = np.arange(len(class_labels))
width = 0.25

plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, f1, width, label="F1 Score")

plt.xlabel("Sentiment Class")
plt.ylabel("Score")
plt.title("Classification Metrics by Class - DistilBERT on NVDA News")
plt.xticks(x, class_labels)
plt.ylim(0, 1.1)
plt.legend(loc="best")

# Add value labels
for i, v in enumerate(precision):
    plt.text(i - width, v + 0.02, f"{v:.2f}", ha="center")
for i, v in enumerate(recall):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
for i, v in enumerate(f1):
    plt.text(i + width, v + 0.02, f"{v:.2f}", ha="center")

plt.tight_layout()
plt.savefig("./distilbert_news_NVDA/class_metrics.png")
plt.close()

print("Visualizations created in the ./distilbert_news_NVDA/ directory.")
