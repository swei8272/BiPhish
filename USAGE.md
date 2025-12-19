# Usage Guide for BiPhish

This guide explains how to use BiPhish for phishing detection, from training models to making predictions.

## Table of Contents

- [Overview](#overview)
- [Complete Training Pipeline](#complete-training-pipeline)
- [Step-by-Step Usage](#step-by-step-usage)
- [Advanced Usage](#advanced-usage)
- [Inference/Prediction](#inferenceprediction)
- [Configuration Options](#configuration-options)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

BiPhish operates in two main stages:

**Stage 1: Feature Extraction**
1. Train CNN on URL character sequences
2. Extract CNN-learned features (128 dimensions)
3. Extract handcrafted features (27+ URL/HTML/reputation features)

**Stage 2: Ensemble Classification**
4. Combine features and train ensemble classifier
5. Evaluate and save models

## Complete Training Pipeline

### Prerequisites

1. Installation complete (see [INSTALL.md](INSTALL.md))
2. Data files in place:
   - `data/phish-58w.txt`
   - `data/legtimate-58w.txt`

### Quick Start (Full Pipeline)

```bash
# Activate your virtual environment
source biphish_env/bin/activate  # Linux/macOS
# or
biphish_env\Scripts\activate     # Windows

# Run the complete pipeline
python train.py                   # Step 1: Train CNN (~10-30 mins)
python CNN_process.py             # Step 2: Extract CNN features (~5-15 mins)
# Step 3: Extract handcrafted features (see below)
python final_train.py             # Step 4: Train ensemble (~5-10 mins)
python test.py                    # Step 5: Evaluate models
```

## Step-by-Step Usage

### Step 1: Train the CNN Model

**Purpose**: Train a CNN to learn character-level patterns in URLs

**Command**:
```bash
python train.py
```

**What it does**:
- Loads URLs from `data/phish-58w.txt` and `data/legtimate-58w.txt`
- Splits data: 70% train, 10% validation, 20% test
- Builds character vocabulary (96 tokens)
- Trains CNN with:
  - Embedding dimension: 128
  - Two convolutional layers (3-gram and 5-gram filters)
  - Max pooling and concatenation
  - 2 output classes (0=Legitimate, 1=Phishing)
- Uses early stopping (patience=50 epochs)
- Saves model to `model_train/model-cc_1.pkl`

**Expected Output**:
```
Loading data...
Building vocabulary...
Vocabulary size: 96
Training samples: 812800
Validation samples: 116114
Test samples: 232228

Epoch 1/50
Train Loss: 0.6234, Train Acc: 65.23%
Val Loss: 0.5891, Val Acc: 68.45%

...

Early stopping triggered at epoch 37
Best model saved to model_train/model-cc_1.pkl
```

**Configuration Options** (edit `train.py`):
```python
num_epoch = 50           # Maximum training epochs
embedding_dim = 128      # Embedding dimension
batch_size = 256         # Batch size (reduce if OOM)
learning_rate = 1e-6     # Learning rate
patience = 50            # Early stopping patience
```

**Troubleshooting**:
- **Out of memory**: Reduce `batch_size` to 128 or 64
- **Slow training**: Enable GPU (CUDA) if available
- **Poor accuracy**: Check data quality, increase epochs

---

### Step 2: Extract CNN Features

**Purpose**: Use trained CNN to extract learned features from URLs

**Command**:
```bash
python CNN_process.py
```

**What it does**:
- Loads trained CNN model from `model_train/model-cc_1.pkl`
- Processes all URLs through the CNN
- Extracts 128-dimensional feature vectors (output of concatenated conv layers)
- Saves features to CSV:
  - `Phishing_url_data_cnn.csv`
  - `Legitimate_url_data_cnn.csv`

**Expected Output**:
```
Loading CNN model...
Processing phishing URLs...
100%|████████████████████| 573963/573963 [05:23<00:00, 1772.45it/s]
Saved to Phishing_url_data_cnn.csv

Processing legitimate URLs...
100%|████████████████████| 584909/584909 [05:31<00:00, 1763.21it/s]
Saved to Legitimate_url_data_cnn.csv

Done!
```

**Output Files**:
- **Phishing_url_data_cnn.csv**: 573,963 rows × 128 columns
- **Legitimate_url_data_cnn.csv**: 584,909 rows × 128 columns

**Configuration Options** (edit `CNN_process.py`):
```python
max_len = 100            # Maximum URL length
batch_size = 256         # Batch size for processing
```

---

### Step 3: Extract Handcrafted Features

**Purpose**: Extract URL, HTML, and reputation-based features

**Note**: This step is complex and requires careful setup due to external dependencies (WHOIS, SSL, HTML parsing).

#### Option A: URL Features Only (Fast)

```python
from feature_extractor import extract_features_url

# For a single URL
url = "http://example.com/path"
features = extract_features_url(url)
print(features)  # Dictionary with 27 URL features
```

#### Option B: Batch Processing (Recommended)

Create a script `extract_all_features.py`:

```python
from feature_extractor import extract_features_url
import pandas as pd
from tqdm import tqdm

def process_urls(input_file, output_file):
    # Read URLs
    with open(input_file, 'r') as f:
        urls = [line.strip() for line in f.readlines()]

    # Extract features for each URL
    features_list = []
    for url in tqdm(urls):
        try:
            features = extract_features_url(url)
            features_list.append(features)
        except Exception as e:
            print(f"Error processing {url}: {e}")
            features_list.append({})  # Add empty dict on error

    # Save to CSV
    df = pd.DataFrame(features_list)
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

# Process phishing URLs
process_urls('data/phish-58w.txt', 'Phishing_url_data_art.csv')

# Process legitimate URLs
process_urls('data/legtimate-58w.txt', 'Legitimate_url_data_art.csv')
```

Run:
```bash
python extract_all_features.py
```

#### Option C: Full Features (URL + HTML + Reputation)

**Warning**: This is time-consuming due to WHOIS lookups and HTML fetching.

```python
from feature_extractor import extract_features_url, extract_features_html, extract_features_rep

def extract_all_features(url):
    features = {}

    # URL features (fast)
    features.update(extract_features_url(url))

    # HTML features (requires fetching page)
    try:
        html_features = extract_features_html(url)
        features.update(html_features)
    except:
        pass

    # Reputation features (slow - WHOIS, SSL, DNS)
    try:
        rep_features = extract_features_rep(url)
        features.update(rep_features)
    except:
        pass

    return features
```

**Output Files**:
- **Phishing_url_data_art.csv**: 573,963 rows × 27+ columns
- **Legitimate_url_data_art.csv**: 584,909 rows × 27+ columns

---

### Step 4: Train Ensemble Classifier

**Purpose**: Train voting ensemble on combined CNN + handcrafted features

**Command**:
```bash
python final_train.py
```

**What it does**:
- Loads CNN features from CSVs
- Loads handcrafted features from CSVs
- Selects 10 key handcrafted features (from 27):
  - `URL_length`, `URL_subdomains`, `URL_totalWordUrl`
  - `URL_shortestWordPath`, `URL_longestWordUrl`
  - `URL_longestWordHost`, `URL_longestWordPath`
  - `URL_averageWordUrl`, `URL_averageWordHost`, `URL_averageWordPath`
- Combines: 128 CNN features + 10 handcrafted = 138 total features
- Handles missing values (NaN → 0, Inf → large value)
- Splits: 80% train, 20% test
- Scales features using StandardScaler
- Trains 5 base classifiers:
  1. Logistic Regression (weight=1)
  2. Naive Bayes (weight=1)
  3. Linear SVM (weight=2)
  4. Decision Tree (weight=1)
  5. Random Forest (weight=2, 100 trees)
- Creates voting classifier (soft voting)
- Saves models:
  - `voting_classifier.pkl` (ensemble)
  - `scaler.pkl` (feature scaler - **REQUIRED for inference**)
  - Individual model files

**Expected Output**:
```
Loading CNN features...
Loaded 573963 phishing, 584909 legitimate samples

Selecting handcrafted features...
Selected 10 features

Combining features...
Total features: 138 (128 CNN + 10 handcrafted)

Handling missing values...
Replaced 234 NaN values, 12 Inf values

Splitting data (80/20)...
Train: 926297, Test: 231644

Scaling features...
Scaler fitted and saved to scaler.pkl

Training Logistic Regression...
Training Naive Bayes...
Training Linear SVM...
Training Decision Tree...
Training Random Forest...
Training Voting Classifier...

Evaluation Results:
Accuracy: 97.82%
Precision: 98.12%
Recall: 97.51%
F1-Score: 97.81%

Confusion Matrix:
[[113245   1823]
 [  2857 113719]]

Models saved successfully!
```

**Saved Models**:
- `voting_classifier.pkl` - Main ensemble model
- `scaler.pkl` - Feature scaler (**must use for predictions**)
- `lr_model.pkl` - Logistic Regression
- `nb_model.pkl` - Naive Bayes
- `svm_model.pkl` - Linear SVM
- `dt_model.pkl` - Decision Tree
- `rf_model.pkl` - Random Forest

**Configuration Options** (edit `final_train.py`):
```python
# Feature selection
selected_features = [...]  # List of feature names to use

# Train/test split
test_size = 0.2            # 20% test set

# Random Forest parameters
n_estimators = 100         # Number of trees
n_jobs = -1                # Use all CPU cores

# Voting weights
weights = [1, 1, 2, 1, 2]  # [LR, NB, SVM, DT, RF]
```

---

### Step 5: Evaluate Models

**Purpose**: Test models on held-out test set

**Command**:
```bash
python test.py
```

**What it does**:
- Loads saved models
- Evaluates on test set
- Generates metrics and visualizations

**Alternative: Use Enhanced Evaluation**:
```bash
python final_train_revised.py
```

This provides:
- Comparative analysis with state-of-the-art methods
- TPR vs FPR at low FPR values
- ROC curves and AUC scores
- Feature importance analysis
- Model complexity metrics (FLOPs, parameters)

---

## Inference/Prediction

### Classify a Single URL

Create `predict.py`:

```python
import joblib
import numpy as np
from CNN_process import get_part_feature
from feature_extractor import extract_features_url

# Load models
voting_clf = joblib.load('voting_classifier.pkl')
scaler = joblib.load('scaler.pkl')
cnn_model = joblib.load('model_train/model-cc_1.pkl')

def predict_url(url):
    # Extract CNN features
    cnn_features = get_part_feature([url], cnn_model)  # 128-dim

    # Extract handcrafted features
    url_features = extract_features_url(url)

    # Select the 10 key features
    selected = [
        url_features.get('URL_length', 0),
        url_features.get('URL_subdomains', 0),
        url_features.get('URL_totalWordUrl', 0),
        url_features.get('URL_shortestWordPath', 0),
        url_features.get('URL_longestWordUrl', 0),
        url_features.get('URL_longestWordHost', 0),
        url_features.get('URL_longestWordPath', 0),
        url_features.get('URL_averageWordUrl', 0),
        url_features.get('URL_averageWordHost', 0),
        url_features.get('URL_averageWordPath', 0),
    ]

    # Combine features
    combined = np.concatenate([cnn_features[0], selected])

    # Handle NaN/Inf
    combined = np.nan_to_num(combined, nan=0.0, posinf=1e10, neginf=-1e10)

    # Scale features (CRITICAL!)
    combined_scaled = scaler.transform([combined])

    # Predict
    prediction = voting_clf.predict(combined_scaled)[0]
    probability = voting_clf.predict_proba(combined_scaled)[0]

    return {
        'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
        'confidence': max(probability),
        'probabilities': {
            'legitimate': probability[0],
            'phishing': probability[1]
        }
    }

# Example usage
url = "http://paypal-verify.suspicious-domain.com/login"
result = predict_url(url)
print(f"URL: {url}")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Phishing Probability: {result['probabilities']['phishing']:.2%}")
```

Run:
```bash
python predict.py
```

### Batch Prediction

```python
def predict_batch(urls):
    results = []
    for url in urls:
        try:
            result = predict_url(url)
            results.append(result)
        except Exception as e:
            results.append({'prediction': 'Error', 'error': str(e)})
    return results

# Example
urls = [
    "http://google.com",
    "http://paypal-secure-login.xyz/verify",
    "http://github.com"
]
results = predict_batch(urls)
for url, result in zip(urls, results):
    print(f"{url}: {result['prediction']}")
```

---

## Configuration Options

### Global Configuration

Edit `utils.py` for global settings:

```python
# Data paths
Phishing_url_data_path = r"./data/phish-58w.txt"
Legitimate_url_data_path = r"./data/legtimate-58w.txt"

# Data split ratios
train_prop = 0.7   # 70% training
val_prop = 0.1     # 10% validation
test_prop = 0.2    # 20% testing
```

### CNN Configuration

Edit `train.py`:

```python
num_epoch = 50              # Training epochs
embedding_dim = 128         # Embedding size
num_class = 2               # Output classes
batch_size = 256            # Batch size
learning_rate = 1e-6        # Learning rate
patience = 50               # Early stopping patience
```

### Feature Extraction Configuration

Edit `CNN_process.py`:

```python
max_len = 100               # Max URL length (characters)
batch_size = 256            # Processing batch size
```

Edit `feature_extractor.py` for timeout settings:

```python
SSL_TIMEOUT = 3             # SSL certificate check timeout
HTTP_TIMEOUT = 5            # HTTP request timeout
WHOIS_TIMEOUT = 10          # WHOIS lookup timeout
```

### Ensemble Configuration

Edit `final_train.py`:

```python
# Feature selection
selected_features = [...]   # Features to use

# Model parameters
rf_n_estimators = 100       # Random Forest trees
test_size = 0.2             # Test set proportion

# Voting weights
voting_weights = [1, 1, 2, 1, 2]  # [LR, NB, SVM, DT, RF]
```

---

## Best Practices

### 1. Data Quality

- Remove duplicate URLs
- Validate URL format before processing
- Handle international domains (punycode) properly
- Balance dataset if highly imbalanced

### 2. Feature Extraction

- **URL features**: Fast, always include
- **HTML features**: Moderate speed, good accuracy boost
- **Reputation features**: Slow, use selectively or cache results
- Handle timeouts gracefully (set reasonable limits)

### 3. Model Training

- Use early stopping to prevent overfitting
- Monitor validation loss during CNN training
- Use cross-validation for ensemble tuning
- Save checkpoints regularly

### 4. Inference

- **Always use the scaler**: Features must be scaled identically to training
- Cache CNN features for repeated predictions
- Implement timeout handling for feature extraction
- Validate input URLs before processing

### 5. Performance Optimization

- Use GPU for CNN training if available
- Enable Intel scikit-learn extensions (scikit-learn-intelex)
- Use parallel processing (`n_jobs=-1`) for ensemble
- Batch process URLs instead of one-by-one

### 6. Production Deployment

- Implement proper error handling
- Add logging for debugging
- Cache frequently accessed features
- Consider model quantization for faster inference
- Monitor prediction latency

---

## Troubleshooting

### Issue: Different Results Between Runs

**Cause**: Random seed not set

**Solution**:
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

### Issue: Feature Extraction Taking Too Long

**Solutions**:
1. Extract only URL features (skip HTML/reputation)
2. Increase timeout limits
3. Process in smaller batches
4. Use multiprocessing

### Issue: Models Loading Incorrectly

**Cause**: Pickle compatibility issues

**Solution**: Ensure same Python and package versions during save/load

### Issue: Poor Prediction Accuracy

**Causes**:
- Forgot to use scaler
- Different feature set than training
- Model not trained properly
- Data distribution mismatch

**Checklist**:
- [ ] Using scaler.pkl?
- [ ] Same 138 features as training?
- [ ] Handling NaN/Inf values?
- [ ] CNN model trained successfully?

### Issue: Memory Error During Batch Processing

**Solutions**:
```python
# Reduce batch size
batch_size = 64  # or smaller

# Process in chunks
for i in range(0, len(urls), 1000):
    chunk = urls[i:i+1000]
    process_chunk(chunk)
```

---

## Next Steps

1. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design details
2. Experiment with different feature combinations
3. Try hyperparameter tuning
4. Implement real-time URL scanning
5. Deploy as a web service or API

## Additional Resources

- **Research Paper**: [Contact authors for citation]
- **Dataset**: [Contact authors for SOTAPhish-1.2M access]
- **Issues**: [GitHub Issues](https://github.com/yourusername/BiPhish/issues)
