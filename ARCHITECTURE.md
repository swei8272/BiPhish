# BiPhish Architecture Documentation

This document provides detailed technical documentation of the BiPhish system architecture.

## Table of Contents

- [System Overview](#system-overview)
- [Dual-Channel Architecture](#dual-channel-architecture)
- [CNN Channel](#cnn-channel)
- [Traditional ML Channel](#traditional-ml-channel)
- [Feature Fusion](#feature-fusion)
- [Ensemble Classifier](#ensemble-classifier)
- [Data Flow](#data-flow)
- [Module Reference](#module-reference)
- [Design Decisions](#design-decisions)

## System Overview

BiPhish is a **dual-channel phishing detection system** that combines:
1. **Deep learning** (CNN) for automatic feature learning from URL character sequences
2. **Traditional ML** with handcrafted features based on domain expertise

```
┌─────────────────────────────────────────────────────────────────┐
│                         BiPhish System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐              ┌────────────────────────┐  │
│  │  CNN Channel    │              │  Traditional Channel   │  │
│  │                 │              │                        │  │
│  │  URL → Chars    │              │  URL → Handcrafted     │  │
│  │  → Embedding    │              │       Features         │  │
│  │  → Conv Layers  │              │  - URL Features (27)   │  │
│  │  → 128 features │              │  - HTML Features (18)  │  │
│  └────────┬────────┘              │  - Reputation (9)      │  │
│           │                       └───────────┬────────────┘  │
│           │                                   │               │
│           └──────────┬────────────────────────┘               │
│                      │                                        │
│              ┌───────▼────────┐                               │
│              │ Feature Fusion │                               │
│              │ 128 + 10 = 138 │                               │
│              └───────┬────────┘                               │
│                      │                                        │
│              ┌───────▼────────┐                               │
│              │    Ensemble    │                               │
│              │  Voting (5ML)  │                               │
│              └───────┬────────┘                               │
│                      │                                        │
│              ┌───────▼────────┐                               │
│              │  Classification │                               │
│              │ Legit/Phishing │                               │
│              └────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## Dual-Channel Architecture

### Why Dual-Channel?

**Problem**: Single-channel approaches have limitations:
- **CNN-only**: May miss semantic/structural patterns
- **Handcrafted-only**: Cannot learn complex implicit patterns

**Solution**: Combine both approaches to leverage their complementary strengths.

| Channel | Strengths | Weaknesses |
|---------|-----------|------------|
| **CNN** | - Learns implicit patterns<br>- Detects DGA domains<br>- No manual engineering | - Black-box<br>- Requires training data<br>- May overfit |
| **Traditional** | - Interpretable<br>- Domain knowledge<br>- Stable | - Manual effort<br>- May miss novel patterns<br>- Feature engineering required |
| **Combined** | - Best of both worlds<br>- Robust<br>- High accuracy | - More complex<br>- Two-stage process |

---

## CNN Channel

### Architecture (model.py)

The CNN extracts character-level patterns from URLs.

```
Input URL: "http://paypal-verify.com/login?user=abc"
                    ↓
        Character Tokenization
                    ↓
     [h, t, t, p, :, /, /, p, a, y, ...]  (max 100 chars)
                    ↓
            Char → ID Mapping
                    ↓
     [23, 45, 45, 41, 58, 47, 47, 41, ...]
                    ↓
         ┌──────────────────────┐
         │  Embedding Layer     │
         │  vocab_size × 128    │
         └──────────┬───────────┘
                    ↓
     [batch, 100, 128]  (100 chars × 128-dim vectors)
                    ↓
         ┌──────────┴───────────┐
         │                      │
    ┌────▼─────┐          ┌────▼─────┐
    │ Conv1d-3 │          │ Conv1d-5 │
    │ (3-gram) │          │ (5-gram) │
    │ 128→64   │          │ 128→64   │
    └────┬─────┘          └────┬─────┘
         │                      │
    ┌────▼─────┐          ┌────▼─────┐
    │ MaxPool  │          │ MaxPool  │
    │ Global   │          │ Global   │
    └────┬─────┘          └────┬─────┘
         │                      │
         │    [64 features]     │
         └──────────┬───────────┘
                    │
            Concatenation
                    ↓
           [128 features]  ← EXTRACTED HERE
                    ↓
         ┌──────────────────┐
         │  Linear Layer    │
         │  128 → 2         │
         └──────────────────┘
                    ↓
         [Legitimate, Phishing]
```

### CNN Model Details

**File**: `model.py`

```python
class CNN_Text(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_class):
        # Embedding: vocab_size (96) → embedding_dim (128)
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        # 3-gram convolution: 128 input channels → 64 output
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3)

        # 5-gram convolution: 128 input channels → 64 output
        self.conv2 = nn.Conv1d(embedding_dim, 64, kernel_size=5)

        # Final classifier: 128 (64+64) → 2 classes
        self.fc = nn.Linear(128, num_class)
```

**Parameters**:
- **Vocabulary size**: 96 (alphanumeric + special chars)
- **Embedding dimension**: 128
- **Convolution filters**: 3-gram (64) + 5-gram (64)
- **Output dimension**: 128 (for feature extraction) or 2 (for classification)

### Character Vocabulary (vocab.py)

**All supported characters**:
```python
a-z, A-Z, 0-9, space, comma, semicolon, period, !, ?, :, /, \, |, @, #, $, %, ^, &, *, ~, `, +, -, =, <, >, (, ), [, ], {, }, ', "
```

**Special tokens**:
- `<unk>`: Unknown characters (out-of-vocabulary)

### CNN Training (train.py)

**Configuration**:
```python
Optimizer: Adam
Learning rate: 1e-6 (very small for stability)
Batch size: 256
Max epochs: 50
Early stopping: patience=50
Loss function: NLLLoss (Negative Log Likelihood)
```

**Data split**:
- Training: 70%
- Validation: 10%
- Test: 20%

**Training workflow**:
```
1. Load URLs from text files
2. Convert URLs to character n-grams
3. Build vocabulary (96 unique tokens)
4. Create DataLoader with batching
5. Train with Adam optimizer
6. Validate after each epoch
7. Early stop if validation loss doesn't improve
8. Save best model to model_train/model-cc_1.pkl
```

### CNN Feature Extraction (CNN_process.py)

**Purpose**: Extract the 128-dimensional learned features (before final classification layer)

```python
def get_part_feature(urls, model):
    # Process URLs through CNN
    # Extract features from the concatenated conv output
    # Return 128-dimensional vectors
```

**Output**:
- `Phishing_url_data_cnn.csv`: 573,963 × 128
- `Legitimate_url_data_cnn.csv`: 584,909 × 128

---

## Traditional ML Channel

### Feature Categories

BiPhish extracts **54+ handcrafted features** across three categories.

#### 1. URL Features (27 features)

**File**: `feature_extractor.py` → `extract_features_url()`

| Feature | Description | Example |
|---------|-------------|---------|
| `URL_length` | Total URL length | 45 |
| `URL_subdomains` | Number of subdomains | 2 (for a.b.example.com) |
| `URL_hasDash` | Contains dash in domain | 1 or 0 |
| `URL_hasDot` | Dots in URL | 5 |
| `URL_hasAt` | Contains @ symbol | 0 |
| `URL_hasDoubleSlash` | // in path | 0 |
| `URL_hasHyphen` | Contains hyphen | 1 |
| `URL_hasNumericChars` | Numeric characters count | 3 |
| `URL_hasDataURI` | Is data: URI | 0 |
| `URL_hasPunycode` | Uses punycode (IDN) | 0 |
| `URL_hasPathExtension` | File extension in path | 1 (.html) |
| `URL_totalWordUrl` | Total words in URL | 5 |
| `URL_totalWordHost` | Total words in hostname | 2 |
| `URL_totalWordPath` | Total words in path | 3 |
| `URL_shortestWordHost` | Shortest word length | 2 |
| `URL_shortestWordPath` | Shortest word in path | 1 |
| `URL_longestWordUrl` | Longest word in URL | 12 |
| `URL_longestWordHost` | Longest word in host | 8 |
| `URL_longestWordPath` | Longest word in path | 7 |
| `URL_averageWordUrl` | Average word length | 4.5 |
| `URL_averageWordHost` | Average in hostname | 5.2 |
| `URL_averageWordPath` | Average in path | 3.8 |
| `URL_hasFakeHTTPS` | HTTPS in subdomain | 0 |
| `URL_hasShortener` | Is URL shortener | 0 |
| `URL_hasRedirect` | Contains redirect | 0 |
| `URL_hasCommonTerm` | Common phishing terms | 1 |
| `URL_hasSensitive` | Bank/payment keywords | 1 |

**Relevance Categories**:
1. **DGA Detection**: Numeric chars, punycode, word patterns
2. **Structure Analysis**: Length, subdomains, special chars
3. **Obfuscation Detection**: Data URI, fake HTTPS, redirects
4. **Brand Impersonation**: Sensitive keywords, common terms

#### 2. HTML Features (18 features)

**File**: `feature_extractor.py` → `extract_features_html()`

Requires fetching and parsing the webpage.

| Feature | Description |
|---------|-------------|
| `HTML_externalObjects` | Ratio of external resources |
| `HTML_externalMeta` | External meta tags |
| `HTML_externalScripts` | External JavaScript count |
| `HTML_externalLinks` | External link ratio |
| `HTML_formHandler` | Suspicious form action |
| `HTML_hasPopup` | PopUp window detection |
| `HTML_hasRightClick` | Right-click disabled |
| `HTML_hasCopyright` | Copyright symbol present |
| `HTML_nullLinks` | Links to # or void |
| `HTML_hasFavicon` | Favicon present |
| `HTML_hasStatusBar` | Status bar manipulation |
| `HTML_cssExternalLinks` | External CSS count |
| `HTML_hasIframe` | Iframe present |
| `HTML_hiddenElements` | Hidden form fields |
| ... | (18 total) |

#### 3. Reputation Features (9 features)

**File**: `feature_extractor.py` → `extract_features_rep()`

**Warning**: These are slow (network I/O intensive).

| Feature | Description | Data Source |
|---------|-------------|-------------|
| `REP_googleIndex` | Indexed by Google | Google Search API |
| `REP_pageRank` | Google PageRank | PageRank API |
| `REP_whoisData` | WHOIS record exists | WHOIS lookup |
| `REP_dnsRecord` | DNS record valid | DNS query |
| `REP_domainAge` | Domain age (days) | WHOIS creation date |
| `REP_registrationLength` | Registration duration | WHOIS expiry |
| `REP_sslValid` | SSL certificate valid | SSL handshake |
| `REP_abnormalWHOIS` | WHOIS anomalies | WHOIS analysis |
| `REP_suspiciousPort` | Non-standard port | URL parsing |

**Timeouts**:
- WHOIS: 10 seconds
- SSL: 3 seconds
- HTTP: 5 seconds

---

## Feature Fusion

### Feature Selection

BiPhish uses **138 features** in the final model:

```
128 CNN features (all)
+
10 selected URL features (from 27)
───────────────────────────
138 total features
```

### Selected URL Features (final_train.py)

Only **10 of 27** URL features are used:

```python
selected_features = [
    'URL_length',
    'URL_subdomains',
    'URL_totalWordUrl',
    'URL_shortestWordPath',
    'URL_longestWordUrl',
    'URL_longestWordHost',
    'URL_longestWordPath',
    'URL_averageWordUrl',
    'URL_averageWordHost',
    'URL_averageWordPath'
]
```

**Why only 10?**
- Feature ablation study showed these 10 provide best performance
- Reduces dimensionality and overfitting
- Focuses on word-based statistics (strong phishing indicators)

### Feature Preprocessing

**File**: `final_train.py`

```python
# 1. Handle missing values
features[features == np.inf] = 1e10      # Inf → large number
features[features == -np.inf] = -1e10    # -Inf → small number
features = np.nan_to_num(features, 0.0)  # NaN → 0

# 2. Standardization (Z-score normalization)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Save scaler for inference!
joblib.dump(scaler, 'scaler.pkl')
```

**Critical**: The scaler **must** be saved and reused during inference!

---

## Ensemble Classifier

### Voting Ensemble Architecture

BiPhish uses **soft voting** with 5 base classifiers:

```
┌────────────────────────────────────────────────────┐
│              Voting Classifier (Soft)              │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │
│  │ Logistic    │  │ Naive       │  │ Linear   │  │
│  │ Regression  │  │ Bayes       │  │ SVM      │  │
│  │ (weight=1)  │  │ (weight=1)  │  │ (w=2)    │  │
│  └──────┬──────┘  └──────┬──────┘  └────┬─────┘  │
│         │                │               │        │
│  ┌──────▼──────┐  ┌──────▼──────┐               │
│  │ Decision    │  │ Random      │               │
│  │ Tree        │  │ Forest      │               │
│  │ (weight=1)  │  │ (w=2)       │               │
│  └──────┬──────┘  └──────┬──────┘               │
│         │                │                       │
│         └────────┬───────┴───────────────────┘  │
│                  │                               │
│          Weighted Average                        │
│          of Probabilities                        │
│                  │                               │
│         ┌────────▼────────┐                      │
│         │ Final Prediction │                      │
│         └─────────────────┘                      │
└────────────────────────────────────────────────────┘
```

### Base Classifiers

| Classifier | Configuration | Weight | Rationale |
|------------|---------------|--------|-----------|
| **Logistic Regression** | solver='saga', n_jobs=-1 | 1 | Linear baseline, fast |
| **Naive Bayes** | GaussianNB() | 1 | Probabilistic, handles independence |
| **Linear SVM** | CalibratedClassifierCV | **2** | Strong linear separator |
| **Decision Tree** | default | 1 | Non-linear, interpretable |
| **Random Forest** | 100 trees, n_jobs=-1 | **2** | Ensemble, robust |

**Voting Method**: Soft voting (average predicted probabilities)

**Formula**:
```
P(class|x) = Σ(weight_i × P_i(class|x)) / Σ(weight_i)
```

### Why These Weights?

```python
weights = [1, 1, 2, 1, 2]
           ↑  ↑  ↑  ↑  ↑
           LR NB SVM DT RF
```

- **SVM (2×)**: Strong performance on high-dimensional data
- **Random Forest (2×)**: Robust to noise, handles non-linearity
- **Others (1×)**: Provide diversity to ensemble

---

## Data Flow

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: Training                        │
└─────────────────────────────────────────────────────────────┘

1. Raw Data
   ├─ data/phish-58w.txt        (573,963 URLs)
   └─ data/legtimate-58w.txt    (584,909 URLs)
                │
                ▼
2. CNN Training (train.py)
   ├─ Convert URLs → character sequences (max 100)
   ├─ Build vocabulary (96 tokens)
   ├─ Train CNN (50 epochs, early stopping)
   └─ Save: model_train/model-cc_1.pkl
                │
                ▼
3. CNN Feature Extraction (CNN_process.py)
   ├─ Load trained CNN
   ├─ Process all URLs through CNN
   ├─ Extract 128-dim features (concat layer output)
   └─ Save: Phishing_url_data_cnn.csv, Legitimate_url_data_cnn.csv
                │
                ▼
4. Handcrafted Feature Extraction (feature_extractor.py)
   ├─ Extract URL features (27)
   ├─ Extract HTML features (18) [optional]
   ├─ Extract reputation features (9) [optional]
   └─ Save: Phishing_url_data_art.csv, Legitimate_url_data_art.csv
                │
                ▼
5. Ensemble Training (final_train.py)
   ├─ Load CNN features (128)
   ├─ Load handcrafted features (select 10/27)
   ├─ Combine: 128 + 10 = 138 features
   ├─ Handle NaN/Inf, scale features
   ├─ Train 5 base classifiers
   ├─ Create voting ensemble
   └─ Save: voting_classifier.pkl, scaler.pkl, etc.

┌─────────────────────────────────────────────────────────────┐
│                  STAGE 2: Inference                         │
└─────────────────────────────────────────────────────────────┘

Input URL
    │
    ├─► CNN Feature Extraction
    │   ├─ Tokenize to characters
    │   ├─ Convert to IDs (vocab)
    │   ├─ Pass through CNN
    │   └─ Extract 128 features
    │
    └─► Handcrafted Feature Extraction
        └─ Extract 10 selected URL features
                │
                ▼
        Combine Features (138)
                │
                ▼
        Handle NaN/Inf
                │
                ▼
        Scale with scaler.pkl
                │
                ▼
        Predict with ensemble
                │
                ▼
        [Legitimate | Phishing]
```

---

## Module Reference

### Core Modules

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `model.py` | CNN architecture | `CNN_Text` |
| `vocab.py` | Character vocabulary | `Vocab.build()`, `convert_tokens_to_ids()` |
| `utils.py` | Data loading, preprocessing | `load_sentence_polarity()`, `get_char_ngrams()`, `EarlyStopping` |
| `train.py` | CNN training | Main training loop |
| `CNN_process.py` | CNN feature extraction | `get_part_feature()` |
| `feature_extractor.py` | Handcrafted features | `extract_features_url()`, `extract_features_html()`, `extract_features_rep()` |
| `final_train.py` | Ensemble training | Feature fusion, voting classifier |
| `final_train_revised.py` | Enhanced evaluation | SOTA comparison, TPR/FPR analysis |
| `test.py` | Model evaluation | Testing pipeline |

### Key Data Structures

**CNN Input**:
```python
{
    'url_chars': List[str],       # ['h', 't', 't', 'p', ...]
    'url_ids': List[int],          # [23, 45, 45, 41, ...]
    'label': int                   # 0=Legit, 1=Phishing
}
```

**CNN Output (features)**:
```python
shape: [batch_size, 128]
dtype: float32
```

**Handcrafted Features**:
```python
{
    'URL_length': float,
    'URL_subdomains': int,
    ...,  # 27 URL features
    'HTML_externalObjects': float,
    ...,  # 18 HTML features (optional)
    'REP_pageRank': float,
    ...   # 9 reputation features (optional)
}
```

**Final Feature Vector**:
```python
shape: [138]  # 128 CNN + 10 selected
dtype: float64 (after scaling)
```

---

## Design Decisions

### Why CNN for URLs?

**Advantages**:
1. **Character-level analysis**: Captures obfuscation, typosquatting
2. **DGA detection**: Learns patterns in algorithmically generated domains
3. **No feature engineering**: Automatically discovers patterns
4. **Transfer learning**: Pre-trained on large datasets

**Architecture choices**:
- **3-gram + 5-gram filters**: Capture short and medium-range patterns
- **Max pooling**: Position-invariant features
- **Embedding dim 128**: Balance between capacity and efficiency

### Why Only 10 Handcrafted Features?

**Feature selection study** (in `final_train_revised.py`) showed:
1. **Diminishing returns**: Beyond 10 features, marginal improvement
2. **Overfitting risk**: More features ≈ more noise
3. **Efficiency**: Faster inference, less feature extraction time
4. **Word statistics**: Most discriminative features are word-based

### Why Ensemble Over Single Model?

| Approach | Accuracy | Robustness | Interpretability |
|----------|----------|------------|------------------|
| CNN only | 94% | Medium | Low |
| Handcrafted only | 91% | High | High |
| **Ensemble (BiPhish)** | **97.8%** | **High** | **Medium** |

**Benefits of ensemble**:
1. **Diversity**: Different models capture different patterns
2. **Robustness**: Less sensitive to outliers
3. **Confidence**: Soft voting provides calibrated probabilities

### Why These Specific Classifiers?

- **Logistic Regression**: Fast, linear baseline
- **Naive Bayes**: Handles probabilistic independence
- **SVM**: Strong margin-based classifier
- **Decision Tree**: Captures non-linear interactions
- **Random Forest**: Ensemble of trees, robust

**Complementary strengths**:
- Linear: LR, SVM
- Non-linear: DT, RF
- Probabilistic: NB, LR

---

## Performance Characteristics

### Computational Complexity

| Stage | Time | Memory | Bottleneck |
|-------|------|--------|------------|
| CNN Training | 10-30 min | 2-4 GB | GPU/CPU |
| CNN Extraction | 5-15 min | 1-2 GB | I/O |
| URL Features | 5-10 min | < 1 GB | CPU |
| HTML Features | 30-60 min | 1-2 GB | Network I/O |
| Reputation Features | 2-4 hours | < 1 GB | Network I/O |
| Ensemble Training | 5-10 min | 2-4 GB | CPU |
| **Inference (per URL)** | **< 100ms** | **< 100 MB** | **Feature extraction** |

### Scalability

**Training**:
- Parallelizable: CNN (GPU), ensemble (multi-core)
- Memory: O(dataset_size × feature_dim)
- Disk: O(dataset_size)

**Inference**:
- Batch-friendly: Process multiple URLs in parallel
- Cacheable: CNN features can be cached
- Stateless: No session state required

---

## Future Enhancements

1. **LSTM/Transformer**: Sequential modeling of URL structure
2. **Graph Neural Networks**: Model URL components as graphs
3. **Transfer Learning**: Pre-train on larger corpora
4. **Active Learning**: Continuously improve with user feedback
5. **Multi-task Learning**: Joint training on related tasks
6. **Model Compression**: Quantization, pruning for deployment
7. **Explainability**: LIME, SHAP for feature attribution

---

## References

- **Dataset**: SOTAPhish-1.2M
- **Framework**: PyTorch 1.9+, scikit-learn 0.24+
- **Methodology**: Dual-channel ensemble learning

For more details, see the research paper (contact authors for citation).
