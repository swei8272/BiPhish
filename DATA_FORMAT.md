# Data Format Specification

This document describes all data formats used in BiPhish, including input files, intermediate outputs, and model files.

## Table of Contents

- [Input Data Formats](#input-data-formats)
- [Intermediate Data Formats](#intermediate-data-formats)
- [Model Files](#model-files)
- [Configuration Files](#configuration-files)
- [Data Validation](#data-validation)
- [Examples](#examples)

---

## Input Data Formats

### 1. URL Text Files

**Purpose**: Raw phishing and legitimate URLs for training

**Files**:
- `data/phish-58w.txt` - Phishing URLs
- `data/legtimate-58w.txt` - Legitimate URLs

**Format**: Plain text, one URL per line

```
http://example.com
https://suspicious-paypal-verify.com/login
http://192.168.1.1/admin
https://secure.bankofamerica.com.fake-domain.com/
...
```

**Specifications**:
- **Encoding**: UTF-8
- **Line endings**: LF (`\n`) or CRLF (`\r\n`)
- **Empty lines**: Allowed (will be skipped)
- **Comments**: Not supported
- **URL encoding**: Both raw and percent-encoded URLs accepted
- **Maximum length**: 2048 characters (longer URLs will be truncated)
- **Protocols**: http, https, ftp, data, etc.

**Validation Rules**:
1. Each line must be a valid URL or empty
2. URLs should include protocol (http://, https://, etc.)
3. No header row
4. No duplicate URLs (recommended but not enforced)

**Example**:
```text
http://google.com
https://github.com/user/repo
http://phishing-site-example.com/fake-login
https://amazon.com
http://192.168.0.1:8080/admin

http://legitimate-site.org
```

### 2. Vocabulary File

**Purpose**: Pre-built character vocabulary for CNN

**File**: `vocab.txt`

**Format**: Plain text, one token per line

```
<unk>
a
b
c
...
Z
0
1
...
9
!
@
#
...
```

**Specifications**:
- **Encoding**: UTF-8
- **Line 1**: `<unk>` (unknown token, required)
- **Lines 2+**: Individual characters
- **Total tokens**: 96 (including `<unk>`)

**Character Set**:
- Lowercase: a-z
- Uppercase: A-Z
- Digits: 0-9
- Special: ` ` (space), `,`, `;`, `.`, `!`, `?`, `:`, `/`, `\`, `|`, `@`, `#`, `$`, `%`, `^`, `&`, `*`, `~`, `` ` ``, `+`, `-`, `=`, `<`, `>`, `(`, `)`, `[`, `]`, `{`, `}`, `'`, `"`

---

## Intermediate Data Formats

### 1. CNN Feature CSV Files

**Purpose**: Store CNN-extracted features for each URL

**Files**:
- `Phishing_url_data_cnn.csv` - Phishing URL features
- `Legitimate_url_data_cnn.csv` - Legitimate URL features

**Format**: CSV (Comma-Separated Values)

**Structure**:
```csv
feature_0,feature_1,feature_2,...,feature_127
0.234,0.567,-0.123,...,0.891
-0.456,0.789,0.321,...,-0.234
...
```

**Specifications**:
- **Header row**: `feature_0` through `feature_127` (128 columns)
- **Rows**: One per URL (matches input file order)
- **Columns**: 128 (CNN output dimension)
- **Data type**: Float (float32 or float64)
- **Missing values**: Not allowed (all features must be present)
- **Encoding**: UTF-8
- **Delimiter**: Comma (`,`)

**Example**:
```csv
feature_0,feature_1,feature_2,feature_3,feature_4,feature_5
0.123,0.456,-0.789,0.234,0.567,-0.890
-0.234,0.567,0.123,-0.456,0.789,0.321
0.345,-0.678,0.901,0.234,-0.567,0.890
```

**Dimensions**:
- Phishing: 573,963 rows × 128 columns
- Legitimate: 584,909 rows × 128 columns

### 2. Handcrafted Feature CSV Files

**Purpose**: Store manually engineered features for each URL

**Files**:
- `Phishing_url_data_art.csv` - Phishing URL features
- `Legitimate_url_data_art.csv` - Legitimate URL features

**Format**: CSV with named columns

**Structure**:
```csv
URL_length,URL_subdomains,URL_hasDash,...,REP_pageRank
45,2,1,...,4.5
123,0,0,...,6.2
...
```

**URL Features (27 columns)**:
```
URL_length                  (int)
URL_subdomains             (int)
URL_hasDash                (0/1)
URL_hasDot                 (int)
URL_hasAt                  (0/1)
URL_hasDoubleSlash         (0/1)
URL_hasHyphen              (0/1)
URL_hasNumericChars        (int)
URL_hasDataURI             (0/1)
URL_hasPunycode            (0/1)
URL_hasPathExtension       (0/1)
URL_totalWordUrl           (int)
URL_totalWordHost          (int)
URL_totalWordPath          (int)
URL_shortestWordHost       (int)
URL_shortestWordPath       (int)
URL_longestWordUrl         (int)
URL_longestWordHost        (int)
URL_longestWordPath        (int)
URL_averageWordUrl         (float)
URL_averageWordHost        (float)
URL_averageWordPath        (float)
URL_hasFakeHTTPS           (0/1)
URL_hasShortener           (0/1)
URL_hasRedirect            (0/1)
URL_hasCommonTerm          (0/1)
URL_hasSensitive           (0/1)
```

**HTML Features (18 columns)** [Optional]:
```
HTML_externalObjects       (float, 0-1)
HTML_externalMeta          (int)
HTML_externalScripts       (int)
HTML_externalLinks         (float, 0-1)
HTML_formHandler           (0/1/-1)
HTML_hasPopup              (0/1)
HTML_hasRightClick         (0/1)
HTML_hasCopyright          (0/1)
HTML_nullLinks             (int)
HTML_hasFavicon            (0/1)
HTML_hasStatusBar          (0/1)
HTML_cssExternalLinks      (int)
HTML_hasIframe             (0/1)
HTML_hiddenElements        (int)
...
```

**Reputation Features (9 columns)** [Optional]:
```
REP_googleIndex            (0/1)
REP_pageRank               (float, 0-10)
REP_whoisData              (0/1)
REP_dnsRecord              (0/1)
REP_domainAge              (int, days)
REP_registrationLength     (int, days)
REP_sslValid               (0/1/-1)
REP_abnormalWHOIS          (0/1)
REP_suspiciousPort         (0/1)
```

**Specifications**:
- **Header row**: Required (feature names)
- **Rows**: One per URL
- **Data types**: int, float, binary (0/1)
- **Missing values**: NaN allowed (will be replaced with 0)
- **Infinite values**: May occur (will be capped at ±1e10)
- **Encoding**: UTF-8
- **Delimiter**: Comma

**Example**:
```csv
URL_length,URL_subdomains,URL_hasDash,URL_longestWordUrl,URL_averageWordUrl
45,2,1,12,5.6
123,0,0,8,4.2
67,3,1,15,6.8
```

### 3. Combined Feature Matrix

**Purpose**: Final feature matrix for ensemble training

**Created in**: `final_train.py` (in-memory, not saved)

**Structure**:
```
[CNN features (128) | Selected URL features (10)]
```

**Dimensions**: N × 138
- N = 1,158,872 (573,963 phishing + 584,909 legitimate)
- 138 = 128 CNN + 10 selected handcrafted

**Selected Features** (10 from 27):
```python
[
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

**Preprocessing**:
1. NaN → 0
2. +Inf → 1e10
3. -Inf → -1e10
4. StandardScaler (Z-score normalization)

**Example Row** (after scaling):
```python
array([
    # CNN features (128)
    0.234, 0.567, -0.123, ..., 0.891,
    # Selected URL features (10)
    1.23,  # URL_length (scaled)
    0.45,  # URL_subdomains (scaled)
    -0.67, # URL_totalWordUrl (scaled)
    ...
])
```

---

## Model Files

### 1. CNN Model

**File**: `model_train/model-cc_1.pkl`

**Format**: PyTorch pickle (`.pkl`)

**Contents**:
```python
{
    'model_state_dict': OrderedDict(...),  # Model weights
    'optimizer_state_dict': OrderedDict(...),  # Optimizer state (optional)
    'epoch': int,  # Training epoch
    'loss': float,  # Best validation loss
    'vocab': Vocab object,  # Vocabulary
    ...
}
```

**Loading**:
```python
import torch
from model import CNN_Text

# Load checkpoint
checkpoint = torch.load('model_train/model-cc_1.pkl')

# Create model
model = CNN_Text(vocab_size=96, embedding_dim=128, num_class=2)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

**Specifications**:
- **Size**: ~2-5 MB
- **Format**: PyTorch serialized state dict
- **Python version**: Should match training environment
- **PyTorch version**: 1.9+ recommended

### 2. Feature Scaler

**File**: `scaler.pkl`

**Format**: Joblib pickle

**Contents**: sklearn StandardScaler object

**Loading**:
```python
import joblib

scaler = joblib.load('scaler.pkl')

# Use for inference
features_scaled = scaler.transform(features)
```

**Critical**: This scaler **must** be used during inference to match training preprocessing!

**Scaler State**:
```python
{
    'mean_': array([...]),      # 138 means
    'var_': array([...]),       # 138 variances
    'scale_': array([...]),     # 138 std deviations
    'n_features_in_': 138,
    'n_samples_seen_': 926297
}
```

### 3. Ensemble Model

**File**: `voting_classifier.pkl`

**Format**: Joblib pickle

**Contents**: sklearn VotingClassifier with 5 fitted base estimators

**Loading**:
```python
import joblib

voting_clf = joblib.load('voting_classifier.pkl')

# Predict
predictions = voting_clf.predict(features_scaled)
probabilities = voting_clf.predict_proba(features_scaled)
```

**Structure**:
```python
VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(...)),
        ('nb', GaussianNB(...)),
        ('svm', CalibratedClassifierCV(...)),
        ('dt', DecisionTreeClassifier(...)),
        ('rf', RandomForestClassifier(...))
    ],
    voting='soft',
    weights=[1, 1, 2, 1, 2]
)
```

**Size**: ~50-200 MB (varies with Random Forest size)

### 4. Individual Base Models

**Files**:
- `lr_model.pkl` - Logistic Regression
- `nb_model.pkl` - Naive Bayes
- `svm_model.pkl` - Linear SVM
- `dt_model.pkl` - Decision Tree
- `rf_model.pkl` - Random Forest

**Format**: Joblib pickle

**Usage**: Typically not used directly; ensemble uses them internally

---

## Configuration Files

### 1. requirements.txt

**Purpose**: Python dependencies

**Format**: pip requirements format

```
package>=version
package==exact_version
package  # any version
```

**Example**:
```
torch>=1.9.0
scikit-learn>=0.24.0
pandas>=1.3.0
```

### 2. Code Configuration

Configuration is embedded in Python files:

**utils.py**:
```python
Phishing_url_data_path = r"./data/phish-58w.txt"
Legitimate_url_data_path = r"./data/legtimate-58w.txt"
train_prop = 0.7
val_prop = 0.1
test_prop = 0.2
```

**train.py**:
```python
num_epoch = 50
embedding_dim = 128
batch_size = 256
learning_rate = 1e-6
```

---

## Data Validation

### Input URL Validation

```python
import re
from urllib.parse import urlparse

def validate_url(url):
    """Validate URL format"""
    try:
        result = urlparse(url)
        # Must have scheme and netloc
        return all([result.scheme, result.netloc])
    except:
        return False

# Usage
url = "http://example.com"
if validate_url(url):
    print("Valid URL")
```

### Feature Validation

```python
import numpy as np
import pandas as pd

def validate_features(df, expected_cols, expected_rows=None):
    """Validate feature DataFrame"""
    # Check columns
    assert df.shape[1] == expected_cols, f"Expected {expected_cols} columns, got {df.shape[1]}"

    # Check rows
    if expected_rows:
        assert df.shape[0] == expected_rows, f"Expected {expected_rows} rows, got {df.shape[0]}"

    # Check for all-NaN rows
    assert not df.isnull().all(axis=1).any(), "Found all-NaN rows"

    print(f"✓ Validation passed: {df.shape[0]} rows × {df.shape[1]} columns")

# Usage
df_cnn = pd.read_csv('Phishing_url_data_cnn.csv')
validate_features(df_cnn, expected_cols=128)
```

### Model Validation

```python
def validate_model_files():
    """Check all required model files exist"""
    import os

    required_files = [
        'model_train/model-cc_1.pkl',
        'voting_classifier.pkl',
        'scaler.pkl'
    ]

    for filepath in required_files:
        assert os.path.exists(filepath), f"Missing: {filepath}"

    print("✓ All model files present")
```

---

## Examples

### Example 1: Creating Input Files

```python
# Create sample input files
phishing_urls = [
    "http://paypal-verify.com/login",
    "http://secure.bankofamerica.fake.com",
    "http://192.168.1.1/admin"
]

legitimate_urls = [
    "http://google.com",
    "https://github.com",
    "https://wikipedia.org"
]

# Write to files
with open('data/phish-58w.txt', 'w') as f:
    f.write('\n'.join(phishing_urls))

with open('data/legtimate-58w.txt', 'w') as f:
    f.write('\n'.join(legitimate_urls))
```

### Example 2: Reading CNN Features

```python
import pandas as pd
import numpy as np

# Read CNN features
df_phish = pd.read_csv('Phishing_url_data_cnn.csv')
df_legit = pd.read_csv('Legitimate_url_data_cnn.csv')

print(f"Phishing: {df_phish.shape}")  # (573963, 128)
print(f"Legitimate: {df_legit.shape}")  # (584909, 128)

# Convert to numpy array
features = df_phish.values  # shape: (573963, 128)
```

### Example 3: Reading Handcrafted Features

```python
import pandas as pd

# Read with specific columns
selected_cols = [
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

df_art = pd.read_csv('Phishing_url_data_art.csv', usecols=selected_cols)
print(df_art.head())
```

### Example 4: Combining Features

```python
import pandas as pd
import numpy as np

# Read both feature sets
df_cnn = pd.read_csv('Phishing_url_data_cnn.csv')
df_art = pd.read_csv('Phishing_url_data_art.csv', usecols=selected_cols)

# Combine horizontally
df_combined = pd.concat([df_cnn, df_art], axis=1)

print(f"Combined shape: {df_combined.shape}")  # (N, 138)

# Add label
df_combined['label'] = 1  # 1 = Phishing

# Handle missing values
df_combined = df_combined.fillna(0)
df_combined = df_combined.replace([np.inf, -np.inf], [1e10, -1e10])
```

### Example 5: Loading and Using Models

```python
import joblib
import numpy as np

# Load models
scaler = joblib.load('scaler.pkl')
voting_clf = joblib.load('voting_classifier.pkl')

# Prepare features (138-dimensional)
features = np.random.rand(1, 138)  # Example

# Scale
features_scaled = scaler.transform(features)

# Predict
prediction = voting_clf.predict(features_scaled)[0]
probability = voting_clf.predict_proba(features_scaled)[0]

print(f"Prediction: {'Phishing' if prediction == 1 else 'Legitimate'}")
print(f"Probability: {probability}")
```

---

## Data Directory Structure

```
BiPhish/
├── data/
│   ├── phish-58w.txt              # Input: Phishing URLs
│   └── legtimate-58w.txt          # Input: Legitimate URLs
├── model_train/
│   └── model-cc_1.pkl             # Trained CNN model
├── Phishing_url_data_cnn.csv      # CNN features (phishing)
├── Legitimate_url_data_cnn.csv    # CNN features (legitimate)
├── Phishing_url_data_art.csv      # Handcrafted features (phishing)
├── Legitimate_url_data_art.csv    # Handcrafted features (legitimate)
├── voting_classifier.pkl          # Ensemble model
├── scaler.pkl                     # Feature scaler
├── lr_model.pkl                   # Logistic Regression
├── nb_model.pkl                   # Naive Bayes
├── svm_model.pkl                  # Linear SVM
├── dt_model.pkl                   # Decision Tree
├── rf_model.pkl                   # Random Forest
└── vocab.txt                      # Character vocabulary
```

---

## Summary Table

| File Type | Format | Purpose | Size |
|-----------|--------|---------|------|
| `phish-58w.txt` | Text | Input phishing URLs | ~50 MB |
| `legtimate-58w.txt` | Text | Input legitimate URLs | ~50 MB |
| `vocab.txt` | Text | Character vocabulary | < 1 KB |
| `*_cnn.csv` | CSV | CNN features (128) | ~500 MB each |
| `*_art.csv` | CSV | Handcrafted features | ~50 MB each |
| `model-cc_1.pkl` | PyTorch | Trained CNN | ~3 MB |
| `voting_classifier.pkl` | Joblib | Ensemble model | ~100 MB |
| `scaler.pkl` | Joblib | Feature scaler | < 1 MB |
| `*_model.pkl` | Joblib | Individual classifiers | 5-50 MB each |

---

For questions about data formats, refer to:
- [USAGE.md](USAGE.md) for usage examples
- [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- [INSTALL.md](INSTALL.md) for setup instructions
