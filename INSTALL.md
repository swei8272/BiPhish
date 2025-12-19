# Installation Guide for BiPhish

This guide provides detailed instructions for installing and setting up BiPhish on your system.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Post-Installation Setup](#post-installation-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Operating System**: Linux, macOS, or Windows 10+
- **Python**: 3.7 or higher
- **RAM**: 8 GB minimum (16 GB recommended for large datasets)
- **Disk Space**: 5 GB for software + space for datasets and models
- **Internet**: Required for dependency installation and some feature extraction (WHOIS, SSL)

### Recommended Setup

- **Python**: 3.8 or 3.9
- **RAM**: 16 GB or more
- **CPU**: Multi-core processor (for parallel processing)
- **GPU**: CUDA-compatible GPU (optional, for faster CNN training)

## Installation Methods

### Method 1: Using pip (Recommended)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/BiPhish.git
cd BiPhish
```

#### Step 2: Create Virtual Environment (Recommended)

**Using venv (Python built-in):**

```bash
# Create virtual environment
python3 -m venv biphish_env

# Activate virtual environment
# On Linux/macOS:
source biphish_env/bin/activate

# On Windows:
biphish_env\Scripts\activate
```

**Using conda:**

```bash
# Create conda environment
conda create -n biphish python=3.8

# Activate environment
conda activate biphish
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

#### Step 4: Install PyTorch

BiPhish requires PyTorch. Install the appropriate version for your system:

**For CPU-only systems:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**For systems with CUDA GPU (recommended for faster training):**

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for more installation options.

#### Step 5: Download NLTK Data

```bash
python -c "import nltk; nltk.download('treebank')"
```

### Method 2: Using conda (Alternative)

```bash
# Clone repository
git clone https://github.com/yourusername/BiPhish.git
cd BiPhish

# Create conda environment from scratch
conda create -n biphish python=3.8

# Activate environment
conda activate biphish

# Install PyTorch via conda
conda install pytorch torchvision torchaudio -c pytorch

# Install other dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('treebank')"
```

### Method 3: Docker (Coming Soon)

A Docker container for BiPhish is in development.

## Post-Installation Setup

### 1. Create Data Directory

```bash
mkdir -p data
```

### 2. Obtain Dataset

The BiPhish dataset is not publicly available in this repository. To obtain it:

1. Contact the authors via email (see README.md)
2. Request access to the SOTAPhish-1.2M dataset
3. Once received, place the files in the `data/` directory:
   - `data/phish-58w.txt`
   - `data/legtimate-58w.txt`

### 3. Create Model Directory

```bash
mkdir -p model_train
```

This directory will store trained CNN models.

### 4. Optional: Install Intel scikit-learn Extensions (For Performance)

If you have an Intel CPU, you can install Intel's optimized scikit-learn:

```bash
pip install scikit-learn-intelex
```

The code will automatically detect and use this if available.

### 5. Configure Feature Extraction (Optional)

Some features require external services or API keys:

- **Google PageRank**: Requires API setup (see `feature_extractor.py`)
- **WHOIS Lookups**: No setup required, but requires internet connectivity
- **SSL Certificate Validation**: No setup required, but requires internet connectivity

## Verification

### Verify Python Version

```bash
python --version
# Should show Python 3.7.x or higher
```

### Verify Package Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
python -c "import pandas; print(f'pandas version: {pandas.__version__}')"
python -c "import nltk; print(f'NLTK version: {nltk.__version__}')"
```

### Verify CUDA (Optional, for GPU users)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

### Run Quick Test

```bash
# Test model definition
python -c "from model import CNN_Text; print('Model definition OK')"

# Test vocabulary
python -c "from vocab import Vocab; print('Vocabulary module OK')"

# Test utilities
python -c "from utils import get_char_ngrams; print('Utils module OK')"
```

## Directory Structure After Installation

```
BiPhish/
├── biphish_env/              # Virtual environment (if using venv)
├── data/                     # Dataset directory (user-created)
│   ├── phish-58w.txt         # Phishing URLs (user-provided)
│   └── legtimate-58w.txt     # Legitimate URLs (user-provided)
├── model_train/              # Model storage (user-created)
├── *.py                      # Python source files
├── vocab.txt                 # Vocabulary file
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
pip install torch
```

### Issue: "SSL: CERTIFICATE_VERIFY_FAILED"

**Solution (macOS):**
```bash
# Run the certificate installer
/Applications/Python\ 3.x/Install\ Certificates.command
```

**Solution (General):**
```bash
pip install --upgrade certifi
```

### Issue: NLTK Data Not Found

**Solution:**
```bash
import nltk
nltk.download('treebank')
```

Or download manually:
```bash
python -m nltk.downloader treebank
```

### Issue: Memory Error During Training

**Solutions:**
1. Reduce batch size in `train.py`:
   ```python
   batch_size = 128  # Instead of 256
   ```

2. Use a machine with more RAM

3. Process smaller subsets of data

### Issue: WHOIS Lookup Failures

**Common Causes:**
- No internet connection
- Domain doesn't exist
- WHOIS server timeout
- Rate limiting by WHOIS servers

**Solutions:**
- Ensure internet connectivity
- Add retry logic (already implemented in code)
- Reduce batch size for WHOIS queries
- Some failures are expected and handled gracefully

### Issue: CUDA Out of Memory

**Solutions:**
1. Reduce batch size
2. Use CPU instead of GPU:
   ```python
   device = 'cpu'  # In train.py
   ```
3. Clear GPU cache:
   ```python
   torch.cuda.empty_cache()
   ```

### Issue: ImportError with BeautifulSoup

**Solution:**
```bash
pip install beautifulsoup4 lxml html5lib
```

### Issue: scikit-learn-intelex Warning

**Note:** This is optional. If you see warnings related to `scikit-learn-intelex`, you can safely ignore them or uninstall:
```bash
pip uninstall scikit-learn-intelex
```

The code will fall back to standard scikit-learn.

## Platform-Specific Notes

### Linux

- No special requirements
- Recommended for production use

### macOS

- May need to install Xcode Command Line Tools:
  ```bash
  xcode-select --install
  ```
- Certificate issues may require running the Python certificate installer

### Windows

- Use Anaconda or Miniconda for easier package management
- Some WHOIS lookups may behave differently
- Use Command Prompt or PowerShell (not Git Bash) for activation commands

## Next Steps

After successful installation:

1. Read [USAGE.md](USAGE.md) for usage instructions
2. Read [DATA_FORMAT.md](DATA_FORMAT.md) for data format specifications
3. Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system architecture
4. Run the training pipeline (see USAGE.md)

## Getting Help

If you encounter issues not covered here:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing GitHub issues
3. Create a new GitHub issue with:
   - Your operating system and version
   - Python version
   - Complete error message
   - Steps to reproduce

## Updating BiPhish

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
```

## Uninstallation

To remove BiPhish:

```bash
# Deactivate virtual environment
deactivate  # or: conda deactivate

# Remove virtual environment
rm -rf biphish_env  # for venv
# or
conda env remove -n biphish  # for conda

# Remove repository
cd ..
rm -rf BiPhish
```
