# BiPhish

A dual-channel phishing detection approach combining deep learning (CNN) with traditional machine learning for robust URL classification.

## Overview

BiPhish employs a novel dual-channel architecture that fuses:
- **Channel 1**: Deep learning features extracted from a Convolutional Neural Network (CNN) trained on URL character sequences
- **Channel 2**: Handcrafted features based on URL structure, HTML content, and domain reputation

The combined features are processed by an ensemble voting classifier to achieve high-accuracy phishing detection.

## Features

- **Dual-Channel Architecture**: Combines learned and engineered features
- **CNN-based Feature Learning**: Automatically learns patterns from URL character sequences
- **Comprehensive Feature Engineering**: 27+ URL features, 18+ HTML features, 9+ reputation features
- **Ensemble Classification**: Voting classifier with 5 base models (Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest)
- **Robust Performance**: Tested on SOTAPhish-1.2M dataset (573,963 phishing + 584,909 legitimate URLs)

## Quick Start

### Prerequisites

- Python 3.7 or higher
- Required packages (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BiPhish.git
cd BiPhish

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for feature extraction)
python -c "import nltk; nltk.download('treebank')"
```

### Data Preparation

1. Create a `data/` directory in the project root
2. Obtain the dataset by contacting the authors (see [Dataset Access](#dataset-access))
3. Place the following files in the `data/` directory:
   - `phish-58w.txt` - Phishing URLs (one per line)
   - `legtimate-58w.txt` - Legitimate URLs (one per line)

### Basic Usage

```bash
# Step 1: Train the CNN model
python train.py

# Step 2: Extract CNN features
python CNN_process.py

# Step 3: Extract handcrafted features (requires feature_extractor.py)
# See USAGE.md for detailed instructions

# Step 4: Train the ensemble classifier
python final_train.py

# Step 5: Test the model
python test.py
```

For detailed usage instructions, see [USAGE.md](USAGE.md).

## Architecture

BiPhish uses a two-stage pipeline:

```
Stage 1: Feature Extraction
├── CNN Channel: train.py → CNN_process.py → 128-dimensional learned features
└── Traditional Channel: feature_extractor.py → 27+ handcrafted features

Stage 2: Ensemble Classification
└── final_train.py → Voting Classifier (5 models) → Prediction
```

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Dataset Access

To obtain the dataset used in BiPhish, please send an email to the authors. The dataset includes:
- **Phishing URLs**: 573,963 samples
- **Legitimate URLs**: 584,909 samples
- **Source**: SOTAPhish-1.2M dataset

## Key Components

### Core Files

- `train.py` - Train the CNN model on URL character sequences
- `CNN_process.py` - Extract learned features from trained CNN
- `feature_extractor.py` - Extract handcrafted URL/HTML/reputation features
- `final_train.py` - Train ensemble classifier on combined features
- `final_train_revised.py` - Enhanced version with comprehensive evaluation
- `test.py` - Evaluate trained models
- `model.py` - CNN architecture definition
- `utils.py` - Data loading and preprocessing utilities
- `vocab.py` - Character vocabulary management

### Feature Categories

- **URL Features (27)**: Length, subdomains, special characters, obfuscation patterns, word statistics
- **HTML Features (18)**: External objects, forms, iframes, JavaScript behaviors
- **Reputation Features (9)**: WHOIS data, DNS records, SSL certificates, domain age
- **CNN Features (128)**: Automatically learned from character n-grams

For complete feature documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Model Performance

The ensemble model combines:
- Logistic Regression
- Naive Bayes
- Linear SVM (weighted: 2x)
- Decision Tree
- Random Forest (weighted: 2x)

Performance metrics and detailed evaluation can be found in `final_train_revised.py`.

## Extending BiPhish

To add custom features:
1. Open `feature_extractor.py`
2. Add your feature extraction function
3. Update the feature list in `final_train.py`

See the inline documentation for guidance.

## Project Structure

```
BiPhish/
├── data/                          # Dataset directory (user-provided)
│   ├── phish-58w.txt
│   └── legtimate-58w.txt
├── model_train/                   # Saved CNN models
│   └── model-cc_1.pkl
├── train.py                       # CNN training
├── CNN_process.py                 # CNN feature extraction
├── feature_extractor.py           # Handcrafted features
├── final_train.py                 # Ensemble training
├── final_train_revised.py         # Enhanced evaluation
├── test.py                        # Model testing
├── model.py                       # CNN architecture
├── utils.py                       # Utilities
├── vocab.py                       # Vocabulary management
├── vocab.txt                      # Pre-built vocabulary
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── INSTALL.md                     # Installation guide
├── USAGE.md                       # Usage guide
├── ARCHITECTURE.md                # Architecture details
└── DATA_FORMAT.md                 # Data format specification
```

## Citation

If you use BiPhish in your research, please cite:

```bibtex
@article{biphish2024,
  title={BiPhish: A Dual-Channel Phishing Detection Approach},
  author={[Author Names]},
  journal={[Journal/Conference Name]},
  year={2024}
}
```

*(Please contact the authors for the complete citation information)*

## License

[License Type] - See [LICENSE](LICENSE) file for details

## Contact

For questions, dataset access, or collaborations, please contact:
- [Author Email]

## Acknowledgments

- Dataset: SOTAPhish-1.2M
- Built with PyTorch and scikit-learn

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure `data/phish-58w.txt` and `data/legtimate-58w.txt` exist
2. **WHOIS lookup failures**: Network connectivity required; some lookups may timeout
3. **SSL certificate errors**: Some URLs may have invalid certificates (expected)
4. **Memory issues**: Large datasets may require batching or more RAM

For more troubleshooting tips, see [USAGE.md](USAGE.md).

## Version History

- **v1.0**: Initial release with dual-channel architecture
- **v1.1**: Added `final_train_revised.py` with enhanced evaluation

## Future Work

- Real-time URL scanning API
- Model compression for deployment
- Additional deep learning architectures (LSTM, Transformer)
- Transfer learning from pre-trained models
