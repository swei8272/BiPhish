import torch
from torch.utils.data import DataLoader, Dataset
from vocab import read_vocab
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from utils import get_char_ngrams, url_cut
import pandas as pd
from tqdm.auto import tqdm
from model import CNN
import os

# Load vocab once globally
vocab = read_vocab("./vocab.txt")


class CnnDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(examples):
    """Collate function for feature extraction"""
    inputs = [torch.tensor(ex) for ex in examples]
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs


def process_urls_to_indices(urls, max_len=100):
    """
    Convert URLs (either raw strings or n-grams) into token indices.

    Args:
        urls: List of URLs (can be strings or lists of n-grams)
        max_len: Maximum sequence length

    Returns:
        List of token index sequences
    """
    processed_data = []
    for url in urls:
        # Check if already processed as n-grams (list) or raw string
        if isinstance(url, list):
            # Already n-grams from load_data()
            ngrams = url
        else:
            # Raw string - convert to n-grams
            ngrams = get_char_ngrams(url)

        # Cut to max length
        ngrams = ngrams[:max_len]

        # Convert to IDs
        indices = vocab.convert_tokens_to_ids(ngrams)
        processed_data.append(indices)

    return processed_data


def get_part_feature(data_urls, model_path="./model_train/model-cc_1.pkl", batch_size=256):
    """
    Extract deep features from URLs using trained CNN model.

    Args:
        data_urls: List of URLs (strings or n-grams)
        model_path: Path to trained model
        batch_size: Batch size for processing

    Returns:
        numpy array of features (N x 128)
    """
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please train the model first using train.py"
        )

    # 1. Preprocess the URLs into indices
    print("Preprocessing URLs to token indices...")
    indices_data = process_urls_to_indices(data_urls, max_len=100)

    # 2. Create Dataset
    dataset = CnnDataset(indices_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Load Model
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    features_list = []

    # 4. Extract features
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting CNN Features"):
            x = batch.to(device)
            _ = model(x)  # Forward pass to populate model.features

            # Extract the features stored during forward pass
            batch_features = model.features.cpu().numpy()
            features_list.append(batch_features)

    # 5. Concatenate all batches
    if features_list:
        all_features = np.concatenate(features_list, axis=0)
        print(f"Extracted features shape: {all_features.shape}")
        return all_features
    else:
        print("Warning: No features extracted!")
        return np.array([])


# Main execution for generating CNN feature CSVs
if __name__ == "__main__":
    from utils import load_data

    print("=" * 60)
    print("CNN Feature Extraction Pipeline")
    print("=" * 60)

    # Paths
    Phishing_url_data_path = r"./data/phish-58w.txt"
    Legitimate_url_data_path = r"./data/legtimate-58w.txt"

    # Process Phishing URLs
    print("\n[1/2] Processing Phishing URLs...")
    try:
        phish_urls = load_data(Phishing_url_data_path)
        print(f"Loaded {len(phish_urls)} phishing URLs")

        phish_features = get_part_feature(phish_urls)
        phish_df = pd.DataFrame(phish_features)
        phish_df.to_csv('./Phishing_url_data_cnn.csv', index=False)
        print(f"✓ Saved to ./Phishing_url_data_cnn.csv")
    except Exception as e:
        print(f"✗ Error processing phishing URLs: {e}")

    # Process Legitimate URLs
    print("\n[2/2] Processing Legitimate URLs...")
    try:
        legit_urls = load_data(Legitimate_url_data_path)
        print(f"Loaded {len(legit_urls)} legitimate URLs")

        legit_features = get_part_feature(legit_urls)
        legit_df = pd.DataFrame(legit_features)
        legit_df.to_csv('./Legitimate_url_data_cnn.csv', index=False)
        print(f"✓ Saved to ./Legitimate_url_data_cnn.csv")
    except Exception as e:
        print(f"✗ Error processing legitimate URLs: {e}")

    print("\n" + "=" * 60)
    print("CNN Feature Extraction Complete!")
    print("=" * 60)